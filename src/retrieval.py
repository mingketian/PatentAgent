from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from src.data_loader import Par4pcCase, PatentCandidate


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True)
class RetrievalResult:
    letter: str
    score: float
    patent_id: str
    title: str


@dataclass(frozen=True)
class PatentSearchResult:
    patent_id: str
    title: str
    score: float
    candidate: PatentCandidate


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _sanitize_scores(scores) -> np.ndarray:
    return np.clip(
        np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-1.0, posinf=1.0, neginf=-1.0),
        -1.0,
        1.0,
    )


def _safe_cosine_scores(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    docs = np.asarray(docs, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    return _sanitize_scores(np.sum(docs * query, axis=1, dtype=np.float64))


def _result_for_letter(case: Par4pcCase, letter: str, score: float) -> RetrievalResult:
    candidate = case.candidates[letter]
    return RetrievalResult(
        letter=letter,
        score=float(score),
        patent_id=candidate.patent_id,
        title=candidate.title,
    )


def rank_candidates_bm25(case: Par4pcCase, top_k: int | None = None) -> list[RetrievalResult]:
    letters = sorted(case.candidates)
    corpus = [tokenize(case.candidates[letter].retrieval_text) for letter in letters]
    query = tokenize(case.target_claim)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)

    results = [_result_for_letter(case, letter, float(score)) for letter, score in zip(letters, scores, strict=True)]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def rank_candidates_openai_embeddings(
    case: Par4pcCase,
    top_k: int | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> list[RetrievalResult]:
    letters = sorted(case.candidates)
    texts = [case.target_claim] + [case.candidates[letter].retrieval_text for letter in letters]
    embeddings = OpenAIEmbeddings(model=embedding_model).embed_documents(texts)
    query = np.array(embeddings[0], dtype=np.float32)
    docs = np.array(embeddings[1:], dtype=np.float32)

    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    scores = _sanitize_scores(docs @ query / np.maximum(doc_norms * query_norm, 1e-12))

    results = [_result_for_letter(case, letter, float(score)) for letter, score in zip(letters, scores, strict=True)]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def _reranker_text(case: Par4pcCase, letter: str, max_claims: int = 6) -> str:
    candidate = case.candidates[letter]
    return "\n".join(
        part
        for part in [
            f"Title: {candidate.title}",
            f"Abstract: {candidate.abstract}",
            "Claims: " + " ".join(candidate.claims[:max_claims]),
        ]
        if part
    )


@lru_cache(maxsize=4)
def _cross_encoder_model(reranker_model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(reranker_model)


def rank_candidates_cross_encoder(
    case: Par4pcCase,
    top_k: int | None = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RetrievalResult]:
    letters = sorted(case.candidates)
    model = _cross_encoder_model(reranker_model)
    pairs = [(case.target_claim, _reranker_text(case, letter)) for letter in letters]
    scores = model.predict(pairs)

    results = [_result_for_letter(case, letter, float(score)) for letter, score in zip(letters, scores, strict=True)]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


@lru_cache(maxsize=4)
def _sentence_transformer_model(embedding_model: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model)


def rank_candidates_local_embeddings(
    case: Par4pcCase,
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
) -> list[RetrievalResult]:
    letters = sorted(case.candidates)
    model = _sentence_transformer_model(embedding_model)
    texts = [case.target_claim] + [case.candidates[letter].retrieval_text for letter in letters]
    embeddings = model.encode(texts, normalize_embeddings=True)
    query = np.array(embeddings[0], dtype=np.float32)
    docs = np.array(embeddings[1:], dtype=np.float32)
    scores = _safe_cosine_scores(docs, query)

    results = [_result_for_letter(case, letter, float(score)) for letter, score in zip(letters, scores, strict=True)]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def results_from_ordered_letters(
    case: Par4pcCase,
    ordered_letters: list[str],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    seen: set[str] = set()
    normalized: list[str] = []
    for letter in ordered_letters:
        letter = letter.strip().upper()
        if letter in case.candidates and letter not in seen:
            normalized.append(letter)
            seen.add(letter)
    for letter in sorted(case.candidates):
        if letter not in seen:
            normalized.append(letter)
            seen.add(letter)

    total = len(normalized)
    results = [
        _result_for_letter(case, letter, score=float(total - index))
        for index, letter in enumerate(normalized)
    ]
    return results[:top_k] if top_k is not None else results


def _patent_search_result(candidate: PatentCandidate, score: float) -> PatentSearchResult:
    return PatentSearchResult(
        patent_id=candidate.patent_id,
        title=candidate.title,
        score=float(score),
        candidate=candidate,
    )


def _candidate_texts(candidates: list[PatentCandidate]) -> tuple[str, ...]:
    return tuple(candidate.retrieval_text for candidate in candidates)


@lru_cache(maxsize=8)
def _cached_bm25(corpus_texts: tuple[str, ...]) -> BM25Okapi:
    return BM25Okapi([tokenize(text) for text in corpus_texts])


@lru_cache(maxsize=8)
def _cached_local_embeddings(embedding_model: str, corpus_texts: tuple[str, ...]) -> np.ndarray:
    model = _sentence_transformer_model(embedding_model)
    embeddings = model.encode(list(corpus_texts), normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def encode_texts_local_embeddings(
    texts: list[str] | tuple[str, ...],
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
) -> np.ndarray:
    return _cached_local_embeddings(embedding_model, tuple(texts))


def rank_patent_pool_bm25(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
) -> list[PatentSearchResult]:
    query = tokenize(query_text)
    bm25 = _cached_bm25(_candidate_texts(candidates))
    scores = bm25.get_scores(query)
    results = [
        _patent_search_result(candidate, float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def rank_patent_pool_local_embeddings(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
) -> list[PatentSearchResult]:
    model = _sentence_transformer_model(embedding_model)
    query = np.array(model.encode([query_text], normalize_embeddings=True)[0], dtype=np.float32)
    docs = _cached_local_embeddings(embedding_model, _candidate_texts(candidates))
    scores = _safe_cosine_scores(docs, query)
    results = [
        _patent_search_result(candidate, float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def rank_patent_pool_cross_encoder(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[PatentSearchResult]:
    model = _cross_encoder_model(reranker_model)
    pairs = [
        (
            query_text,
            "\n".join(
                part
                for part in [
                    f"Title: {candidate.title}",
                    f"Abstract: {candidate.abstract}",
                    "Claims: " + " ".join(candidate.claims[:6]),
                ]
                if part
            ),
        )
        for candidate in candidates
    ]
    scores = model.predict(pairs)
    results = [
        _patent_search_result(candidate, float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results
