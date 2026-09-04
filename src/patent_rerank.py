from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from src.claim_analysis import PATENT_STOPWORDS, ClaimLimitation, decompose_claim_heuristic, rank_candidate_segments
from src.data_loader import Par4pcCase, PatentCandidate
from src.llm_tools import decompose_claim_llm, expand_query_llm, openai_available
from src.retrieval import (
    PatentSearchResult,
    RetrievalResult,
    encode_texts_local_embeddings,
    rank_patent_pool_bm25,
    rank_patent_pool_local_embeddings,
    tokenize,
)


@dataclass(frozen=True)
class PatentScoreBreakdown:
    retrieval_score: float
    bm25_score: float
    dense_score: float
    coverage_score: float
    evidence_score: float
    final_score: float


@dataclass(frozen=True)
class PatentFeatureVector:
    patent_id: str
    dense_score: float
    bm25_score: float
    field_dense_score: float
    field_lexical_score: float
    field_rarity_score: float
    coverage_score: float
    evidence_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "field_dense_score": self.field_dense_score,
            "field_lexical_score": self.field_lexical_score,
            "field_rarity_score": self.field_rarity_score,
            "coverage_score": self.coverage_score,
            "evidence_score": self.evidence_score,
        }


FIELD_WEIGHTS = {
    "title": 0.15,
    "abstract": 0.30,
    "claims": 0.55,
}

STRICT_PATENT_STOPWORDS = PATENT_STOPWORDS | {
    "access",
    "associated",
    "comprise",
    "comprises",
    "data",
    "determine",
    "determining",
    "event",
    "first",
    "following",
    "gathering",
    "identified",
    "identify",
    "include",
    "includes",
    "information",
    "instructions",
    "participant",
    "participants",
    "physical",
    "program",
    "provide",
    "providing",
    "receive",
    "received",
    "receiving",
    "requested",
    "request",
    "requests",
    "second",
}


def _normalize_scores(items: dict[str, float]) -> dict[str, float]:
    if not items:
        return {}
    values = list(items.values())
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        return {key: 1.0 for key in items}
    return {key: (value - low) / (high - low) for key, value in items.items()}


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in STRICT_PATENT_STOPWORDS and len(token) > 2
    }


def _focused_query_text(query_text: str) -> str:
    normalized = " ".join(query_text.split())
    lowered = normalized.lower()
    for marker in ("one or more of the following:", "selected from the group consisting of", "selected from the group consisting"):
        index = lowered.find(marker)
        if index >= 0:
            return normalized[index + len(marker) :].strip(" :;,.") or normalized
    for marker in (" wherein ", " wherein", "comprising:", "comprises:", "steps of:"):
        index = lowered.find(marker)
        if index >= 0:
            return normalized[index + len(marker) :].strip(" :;,.") or normalized
    return normalized


def _weighted_term_overlap(query_text: str, evidence_text: str) -> float:
    query_terms = _content_terms(query_text)
    if not query_terms:
        return 0.0
    evidence_terms = _content_terms(evidence_text)
    numerator = sum(1.0 + 0.1 * max(len(term) - 4, 0) for term in query_terms if term in evidence_terms)
    denominator = sum(1.0 + 0.1 * max(len(term) - 4, 0) for term in query_terms)
    return numerator / denominator if denominator else 0.0


def _query_phrases(query_text: str, max_phrases: int = 20) -> list[str]:
    tokens = [token for token in tokenize(query_text) if token not in PATENT_STOPWORDS and len(token) > 2]
    phrases: list[str] = []
    seen: set[str] = set()
    for size in (4, 3, 2):
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[index : index + size])
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def _phrase_overlap(query_text: str, evidence_text: str) -> float:
    phrases = _query_phrases(query_text)
    if not phrases:
        return 0.0
    haystack = " ".join(evidence_text.lower().split())
    matched = 0.0
    total = 0.0
    for phrase in phrases:
        weight = 1.0 + 0.25 * (len(phrase.split()) - 2)
        total += weight
        if phrase in haystack:
            matched += weight
    return matched / total if total else 0.0


def _lexical_match_score(query_text: str, evidence_text: str) -> float:
    term_score = _weighted_term_overlap(query_text, evidence_text)
    phrase_score = _phrase_overlap(query_text, evidence_text)
    return 0.7 * term_score + 0.3 * phrase_score


def _candidate_term_document_frequency(candidates: list[PatentCandidate]) -> dict[str, int]:
    document_frequency: dict[str, int] = {}
    for candidate in candidates:
        seen = _content_terms(candidate.retrieval_text)
        for term in seen:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    return document_frequency


def _rarity_overlap_score(
    query_text: str,
    evidence_text: str,
    document_frequency: dict[str, int],
    total_documents: int,
) -> float:
    query_terms = _content_terms(query_text)
    if not query_terms:
        return 0.0
    evidence_terms = _content_terms(evidence_text)
    numerator = 0.0
    denominator = 0.0
    for term in query_terms:
        df = document_frequency.get(term, 0)
        idf = log((total_documents + 1.0) / (df + 1.0)) + 1.0
        denominator += idf
        if term in evidence_terms:
            numerator += idf
    return numerator / denominator if denominator else 0.0


def expand_query_heuristic(query_text: str) -> list[str]:
    variants = [query_text]
    replacements = [
        ("participant", "attendee"),
        ("attendee", "participant"),
        ("request for information", "inquiry for contextual information"),
        ("contextual information", "personalized information"),
        ("profile", "data attributes"),
        ("physical gathering", "event"),
        ("access", "authorization"),
    ]
    lower = query_text.lower()
    for source, target in replacements:
        if source in lower:
            variants.append(lower.replace(source, target))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = " ".join(item.split())
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped


def _get_limitations(
    query_text: str,
    use_llm_decompose: bool = False,
    llm_model: str = "",
    use_focused_query: bool = True,
) -> list[ClaimLimitation]:
    focused = _focused_query_text(query_text) if use_focused_query else query_text
    if use_llm_decompose and openai_available():
        return decompose_claim_llm(focused, model=llm_model or None)
    return decompose_claim_heuristic(focused)


def _query_variants(
    query_text: str,
    use_query_expansion: bool,
    use_llm_expansion: bool = False,
    llm_model: str = "",
    use_focused_query: bool = True,
) -> list[str]:
    focused = _focused_query_text(query_text) if use_focused_query else query_text
    variants = [focused, query_text] if focused != query_text else [query_text]
    if use_query_expansion:
        variants.extend(expand_query_heuristic(query_text))
    if use_llm_expansion and openai_available():
        variants.extend(expand_query_llm(query_text, model=llm_model or None))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = " ".join(item.split())
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped


def _field_texts(candidate: PatentCandidate) -> dict[str, str]:
    return {
        "title": candidate.title,
        "abstract": candidate.abstract,
        "claims": " ".join(candidate.claims),
    }


def _field_aware_lexical_score(query_variants: list[str], candidate: PatentCandidate) -> float:
    total = 0.0
    fields = _field_texts(candidate)
    for field, text in fields.items():
        if not text.strip():
            continue
        best = 0.0
        for query in query_variants:
            score = _lexical_match_score(query, text)
            best = max(best, score)
        total += FIELD_WEIGHTS[field] * best
    return total


def _field_aware_rarity_score(
    query_variants: list[str],
    candidate: PatentCandidate,
    document_frequency: dict[str, int],
    total_documents: int,
) -> float:
    total = 0.0
    fields = _field_texts(candidate)
    for field, text in fields.items():
        if not text.strip():
            continue
        best = 0.0
        for query in query_variants:
            score = _rarity_overlap_score(query, text, document_frequency, total_documents)
            best = max(best, score)
        total += FIELD_WEIGHTS[field] * best
    return total


def _field_dense_scores(
    query_variants: list[str],
    candidates: list[PatentCandidate],
    embedding_model: str,
) -> dict[str, float]:
    if not candidates:
        return {}
    query_embeddings = encode_texts_local_embeddings(query_variants, embedding_model=embedding_model)
    results: dict[str, float] = {}
    for candidate in candidates:
        fields = _field_texts(candidate)
        field_names = [field for field, text in fields.items() if text.strip()]
        if not field_names:
            results[candidate.patent_id] = 0.0
            continue
        field_texts = [fields[field] for field in field_names]
        field_embeddings = encode_texts_local_embeddings(field_texts, embedding_model=embedding_model)
        similarities = np.clip(query_embeddings @ field_embeddings.T, -1.0, 1.0)
        best_by_field = similarities.max(axis=0)
        score = sum(
            FIELD_WEIGHTS[field] * float(best)
            for field, best in zip(field_names, best_by_field, strict=True)
        )
        results[candidate.patent_id] = score
    return results


def _limitation_coverage_score(query_text: str, candidate: PatentCandidate) -> tuple[float, float]:
    limitations = decompose_claim_heuristic(query_text)
    if not limitations:
        limitations = [type("Tmp", (), {"text": query_text})()]

    supported = 0.0
    best_segment = 0.0
    for limitation in limitations:
        ranked_segments = rank_candidate_segments(limitation.text, candidate)
        if not ranked_segments:
            continue
        source, evidence, segment_score = ranked_segments[0]
        best_segment = max(best_segment, segment_score)
        overlap = _lexical_match_score(limitation.text, evidence)
        supported += overlap

    coverage = supported / max(len(limitations), 1)
    return coverage, best_segment


def _limitation_fusion_score(
    limitations: list[ClaimLimitation],
    candidate: PatentCandidate,
) -> tuple[float, float]:
    if not limitations:
        return _limitation_coverage_score("", candidate)

    supported = 0.0
    best_segment = 0.0
    for limitation in limitations:
        ranked_segments = rank_candidate_segments(limitation.text, candidate)
        if not ranked_segments:
            continue
        source, evidence, segment_score = ranked_segments[0]
        best_segment = max(best_segment, segment_score)
        supported += _lexical_match_score(limitation.text, evidence)
    return supported / max(len(limitations), 1), best_segment


def _hybrid_breakdowns(
    query_text: str,
    candidates: list[PatentCandidate],
    embedding_model: str,
) -> dict[str, PatentScoreBreakdown]:
    bm25_ranked = rank_patent_pool_bm25(query_text, candidates, top_k=None)
    dense_ranked = rank_patent_pool_local_embeddings(
        query_text,
        candidates,
        top_k=None,
        embedding_model=embedding_model,
    )

    bm25_map = {item.patent_id: item.score for item in bm25_ranked}
    dense_map = {item.patent_id: item.score for item in dense_ranked}
    bm25_norm = _normalize_scores(bm25_map)
    dense_norm = _normalize_scores(dense_map)

    coverage_map: dict[str, float] = {}
    evidence_map: dict[str, float] = {}
    for candidate in candidates:
        coverage, evidence = _limitation_coverage_score(query_text, candidate)
        coverage_map[candidate.patent_id] = coverage
        evidence_map[candidate.patent_id] = evidence

    evidence_norm = _normalize_scores(evidence_map)
    breakdowns: dict[str, PatentScoreBreakdown] = {}
    for candidate in candidates:
        patent_id = candidate.patent_id
        final_score = (
            0.35 * dense_norm.get(patent_id, 0.0)
            + 0.25 * bm25_norm.get(patent_id, 0.0)
            + 0.30 * coverage_map.get(patent_id, 0.0)
            + 0.10 * evidence_norm.get(patent_id, 0.0)
        )
        breakdowns[patent_id] = PatentScoreBreakdown(
            retrieval_score=final_score,
            bm25_score=bm25_norm.get(patent_id, 0.0),
            dense_score=dense_norm.get(patent_id, 0.0),
            coverage_score=coverage_map.get(patent_id, 0.0),
            evidence_score=evidence_norm.get(patent_id, 0.0),
            final_score=final_score,
        )
    return breakdowns


def rank_patent_pool_hybrid_coverage(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
) -> list[PatentSearchResult]:
    breakdowns = _hybrid_breakdowns(query_text, candidates, embedding_model=embedding_model)
    results = [
        PatentSearchResult(
            patent_id=candidate.patent_id,
            title=candidate.title,
            score=breakdowns[candidate.patent_id].final_score,
            candidate=candidate,
        )
        for candidate in candidates
    ]
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def rank_patent_pool_patent_specialized(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    use_query_expansion: bool = True,
    use_llm_expansion: bool = False,
    use_llm_decompose: bool = False,
    llm_model: str = "",
    use_focused_query: bool = True,
    use_field_dense: bool = True,
    use_field_lexical: bool = True,
    use_field_rarity: bool = True,
    use_limitation_fusion: bool = True,
    use_evidence_score: bool = True,
) -> list[PatentSearchResult]:
    feature_vectors = patent_specialized_feature_vectors(
        query_text=query_text,
        candidates=candidates,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_llm_expansion=use_llm_expansion,
        use_llm_decompose=use_llm_decompose,
        llm_model=llm_model,
        use_focused_query=use_focused_query,
    )

    results = []
    for candidate in candidates:
        patent_id = candidate.patent_id
        feature = feature_vectors[patent_id]
        weighted_parts = [
            (0.35, feature.dense_score, True),
            (0.10, feature.bm25_score, True),
            (0.10, feature.field_dense_score, use_field_dense),
            (0.15, feature.field_lexical_score, use_field_lexical),
            (0.20, feature.field_rarity_score, use_field_rarity),
            (0.05, feature.coverage_score, use_limitation_fusion),
            (0.05, feature.evidence_score, use_evidence_score),
        ]
        active_weight = sum(weight for weight, _, enabled in weighted_parts if enabled)
        score = sum(weight * value for weight, value, enabled in weighted_parts if enabled) / max(active_weight, 1e-9)
        results.append(
            PatentSearchResult(
                patent_id=patent_id,
                title=candidate.title,
                score=score,
                candidate=candidate,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k] if top_k is not None else results


def patent_specialized_feature_vectors(
    query_text: str,
    candidates: list[PatentCandidate],
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    use_query_expansion: bool = True,
    use_llm_expansion: bool = False,
    use_llm_decompose: bool = False,
    llm_model: str = "",
    use_focused_query: bool = True,
) -> dict[str, PatentFeatureVector]:
    query_variants = _query_variants(
        query_text,
        use_query_expansion=use_query_expansion,
        use_llm_expansion=use_llm_expansion,
        llm_model=llm_model,
        use_focused_query=use_focused_query,
    )
    limitations = _get_limitations(
        query_text,
        use_llm_decompose=use_llm_decompose,
        llm_model=llm_model,
        use_focused_query=use_focused_query,
    )

    base_dense = rank_patent_pool_local_embeddings(
        query_text,
        candidates,
        top_k=None,
        embedding_model=embedding_model,
    )
    base_bm25 = rank_patent_pool_bm25(query_text, candidates, top_k=None)
    dense_map = {item.patent_id: item.score for item in base_dense}
    bm25_map = {item.patent_id: item.score for item in base_bm25}
    dense_norm = _normalize_scores(dense_map)
    bm25_norm = _normalize_scores(bm25_map)

    field_dense_map = _field_dense_scores(query_variants, candidates, embedding_model)
    field_lexical_map: dict[str, float] = {}
    field_rarity_map: dict[str, float] = {}
    coverage_map: dict[str, float] = {}
    evidence_map: dict[str, float] = {}
    document_frequency = _candidate_term_document_frequency(candidates)
    total_documents = max(len(candidates), 1)
    for candidate in candidates:
        patent_id = candidate.patent_id
        field_lexical_map[patent_id] = _field_aware_lexical_score(query_variants, candidate)
        field_rarity_map[patent_id] = _field_aware_rarity_score(
            query_variants,
            candidate,
            document_frequency,
            total_documents,
        )
        coverage, evidence = _limitation_fusion_score(limitations, candidate)
        coverage_map[patent_id] = coverage
        evidence_map[patent_id] = evidence

    field_dense_norm = _normalize_scores(field_dense_map)
    field_lexical_norm = _normalize_scores(field_lexical_map)
    field_rarity_norm = _normalize_scores(field_rarity_map)
    evidence_norm = _normalize_scores(evidence_map)

    features: dict[str, PatentFeatureVector] = {}
    for candidate in candidates:
        patent_id = candidate.patent_id
        features[patent_id] = PatentFeatureVector(
            patent_id=patent_id,
            dense_score=dense_norm.get(patent_id, 0.0),
            bm25_score=bm25_norm.get(patent_id, 0.0),
            field_dense_score=field_dense_norm.get(patent_id, 0.0),
            field_lexical_score=field_lexical_norm.get(patent_id, 0.0),
            field_rarity_score=field_rarity_norm.get(patent_id, 0.0),
            coverage_score=coverage_map.get(patent_id, 0.0),
            evidence_score=evidence_norm.get(patent_id, 0.0),
        )
    return features


def rank_candidates_hybrid_coverage(
    case: Par4pcCase,
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
) -> list[RetrievalResult]:
    candidates = list(case.candidates.values())
    hybrid_ranked = rank_patent_pool_hybrid_coverage(
        case.target_claim,
        candidates,
        top_k=top_k,
        embedding_model=embedding_model,
    )
    letter_by_patent_id = {candidate.patent_id: candidate.letter for candidate in candidates}
    results = [
        RetrievalResult(
            letter=letter_by_patent_id[item.patent_id],
            score=item.score,
            patent_id=item.patent_id,
            title=item.title,
        )
        for item in hybrid_ranked
    ]
    return results


def rank_candidates_patent_specialized(
    case: Par4pcCase,
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    use_query_expansion: bool = True,
    use_llm_expansion: bool = False,
    use_llm_decompose: bool = False,
    llm_model: str = "",
    use_focused_query: bool = True,
    use_field_dense: bool = True,
    use_field_lexical: bool = True,
    use_field_rarity: bool = True,
    use_limitation_fusion: bool = True,
    use_evidence_score: bool = True,
) -> list[RetrievalResult]:
    candidates = list(case.candidates.values())
    ranked = rank_patent_pool_patent_specialized(
        case.target_claim,
        candidates,
        top_k=top_k,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_llm_expansion=use_llm_expansion,
        use_llm_decompose=use_llm_decompose,
        llm_model=llm_model,
        use_focused_query=use_focused_query,
        use_field_dense=use_field_dense,
        use_field_lexical=use_field_lexical,
        use_field_rarity=use_field_rarity,
        use_limitation_fusion=use_limitation_fusion,
        use_evidence_score=use_evidence_score,
    )
    letter_by_patent_id = {candidate.patent_id: candidate.letter for candidate in candidates}
    return [
        RetrievalResult(
            letter=letter_by_patent_id[item.patent_id],
            score=item.score,
            patent_id=item.patent_id,
            title=item.title,
        )
        for item in ranked
    ]
