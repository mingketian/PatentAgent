from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PatentCandidate
from src.retrieval import PatentSearchResult, _sentence_transformer_model


MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.parquet"
FAISS_FILENAME = "index.faiss"


@dataclass(frozen=True)
class PersistentIndexManifest:
    embedding_model: str
    patent_count: int
    dimension: int


def _manifest_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / MANIFEST_FILENAME


def _metadata_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / METADATA_FILENAME


def _faiss_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / FAISS_FILENAME


def index_exists(index_dir: str | Path) -> bool:
    root = Path(index_dir)
    return (
        _manifest_path(root).exists()
        and _metadata_path(root).exists()
        and _faiss_path(root).exists()
    )


def _candidate_rows(candidates: list[PatentCandidate]) -> list[dict[str, str]]:
    return [
        {
            "letter": candidate.letter,
            "patent_id": candidate.patent_id,
            "title": candidate.title,
            "abstract": candidate.abstract,
            "claims_json": json.dumps(candidate.claims),
            "retrieval_text": candidate.retrieval_text,
        }
        for candidate in candidates
    ]


def _row_to_candidate(row: pd.Series) -> PatentCandidate:
    claims_value = row["claims_json"]
    claims = json.loads(claims_value) if isinstance(claims_value, str) and claims_value else []
    return PatentCandidate(
        letter=str(row.get("letter", "")),
        patent_id=str(row.get("patent_id", "")),
        title=str(row.get("title", "")),
        abstract=str(row.get("abstract", "")),
        claims=claims,
    )


def _load_manifest(index_dir: str | Path) -> PersistentIndexManifest:
    payload = json.loads(_manifest_path(index_dir).read_text(encoding="utf-8"))
    return PersistentIndexManifest(
        embedding_model=str(payload["embedding_model"]),
        patent_count=int(payload["patent_count"]),
        dimension=int(payload["dimension"]),
    )


@lru_cache(maxsize=4)
def load_persistent_candidates(index_dir: str) -> list[PatentCandidate]:
    frame = pd.read_parquet(_metadata_path(index_dir))
    return [_row_to_candidate(row) for _, row in frame.iterrows()]


@lru_cache(maxsize=4)
def load_persistent_manifest(index_dir: str) -> PersistentIndexManifest:
    return _load_manifest(index_dir)


@lru_cache(maxsize=4)
def _load_faiss_index(index_dir: str):
    import faiss

    return faiss.read_index(str(_faiss_path(index_dir)))


def build_persistent_index(
    candidates: list[PatentCandidate],
    index_dir: str | Path,
    embedding_model: str,
    batch_size: int = 128,
) -> PersistentIndexManifest:
    import faiss

    output_dir = Path(index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [candidate.retrieval_text for candidate in candidates]
    model = _sentence_transformer_model(embedding_model)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    dimension = int(matrix.shape[1])

    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)
    faiss.write_index(index, str(_faiss_path(output_dir)))

    frame = pd.DataFrame(_candidate_rows(candidates))
    frame.to_parquet(_metadata_path(output_dir), index=False)

    manifest = PersistentIndexManifest(
        embedding_model=embedding_model,
        patent_count=len(candidates),
        dimension=dimension,
    )
    _manifest_path(output_dir).write_text(
        json.dumps(
            {
                "embedding_model": manifest.embedding_model,
                "patent_count": manifest.patent_count,
                "dimension": manifest.dimension,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    load_persistent_candidates.cache_clear()
    load_persistent_manifest.cache_clear()
    _load_faiss_index.cache_clear()
    return manifest


def search_persistent_index(
    query_text: str,
    index_dir: str | Path,
    top_k: int = 3,
    embedding_model: str = "",
) -> list[PatentSearchResult]:
    index_dir = str(index_dir)
    manifest = load_persistent_manifest(index_dir)
    effective_model = embedding_model or manifest.embedding_model
    if effective_model != manifest.embedding_model:
        raise ValueError(
            f"Persistent index was built with {manifest.embedding_model}, "
            f"but search requested {effective_model}."
        )

    model = _sentence_transformer_model(manifest.embedding_model)
    query = np.asarray(
        model.encode([query_text], normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    index = _load_faiss_index(index_dir)
    candidates = load_persistent_candidates(index_dir)
    top_k = min(top_k, len(candidates))
    scores, indices = index.search(query, top_k)

    results: list[PatentSearchResult] = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        candidate = candidates[int(idx)]
        results.append(
            PatentSearchResult(
                patent_id=candidate.patent_id,
                title=candidate.title,
                score=float(score),
                candidate=candidate,
            )
        )
    return results
