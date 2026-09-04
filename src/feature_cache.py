from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import Par4pcCase, load_hf_par4pc_cases, load_par4pc_dir
from src.patent_rerank import patent_specialized_feature_vectors


DEFAULT_FEATURE_CACHE_DIR = Path("data/cache/features")
FEATURE_NAMES = [
    "dense_score",
    "bm25_score",
    "field_dense_score",
    "field_lexical_score",
    "field_rarity_score",
    "coverage_score",
    "evidence_score",
]


def _cases_signature(cases: list[Par4pcCase]) -> str:
    payload = "||".join(str(case.source_path) for case in cases)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cache_stem(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    namespace: str | None,
) -> str:
    model_slug = embedding_model.split("/")[-1].replace(".", "_").replace("-", "_").lower()
    prefix = namespace or "cases"
    signature = _cases_signature(cases)
    return (
        f"{prefix}_{len(cases)}cases_{model_slug}"
        f"_qe{int(use_query_expansion)}_fq{int(use_focused_query)}_{signature}"
    )


def _paths_for_cache(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    cache_dir: str | Path,
    namespace: str | None,
) -> tuple[Path, Path]:
    root = Path(cache_dir)
    stem = _cache_stem(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        namespace=namespace,
    )
    return root / f"{stem}.parquet", root / f"{stem}.json"


def build_feature_row_dicts(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        feature_vectors = patent_specialized_feature_vectors(
            query_text=case.target_claim,
            candidates=list(case.candidates.values()),
            embedding_model=embedding_model,
            use_query_expansion=use_query_expansion,
            use_focused_query=use_focused_query,
        )
        for letter, candidate in sorted(case.candidates.items()):
            vector = feature_vectors[candidate.patent_id]
            row = {
                "case_index": case_index,
                "source_path": str(case.source_path),
                "application_number": case.application_number,
                "claim_number": case.claim_number,
                "letter": letter,
                "patent_id": candidate.patent_id,
                "is_gold": int(letter in case.gold_answers),
            }
            row.update(vector.as_dict())
            rows.append(row)
    return rows


def write_feature_cache(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    cache_dir: str | Path = DEFAULT_FEATURE_CACHE_DIR,
    namespace: str | None = None,
) -> tuple[Path, Path]:
    rows = build_feature_row_dicts(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
    )
    parquet_path, metadata_path = _paths_for_cache(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_dir=cache_dir,
        namespace=namespace,
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "use_query_expansion": use_query_expansion,
                "use_focused_query": use_focused_query,
                "namespace": namespace,
                "num_cases": len(cases),
                "num_rows": len(rows),
                "case_sources": [str(case.source_path) for case in cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parquet_path, metadata_path


def load_or_build_feature_rows(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    cache_dir: str | Path = DEFAULT_FEATURE_CACHE_DIR,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    parquet_path, _ = _paths_for_cache(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_dir=cache_dir,
        namespace=namespace,
    )
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).to_dict(orient="records")
    write_feature_cache(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_dir=cache_dir,
        namespace=namespace,
    )
    return pd.read_parquet(parquet_path).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild patent reranker feature caches.")
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--max-rows-per-split", type=int, default=100)
    parser.add_argument("--data-dir", default="../PANORAMA/data/benchmark/par4pc")
    parser.add_argument("--embedding-model", default="AI-Growth-Lab/PatentSBERTa")
    parser.add_argument("--use-query-expansion", action="store_true")
    parser.add_argument("--no-focused-query", action="store_true")
    parser.add_argument("--cache-dir", default=str(DEFAULT_FEATURE_CACHE_DIR))
    parser.add_argument("--namespace", default="")
    args = parser.parse_args()

    if args.source == "hf":
        cases = load_hf_par4pc_cases(
            splits=tuple(args.splits),
            max_rows_per_split=args.max_rows_per_split,
        )
    else:
        cases = load_par4pc_dir(args.data_dir)

    parquet_path, metadata_path = write_feature_cache(
        cases=cases,
        embedding_model=args.embedding_model,
        use_query_expansion=args.use_query_expansion,
        use_focused_query=not args.no_focused_query,
        cache_dir=args.cache_dir,
        namespace=args.namespace or None,
    )
    print(f"Wrote feature cache to {parquet_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
