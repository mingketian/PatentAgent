from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import combine_patent_pools, load_hf_par4pc_patent_pool, load_unique_patent_pool
from src.persistent_index import build_persistent_index


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")
DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_demo")
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent local patent FAISS index.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--pool-source",
        choices=["local", "hub", "combined"],
        default="hub",
    )
    parser.add_argument(
        "--hub-rows-per-split",
        type=int,
        default=2000,
        help="0 loads full train/validation/test splits.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    local_pool = load_unique_patent_pool(args.data_dir)
    hf_limit = None if args.hub_rows_per_split <= 0 else args.hub_rows_per_split

    if args.pool_source == "local":
        candidates = local_pool
    elif args.pool_source == "hub":
        candidates = load_hf_par4pc_patent_pool(max_rows_per_split=hf_limit)
    else:
        candidates = combine_patent_pools(
            local_pool,
            load_hf_par4pc_patent_pool(max_rows_per_split=hf_limit),
        )

    manifest = build_persistent_index(
        candidates=candidates,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(
        f"Built index at {args.index_dir} with {manifest.patent_count} patents "
        f"using {manifest.embedding_model} (dim={manifest.dimension})."
    )


if __name__ == "__main__":
    main()
