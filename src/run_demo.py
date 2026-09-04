from __future__ import annotations

import argparse
from pathlib import Path

from src.graph import run_graph


DEFAULT_CASE = Path("../PANORAMA/data/benchmark/par4pc/par4pc_r00011_14704145_cl1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the patent prior-art agent MVP graph.")
    parser.add_argument("--case", default=str(DEFAULT_CASE))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="")
    parser.add_argument("--llm-decompose", action="store_true")
    parser.add_argument("--llm-verify", action="store_true")
    parser.add_argument(
        "--retrieval-method",
        choices=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized", "local-cross-encoder", "openai-embedding", "llm-rerank"],
        default="local-embedding",
    )
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    result = run_graph(
        args.case,
        top_k=args.top_k,
        use_llm_decompose=args.llm_decompose,
        use_llm_verify=args.llm_verify,
        retrieval_method=args.retrieval_method,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
    )
    report = result["report"]
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
