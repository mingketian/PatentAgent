from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_par4pc_dir
from src.graph import run_graph
from src.patent_rerank import rank_candidates_hybrid_coverage, rank_candidates_patent_specialized
from src.retrieval import rank_candidates_bm25


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")


def evaluate(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    retrieval_method: str = "bm25",
    top_k: int = 3,
    embedding_model: str = "",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm_decompose: bool = False,
    llm_query_expansion: bool = False,
    llm_model: str = "",
) -> None:
    cases = load_par4pc_dir(data_dir)

    hit_at_1 = 0
    hit_at_3 = 0
    recall_at_3_total = 0.0
    exact_top_gold_count = 0

    print(f"Loaded {len(cases)} PAR4PC cases from {Path(data_dir)}")
    print(f"Retrieval method: {retrieval_method}")
    print()

    for case in cases:
        if retrieval_method == "bm25":
            ranked = rank_candidates_bm25(case)
        elif retrieval_method == "hybrid-coverage":
            ranked = rank_candidates_hybrid_coverage(
                case,
                top_k=top_k,
                embedding_model=embedding_model or "AI-Growth-Lab/PatentSBERTa",
            )
        elif retrieval_method == "patent-specialized":
            ranked = rank_candidates_patent_specialized(
                case,
                top_k=top_k,
                embedding_model=embedding_model or "AI-Growth-Lab/PatentSBERTa",
                use_llm_expansion=llm_query_expansion,
                use_llm_decompose=llm_decompose,
                llm_model=llm_model,
            )
        else:
            result = run_graph(
                case.source_path,
                top_k=top_k,
                retrieval_method=retrieval_method,
                llm_model=llm_model,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
            )
            ranked = result["ranked"]
        predicted = [result.letter for result in ranked]
        gold = set(case.gold_answers)
        top_1 = set(predicted[:1])
        top_n = set(predicted[:top_k])
        top_gold_count = set(predicted[: len(gold)])

        hit_at_1 += bool(top_1 & gold)
        hit_at_3 += bool(top_n & gold)
        recall_at_3_total += len(top_n & gold) / len(gold) if gold else 0.0
        exact_top_gold_count += top_gold_count == gold

        print(
            f"{case.source_path.name}: "
            f"claim={case.claim_number} gold={case.gold_answers} "
            f"top{top_k}={predicted[:top_k]}"
        )

    n = len(cases)
    print()
    print(f"{retrieval_method} retrieval")
    print(f"hit@1: {hit_at_1 / n:.3f}")
    print(f"hit@{top_k}: {hit_at_3 / n:.3f}")
    print(f"recall@{top_k}: {recall_at_3_total / n:.3f}")
    print(f"exact@|gold|: {exact_top_gold_count / n:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval on PANORAMA PAR4PC samples.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval-method",
        choices=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized", "linear-patent-reranker", "local-cross-encoder", "openai-embedding", "llm-rerank"],
        default="bm25",
    )
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--llm-decompose", action="store_true")
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()
    evaluate(
        args.data_dir,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        llm_decompose=args.llm_decompose,
        llm_query_expansion=args.llm_query_expansion,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
