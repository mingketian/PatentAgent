from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_hf_par4pc_cases
from src.patent_rerank import rank_candidates_hybrid_coverage, rank_candidates_patent_specialized
from src.retrieval import rank_candidates_bm25, rank_candidates_local_embeddings
from src.train_linear_patent_reranker import rank_case_with_default_linear_reranker


def evaluate_cases(cases, retrieval_method: str, top_k: int, embedding_model: str) -> dict[str, float]:
    hit_at_1 = 0
    hit_at_k = 0
    recall_at_k = 0.0
    exact_at_gold = 0

    for case in cases:
        if retrieval_method == "bm25":
            ranked = rank_candidates_bm25(case, top_k=top_k)
        elif retrieval_method == "local-embedding":
            ranked = rank_candidates_local_embeddings(
                case,
                top_k=top_k,
                embedding_model=embedding_model,
            )
        elif retrieval_method == "hybrid-coverage":
            ranked = rank_candidates_hybrid_coverage(
                case,
                top_k=top_k,
                embedding_model=embedding_model,
            )
        elif retrieval_method == "patent-specialized":
            ranked = rank_candidates_patent_specialized(
                case,
                top_k=top_k,
                embedding_model=embedding_model,
            )
        elif retrieval_method == "linear-patent-reranker":
            ranked_letters = rank_case_with_default_linear_reranker(
                case,
                top_k=top_k,
                embedding_model=embedding_model,
            )
            letter_to_result = {
                result.letter: result
                for result in rank_candidates_local_embeddings(
                    case,
                    top_k=None,
                    embedding_model=embedding_model,
                )
            }
            ranked = [
                letter_to_result[letter].__class__(
                    letter=letter_to_result[letter].letter,
                    score=score,
                    patent_id=letter_to_result[letter].patent_id,
                    title=letter_to_result[letter].title,
                )
                for letter, score in ranked_letters
            ]
        else:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

        predicted = [result.letter for result in ranked]
        gold = set(case.gold_answers)
        top_1 = set(predicted[:1])
        top_n = set(predicted[:top_k])
        top_gold_count = set(predicted[: len(gold)])

        hit_at_1 += bool(top_1 & gold)
        hit_at_k += bool(top_n & gold)
        recall_at_k += len(top_n & gold) / len(gold) if gold else 0.0
        exact_at_gold += top_gold_count == gold

    n = len(cases)
    return {
        "hit@1": hit_at_1 / n,
        f"hit@{top_k}": hit_at_k / n,
        f"recall@{top_k}": recall_at_k / n,
        "exact@|gold|": exact_at_gold / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAR4PC retrieval on larger HF splits.")
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--max-rows-per-split", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default="AI-Growth-Lab/PatentSBERTa")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized", "linear-patent-reranker"],
        choices=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized", "linear-patent-reranker"],
    )
    args = parser.parse_args()

    cases = load_hf_par4pc_cases(
        splits=tuple(args.splits),
        max_rows_per_split=args.max_rows_per_split,
    )
    print(
        f"Loaded {len(cases)} HF PAR4PC cases from splits={args.splits} "
        f"(max_rows_per_split={args.max_rows_per_split})."
    )

    for method in args.methods:
        metrics = evaluate_cases(
            cases=cases,
            retrieval_method=method,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
        )
        print(f"\n{method}")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
