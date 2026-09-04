from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from src.data_loader import load_par4pc_dir
from src.graph import run_graph
from src.retrieval import rank_candidates_bm25


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")
DEFAULT_OUTPUT = Path("outputs/retrieval_comparison.csv")


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    method: str
    embedding_model: str = ""
    reranker_model: str = ""


def evaluate_config(
    config: RetrievalConfig,
    data_dir: str | Path,
    top_k: int,
) -> dict[str, object]:
    cases = load_par4pc_dir(data_dir)
    hit_at_1 = 0
    hit_at_k = 0
    recall_at_k = 0.0
    exact_at_gold = 0
    predictions: list[str] = []

    for case in cases:
        if config.method == "bm25":
            ranked = rank_candidates_bm25(case)
        else:
            state = run_graph(
                case.source_path,
                top_k=top_k,
                retrieval_method=config.method,
                embedding_model=config.embedding_model,
                reranker_model=config.reranker_model,
            )
            ranked = state["ranked"]

        predicted = [result.letter for result in ranked]
        gold = set(case.gold_answers)
        hit_at_1 += bool(set(predicted[:1]) & gold)
        hit_at_k += bool(set(predicted[:top_k]) & gold)
        recall_at_k += len(set(predicted[:top_k]) & gold) / len(gold) if gold else 0.0
        exact_at_gold += set(predicted[: len(gold)]) == gold
        predictions.append(f"{case.source_path.name}:{'/'.join(predicted[:top_k])}")

    n = len(cases)
    return {
        "name": config.name,
        "method": config.method,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "num_cases": n,
        "hit@1": round(hit_at_1 / n, 3),
        f"hit@{top_k}": round(hit_at_k / n, 3),
        f"recall@{top_k}": round(recall_at_k / n, 3),
        "exact@|gold|": round(exact_at_gold / n, 3),
        "predictions": " ; ".join(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local retrieval methods on PAR4PC samples.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    configs = [
        RetrievalConfig(name="BM25", method="bm25"),
        RetrievalConfig(
            name="Hybrid Coverage",
            method="hybrid-coverage",
            embedding_model="AI-Growth-Lab/PatentSBERTa",
        ),
        RetrievalConfig(
            name="Patent Specialized",
            method="patent-specialized",
            embedding_model="AI-Growth-Lab/PatentSBERTa",
        ),
        RetrievalConfig(
            name="Linear Patent Reranker",
            method="linear-patent-reranker",
            embedding_model="AI-Growth-Lab/PatentSBERTa",
        ),
        RetrievalConfig(
            name="PatentSBERTa",
            method="local-embedding",
            embedding_model="AI-Growth-Lab/PatentSBERTa",
        ),
        RetrievalConfig(
            name="PaECTER",
            method="local-embedding",
            embedding_model="mpi-inno-comp/paecter",
        ),
        RetrievalConfig(
            name="BGE-small",
            method="local-embedding",
            embedding_model="BAAI/bge-small-en-v1.5",
        ),
        RetrievalConfig(
            name="MSMARCO MiniLM CrossEncoder",
            method="local-cross-encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        RetrievalConfig(
            name="BGE Reranker Base",
            method="local-cross-encoder",
            reranker_model="BAAI/bge-reranker-base",
        ),
    ]

    rows = [evaluate_config(config, args.data_dir, args.top_k) for config in configs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['name']}: hit@1={row['hit@1']:.3f}, "
            f"hit@{args.top_k}={row[f'hit@{args.top_k}']:.3f}, "
            f"recall@{args.top_k}={row[f'recall@{args.top_k}']:.3f}, "
            f"exact@|gold|={row['exact@|gold|']:.3f}"
        )
    print(f"Wrote comparison to {output}")


if __name__ == "__main__":
    main()
