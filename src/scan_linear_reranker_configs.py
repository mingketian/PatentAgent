from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data_loader import load_hf_par4pc_cases
from src.feature_cache import load_or_build_feature_rows
from src.retrieval import rank_candidates_local_embeddings
from src.train_linear_patent_reranker import (
    DEFAULT_LINEAR_C,
    DEFAULT_LINEAR_CLASS_WEIGHT,
    DEFAULT_LINEAR_FEATURE_NAMES,
    DEFAULT_LINEAR_SOLVER,
    _fit_model_with_params,
)


DEFAULT_OUTPUT = Path("outputs/linear_reranker_scan.csv")


def _evaluate_rankings(rankings: list[tuple[list[str], set[str]]], top_k: int = 3) -> dict[str, float]:
    hit_at_1 = 0
    hit_at_k = 0
    recall_at_k = 0.0
    exact_at_gold = 0
    for predicted, gold in rankings:
        top_1 = set(predicted[:1])
        top_n = set(predicted[:top_k])
        top_gold_count = set(predicted[: len(gold)])
        hit_at_1 += bool(top_1 & gold)
        hit_at_k += bool(top_n & gold)
        recall_at_k += len(top_n & gold) / len(gold) if gold else 0.0
        exact_at_gold += top_gold_count == gold
    n = len(rankings)
    return {
        "hit@1": hit_at_1 / n,
        f"hit@{top_k}": hit_at_k / n,
        f"recall@{top_k}": recall_at_k / n,
        "exact@|gold|": exact_at_gold / n,
    }


def _baseline_metrics(cases, embedding_model: str) -> dict[str, float]:
    rankings: list[tuple[list[str], set[str]]] = []
    for case in cases:
        ranked = rank_candidates_local_embeddings(case, top_k=3, embedding_model=embedding_model)
        rankings.append(([item.letter for item in ranked], set(case.gold_answers)))
    return _evaluate_rankings(rankings)


def _group_feature_rows(row_dicts: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in row_dicts:
        grouped.setdefault(int(row["case_index"]), []).append(row)
    return grouped


def _train_matrix(grouped_rows, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    for _, rows in grouped_rows.items():
        for row in rows:
            X_rows.append([float(row[name]) for name in feature_names])
            y_rows.append(int(row["is_gold"]))
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32)


def _eval_with_model(cases, grouped_rows, model, feature_names: list[str]) -> dict[str, float]:
    rankings: list[tuple[list[str], set[str]]] = []
    for case_index, case in enumerate(cases):
        rows = grouped_rows[case_index]
        rows_by_letter = {str(row["letter"]): row for row in rows}
        letters = sorted(case.candidates)
        X_case = np.asarray(
            [
                [float(rows_by_letter[letter][name]) for name in feature_names]
                for letter in letters
            ],
            dtype=np.float32,
        )
        probs = model.predict_proba(X_case)[:, 1]
        ranked_letters = [letter for letter, _ in sorted(zip(letters, probs, strict=True), key=lambda item: item[1], reverse=True)]
        rankings.append((ranked_letters[:3], set(case.gold_answers)))
    return _evaluate_rankings(rankings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan train-size / feature-set configs for the linear patent reranker.")
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--train-rows", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--eval-splits", nargs="+", default=["validation"])
    parser.add_argument("--eval-rows", type=int, default=100)
    parser.add_argument("--embedding-model", default="AI-Growth-Lab/PatentSBERTa")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    eval_cases = load_hf_par4pc_cases(
        splits=tuple(args.eval_splits),
        max_rows_per_split=args.eval_rows,
    )
    eval_rows = load_or_build_feature_rows(
        cases=eval_cases,
        embedding_model=args.embedding_model,
        use_query_expansion=False,
        use_focused_query=True,
        namespace=f"scan_eval_{len(eval_cases)}cases",
    )
    grouped_eval = _group_feature_rows(eval_rows)
    baseline = _baseline_metrics(eval_cases, embedding_model=args.embedding_model)
    print("local-embedding baseline", {key: round(value, 3) for key, value in baseline.items()})

    feature_sets = [
        ("dense+bm25+lexical", ["dense_score", "bm25_score", "field_lexical_score"]),
        ("dense+bm25+lexical+rarity", ["dense_score", "bm25_score", "field_lexical_score", "field_rarity_score"]),
        ("default_linear", list(DEFAULT_LINEAR_FEATURE_NAMES)),
    ]

    rows: list[dict[str, object]] = []
    for train_rows in args.train_rows:
        train_cases = load_hf_par4pc_cases(
            splits=tuple(args.train_splits),
            max_rows_per_split=train_rows,
        )
        train_rows_dicts = load_or_build_feature_rows(
            cases=train_cases,
            embedding_model=args.embedding_model,
            use_query_expansion=False,
            use_focused_query=True,
            namespace=f"scan_train_{train_rows}rows",
        )
        grouped_train = _group_feature_rows(train_rows_dicts)
        for feature_set_name, feature_names in feature_sets:
            X_train, y_train = _train_matrix(grouped_train, feature_names)
            model = _fit_model_with_params(
                X_train,
                y_train,
                solver=DEFAULT_LINEAR_SOLVER,
                C=DEFAULT_LINEAR_C,
                class_weight=DEFAULT_LINEAR_CLASS_WEIGHT,
            )
            metrics = _eval_with_model(eval_cases, grouped_eval, model, feature_names)
            row = {
                "train_rows": train_rows,
                "feature_set": feature_set_name,
                "num_features": len(feature_names),
                "solver": DEFAULT_LINEAR_SOLVER,
                "c_value": DEFAULT_LINEAR_C,
                "class_weight": str(DEFAULT_LINEAR_CLASS_WEIGHT),
                **{key: round(value, 3) for key, value in metrics.items()},
            }
            rows.append(row)
            print(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote scan results to {output_path}")


if __name__ == "__main__":
    main()
