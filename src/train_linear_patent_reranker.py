from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import Par4pcCase, PatentCandidate, load_hf_par4pc_cases
from src.feature_cache import load_or_build_feature_rows
from src.patent_rerank import patent_specialized_feature_vectors, rank_candidates_patent_specialized
from src.retrieval import PatentSearchResult, rank_candidates_local_embeddings


FEATURE_ORDER = [
    "dense_score",
    "bm25_score",
    "field_lexical_score",
    "field_rarity_score",
    "coverage_score",
    "evidence_score",
    "field_dense_score",
]

DEFAULT_LINEAR_FEATURE_NAMES = [
    "dense_score",
    "bm25_score",
    "field_lexical_score",
]
DEFAULT_LINEAR_SOLVER = "liblinear"
DEFAULT_LINEAR_C = 4.0
DEFAULT_LINEAR_CLASS_WEIGHT: str | None = None
DEFAULT_LINEAR_TRAIN_SPLITS = ("train",)
DEFAULT_LINEAR_TRAIN_MAX_ROWS = 200
DEFAULT_LINEAR_MODEL_PATH = Path("data/models/linear_patent_reranker_patentsberta_train200_3feat.joblib")


@dataclass(frozen=True)
class CandidateRow:
    case_index: int
    letter: str
    is_gold: int
    features: dict[str, float]


def _build_candidate_rows(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    cache_namespace: str | None = None,
) -> list[CandidateRow]:
    row_dicts = load_or_build_feature_rows(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        namespace=cache_namespace,
    )
    rows: list[CandidateRow] = []
    for row in row_dicts:
        rows.append(
            CandidateRow(
                case_index=int(row["case_index"]),
                letter=str(row["letter"]),
                is_gold=int(row["is_gold"]),
                features={name: float(row[name]) for name in FEATURE_ORDER},
            )
        )
    return rows


def _rows_to_matrix(rows: list[CandidateRow], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[row.features[name] for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([row.is_gold for row in rows], dtype=np.int32)
    return X, y


def _metrics_from_rankings(rankings: list[tuple[list[str], set[str]]], top_k: int = 3) -> dict[str, float]:
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


def _evaluate_baseline(cases: list[Par4pcCase], method: str, embedding_model: str) -> dict[str, float]:
    rankings: list[tuple[list[str], set[str]]] = []
    for case in cases:
        if method == "local-embedding":
            ranked = rank_candidates_local_embeddings(case, top_k=3, embedding_model=embedding_model)
        elif method == "patent-specialized":
            ranked = rank_candidates_patent_specialized(case, top_k=3, embedding_model=embedding_model)
        else:
            raise ValueError(method)
        rankings.append(([item.letter for item in ranked], set(case.gold_answers)))
    return _metrics_from_rankings(rankings)


def _fit_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    return _fit_model_with_params(X, y)


def _fit_model_with_params(
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = "liblinear",
    C: float = 1.0,
    class_weight: str | None = "balanced",
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver=solver,
                    C=C,
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def _cross_validated_rankings(
    cases: list[Par4pcCase],
    rows: list[CandidateRow],
    feature_names: list[str],
    n_splits: int,
    *,
    solver: str = "liblinear",
    C: float = 1.0,
    class_weight: str | None = "balanced",
) -> list[tuple[list[str], set[str]]]:
    if not cases:
        return []
    case_indices = np.arange(len(cases))
    split_count = min(n_splits, len(cases))
    splitter = GroupKFold(n_splits=split_count)
    rankings: list[tuple[list[str], set[str]]] = []
    rows_by_case: dict[int, list[CandidateRow]] = {}
    for row in rows:
        rows_by_case.setdefault(row.case_index, []).append(row)

    for train_idx, test_idx in splitter.split(case_indices, groups=case_indices):
        train_rows = [row for idx in train_idx for row in rows_by_case[idx]]
        X_train = np.array([[row.features[name] for name in feature_names] for row in train_rows], dtype=np.float32)
        y_train = np.array([row.is_gold for row in train_rows], dtype=np.int32)
        model = _fit_model_with_params(
            X_train,
            y_train,
            solver=solver,
            C=C,
            class_weight=class_weight,
        )

        for case_index in test_idx:
            case_rows = rows_by_case[case_index]
            X_case = np.array([[row.features[name] for name in feature_names] for row in case_rows], dtype=np.float32)
            probs = model.predict_proba(X_case)[:, 1]
            ranked_letters = [
                row.letter
                for row, _ in sorted(
                    zip(case_rows, probs, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            rankings.append((ranked_letters[:3], set(cases[case_index].gold_answers)))
    return rankings


def evaluate_forward_selection(
    cases: list[Par4pcCase],
    embedding_model: str,
    n_splits: int,
    use_query_expansion: bool,
    use_focused_query: bool,
    solver: str = "liblinear",
    C: float = 1.0,
    class_weight: str | None = "balanced",
) -> list[dict[str, object]]:
    rows = _build_candidate_rows(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_namespace=f"linear_cv_{len(cases)}cases",
    )
    output_rows: list[dict[str, object]] = []
    for prefix_end in range(1, len(FEATURE_ORDER) + 1):
        feature_names = FEATURE_ORDER[:prefix_end]
        rankings = _cross_validated_rankings(
            cases,
            rows,
            feature_names,
            n_splits=n_splits,
            solver=solver,
            C=C,
            class_weight=class_weight,
        )
        metrics = _metrics_from_rankings(rankings)
        output_rows.append(
            {
                "feature_set": " + ".join(feature_names),
                "num_features": len(feature_names),
                **{key: round(value, 3) for key, value in metrics.items()},
            }
        )
    return output_rows


def evaluate_single_feature_set(
    cases: list[Par4pcCase],
    embedding_model: str,
    n_splits: int,
    use_query_expansion: bool,
    use_focused_query: bool,
    feature_names: list[str],
    solver: str,
    C: float,
    class_weight: str | None,
) -> dict[str, float]:
    rows = _build_candidate_rows(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_namespace=f"linear_single_{len(cases)}cases",
    )
    rankings = _cross_validated_rankings(
        cases,
        rows,
        feature_names,
        n_splits=n_splits,
        solver=solver,
        C=C,
        class_weight=class_weight,
    )
    return _metrics_from_rankings(rankings)


def train_linear_reranker_from_cases(
    cases: list[Par4pcCase],
    embedding_model: str,
    use_query_expansion: bool,
    use_focused_query: bool,
    feature_names: list[str],
    solver: str,
    C: float,
    class_weight: str | None,
) -> Pipeline:
    rows = _build_candidate_rows(
        cases=cases,
        embedding_model=embedding_model,
        use_query_expansion=use_query_expansion,
        use_focused_query=use_focused_query,
        cache_namespace=f"linear_train_{len(cases)}cases",
    )
    X, y = _rows_to_matrix(rows, feature_names)
    return _fit_model_with_params(
        X,
        y,
        solver=solver,
        C=C,
        class_weight=class_weight,
    )


def _linear_model_metadata(
    *,
    embedding_model: str,
    train_splits: tuple[str, ...],
    max_rows_per_split: int,
) -> dict[str, object]:
    return {
        "embedding_model": embedding_model,
        "feature_names": DEFAULT_LINEAR_FEATURE_NAMES,
        "solver": DEFAULT_LINEAR_SOLVER,
        "c_value": DEFAULT_LINEAR_C,
        "class_weight": DEFAULT_LINEAR_CLASS_WEIGHT,
        "train_splits": list(train_splits),
        "max_rows_per_split": max_rows_per_split,
    }


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def save_linear_reranker(
    model: Pipeline,
    *,
    model_path: str | Path,
    embedding_model: str,
    train_splits: tuple[str, ...],
    max_rows_per_split: int,
) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    _metadata_path(path).write_text(
        json.dumps(
            _linear_model_metadata(
                embedding_model=embedding_model,
                train_splits=train_splits,
                max_rows_per_split=max_rows_per_split,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def train_and_save_default_linear_reranker(
    *,
    model_path: str | Path = DEFAULT_LINEAR_MODEL_PATH,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    train_splits: tuple[str, ...] = DEFAULT_LINEAR_TRAIN_SPLITS,
    max_rows_per_split: int = DEFAULT_LINEAR_TRAIN_MAX_ROWS,
) -> Path:
    train_cases = load_hf_par4pc_cases(
        splits=train_splits,
        max_rows_per_split=max_rows_per_split,
    )
    model = train_linear_reranker_from_cases(
        cases=train_cases,
        embedding_model=embedding_model,
        use_query_expansion=False,
        use_focused_query=True,
        feature_names=DEFAULT_LINEAR_FEATURE_NAMES,
        solver=DEFAULT_LINEAR_SOLVER,
        C=DEFAULT_LINEAR_C,
        class_weight=DEFAULT_LINEAR_CLASS_WEIGHT,
    )
    return save_linear_reranker(
        model,
        model_path=model_path,
        embedding_model=embedding_model,
        train_splits=train_splits,
        max_rows_per_split=max_rows_per_split,
    )


def load_linear_reranker(model_path: str | Path) -> Pipeline:
    return joblib.load(Path(model_path))


@lru_cache(maxsize=4)
def get_default_linear_reranker(
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    train_splits: tuple[str, ...] = DEFAULT_LINEAR_TRAIN_SPLITS,
    max_rows_per_split: int = DEFAULT_LINEAR_TRAIN_MAX_ROWS,
    model_path: str | Path = DEFAULT_LINEAR_MODEL_PATH,
) -> Pipeline:
    path = Path(model_path)
    if path.exists():
        return load_linear_reranker(path)
    saved_path = train_and_save_default_linear_reranker(
        model_path=path,
        embedding_model=embedding_model,
        train_splits=train_splits,
        max_rows_per_split=max_rows_per_split,
    )
    return load_linear_reranker(saved_path)


def rank_case_with_default_linear_reranker(
    case: Par4pcCase,
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    train_splits: tuple[str, ...] = DEFAULT_LINEAR_TRAIN_SPLITS,
    max_rows_per_split: int = DEFAULT_LINEAR_TRAIN_MAX_ROWS,
    model_path: str | Path = DEFAULT_LINEAR_MODEL_PATH,
) -> list[tuple[str, float]]:
    model = get_default_linear_reranker(
        embedding_model=embedding_model,
        train_splits=train_splits,
        max_rows_per_split=max_rows_per_split,
        model_path=model_path,
    )
    feature_vectors = patent_specialized_feature_vectors(
        query_text=case.target_claim,
        candidates=list(case.candidates.values()),
        embedding_model=embedding_model,
        use_query_expansion=False,
        use_focused_query=True,
    )
    letters = sorted(case.candidates)
    X_case = np.array(
        [
            [feature_vectors[case.candidates[letter].patent_id].as_dict()[name] for name in DEFAULT_LINEAR_FEATURE_NAMES]
            for letter in letters
        ],
        dtype=np.float32,
    )
    probs = model.predict_proba(X_case)[:, 1]
    ranked = sorted(zip(letters, probs, strict=True), key=lambda item: item[1], reverse=True)
    return [(letter, float(score)) for letter, score in (ranked[:top_k] if top_k is not None else ranked)]


def rank_patent_pool_with_default_linear_reranker(
    query_text: str,
    candidates: list[PatentCandidate],
    top_k: int | None = None,
    embedding_model: str = "AI-Growth-Lab/PatentSBERTa",
    train_splits: tuple[str, ...] = DEFAULT_LINEAR_TRAIN_SPLITS,
    max_rows_per_split: int = DEFAULT_LINEAR_TRAIN_MAX_ROWS,
    model_path: str | Path = DEFAULT_LINEAR_MODEL_PATH,
) -> list[PatentSearchResult]:
    model = get_default_linear_reranker(
        embedding_model=embedding_model,
        train_splits=train_splits,
        max_rows_per_split=max_rows_per_split,
        model_path=model_path,
    )
    feature_vectors = patent_specialized_feature_vectors(
        query_text=query_text,
        candidates=candidates,
        embedding_model=embedding_model,
        use_query_expansion=False,
        use_focused_query=True,
    )
    scored: list[PatentSearchResult] = []
    X_case = np.array(
        [
            [feature_vectors[candidate.patent_id].as_dict()[name] for name in DEFAULT_LINEAR_FEATURE_NAMES]
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    probs = model.predict_proba(X_case)[:, 1]
    for candidate, score in zip(candidates, probs, strict=True):
        scored.append(
            PatentSearchResult(
                patent_id=candidate.patent_id,
                title=candidate.title,
                score=float(score),
                candidate=candidate,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k] if top_k is not None else scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a learned linear patent reranker.")
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--max-rows-per-split", type=int, default=100)
    parser.add_argument("--embedding-model", default="AI-Growth-Lab/PatentSBERTa")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default="outputs/linear_reranker_forward_selection.csv")
    parser.add_argument("--use-query-expansion", action="store_true")
    parser.add_argument("--no-focused-query", action="store_true")
    parser.add_argument("--mode", choices=["forward-selection", "single", "train-default-model"], default="forward-selection")
    parser.add_argument("--feature-names", nargs="+", default=FEATURE_ORDER)
    parser.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs"])
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--model-path", default=str(DEFAULT_LINEAR_MODEL_PATH))
    args = parser.parse_args()

    if args.mode == "train-default-model":
        saved_path = train_and_save_default_linear_reranker(
            model_path=args.model_path,
            embedding_model=args.embedding_model,
            train_splits=tuple(args.splits),
            max_rows_per_split=args.max_rows_per_split,
        )
        print(f"Saved default linear patent reranker to {saved_path}")
        print(f"Wrote metadata to {_metadata_path(Path(saved_path))}")
        return

    cases = load_hf_par4pc_cases(
        splits=tuple(args.splits),
        max_rows_per_split=args.max_rows_per_split,
    )
    print(
        f"Loaded {len(cases)} HF PAR4PC cases from splits={args.splits} "
        f"(max_rows_per_split={args.max_rows_per_split})."
    )

    local_metrics = _evaluate_baseline(cases, method="local-embedding", embedding_model=args.embedding_model)
    patent_metrics = _evaluate_baseline(cases, method="patent-specialized", embedding_model=args.embedding_model)
    print("\nlocal-embedding baseline")
    for key, value in local_metrics.items():
        print(f"{key}: {value:.3f}")
    print("\npatent-specialized baseline")
    for key, value in patent_metrics.items():
        print(f"{key}: {value:.3f}")

    class_weight = None if args.class_weight == "none" else args.class_weight

    if args.mode == "forward-selection":
        rows = evaluate_forward_selection(
            cases=cases,
            embedding_model=args.embedding_model,
            n_splits=args.n_splits,
            use_query_expansion=args.use_query_expansion,
            use_focused_query=not args.no_focused_query,
            solver=args.solver,
            C=args.c_value,
            class_weight=class_weight,
        )
        for row in rows:
            print(
                f"\n{row['feature_set']}\n"
                f"hit@1: {row['hit@1']:.3f}\n"
                f"hit@3: {row['hit@3']:.3f}\n"
                f"recall@3: {row['recall@3']:.3f}\n"
                f"exact@|gold|: {row['exact@|gold|']:.3f}"
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote forward-selection table to {output_path}")
    else:
        metrics = evaluate_single_feature_set(
            cases=cases,
            embedding_model=args.embedding_model,
            n_splits=args.n_splits,
            use_query_expansion=args.use_query_expansion,
            use_focused_query=not args.no_focused_query,
            feature_names=args.feature_names,
            solver=args.solver,
            C=args.c_value,
            class_weight=class_weight,
        )
        print(f"\nfeature set: {' + '.join(args.feature_names)}")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
