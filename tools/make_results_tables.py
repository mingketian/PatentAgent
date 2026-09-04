"""Derive the committed result tables in results/tables/ from the raw evaluation data.

Sources
-------
* PANORAMA PAR4PC retrieval metrics : docs/EVALUATION_RESULTS.md (full validation run)
* Human product-QA study            : docs/eval/product_qa_manual_eval_annotated.xlsx

Run with:  python tools/make_results_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

# Full PANORAMA PAR4PC validation split (3,029 cases).
# Source: docs/EVALUATION_RESULTS.md, produced by src/evaluate_par4pc_hf.py
RETRIEVAL_ROWS = [
    ("bm25", "Lexical baseline", 0.718, 0.926, 0.843, 0.560),
    ("local-embedding", "PatentSBERTa baseline", 0.714, 0.950, 0.885, 0.570),
    ("linear-patent-reranker", "Ours (learned rerank)", 0.754, 0.959, 0.901, 0.622),
]

# validation-100 subset, used during development for the reranker config scan.
RETRIEVAL_100_ROWS = [
    ("local-embedding", "PatentSBERTa baseline", 0.590, 0.860, 0.802, 0.470),
    ("linear-patent-reranker", "Ours (learned rerank)", 0.600, 0.910, 0.850, 0.510),
]


def write_retrieval_tables() -> None:
    header = ["method", "label", "hit@1", "hit@3", "recall@3", "exact@gold"]
    for name, rows in (
        ("retrieval_validation_full.csv", RETRIEVAL_ROWS),
        ("retrieval_validation_100.csv", RETRIEVAL_100_ROWS),
    ):
        with (TABLES / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"wrote {TABLES / name}")


def write_human_eval_tables() -> None:
    import pandas as pd

    book = pd.ExcelFile(ROOT / "docs" / "eval" / "product_qa_manual_eval_annotated.xlsx")

    summary = book.parse("Method Summary")
    summary.to_csv(TABLES / "human_eval_summary.csv", index=False)
    print(f"wrote {TABLES / 'human_eval_summary.csv'}")

    counts = book.parse("Label Counts")
    counts.to_csv(TABLES / "human_eval_label_counts.csv", index=False)
    print(f"wrote {TABLES / 'human_eval_label_counts.csv'}")

    cases = book.parse("Annotated Cases")
    per_category = (
        cases.groupby(["category", "system_name"])[
            ["score_groundedness", "score_helpfulness", "score_hallucination", "score_context_reuse"]
        ]
        .mean()
        .round(3)
        .reset_index()
    )
    per_category.to_csv(TABLES / "human_eval_by_category.csv", index=False)
    print(f"wrote {TABLES / 'human_eval_by_category.csv'}")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_retrieval_tables()
    write_human_eval_tables()


if __name__ == "__main__":
    main()
