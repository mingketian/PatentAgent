from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from src.data_loader import load_hf_par4pc_cases
from src.patent_rerank import rank_candidates_patent_specialized
from src.retrieval import rank_candidates_local_embeddings


DEFAULT_OUTPUT = Path("outputs/patent_specialized_ablation.csv")


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_focused_query: bool = True
    use_query_expansion: bool = True
    use_field_dense: bool = True
    use_field_lexical: bool = True
    use_field_rarity: bool = True
    use_limitation_fusion: bool = True
    use_evidence_score: bool = True


def evaluate_configs(max_rows_per_split: int, splits: tuple[str, ...], output: Path) -> None:
    cases = load_hf_par4pc_cases(splits=splits, max_rows_per_split=max_rows_per_split)
    configs = [
        AblationConfig(name="PatentSBERTa Baseline"),
        AblationConfig(name="No Focused Query", use_focused_query=False),
        AblationConfig(name="No Field Lexical", use_field_lexical=False),
        AblationConfig(name="No Field Rarity", use_field_rarity=False),
        AblationConfig(name="No Limitation Fusion", use_limitation_fusion=False),
        AblationConfig(name="Full Patent Specialized"),
    ]

    rows: list[dict[str, object]] = []
    for config in configs:
        hit_at_1 = 0
        hit_at_3 = 0
        recall_at_3 = 0.0
        exact_at_gold = 0
        for case in cases:
            if config.name == "PatentSBERTa Baseline":
                ranked = rank_candidates_local_embeddings(
                    case,
                    top_k=3,
                    embedding_model="AI-Growth-Lab/PatentSBERTa",
                )
            else:
                ranked = rank_candidates_patent_specialized(
                    case,
                    top_k=3,
                    embedding_model="AI-Growth-Lab/PatentSBERTa",
                    use_query_expansion=config.use_query_expansion,
                    use_focused_query=config.use_focused_query,
                    use_field_dense=config.use_field_dense,
                    use_field_lexical=config.use_field_lexical,
                    use_field_rarity=config.use_field_rarity,
                    use_limitation_fusion=config.use_limitation_fusion,
                    use_evidence_score=config.use_evidence_score,
                )
            predicted = [result.letter for result in ranked]
            gold = set(case.gold_answers)
            hit_at_1 += bool(set(predicted[:1]) & gold)
            hit_at_3 += bool(set(predicted[:3]) & gold)
            recall_at_3 += len(set(predicted[:3]) & gold) / len(gold) if gold else 0.0
            exact_at_gold += set(predicted[: len(gold)]) == gold

        n = len(cases)
        rows.append(
            {
                "name": config.name,
                "num_cases": n,
                "hit@1": round(hit_at_1 / n, 3),
                "hit@3": round(hit_at_3 / n, 3),
                "recall@3": round(recall_at_3 / n, 3),
                "exact@|gold|": round(exact_at_gold / n, 3),
            }
        )
        print(
            f"{config.name}: hit@1={rows[-1]['hit@1']:.3f}, hit@3={rows[-1]['hit@3']:.3f}, "
            f"recall@3={rows[-1]['recall@3']:.3f}, exact@|gold|={rows[-1]['exact@|gold|']:.3f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Loaded {len(cases)} HF PAR4PC cases from splits={list(splits)} (max_rows_per_split={max_rows_per_split}).")
    print(f"Wrote ablation table to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate patent-specialized retrieval components.")
    parser.add_argument("--max-rows-per-split", type=int, default=30)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    evaluate_configs(
        max_rows_per_split=args.max_rows_per_split,
        splits=tuple(args.splits),
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
