from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_QUERY_SET = Path("docs/eval/product_qa_queries.json")
DEFAULT_RAG_CSV = Path("docs/eval/product_qa_eval_rag_baseline_prefilled.csv")
DEFAULT_CHATGPT_CSV = Path("docs/eval/product_qa_eval_chatgpt_auto_prefilled.csv")
DEFAULT_GEMINI_CSV = Path("docs/eval/product_qa_eval_gemini_fast_prefilled.csv")
DEFAULT_AGENT_CSV = Path("outputs/product_qa_eval_q1_openai_answer_only.csv")
DEFAULT_OUTPUT_CSV = Path("docs/eval/product_qa_manual_eval_longform.csv")


def _load_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["query_id"]: row for row in csv.DictReader(handle)}


def _base_row(query: dict[str, object]) -> dict[str, str]:
    requires_context = bool(query["requires_context"])
    return {
        "query_id": str(query["query_id"]),
        "category": str(query["category"]),
        "requires_context": str(requires_context).lower(),
        "recommended_context_query_ids": "|".join(query["recommended_context_query_ids"]),
        "query_text": str(query["query_text"]),
        "evaluation_goal": str(query["evaluation_goal"]),
        "system_id": "",
        "system_name": "",
        "response_text": "",
        "response_verification": "",
        "response_source_file": "",
        "response_source_column": "",
        "manual_groundedness_label": "",
        "manual_helpfulness_label": "",
        "manual_context_reuse_label": "n/a" if not requires_context else "",
        "manual_hallucination_label": "",
        "manual_notes": "",
    }


def _append_system_row(
    rows: list[dict[str, str]],
    *,
    query: dict[str, object],
    system_id: str,
    system_name: str,
    response_text: str,
    response_verification: str,
    response_source_file: Path,
    response_source_column: str,
) -> None:
    row = _base_row(query)
    row["system_id"] = system_id
    row["system_name"] = system_name
    row["response_text"] = response_text
    row["response_verification"] = response_verification
    row["response_source_file"] = str(response_source_file)
    row["response_source_column"] = response_source_column
    rows.append(row)


def build_manual_eval_sheet(
    *,
    query_set_path: Path,
    rag_csv: Path,
    chatgpt_csv: Path,
    gemini_csv: Path,
    agent_csv: Path,
    output_csv: Path,
) -> None:
    queries = json.loads(query_set_path.read_text(encoding="utf-8"))
    rag_rows = _load_csv_rows(rag_csv)
    chatgpt_rows = _load_csv_rows(chatgpt_csv)
    gemini_rows = _load_csv_rows(gemini_csv)
    agent_rows = _load_csv_rows(agent_csv)

    output_rows: list[dict[str, str]] = []
    for query in queries:
        query_id = str(query["query_id"])
        rag = rag_rows[query_id]
        chatgpt = chatgpt_rows[query_id]
        gemini = gemini_rows[query_id]
        agent = agent_rows[query_id]

        _append_system_row(
            output_rows,
            query=query,
            system_id="rag_only",
            system_name="RAG only",
            response_text=rag["baseline_answer"],
            response_verification="",
            response_source_file=rag_csv,
            response_source_column="baseline_answer",
        )
        _append_system_row(
            output_rows,
            query=query,
            system_id="chatgpt_auto",
            system_name="ChatGPT Auto",
            response_text=chatgpt["baseline_answer"],
            response_verification="",
            response_source_file=chatgpt_csv,
            response_source_column="baseline_answer",
        )
        _append_system_row(
            output_rows,
            query=query,
            system_id="gemini_fast",
            system_name="Gemini Fast",
            response_text=gemini["baseline_answer"],
            response_verification="",
            response_source_file=gemini_csv,
            response_source_column="baseline_answer",
        )
        _append_system_row(
            output_rows,
            query=query,
            system_id="our_agent",
            system_name="Our Agent",
            response_text=agent["system_answer"],
            response_verification=agent.get("answer_verification", ""),
            response_source_file=agent_csv,
            response_source_column="system_answer",
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one long-form manual product-QA evaluation sheet across RAG, ChatGPT Auto, Gemini Fast, and Our Agent."
    )
    parser.add_argument("--query-set", default=str(DEFAULT_QUERY_SET))
    parser.add_argument("--rag-csv", default=str(DEFAULT_RAG_CSV))
    parser.add_argument("--chatgpt-csv", default=str(DEFAULT_CHATGPT_CSV))
    parser.add_argument("--gemini-csv", default=str(DEFAULT_GEMINI_CSV))
    parser.add_argument("--agent-csv", default=str(DEFAULT_AGENT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    build_manual_eval_sheet(
        query_set_path=Path(args.query_set),
        rag_csv=Path(args.rag_csv),
        chatgpt_csv=Path(args.chatgpt_csv),
        gemini_csv=Path(args.gemini_csv),
        agent_csv=Path(args.agent_csv),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
