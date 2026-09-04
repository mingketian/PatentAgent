from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.free_text_qa import gather_query_evidence, heuristic_rag_answer, verify_rag_answer_heuristic
from src.llm_tools import answer_query_with_rag, openai_available, verify_rag_answer_llm
from src.patent_rerank import rank_patent_pool_hybrid_coverage, rank_patent_pool_patent_specialized
from src.persistent_index import index_exists, load_persistent_manifest, search_persistent_index
from src.query_planner import TurnPlan, classify_turn, enrich_query_with_context
from src.retrieval import rank_patent_pool_bm25, rank_patent_pool_local_embeddings
from src.train_linear_patent_reranker import rank_patent_pool_with_default_linear_reranker


DEFAULT_QUERY_SET = Path("docs/eval/product_qa_queries.json")
DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_full")
DEFAULT_OUTPUT_CSV = Path("outputs/product_qa_eval_full_linear.csv")
DEFAULT_OUTPUT_SUMMARY = Path("outputs/product_qa_eval_full_linear_summary.json")
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEFAULT_BASELINE_NAME = "Baseline RAG"
DEFAULT_BASELINE_TYPE = "local_retrieval_rag_baseline"


def _search_patents(
    *,
    query_text: str,
    retrieval_method: str,
    top_k: int,
    index_dir: str,
    embedding_model: str,
    pool=None,
    force_subset: bool = False,
):
    if retrieval_method == "linear-patent-reranker":
        if not force_subset:
            initial = search_persistent_index(
                query_text,
                index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        return rank_patent_pool_with_default_linear_reranker(
            query_text=query_text,
            candidates=pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )

    if retrieval_method == "patent-specialized":
        if not force_subset:
            initial = search_persistent_index(
                query_text,
                index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        return rank_patent_pool_patent_specialized(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            use_query_expansion=True,
        )

    if retrieval_method == "local-embedding" and not force_subset:
        return search_persistent_index(
            query_text,
            index_dir=index_dir,
            top_k=top_k,
            embedding_model=embedding_model,
        )

    if retrieval_method == "local-embedding":
        return rank_patent_pool_local_embeddings(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )

    if retrieval_method == "hybrid-coverage":
        return rank_patent_pool_hybrid_coverage(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )

    return rank_patent_pool_bm25(query_text, pool, top_k=top_k)


def _baseline_answer(ranked) -> str:
    parts = [f"{item.patent_id} | {item.title} | {item.score:.4f}" for item in ranked]
    return f"Top-{len(ranked)} patents: " + "; ".join(parts) + "."


def _top_note(label: str, ranked) -> str:
    parts = [f"{item.patent_id}|{item.title}|{item.score:.4f}" for item in ranked]
    return f"{label}=" + " ; ".join(parts)


def _citation_note(snippets) -> str:
    citations = [f"[{snippet.citation}]" for snippet in snippets[:3]]
    return "citations=" + ", ".join(citations) if citations else "citations="


def _baseline_note_only(notes: str) -> str:
    if not notes:
        return ""
    kept = [part for part in notes.split("; ") if part.startswith("baseline_top3=")]
    return "; ".join(kept)


def _baseline_only_rows(rows: list[dict[str, str]], baseline_name: str) -> list[dict[str, str]]:
    baseline_rows: list[dict[str, str]] = []
    for row in rows:
        baseline_rows.append(
            {
                **row,
                "baseline_name": baseline_name or row.get("baseline_name", DEFAULT_BASELINE_NAME),
                "system_name": "",
                "system_answer": "",
                "answer_verification": "",
                "notes": _baseline_note_only(row.get("notes", "")),
            }
        )
    return baseline_rows


def _write_baseline_markdown(
    rows: list[dict[str, str]],
    *,
    baseline_name: str,
    baseline_type: str,
    baseline_method: str,
    index_dir: str | Path,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {baseline_name} Artifact",
        "",
        "This file organizes the batch product-QA run into a per-query baseline artifact for qualitative evaluation.",
        "",
        "Baseline label:",
        "",
        f"- `{baseline_name}`",
        f"- baseline type: `{baseline_type}`",
        f"- retrieval method: `{baseline_method}`",
        f"- source index: `{index_dir}`",
        "",
        "Important notes:",
        "",
        "- Context-dependent follow-up questions should be interpreted relative to their recommended prior turns.",
        "- This artifact is for qualitative baseline comparison, not a formal answer key.",
        "",
        "---",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"## {row['query_id']}",
                "",
                "**Query**",
                "",
                "```text",
                row["query_text"],
                "```",
                "",
                "**Evaluation Goal**",
                "",
                row["evaluation_goal"],
                "",
                "**Baseline Answer**",
                "",
                row["baseline_answer"],
                "",
            ]
        )
        if row.get("notes"):
            lines.extend(
                [
                    "**Notes**",
                    "",
                    row["notes"],
                    "",
                ]
            )
        lines.extend(["---", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_baseline_manifest(
    rows: list[dict[str, str]],
    *,
    baseline_name: str,
    baseline_type: str,
    markdown_path: str | Path,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = Path(markdown_path)

    fieldnames = [
        "query_id",
        "baseline_name",
        "baseline_type",
        "requires_context",
        "context_source_query_ids",
        "artifact_file",
        "section",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "baseline_name": baseline_name,
                    "baseline_type": baseline_type,
                    "requires_context": row["requires_context"],
                    "context_source_query_ids": row["recommended_context_query_ids"],
                    "artifact_file": str(markdown_path),
                    "section": row["query_id"],
                }
            )


def _execute_optimized_query(
    *,
    query_text: str,
    state: dict,
    retrieval_method: str,
    top_k: int,
    index_dir: str,
    embedding_model: str,
    use_llm_answer: bool,
    use_llm_answer_verification: bool,
    llm_model: str,
):
    local_state = {
        "last_ranked": list(state.get("last_ranked", [])),
        "working_patents": list(state.get("working_patents", [])),
    }
    plan = classify_turn(query_text, has_context=bool(local_state["working_patents"]))
    effective_query = enrich_query_with_context(query_text, plan, local_state["last_ranked"])

    if plan.action == "retrieve_new":
        ranked = _search_patents(
            query_text=effective_query,
            retrieval_method=retrieval_method,
            top_k=top_k,
            index_dir=index_dir,
            embedding_model=embedding_model,
        )
        working_patents = [item.candidate for item in ranked]
    elif plan.action == "reuse_context":
        ranked = local_state["last_ranked"]
        working_patents = local_state["working_patents"]
    else:
        working_patents = local_state["working_patents"]
        ranked = _search_patents(
            query_text=query_text,
            retrieval_method=retrieval_method,
            top_k=min(top_k, len(working_patents) or top_k),
            index_dir=index_dir,
            embedding_model=embedding_model,
            pool=working_patents,
            force_subset=True,
        )

    snippets = gather_query_evidence(query_text, ranked, snippets_per_patent=2)
    if use_llm_answer and openai_available():
        response = answer_query_with_rag(query_text, snippets, model=llm_model or None)
        answer = response.answer
        if response.insufficiency_note:
            answer += f"\n\nNote: {response.insufficiency_note}"
    else:
        answer = heuristic_rag_answer(query_text, ranked, snippets, plan=plan)
    if use_llm_answer_verification and openai_available():
        verification = verify_rag_answer_llm(answer, snippets, model=llm_model or None)
    else:
        verification = verify_rag_answer_heuristic(answer, snippets)

    local_state["last_ranked"] = ranked
    local_state["working_patents"] = working_patents
    return {
        "plan": plan,
        "effective_query": effective_query,
        "ranked": ranked,
        "snippets": snippets,
        "answer": answer,
        "verification": verification,
        "state": local_state,
    }


def _blank_eval_row(query: dict) -> dict[str, str]:
    return {
        "query_id": str(query["query_id"]),
        "category": str(query["category"]),
        "requires_context": str(query["requires_context"]).lower(),
        "recommended_context_query_ids": "|".join(query["recommended_context_query_ids"]),
        "query_text": str(query["query_text"]),
        "evaluation_goal": str(query["evaluation_goal"]),
        "baseline_name": "",
        "baseline_answer": "",
        "system_name": "",
        "system_answer": "",
        "answer_verification": "",
        "human_groundedness_label": "",
        "human_helpfulness_label": "",
        "human_context_reuse_label": "",
        "human_hallucination_label": "",
        "notes": "",
    }


def run_eval(
    *,
    query_set_path: str | Path,
    index_dir: str | Path,
    top_k: int,
    baseline_method: str,
    optimized_method: str,
    embedding_model: str,
    output_csv: str | Path,
    output_summary: str | Path,
    query_ids: tuple[str, ...] = (),
    use_llm_answer: bool = False,
    use_llm_answer_verification: bool = False,
    llm_model: str = "",
    baseline_only: bool = False,
    baseline_name: str = DEFAULT_BASELINE_NAME,
    baseline_markdown: str | Path = "",
    baseline_manifest: str | Path = "",
    baseline_type: str = DEFAULT_BASELINE_TYPE,
) -> None:
    query_set_path = Path(query_set_path)
    index_dir = Path(index_dir)
    output_csv = Path(output_csv)
    output_summary = Path(output_summary)

    if not index_exists(index_dir):
        raise SystemExit(
            f"Persistent index not found at {index_dir}. "
            "Build it first with python -m src.build_patent_index."
        )

    queries = json.loads(query_set_path.read_text(encoding="utf-8"))
    if query_ids:
        selected = {query_id.strip() for query_id in query_ids if query_id.strip()}
        queries = [query for query in queries if str(query["query_id"]) in selected]
        if not queries:
            raise SystemExit(f"No queries matched --query-ids={','.join(query_ids)}")
    manifest = load_persistent_manifest(str(index_dir))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(_blank_eval_row(queries[0]).keys())
    rows: list[dict[str, str]] = []
    optimized_state = {"last_ranked": [], "working_patents": []}
    verification_counts: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()

    print(f"Loaded {len(queries)} product QA queries from {query_set_path}")
    print(f"Using persistent index: {index_dir} ({manifest.patent_count} patents)")
    print(f"Baseline method: {baseline_method}")
    if baseline_only:
        print("Optimized method: skipped (baseline-only run)")
    else:
        print(f"Optimized method: {optimized_method}")
        print(f"LLM grounded answer: {'on' if use_llm_answer else 'off'}")
        print(f"LLM answer verification: {'on' if use_llm_answer_verification else 'off'}")
    print()

    for query in queries:
        row = _blank_eval_row(query)
        baseline_ranked = _search_patents(
            query_text=row["query_text"],
            retrieval_method=baseline_method,
            top_k=top_k,
            index_dir=str(index_dir),
            embedding_model=embedding_model,
        )
        row["baseline_name"] = baseline_name
        row["baseline_answer"] = _baseline_answer(baseline_ranked)
        baseline_notes = [_top_note("baseline_top3", baseline_ranked)]

        if baseline_only:
            row["notes"] = "; ".join(baseline_notes)
            rows.append(row)
            print(
                f"{row['query_id']}: "
                f"baseline_top1={baseline_ranked[0].patent_id if baseline_ranked else 'NA'}"
            )
            continue

        optimized = _execute_optimized_query(
            query_text=row["query_text"],
            state=optimized_state,
            retrieval_method=optimized_method,
            top_k=top_k,
            index_dir=str(index_dir),
            embedding_model=embedding_model,
            use_llm_answer=use_llm_answer,
            use_llm_answer_verification=use_llm_answer_verification,
            llm_model=llm_model,
        )
        optimized_state = optimized["state"]
        verification = optimized["verification"]
        plan: TurnPlan = optimized["plan"]
        verification_counts[verification.status] += 1
        plan_counts[f"{plan.intent}/{plan.action}"] += 1

        row["system_name"] = "Our optimized patent agent"
        row["system_answer"] = optimized["answer"]
        row["answer_verification"] = f"{verification.status}: {verification.reason}"

        notes = [
            *baseline_notes,
            _top_note("optimized_top3", optimized["ranked"]),
            _citation_note(optimized["snippets"]),
            f"plan={plan.intent}/{plan.action}",
        ]
        if optimized["effective_query"] != row["query_text"]:
            notes.append(f"effective_query={optimized['effective_query']}")
        row["notes"] = "; ".join(notes)
        rows.append(row)

        print(
            f"{row['query_id']}: "
            f"baseline_top1={baseline_ranked[0].patent_id if baseline_ranked else 'NA'} | "
            f"optimized_top1={optimized['ranked'][0].patent_id if optimized['ranked'] else 'NA'} | "
            f"verification={verification.status} | "
            f"plan={plan.intent}/{plan.action}"
        )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "query_count": len(rows),
        "index_dir": str(index_dir),
        "patent_count": manifest.patent_count,
        "embedding_model": manifest.embedding_model,
        "baseline_method": baseline_method,
        "optimized_method": "" if baseline_only else optimized_method,
        "top_k": top_k,
        "query_ids": list(query_ids),
        "llm_grounded_answer": False if baseline_only else use_llm_answer,
        "llm_answer_verification": False if baseline_only else use_llm_answer_verification,
        "llm_model": "" if baseline_only else llm_model,
        "baseline_only": baseline_only,
        "baseline_name": baseline_name,
        "verification_counts": dict(verification_counts),
        "plan_counts": dict(plan_counts),
        "csv_output": str(output_csv),
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    baseline_rows = rows if baseline_only else _baseline_only_rows(rows, baseline_name)
    if baseline_markdown:
        _write_baseline_markdown(
            baseline_rows,
            baseline_name=baseline_name,
            baseline_type=baseline_type,
            baseline_method=baseline_method,
            index_dir=index_dir,
            output_path=baseline_markdown,
        )
    if baseline_manifest and baseline_markdown:
        _write_baseline_manifest(
            baseline_rows,
            baseline_name=baseline_name,
            baseline_type=baseline_type,
            markdown_path=baseline_markdown,
            output_path=baseline_manifest,
        )

    print()
    print(f"Wrote CSV to {output_csv}")
    print(f"Wrote summary to {output_summary}")
    if baseline_markdown:
        print(f"Wrote baseline markdown to {baseline_markdown}")
    if baseline_manifest:
        print(f"Wrote baseline manifest to {baseline_manifest}")
    if not baseline_only:
        print("Verification counts:")
        for status, count in sorted(verification_counts.items()):
            print(f"  {status}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch product QA evaluation over the persistent patent index.")
    parser.add_argument("--query-set", default=str(DEFAULT_QUERY_SET))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--baseline-method",
        choices=["bm25", "local-embedding"],
        default="local-embedding",
    )
    parser.add_argument(
        "--optimized-method",
        choices=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized", "linear-patent-reranker"],
        default="linear-patent-reranker",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--query-ids", nargs="+", default=[])
    parser.add_argument("--llm-answer", action="store_true")
    parser.add_argument("--llm-answer-verification", action="store_true")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--baseline-name", default=DEFAULT_BASELINE_NAME)
    parser.add_argument("--baseline-type", default=DEFAULT_BASELINE_TYPE)
    parser.add_argument("--baseline-markdown", default="")
    parser.add_argument("--baseline-manifest", default="")
    args = parser.parse_args()
    run_eval(
        query_set_path=args.query_set,
        index_dir=args.index_dir,
        top_k=args.top_k,
        baseline_method=args.baseline_method,
        optimized_method=args.optimized_method,
        embedding_model=args.embedding_model,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        query_ids=tuple(args.query_ids),
        use_llm_answer=args.llm_answer,
        use_llm_answer_verification=args.llm_answer_verification,
        llm_model=args.llm_model,
        baseline_only=args.baseline_only,
        baseline_name=args.baseline_name,
        baseline_markdown=args.baseline_markdown,
        baseline_manifest=args.baseline_manifest,
        baseline_type=args.baseline_type,
    )


if __name__ == "__main__":
    main()
