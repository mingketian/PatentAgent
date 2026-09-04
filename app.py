from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.claim_analysis import ClaimLimitation, VerificationResult, extract_evidence_for_candidate
from src.data_loader import combine_patent_pools, load_hf_par4pc_patent_pool, load_par4pc_case, load_unique_patent_pool
from src.free_text_qa import gather_query_evidence, heuristic_rag_answer, verify_rag_answer_heuristic
from src.graph import run_graph
from src.llm_tools import answer_query_with_rag, openai_available, plan_turn_llm, verify_rag_answer_llm
from src.patent_rerank import rank_patent_pool_hybrid_coverage
from src.patent_rerank import rank_patent_pool_patent_specialized
from src.persistent_index import (
    index_exists,
    load_persistent_candidates,
    load_persistent_manifest,
    search_persistent_index,
)
from src.retrieval import (
    _sentence_transformer_model,
    rank_patent_pool_bm25,
    rank_patent_pool_cross_encoder,
    rank_patent_pool_local_embeddings,
)
from src.query_planner import TurnPlan, classify_turn, enrich_query_with_context
from src.train_linear_patent_reranker import rank_patent_pool_with_default_linear_reranker


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")
DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_demo")
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASELINE_FREE_TEXT_METHOD = "local-embedding"
OPTIMIZED_FREE_TEXT_METHOD = "linear-patent-reranker"
FREE_TEXT_PRIMARY_RETRIEVAL_OPTIONS = ["bm25", "local-embedding"]
BENCHMARK_PRIMARY_RETRIEVAL_OPTIONS = ["local-embedding", "linear-patent-reranker"]
FREE_TEXT_EXPERIMENTAL_OPTIONS = ["linear-patent-reranker", "patent-specialized", "hybrid-coverage", "local-cross-encoder"]
BENCHMARK_EXPERIMENTAL_OPTIONS = ["linear-patent-reranker", "patent-specialized", "hybrid-coverage", "local-cross-encoder", "openai-embedding", "llm-rerank"]
DEFAULT_FREE_TEXT = (
    "1. A method for leveraging social networks in physical gatherings, the method comprising: "
    "generating, by one or more computer processors, a profile for each participant of one or more "
    "participants at a physical gathering; receiving, by one or more computer processors, data from "
    "one or more computer systems associated with the one or more participants of the physical gathering, "
    "wherein each participant of the one or more participants is associated with a computer system; "
    "receiving, by one or more computer processors, a request for information from a computer system "
    "associated with a first participant of the one or more participants of the physical gathering; "
    "determining, by one or more computer processors, whether the first participant has access to the "
    "information requested based on the profile for the first participant; responsive to determining that "
    "the first participant has access to the information requested, analyzing, by one or more computer "
    "processors, the data received from the one or more computer systems associated with the one or more "
    "participants of the physical gathering to identify data to provide to the first participant to "
    "fulfill the request for information; and providing, by one or more computer processors, the "
    "identified data to the computer system associated with the first participant of the physical gathering."
)

METHOD_LABELS = {
    "bm25": "bm25 - lexical baseline",
    "local-embedding": "local-embedding - PatentSBERTa semantic baseline",
    "linear-patent-reranker": "linear-patent-reranker - learned benchmark reranker",
    "patent-specialized": "patent-specialized - hand-tuned patent-aware reranker",
    "hybrid-coverage": "hybrid-coverage - dense + lexical + coverage",
    "local-cross-encoder": "local-cross-encoder - slow local reranker",
    "openai-embedding": "openai-embedding - OpenAI embedding baseline",
    "llm-rerank": "llm-rerank - LLM candidate reranking",
}

FREE_TEXT_METHOD_HELP = {
    "bm25": "Best current free-text default. Fast and stable on the larger indexed patent pool.",
    "local-embedding": "PatentSBERTa semantic retrieval. Useful for comparison, but not the current free-text default.",
    "linear-patent-reranker": "Two-stage product reranker: coarse recall from the persistent index, then learned linear patent reranking.",
    "patent-specialized": "Patent-aware reranker path. Better suited to benchmark-style reranking than general free-text search.",
    "hybrid-coverage": "Combines semantic retrieval, lexical overlap, and limitation coverage. Experimental for free-text.",
    "local-cross-encoder": "Slow local reranker. Keep this for ablation only.",
}

BENCHMARK_METHOD_HELP = {
    "local-embedding": "Stable benchmark baseline using PatentSBERTa over the A-H candidate patents.",
    "bm25": "Simple lexical baseline over the A-H benchmark candidates.",
    "linear-patent-reranker": "Current learned benchmark experiment. Uses a cached linear model trained on HF train cases.",
    "patent-specialized": "Older hand-tuned patent-aware reranker. Keep for comparison, not as the main experiment.",
    "hybrid-coverage": "Ablation path combining dense retrieval, lexical overlap, and coverage scoring.",
    "local-cross-encoder": "Slow local reranker for ablation.",
    "openai-embedding": "OpenAI embedding baseline. Requires API access.",
    "llm-rerank": "LLM reranking over candidate patents. Requires API access.",
}

POOL_HELP = {
    "Persistent local index": "A local FAISS-backed patent store. This is the recommended free-text pool.",
    "Local sample pool": "Only the small bundled sample patent set. Good for quick smoke tests, not for realistic search.",
    "Hub PAR4PC pool": "A larger patent pool loaded from Hugging Face PAR4PC parquet files.",
    "Combined": "Local sample patents plus a Hub slice combined into one free-text pool.",
}

PRODUCT_VARIANT_HELP = {
    "Normal RAG baseline": (
        "Baseline product path: persistent local index retrieval returning top patents only. "
        "Evidence snippets can be shown, but there is no planner, reranker, grounded answer synthesis, or verification."
    ),
    "Our optimized patent agent": (
        "Our product path: persistent-index recall, learned linear patent reranking, conversational "
        "working-set reuse, evidence extraction, grounded answer synthesis, and answer verification."
    ),
}


@st.cache_data
def list_case_paths(data_dir: str) -> list[str]:
    return [str(path) for path in sorted(Path(data_dir).glob("par4pc_*.json"))]


@st.cache_data
def preview_case(case_path: str):
    return load_par4pc_case(case_path)


@st.cache_data
def load_pool(
    data_dir: str,
    pool_source: str,
    hub_rows_per_split: int,
):
    local_pool = load_unique_patent_pool(data_dir)
    if pool_source == "Local sample pool":
        return local_pool

    hf_limit = None if hub_rows_per_split <= 0 else hub_rows_per_split
    hub_pool = load_hf_par4pc_patent_pool(max_rows_per_split=hf_limit)
    if pool_source == "Hub PAR4PC pool":
        return hub_pool
    return combine_patent_pools(local_pool, hub_pool)


def warm_up_search_backend(
    data_dir: str,
    pool_source: str,
    hub_rows_per_split: int,
    retrieval_method: str,
    embedding_model: str,
    index_dir: str,
) -> tuple[int, str]:
    if pool_source == "Persistent local index":
        manifest = load_persistent_manifest(index_dir)
        model_name = ""
        if retrieval_method == "local-embedding":
            model_name = embedding_model or manifest.embedding_model
            _sentence_transformer_model(model_name)
        return manifest.patent_count, model_name

    pool = load_pool(data_dir, pool_source, hub_rows_per_split)
    model_name = ""
    if retrieval_method == "local-embedding":
        model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
        _sentence_transformer_model(model_name)
    return len(pool), model_name


def ranked_table(ranked) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": index,
                "label": getattr(result, "letter", ""),
                "patent_id": result.patent_id,
                "score": round(result.score, 3),
                "title": result.title,
            }
            for index, result in enumerate(ranked, start=1)
        ]
    )


def limitation_table(limitations) -> pd.DataFrame:
    return pd.DataFrame(
        [{"label": item.label, "limitation": item.text} for item in limitations]
    )


def claim_chart_table(chart) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "limitation": row.limitation_label,
                "candidate": f"{row.candidate_letter} ({row.patent_id})",
                "source": row.source,
                "score": round(row.score, 3),
                "verification": row.verification,
                "reason": row.verification_reason,
                "evidence": row.evidence,
            }
            for row in chart
        ]
    )


def query_evidence_table(snippets) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patent_id": snippet.patent_id,
                "title": snippet.title,
                "source": snippet.source,
                "retrieval_score": round(snippet.retrieval_score, 3),
                "segment_score": round(snippet.segment_score, 3),
                "evidence": snippet.evidence,
            }
            for snippet in snippets
        ]
    )


def search_patents(
    query_text: str,
    pool,
    retrieval_method: str,
    embedding_model: str,
    reranker_model: str,
    top_k: int,
    pool_source: str,
    index_dir: str,
    force_subset: bool = False,
    llm_model: str = "",
    use_llm_retrieval_decompose: bool = False,
    use_llm_query_expansion: bool = False,
):
    if retrieval_method == "linear-patent-reranker":
        if pool_source == "Persistent local index" and not force_subset:
            initial = search_persistent_index(
                query_text,
                index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        elif pool_source == "Persistent local index":
            pool = load_persistent_candidates(index_dir)
        return rank_patent_pool_with_default_linear_reranker(
            query_text=query_text,
            candidates=pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )
    if retrieval_method == "patent-specialized":
        if pool_source == "Persistent local index" and not force_subset:
            initial = search_persistent_index(
                query_text,
                index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        elif pool_source == "Persistent local index":
            pool = load_persistent_candidates(index_dir)
        return rank_patent_pool_patent_specialized(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            use_query_expansion=True,
            use_llm_expansion=use_llm_query_expansion,
            use_llm_decompose=use_llm_retrieval_decompose,
            llm_model=llm_model,
        )
    if retrieval_method == "hybrid-coverage":
        if pool_source == "Persistent local index" and not force_subset:
            pool = load_persistent_candidates(index_dir)
        return rank_patent_pool_hybrid_coverage(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )
    if pool_source == "Persistent local index" and retrieval_method == "local-embedding" and not force_subset:
        return search_persistent_index(
            query_text,
            index_dir=index_dir,
            top_k=top_k,
            embedding_model=embedding_model,
        )
    if pool_source == "Persistent local index" and not force_subset:
        pool = load_persistent_candidates(index_dir)
    if retrieval_method == "local-embedding":
        return rank_patent_pool_local_embeddings(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )
    if retrieval_method == "local-cross-encoder":
        return rank_patent_pool_cross_encoder(
            query_text,
            pool,
            top_k=top_k,
            reranker_model=reranker_model or DEFAULT_RERANKER_MODEL,
        )
    return rank_patent_pool_bm25(query_text, pool, top_k=top_k)


def free_text_summary(query_text: str, ranked) -> tuple[str, pd.DataFrame]:
    rows = []
    summary_lines = [
        "I searched the indexed patent pool and ranked the closest patents to your query.",
        "",
    ]
    for index, result in enumerate(ranked, start=1):
        limitation = ClaimLimitation(label="Q1", text=query_text)
        evidence = extract_evidence_for_candidate(limitation, result.candidate)
        summary_lines.append(
            f"{index}. `{result.patent_id}` {result.title} "
            f"(source: {evidence.source}, score={result.score:.3f})"
        )
        rows.append(
            {
                "rank": index,
                "patent_id": result.patent_id,
                "title": result.title,
                "score": round(result.score, 3),
                "source": evidence.source,
                "evidence": evidence.evidence,
            }
        )
    return "\n".join(summary_lines), pd.DataFrame(rows)


def generate_free_text_answer(
    query_text: str,
    ranked,
    llm_model: str,
    use_llm_answer: bool,
    plan=None,
):
    snippets = gather_query_evidence(query_text, ranked, snippets_per_patent=2)
    warnings: list[str] = []
    if use_llm_answer:
        if openai_available():
            response = answer_query_with_rag(query_text, snippets, model=llm_model)
            answer = response.answer
            if response.insufficiency_note:
                answer += f"\n\nNote: {response.insufficiency_note}"
            return answer, snippets, warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic grounded answer.")
    return heuristic_rag_answer(query_text, ranked, snippets, plan=plan), snippets, warnings


def verify_free_text_answer(
    answer_text: str,
    snippets,
    llm_model: str,
    use_llm_answer_verification: bool,
) -> tuple[VerificationResult, list[str]]:
    warnings: list[str] = []
    if use_llm_answer_verification:
        if openai_available():
            return verify_rag_answer_llm(answer_text, snippets, model=llm_model), warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic answer verification.")
    return verify_rag_answer_heuristic(answer_text, snippets), warnings


def execute_free_text_path(
    *,
    query_text: str,
    agent_state: dict,
    pool,
    retrieval_method: str,
    embedding_model: str,
    reranker_model: str,
    top_k: int,
    pool_source: str,
    index_dir: str,
    llm_model: str,
    use_llm_answer: bool,
    use_llm_answer_verification: bool,
    use_llm_planner: bool,
    use_llm_retrieval_decompose: bool,
    use_llm_query_expansion: bool,
    use_context: bool,
) -> dict:
    local_state = {
        "last_ranked": list(agent_state.get("last_ranked", [])),
        "last_snippets": list(agent_state.get("last_snippets", [])),
        "last_plan": agent_state.get("last_plan"),
        "working_patents": list(agent_state.get("working_patents", [])),
        "last_query": agent_state.get("last_query", ""),
    }
    if use_context:
        if use_llm_planner and openai_available():
            plan = plan_turn_llm(
                query_text=query_text,
                has_context=bool(local_state["working_patents"]),
                previous_titles=[result.title for result in local_state["last_ranked"][:5]],
                model=llm_model or None,
            )
        else:
            plan = classify_turn(query_text, has_context=bool(local_state["working_patents"]))
        effective_query = enrich_query_with_context(query_text, plan, local_state["last_ranked"])
    else:
        plan = TurnPlan(
            intent="new_search",
            action="retrieve_new",
            reason="Baseline RAG path always performs a fresh retrieval.",
            query_text=query_text,
        )
        effective_query = query_text

    if plan.action == "retrieve_new":
        ranked = search_patents(
            query_text=effective_query,
            pool=pool,
            retrieval_method=retrieval_method,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            top_k=top_k,
            pool_source=pool_source,
            index_dir=index_dir,
            llm_model=llm_model,
            use_llm_retrieval_decompose=use_llm_retrieval_decompose and use_context,
            use_llm_query_expansion=use_llm_query_expansion and use_context,
        )
        working_patents = [result.candidate for result in ranked]
    elif plan.action == "reuse_context":
        ranked = local_state["last_ranked"]
        working_patents = local_state["working_patents"]
    else:
        working_patents = local_state["working_patents"]
        ranked = search_patents(
            query_text=query_text,
            pool=working_patents,
            retrieval_method=retrieval_method,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            top_k=min(top_k, len(working_patents) or top_k),
            pool_source=pool_source,
            index_dir=index_dir,
            force_subset=True,
            llm_model=llm_model,
            use_llm_retrieval_decompose=use_llm_retrieval_decompose and use_context,
            use_llm_query_expansion=use_llm_query_expansion and use_context,
        )

    answer, snippets, warnings = generate_free_text_answer(
        query_text=query_text,
        ranked=ranked,
        llm_model=llm_model,
        use_llm_answer=use_llm_answer,
        plan=plan,
    )
    answer_verification, verification_warnings = verify_free_text_answer(
        answer_text=answer,
        snippets=snippets,
        llm_model=llm_model,
        use_llm_answer_verification=use_llm_answer_verification and use_context,
    )
    warnings.extend(verification_warnings)
    summary, evidence_df = free_text_summary(query_text, ranked)

    local_state["last_ranked"] = ranked
    local_state["last_snippets"] = snippets
    local_state["last_plan"] = plan
    local_state["working_patents"] = working_patents
    local_state["last_query"] = query_text
    return {
        "query_text": query_text,
        "pipeline_name": "Our optimized patent agent" if use_context else "Normal RAG baseline",
        "answer_mode": "llm_grounded" if use_llm_answer and openai_available() else "heuristic_grounded",
        "plan": plan,
        "effective_query": effective_query,
        "ranked": ranked,
        "answer": answer,
        "snippets": snippets,
        "warnings": warnings,
        "answer_verification": answer_verification,
        "summary": summary,
        "evidence_df": evidence_df,
        "state": local_state,
    }


def render_free_text_result_block(title: str, result: dict, *, expanded: bool = False) -> None:
    st.subheader(title)
    st.write(result["answer"])
    if result["warnings"]:
        st.warning("\n".join(result["warnings"]))
    st.caption(
        f"Pipeline: `{result['pipeline_name']}` | "
        f"Answer mode: `{result['answer_mode']}` | "
        f"Planner intent: `{result['plan'].intent}` | action: `{result['plan'].action}` | "
        f"Verification: `{result['answer_verification'].status}`"
    )
    if result["effective_query"] != result["query_text"]:
        st.code(result["effective_query"], language="text")
    with st.expander(f"{title}: ranked patents", expanded=expanded):
        st.dataframe(ranked_table(result["ranked"]), use_container_width=True, hide_index=True)
    with st.expander(f"{title}: supporting evidence", expanded=expanded):
        st.dataframe(
            query_evidence_table(result["snippets"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "evidence": st.column_config.TextColumn("evidence", width="large"),
                "title": st.column_config.TextColumn("title", width="medium"),
            },
        )


def render_baseline_retrieval_block(title: str, result: dict, *, show_evidence: bool) -> None:
    st.subheader(title)
    st.write(
        "Baseline retrieval only. This path returns top patents from the persistent local index "
        "without planner-based context reuse, learned reranking, answer synthesis, or verification."
    )
    with st.expander(f"{title}: top patents", expanded=True):
        st.dataframe(ranked_table(result["ranked"]), use_container_width=True, hide_index=True)
    if show_evidence:
        with st.expander(f"{title}: optional evidence snippets", expanded=False):
            st.dataframe(
                query_evidence_table(result["snippets"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "evidence": st.column_config.TextColumn("evidence", width="large"),
                    "title": st.column_config.TextColumn("title", width="medium"),
                },
            )


def render_benchmark_mode(
    data_dir: str,
    top_k: int,
    retrieval_method: str,
    use_llm_decompose: bool,
    use_llm_verify: bool,
    llm_model: str,
    embedding_model: str,
    reranker_model: str,
) -> None:
    st.info(
        "Benchmark is the labeled PAR4PC evaluation path. "
        "Use `local-embedding` as the stable default. "
        "Compare against `linear-patent-reranker` as the current learned benchmark experiment."
    )
    case_paths = list_case_paths(data_dir)
    if not case_paths:
        st.error("No PAR4PC JSON files found.")
        return

    case_path = st.selectbox(
        "Benchmark case",
        options=case_paths,
        format_func=lambda path: Path(path).name,
    )
    case = preview_case(case_path)

    st.chat_message("user").write(
        f"Analyze this claim against the benchmark candidate patents:\n\n{case.target_claim}"
    )

    run = st.button("Analyze Benchmark Case", type="primary")
    if not run:
        st.info("Choose a benchmark case and run the agent.")
        return

    with st.spinner("Running patent agent..."):
        result = run_graph(
            case_path=case_path,
            top_k=top_k,
            use_llm_decompose=use_llm_decompose,
            use_llm_verify=use_llm_verify,
            retrieval_method=retrieval_method,
            llm_model=llm_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    ranked = result["ranked"]
    limitations = result["limitations"]
    chart = result["claim_chart"]
    report = result["report"]
    predicted = [item.letter for item in ranked]
    gold = set(result["case"].gold_answers)
    recall = len(set(predicted) & gold) / len(gold) if gold else 0.0

    summary = (
        f"I ranked the candidate prior art for claim {case.claim_number}. "
        f"Top-{top_k}: {', '.join(predicted)}. "
        f"Gold: {', '.join(case.gold_answers) or 'N/A'}. "
        f"Hit@1={'yes' if set(predicted[:1]) & gold else 'no'}, "
        f"Hit@{top_k}={'yes' if set(predicted[:top_k]) & gold else 'no'}, "
        f"Recall@{top_k}={recall:.3f}."
    )
    st.chat_message("assistant").write(summary)

    tab_summary, tab_evidence, tab_report = st.tabs(["Summary", "Evidence", "Report"])
    with tab_summary:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Application", case.application_number)
        meta_cols[1].metric("Claim", str(case.claim_number))
        meta_cols[2].metric("Gold", ", ".join(case.gold_answers) or "N/A")
        meta_cols[3].metric("Candidates", str(len(case.candidates)))
        st.dataframe(ranked_table(ranked), use_container_width=True, hide_index=True)
        st.dataframe(limitation_table(limitations), use_container_width=True, hide_index=True)
    with tab_evidence:
        st.dataframe(
            claim_chart_table(chart),
            use_container_width=True,
            hide_index=True,
            column_config={
                "evidence": st.column_config.TextColumn("evidence", width="large"),
                "reason": st.column_config.TextColumn("reason", width="medium"),
            },
        )
    with tab_report:
        st.markdown(report)
        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name=f"{Path(case_path).stem}_report.md",
            mime="text/markdown",
        )

    if result.get("warnings"):
        st.warning("\n".join(result["warnings"]))


def render_free_text_mode(
    data_dir: str,
    top_k: int,
    product_variant: str,
    retrieval_method: str,
    baseline_retrieval_method: str,
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    use_llm_answer: bool,
    use_llm_planner: bool,
    use_llm_retrieval_decompose: bool,
    use_llm_query_expansion: bool,
    use_llm_answer_verification: bool,
    pool_source: str,
    hub_rows_per_split: int,
    index_dir: str,
    show_baseline_evidence: bool,
) -> None:
    st.info(
        "Our product UI compares a normal patent RAG baseline against our optimized patent agent. "
        "The baseline uses persistent-index retrieval only. "
        "Our optimized path uses persistent-index coarse recall, learned linear patent reranking, "
        "conversational context reuse, evidence extraction, grounded answer generation, and answer verification."
    )
    pool = None
    if pool_source == "Persistent local index":
        if not index_exists(index_dir):
            st.error(
                "Persistent index not found. Build it first with "
                f"`python -m src.build_patent_index --index-dir \"{index_dir}\"`."
            )
            return
        manifest = load_persistent_manifest(index_dir)
        st.caption(
            f"Persistent local index: {manifest.patent_count} patents | model={manifest.embedding_model}"
        )
    else:
        if pool_source != "Local sample pool":
            st.info("The first Hub-backed search downloads and caches PAR4PC parquet files locally.")
        pool = load_pool(data_dir, pool_source, hub_rows_per_split)
        st.caption(f"{pool_source}: {len(pool)} patents")

    if "free_text_messages" not in st.session_state:
        st.session_state.free_text_messages = [
            {
                "role": "assistant",
                "content": (
                    "Send patent-related text, a claim, or an invention description. "
                    "I will search the indexed patent pool, rank the closest patents, "
                    "and show supporting evidence snippets."
                ),
            }
        ]
    if "free_text_agent_state" not in st.session_state:
        st.session_state.free_text_agent_state = {
            "last_ranked": [],
            "last_snippets": [],
            "last_plan": None,
            "working_patents": [],
            "last_query": "",
        }

    for message in st.session_state.free_text_messages:
        st.chat_message(message["role"]).write(message["content"])

    if st.button("Use example query"):
        st.session_state.free_text_pending_query = DEFAULT_FREE_TEXT

    query = st.chat_input("Ask for related patents")
    if not query:
        query = st.session_state.pop("free_text_pending_query", "")
    if not query:
        return

    st.session_state.free_text_messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    agent_state = st.session_state.free_text_agent_state
    with st.spinner("Searching related patents..."):
        optimized_result = execute_free_text_path(
            query_text=query,
            agent_state=agent_state,
            pool=pool,
            retrieval_method=retrieval_method,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            top_k=top_k,
            pool_source=pool_source,
            index_dir=index_dir,
            llm_model=llm_model,
            use_llm_answer=use_llm_answer,
            use_llm_answer_verification=use_llm_answer_verification,
            use_llm_planner=use_llm_planner,
            use_llm_retrieval_decompose=use_llm_retrieval_decompose,
            use_llm_query_expansion=use_llm_query_expansion,
            use_context=True,
        )
        baseline_result = None
        if product_variant in {"Normal RAG baseline", "Side-by-side comparison"}:
            baseline_result = execute_free_text_path(
                query_text=query,
                agent_state={"last_ranked": [], "working_patents": [], "last_snippets": [], "last_plan": None, "last_query": ""},
                pool=pool,
                retrieval_method=baseline_retrieval_method,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                top_k=top_k,
                pool_source=pool_source,
                index_dir=index_dir,
                llm_model=llm_model,
                use_llm_answer=False,
                use_llm_answer_verification=False,
                use_llm_planner=False,
                use_llm_retrieval_decompose=False,
                use_llm_query_expansion=False,
                use_context=False,
            )
        if product_variant == "Our optimized patent agent":
            selected_result = optimized_result
        elif product_variant == "Normal RAG baseline":
            selected_result = baseline_result
        else:
            selected_result = optimized_result

    agent_state.update(optimized_result["state"])

    if product_variant == "Side-by-side comparison":
        compare_cols = st.columns(2)
        with compare_cols[0]:
            render_baseline_retrieval_block(
                "Normal RAG baseline",
                baseline_result,
                show_evidence=show_baseline_evidence,
            )
        with compare_cols[1]:
            render_free_text_result_block("Our optimized patent agent", optimized_result, expanded=True)
        st.session_state.free_text_messages.append(
            {
                "role": "assistant",
                "content": (
                    "Displayed side-by-side comparison: baseline retrieval only versus the full optimized patent agent."
                ),
            }
        )
        return
    else:
        if product_variant == "Normal RAG baseline":
            st.session_state.free_text_messages.append(
                {
                    "role": "assistant",
                    "content": "Baseline retrieval completed. Review the top patents below.",
                }
            )
            st.chat_message("assistant").write("Baseline retrieval completed. Review the top patents below.")
            if selected_result["warnings"]:
                st.warning("\n".join(selected_result["warnings"]))
            render_baseline_retrieval_block(
                "Normal RAG baseline",
                selected_result,
                show_evidence=show_baseline_evidence,
            )
            return

        st.session_state.free_text_messages.append({"role": "assistant", "content": selected_result["answer"]})
        st.chat_message("assistant").write(selected_result["answer"])
        if selected_result["warnings"]:
            st.warning("\n".join(selected_result["warnings"]))

        with st.expander("Agent Decision"):
            st.write(f"Intent: `{selected_result['plan'].intent}`")
            st.write(f"Action: `{selected_result['plan'].action}`")
            st.write(f"Reason: {selected_result['plan'].reason}")
            if selected_result["effective_query"] != query:
                st.write("Context-enriched query:")
                st.code(selected_result["effective_query"])
        with st.expander("Answer Verification"):
            st.write(f"Status: `{selected_result['answer_verification'].status}`")
            st.write(f"Reason: {selected_result['answer_verification'].reason}")

        with st.expander("Ranked Patents", expanded=True):
            st.dataframe(ranked_table(selected_result["ranked"]), use_container_width=True, hide_index=True)

        with st.expander("Supporting Evidence", expanded=True):
            st.dataframe(
                query_evidence_table(selected_result["snippets"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "evidence": st.column_config.TextColumn("evidence", width="large"),
                    "title": st.column_config.TextColumn("title", width="medium"),
                },
            )
        with st.expander("Retrieval Summary"):
            st.dataframe(
                selected_result["evidence_df"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "evidence": st.column_config.TextColumn("evidence", width="large"),
                },
            )
            st.markdown(selected_result["summary"])


def main() -> None:
    st.set_page_config(page_title="Patent Prior-Art Agent", layout="wide")
    st.title("Patent Prior-Art Agent")
    st.caption("Use the product-style agent for exploratory patent search, or run the labeled benchmark path.")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", options=["Our Patent Agent", "Benchmark"], index=0)
        if mode == "Our Patent Agent":
            st.caption("Default UI path. Compares a normal RAG baseline against our optimized product pipeline.")
        else:
            st.caption("Evaluation path. Compare the PatentSBERTa baseline against our learned reranker.")
        data_dir = st.text_input("PAR4PC data directory", value=str(DEFAULT_DATA_DIR))
        top_k = st.slider("Top-k prior art", min_value=1, max_value=8, value=3)
        show_experimental = False
        retrieval_method = OPTIMIZED_FREE_TEXT_METHOD
        baseline_retrieval_method = BASELINE_FREE_TEXT_METHOD
        pool_source = "Persistent local index"
        index_dir = str(DEFAULT_INDEX_DIR)
        hub_rows_per_split = 500
        product_variant = "Our optimized patent agent"
        llm_model = "gpt-4o-mini"
        embedding_model = ""
        reranker_model = DEFAULT_RERANKER_MODEL
        use_llm_answer = False
        use_llm_planner = False
        use_llm_retrieval_decompose = False
        use_llm_query_expansion = False
        use_llm_answer_verification = False
        use_llm_decompose = False
        use_llm_verify = False
        show_baseline_evidence = True

        if mode == "Benchmark":
            benchmark_choice = st.selectbox(
                "Benchmark method",
                options=["PatentSBERTa baseline", "Our learned reranker"],
                index=0,
            )
            retrieval_method = (
                "linear-patent-reranker"
                if benchmark_choice == "Our learned reranker"
                else "local-embedding"
            )
            st.caption(BENCHMARK_METHOD_HELP[retrieval_method])
            if retrieval_method == "linear-patent-reranker":
                st.caption(
                    "First use trains and saves a small linear reranker under `data/models/`. "
                    "Later runs load the cached model."
                )
        else:
            product_variant = st.selectbox(
                "Product view",
                options=["Our optimized patent agent", "Normal RAG baseline", "Side-by-side comparison"],
                index=0,
            )
            st.caption(PRODUCT_VARIANT_HELP[product_variant])
            if product_variant == "Normal RAG baseline":
                st.caption("Pipeline: persistent local index -> top patents. Optional evidence snippets only.")
            else:
                st.caption(
                    "Pipeline: persistent local index coarse recall -> linear-patent-reranker second-stage rerank "
                    "-> planner / working-set reuse -> evidence extraction -> grounded answer -> answer verification."
                )
            st.caption(f"Patent pool: {POOL_HELP['Persistent local index']}")

        with st.expander("Advanced Settings"):
            if mode == "Benchmark":
                show_experimental = st.checkbox("Show additional benchmark methods", value=False)
                if show_experimental:
                    benchmark_method_override = st.selectbox(
                        "Benchmark retrieval override",
                        options=BENCHMARK_EXPERIMENTAL_OPTIONS,
                        format_func=lambda method: METHOD_LABELS.get(method, method),
                    )
                    retrieval_method = benchmark_method_override
                    st.caption(BENCHMARK_METHOD_HELP.get(retrieval_method, ""))
            else:
                if product_variant in {"Normal RAG baseline", "Side-by-side comparison"}:
                    st.caption("Baseline controls")
                    show_baseline_evidence = st.checkbox("Show optional evidence snippets for baseline view", value=True)
                    override_baseline = st.checkbox("Override baseline retrieval", value=False)
                    if override_baseline:
                        baseline_retrieval_method = st.selectbox(
                            "Baseline retrieval method",
                            options=["local-embedding", "bm25"],
                            index=0,
                            format_func=lambda method: METHOD_LABELS.get(method, method),
                        )
                        st.caption(FREE_TEXT_METHOD_HELP.get(baseline_retrieval_method, ""))

                if product_variant in {"Our optimized patent agent", "Side-by-side comparison"}:
                    st.caption("Optimized agent controls")
                    override_search_backend = st.checkbox("Override optimized retrieval defaults", value=False)
                    if override_search_backend:
                        retrieval_options = list(FREE_TEXT_PRIMARY_RETRIEVAL_OPTIONS)
                        show_experimental = st.checkbox("Show experimental optimized methods", value=False)
                        if show_experimental:
                            retrieval_options.extend(FREE_TEXT_EXPERIMENTAL_OPTIONS)
                        retrieval_method = st.selectbox(
                            "Optimized retrieval method",
                            options=retrieval_options,
                            index=retrieval_options.index(OPTIMIZED_FREE_TEXT_METHOD) if OPTIMIZED_FREE_TEXT_METHOD in retrieval_options else 0,
                            format_func=lambda method: METHOD_LABELS.get(method, method),
                        )
                        st.caption(FREE_TEXT_METHOD_HELP.get(retrieval_method, ""))
                        pool_source = st.selectbox(
                            "Optimized patent pool",
                            options=["Persistent local index", "Local sample pool", "Hub PAR4PC pool", "Combined"],
                            index=0,
                            help="Optimized product mode searches this pool.",
                        )
                        st.caption(f"Selected patent pool: {POOL_HELP.get(pool_source, '')}")
                        hub_rows_per_split = st.number_input(
                            "Hub rows per split for optimized pool",
                            min_value=0,
                            max_value=54028,
                            value=500,
                            step=250,
                            help="0 loads full train/validation/test splits. Smaller values start faster.",
                        )
                    use_llm_answer = st.checkbox("Use LLM grounded answer", value=False)
                    use_llm_planner = st.checkbox("Use LLM planner", value=False)
                    use_llm_retrieval_decompose = st.checkbox("Use LLM retrieval decomposition", value=False)
                    use_llm_query_expansion = st.checkbox("Use LLM query expansion", value=False)
                    use_llm_answer_verification = st.checkbox("Use LLM answer verification", value=False)
                    llm_model = st.text_input("LLM model", value="gpt-4o-mini")

                embedding_model = st.text_input(
                    "Embedding model override",
                    value="",
                    help="Leave blank for the method default.",
                )
                index_dir = st.text_input("Persistent index directory", value=str(DEFAULT_INDEX_DIR))

            if mode == "Benchmark":
                use_llm_answer = False
                use_llm_planner = False
                use_llm_retrieval_decompose = False
                use_llm_query_expansion = False
                use_llm_answer_verification = False
                use_llm_decompose = st.checkbox("Use LLM claim decomposition")
                use_llm_verify = st.checkbox("Use LLM evidence verification")
                llm_model = st.text_input("LLM model", value="gpt-4o-mini")
                embedding_model = st.text_input(
                    "Embedding model override",
                    value="",
                    help="Leave blank for the method default.",
                )
                reranker_model = st.text_input("Local reranker model", value=DEFAULT_RERANKER_MODEL)
                index_dir = st.text_input("Persistent index directory", value=str(DEFAULT_INDEX_DIR))

        if mode == "Our Patent Agent" and st.button("Preload search backend"):
            with st.spinner("Loading search pool and model cache..."):
                pool_size, model_name = warm_up_search_backend(
                    data_dir=data_dir,
                    pool_source=pool_source,
                    hub_rows_per_split=int(hub_rows_per_split),
                    retrieval_method=retrieval_method,
                    embedding_model=embedding_model,
                    index_dir=index_dir,
                )
            detail = f"Loaded {pool_size} patents"
            if model_name:
                detail += f" and warmed {model_name}"
            st.success(detail)

        needs_openai = (
            use_llm_answer
            or use_llm_planner
            or use_llm_retrieval_decompose
            or use_llm_query_expansion
            or use_llm_answer_verification
            or use_llm_decompose
            or use_llm_verify
            or retrieval_method in {"openai-embedding", "llm-rerank"}
        )
        if needs_openai and not openai_available():
            st.warning("OPENAI_API_KEY is not set. OpenAI-dependent paths will fall back.")

    if mode == "Benchmark":
        render_benchmark_mode(
            data_dir=data_dir,
            top_k=top_k,
            retrieval_method=retrieval_method,
            use_llm_decompose=use_llm_decompose,
            use_llm_verify=use_llm_verify,
            llm_model=llm_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    else:
        render_free_text_mode(
            data_dir=data_dir,
            top_k=top_k,
            product_variant=product_variant,
            retrieval_method=retrieval_method,
            baseline_retrieval_method=baseline_retrieval_method,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            use_llm_answer=use_llm_answer,
            use_llm_planner=use_llm_planner,
            use_llm_retrieval_decompose=use_llm_retrieval_decompose,
            use_llm_query_expansion=use_llm_query_expansion,
            use_llm_answer_verification=use_llm_answer_verification,
            pool_source=pool_source,
            hub_rows_per_split=int(hub_rows_per_split),
            index_dir=index_dir,
            show_baseline_evidence=show_baseline_evidence,
        )


if __name__ == "__main__":
    main()
