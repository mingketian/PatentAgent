from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.claim_analysis import (
    apply_verification_heuristic,
    build_claim_chart,
    decompose_claim_heuristic,
    render_report,
)
from src.data_loader import load_par4pc_case
from src.llm_tools import decompose_claim_llm, openai_available, rerank_prior_art_llm, verify_evidence_llm
from src.patent_rerank import rank_candidates_hybrid_coverage, rank_candidates_patent_specialized
from src.retrieval import (
    rank_candidates_bm25,
    rank_candidates_cross_encoder,
    rank_candidates_local_embeddings,
    rank_candidates_openai_embeddings,
    results_from_ordered_letters,
)
from src.train_linear_patent_reranker import rank_case_with_default_linear_reranker


class PatentAgentState(TypedDict, total=False):
    case_path: str
    top_k: int
    use_llm_decompose: bool
    use_llm_verify: bool
    retrieval_method: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    warnings: list[str]
    case: Any
    limitations: list[Any]
    ranked: list[Any]
    claim_chart: list[Any]
    report: str


def load_case_node(state: PatentAgentState) -> PatentAgentState:
    return {"case": load_par4pc_case(state["case_path"])}


def decompose_claim_node(state: PatentAgentState) -> PatentAgentState:
    case = state["case"]
    if state.get("use_llm_decompose"):
        if openai_available():
            return {"limitations": decompose_claim_llm(case.target_claim, model=state.get("llm_model"))}
        warnings = list(state.get("warnings", []))
        warnings.append("OPENAI_API_KEY not set; used heuristic claim decomposition.")
        return {
            "limitations": decompose_claim_heuristic(case.target_claim),
            "warnings": warnings,
        }
    return {"limitations": decompose_claim_heuristic(case.target_claim)}


def retrieve_prior_art_node(state: PatentAgentState) -> PatentAgentState:
    case = state["case"]
    top_k = int(state.get("top_k", 3))
    retrieval_method = state.get("retrieval_method", "bm25")

    if retrieval_method == "bm25":
        return {"ranked": rank_candidates_bm25(case, top_k=top_k)}

    if retrieval_method == "openai-embedding":
        if openai_available():
            return {
                "ranked": rank_candidates_openai_embeddings(
                    case,
                    top_k=top_k,
                    embedding_model=state.get("embedding_model") or "text-embedding-3-small",
                )
            }
        warnings = list(state.get("warnings", []))
        warnings.append("OPENAI_API_KEY not set; used BM25 retrieval instead of OpenAI embeddings.")
        return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "local-cross-encoder":
        try:
            return {
                "ranked": rank_candidates_cross_encoder(
                    case,
                    top_k=top_k,
                    reranker_model=state.get("reranker_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                )
            }
        except ImportError as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(f"sentence-transformers not installed; used BM25 instead. Detail: {exc}")
            return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "local-embedding":
        try:
            return {
                "ranked": rank_candidates_local_embeddings(
                    case,
                    top_k=top_k,
                    embedding_model=state.get("embedding_model") or "AI-Growth-Lab/PatentSBERTa",
                )
            }
        except ImportError as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(f"sentence-transformers not installed; used BM25 instead. Detail: {exc}")
            return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "hybrid-coverage":
        try:
            return {
                "ranked": rank_candidates_hybrid_coverage(
                    case,
                    top_k=top_k,
                    embedding_model=state.get("embedding_model") or "AI-Growth-Lab/PatentSBERTa",
                )
            }
        except ImportError as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(f"sentence-transformers not installed; used BM25 instead. Detail: {exc}")
            return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "patent-specialized":
        try:
            return {
                "ranked": rank_candidates_patent_specialized(
                    case,
                    top_k=top_k,
                    embedding_model=state.get("embedding_model") or "AI-Growth-Lab/PatentSBERTa",
                    use_query_expansion=True,
                    use_llm_expansion=False,
                    use_llm_decompose=bool(state.get("use_llm_decompose")),
                    llm_model=state.get("llm_model") or "",
                )
            }
        except ImportError as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(f"sentence-transformers not installed; used BM25 instead. Detail: {exc}")
            return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "linear-patent-reranker":
        try:
            ranked_letters = rank_case_with_default_linear_reranker(
                case,
                top_k=top_k,
                embedding_model=state.get("embedding_model") or "AI-Growth-Lab/PatentSBERTa",
            )
            letter_to_result = {
                result.letter: result
                for result in rank_candidates_local_embeddings(
                    case,
                    top_k=None,
                    embedding_model=state.get("embedding_model") or "AI-Growth-Lab/PatentSBERTa",
                )
            }
            ranked = [
                replace(letter_to_result[letter], score=score)
                for letter, score in ranked_letters
            ]
            return {"ranked": ranked}
        except ImportError as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(f"linear patent reranker unavailable; used BM25 instead. Detail: {exc}")
            return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}

    if retrieval_method == "llm-rerank":
        base_ranked = rank_candidates_bm25(case, top_k=None)
        if not openai_available():
            warnings = list(state.get("warnings", []))
            warnings.append("OPENAI_API_KEY not set; used BM25 retrieval instead of LLM reranking.")
            return {"ranked": base_ranked[:top_k], "warnings": warnings}
        candidates = {
            result.letter: case.candidates[result.letter].retrieval_text
            for result in base_ranked
        }
        reranked = rerank_prior_art_llm(
            target_claim=case.target_claim,
            candidates=candidates,
            model=state.get("llm_model"),
        )
        warnings = list(state.get("warnings", []))
        warnings.append(f"LLM reranker reason: {reranked.reason}")
        return {
            "ranked": results_from_ordered_letters(case, reranked.ordered_letters, top_k=top_k),
            "warnings": warnings,
        }

    warnings = list(state.get("warnings", []))
    warnings.append(f"Unknown retrieval_method={retrieval_method!r}; used BM25 retrieval.")
    return {"ranked": rank_candidates_bm25(case, top_k=top_k), "warnings": warnings}


def extract_evidence_node(state: PatentAgentState) -> PatentAgentState:
    case = state["case"]
    top_k = int(state.get("top_k", 3))
    return {
        "claim_chart": build_claim_chart(
            case=case,
            ranked_candidates=state["ranked"],
            limitations=state["limitations"],
            top_candidates=top_k,
        )
    }


def verify_evidence_node(state: PatentAgentState) -> PatentAgentState:
    chart = state["claim_chart"]
    if state.get("use_llm_verify"):
        if not openai_available():
            warnings = list(state.get("warnings", []))
            warnings.append("OPENAI_API_KEY not set; used heuristic evidence verification.")
            return {
                "claim_chart": apply_verification_heuristic(chart),
                "warnings": warnings,
            }
        verified = []
        for row in chart:
            decision = verify_evidence_llm(row, model=state.get("llm_model"))
            verified.append(
                replace(
                    row,
                    verification=decision.status,
                    verification_reason=decision.reason,
                )
            )
        return {"claim_chart": verified}
    return {"claim_chart": apply_verification_heuristic(chart)}


def render_report_node(state: PatentAgentState) -> PatentAgentState:
    report = render_report(
        case=state["case"],
        limitations=state["limitations"],
        ranked=state["ranked"],
        chart=state["claim_chart"],
    )
    warnings = state.get("warnings", [])
    if warnings:
        warning_block = "\n".join(f"- {warning}" for warning in warnings)
        report += f"\n\n## Warnings\n\n{warning_block}\n"
    return {"report": report}


def build_graph():
    graph = StateGraph(PatentAgentState)
    graph.add_node("load_case", load_case_node)
    graph.add_node("decompose_claim", decompose_claim_node)
    graph.add_node("retrieve_prior_art", retrieve_prior_art_node)
    graph.add_node("extract_evidence", extract_evidence_node)
    graph.add_node("verify_evidence", verify_evidence_node)
    graph.add_node("render_report", render_report_node)

    graph.add_edge(START, "load_case")
    graph.add_edge("load_case", "decompose_claim")
    graph.add_edge("decompose_claim", "retrieve_prior_art")
    graph.add_edge("retrieve_prior_art", "extract_evidence")
    graph.add_edge("extract_evidence", "verify_evidence")
    graph.add_edge("verify_evidence", "render_report")
    graph.add_edge("render_report", END)
    return graph.compile()


def run_graph(
    case_path: str | Path,
    top_k: int = 3,
    use_llm_decompose: bool = False,
    use_llm_verify: bool = False,
    retrieval_method: str = "bm25",
    llm_model: str = "",
    embedding_model: str = "",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> PatentAgentState:
    app = build_graph()
    return app.invoke(
        {
            "case_path": str(case_path),
            "top_k": top_k,
            "use_llm_decompose": use_llm_decompose,
            "use_llm_verify": use_llm_verify,
            "retrieval_method": retrieval_method,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "warnings": [],
        }
    )
