"""
Patent Prior-Art Agent — Web Demo Backend
FastAPI server wrapping the existing patent agent pipeline.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.claim_analysis import ClaimLimitation, VerificationResult, extract_evidence_for_candidate
from src.data_loader import (
    PatentCandidate,
    combine_patent_pools,
    load_hf_par4pc_patent_pool,
    load_par4pc_case,
    load_unique_patent_pool,
)
from src.free_text_qa import (
    gather_query_evidence,
    heuristic_rag_answer,
    verify_rag_answer_heuristic,
)
from src.llm_tools import (
    answer_query_with_rag,
    openai_available,
    plan_turn_llm,
    verify_rag_answer_llm,
)
from src.patent_rerank import (
    rank_patent_pool_hybrid_coverage,
    rank_patent_pool_patent_specialized,
)
from src.persistent_index import (
    index_exists,
    load_persistent_candidates,
    load_persistent_manifest,
    search_persistent_index,
)
from src.query_planner import TurnPlan, classify_turn, enrich_query_with_context
from src.retrieval import (
    _sentence_transformer_model,
    rank_patent_pool_bm25,
    rank_patent_pool_cross_encoder,
    rank_patent_pool_local_embeddings,
)
from src.train_linear_patent_reranker import rank_patent_pool_with_default_linear_reranker

# ── constants ─────────────────────────────────────────────────
DEFAULT_DATA_DIR = str(ROOT / ".." / "PANORAMA" / "data" / "benchmark" / "par4pc")
_FULL_INDEX_DIR = str(ROOT / "data" / "indexes" / "par4pc_patentsberta_full")
_DEMO_INDEX_DIR = str(ROOT / "data" / "indexes" / "par4pc_patentsberta_demo")


def _resolve_index_dir() -> str:
    """Prefer the full 17,877-patent index; fall back to the committed demo index.

    PATENT_AGENT_INDEX_DIR overrides both. The full index is ~133 MB and is not
    committed (see data/README.md), so a fresh clone would otherwise start with no
    searchable corpus at all.
    """
    override = os.environ.get("PATENT_AGENT_INDEX_DIR", "").strip()
    if override:
        return override
    if index_exists(_FULL_INDEX_DIR):
        return _FULL_INDEX_DIR
    return _DEMO_INDEX_DIR


DEFAULT_INDEX_DIR = _resolve_index_dir()
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BASELINE_METHOD = "local-embedding"
OPTIMIZED_METHOD = "linear-patent-reranker"

SESSION_TTL_SECONDS = 3600  # expire sessions after 1 hour

# ── app ───────────────────────────────────────────────────────
app = FastAPI(title="Patent Prior-Art Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── session store with TTL ────────────────────────────────────
sessions: dict[str, dict] = {}


def _get_session(sid: str) -> dict:
    now = time.time()
    # garbage-collect expired sessions (max 50 at a time to stay fast)
    expired = [k for k, v in list(sessions.items())[:200] if now - v.get("_ts", 0) > SESSION_TTL_SECONDS]
    for k in expired[:50]:
        sessions.pop(k, None)

    if sid not in sessions:
        sessions[sid] = {
            "last_ranked": [],
            "last_snippets": [],
            "last_plan": None,
            "working_patents": [],
            "last_query": "",
            "_ts": now,
        }
    sessions[sid]["_ts"] = now
    return sessions[sid]


# ── serialisation helpers ─────────────────────────────────────
def _ser_ranked(ranked: list, limit: int = 20) -> list[dict]:
    out = []
    for i, r in enumerate(ranked[:limit]):
        out.append({
            "rank": i + 1,
            "patent_id": r.patent_id,
            "title": r.title,
            "score": round(float(r.score), 4),
            "letter": getattr(r, "letter", ""),
        })
    return out


def _ser_snippets(snippets: list) -> list[dict]:
    return [
        {
            "patent_id": s.patent_id,
            "title": s.title,
            "source": s.source,
            "evidence": s.evidence[:600],
            "retrieval_score": round(float(s.retrieval_score), 4),
            "segment_score": round(float(s.segment_score), 4),
        }
        for s in snippets
    ]


def _ser_plan(plan: TurnPlan | None) -> dict | None:
    if plan is None:
        return None
    return {
        "intent": plan.intent,
        "action": plan.action,
        "reason": plan.reason,
    }


def _ser_verification(v) -> dict:
    return {"status": v.status, "reason": v.reason}


# ── search logic ──────────────────────────────────────────────
def _search_patents(
    query: str,
    method: str,
    pool: list[PatentCandidate] | None,
    top_k: int,
    index_dir: str,
    embedding_model: str,
    force_subset: bool = False,
) -> list:
    """
    Retrieve and rank patents.  When *force_subset* is True the provided
    *pool* is used directly (for reranking the working set) and the
    persistent index is never queried.
    """
    if method == "linear-patent-reranker":
        if not force_subset and pool is None and index_exists(index_dir):
            initial = search_persistent_index(
                query, index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        elif not force_subset and pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_with_default_linear_reranker(
            query_text=query, candidates=pool or [],
            top_k=top_k, embedding_model=embedding_model,
        )

    if method == "patent-specialized":
        if not force_subset and pool is None and index_exists(index_dir):
            initial = search_persistent_index(
                query, index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        elif not force_subset and pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_patent_specialized(
            query, pool or [], top_k=top_k,
            embedding_model=embedding_model,
            use_query_expansion=True,
        )

    if method == "hybrid-coverage":
        if not force_subset and pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_hybrid_coverage(
            query, pool or [], top_k=top_k,
            embedding_model=embedding_model,
        )

    if method == "local-cross-encoder":
        if not force_subset and pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_cross_encoder(
            query, pool or [], top_k=top_k,
            reranker_model=DEFAULT_RERANKER_MODEL,
        )

    if method == "local-embedding":
        if not force_subset and pool is None and index_exists(index_dir):
            return search_persistent_index(
                query, index_dir=index_dir,
                top_k=top_k, embedding_model=embedding_model,
            )
        if pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_local_embeddings(
            query, pool or [], top_k=top_k, embedding_model=embedding_model,
        )

    if method == "bm25":
        if pool is None:
            pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
        return rank_patent_pool_bm25(query, pool or [], top_k=top_k)

    # fallback → local-embedding
    if pool is None:
        pool = load_persistent_candidates(index_dir) if index_exists(index_dir) else []
    return rank_patent_pool_local_embeddings(
        query, pool or [], top_k=top_k, embedding_model=embedding_model,
    )


def _generate_answer(query: str, ranked, snippets, plan, use_llm: bool, llm_model: str):
    """Generate a grounded answer, with LLM if available and requested."""
    warnings: list[str] = []
    if use_llm:
        if openai_available():
            response = answer_query_with_rag(query, snippets, model=llm_model or None)
            answer = response.answer
            if response.insufficiency_note:
                answer += f"\n\nNote: {response.insufficiency_note}"
            return answer, warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic grounded answer.")
    return heuristic_rag_answer(query, ranked, snippets, plan=plan), warnings


def _verify_answer(answer: str, snippets, use_llm: bool, llm_model: str):
    """Verify an answer against evidence."""
    warnings: list[str] = []
    if use_llm:
        if openai_available():
            return verify_rag_answer_llm(answer, snippets, model=llm_model or None), warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic answer verification.")
    return verify_rag_answer_heuristic(answer, snippets), warnings


def _execute_pipeline(
    query: str,
    agent_state: dict,
    method: str,
    top_k: int,
    index_dir: str,
    embedding_model: str,
    use_context: bool,
    use_llm_answer: bool = False,
    use_llm_planner: bool = False,
    use_llm_verification: bool = False,
    llm_model: str = "",
) -> dict:
    """Run one pass of the product pipeline."""
    local = {
        "last_ranked": list(agent_state.get("last_ranked", [])),
        "last_snippets": list(agent_state.get("last_snippets", [])),
        "last_plan": agent_state.get("last_plan"),
        "working_patents": list(agent_state.get("working_patents", [])),
        "last_query": agent_state.get("last_query", ""),
    }
    all_warnings: list[str] = []

    # ── plan ──
    if use_context:
        if use_llm_planner and openai_available():
            plan = plan_turn_llm(
                query_text=query,
                has_context=bool(local["working_patents"]),
                previous_titles=[r.title for r in local["last_ranked"][:5]],
                model=llm_model or None,
            )
        else:
            plan = classify_turn(query, has_context=bool(local["working_patents"]))
            if use_llm_planner:
                all_warnings.append("OPENAI_API_KEY not set; used heuristic planner.")
        effective_query = enrich_query_with_context(query, plan, local["last_ranked"])
    else:
        plan = TurnPlan(intent="new_search", action="retrieve_new",
                        reason="Baseline path – fresh retrieval.", query_text=query)
        effective_query = query

    # ── retrieve ──
    if plan.action == "retrieve_new":
        ranked = _search_patents(
            effective_query, method, None, top_k, index_dir, embedding_model,
        )
        working = [r.candidate for r in ranked]
    elif plan.action == "reuse_context":
        ranked = local["last_ranked"]
        working = local["working_patents"]
    else:  # rerank_existing — force_subset to stay within working set
        working = local["working_patents"]
        ranked = _search_patents(
            query, method, working,
            min(top_k, len(working) or top_k),
            index_dir, embedding_model,
            force_subset=True,
        )

    # ── evidence ──
    snippets = gather_query_evidence(query, ranked, snippets_per_patent=2)

    # ── answer ──
    answer, answer_warnings = _generate_answer(
        query, ranked, snippets, plan,
        use_llm=use_llm_answer and use_context,
        llm_model=llm_model,
    )
    all_warnings.extend(answer_warnings)

    # ── verify ──
    verification, verify_warnings = _verify_answer(
        answer, snippets,
        use_llm=use_llm_verification and use_context,
        llm_model=llm_model,
    )
    all_warnings.extend(verify_warnings)

    # ── update state ──
    local["last_ranked"] = ranked
    local["last_snippets"] = snippets
    local["last_plan"] = plan
    local["working_patents"] = working
    local["last_query"] = query

    return {
        "pipeline": "optimized" if use_context else "baseline",
        "answer": answer,
        "plan": _ser_plan(plan),
        "effective_query": effective_query if effective_query != query else None,
        "ranked": _ser_ranked(ranked),
        "snippets": _ser_snippets(snippets),
        "verification": _ser_verification(verification),
        "warnings": all_warnings,
        "_state": local,
    }


# ── request / response models ────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    variant: str = "comparison"  # optimized | baseline | comparison
    top_k: int = 3
    session_id: str = ""
    use_llm_answer: bool = False
    use_llm_planner: bool = False
    use_llm_verification: bool = False
    llm_model: str = ""


class BenchmarkRequest(BaseModel):
    case_path: str
    method: str = "local-embedding"
    top_k: int = 3
    use_llm_decompose: bool = False
    use_llm_verify: bool = False
    llm_model: str = ""


# ── routes: static ───────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── routes: api ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    idx = index_exists(DEFAULT_INDEX_DIR)
    count = 0
    model = ""
    if idx:
        m = load_persistent_manifest(DEFAULT_INDEX_DIR)
        count = m.patent_count
        model = m.embedding_model
    return {
        "index_exists": idx,
        "patent_count": count,
        "embedding_model": model,
        "openai_available": openai_available(),
        "index_dir": DEFAULT_INDEX_DIR,
    }


@app.post("/api/preload")
def preload():
    if not index_exists(DEFAULT_INDEX_DIR):
        raise HTTPException(400, "Persistent index not found. Build it first.")
    m = load_persistent_manifest(DEFAULT_INDEX_DIR)
    _sentence_transformer_model(m.embedding_model)
    return {"status": "ok", "patent_count": m.patent_count, "model": m.embedding_model}


@app.post("/api/search")
def search(req: SearchRequest):
    sid = req.session_id or str(uuid.uuid4())
    session = _get_session(sid)

    results: dict[str, Any] = {"session_id": sid}

    if req.variant in ("optimized", "comparison"):
        opt = _execute_pipeline(
            query=req.query,
            agent_state=session,
            method=OPTIMIZED_METHOD,
            top_k=req.top_k,
            index_dir=DEFAULT_INDEX_DIR,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            use_context=True,
            use_llm_answer=req.use_llm_answer,
            use_llm_planner=req.use_llm_planner,
            use_llm_verification=req.use_llm_verification,
            llm_model=req.llm_model,
        )
        session.update(opt.pop("_state"))
        results["optimized"] = opt

    if req.variant in ("baseline", "comparison"):
        bl = _execute_pipeline(
            query=req.query,
            agent_state={
                "last_ranked": [], "last_snippets": [],
                "last_plan": None, "working_patents": [], "last_query": "",
            },
            method=BASELINE_METHOD,
            top_k=req.top_k,
            index_dir=DEFAULT_INDEX_DIR,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            use_context=False,
        )
        bl.pop("_state", None)
        results["baseline"] = bl

    return results


@app.get("/api/benchmark/cases")
def benchmark_cases():
    # Try local files first
    data_dir = Path(DEFAULT_DATA_DIR)
    if data_dir.exists():
        paths = sorted(data_dir.glob("par4pc_*.json"))
        if paths:
            return {"cases": [{"path": str(p), "name": p.name, "source": "local"} for p in paths]}

    # Fall back to HuggingFace cases
    try:
        from src.data_loader import load_hf_par4pc_cases
        hf_cases = load_hf_par4pc_cases(splits=("validation",), max_rows_per_split=20)
        return {
            "cases": [
                {
                    "path": f"hf:{i}",
                    "name": f"Case {c.application_number} (claim {c.claim_number})",
                    "source": "huggingface",
                }
                for i, c in enumerate(hf_cases)
            ],
            "_hf_cases_count": len(hf_cases),
        }
    except Exception as e:
        return {"cases": [], "error": f"No local data and HF load failed: {e}"}


@app.post("/api/benchmark/analyze")
def benchmark_analyze(req: BenchmarkRequest):
    try:
        from src.graph import run_graph
    except ImportError as e:
        raise HTTPException(500, f"Missing dependency: {e}")

    # Handle HuggingFace cases (path like "hf:3")
    if req.case_path.startswith("hf:"):
        hf_index = int(req.case_path.split(":")[1])
        from src.data_loader import load_hf_par4pc_cases
        from src.claim_analysis import decompose_claim_heuristic, build_claim_chart, apply_verification_heuristic, render_report
        from src.retrieval import rank_candidates_local_embeddings, rank_candidates_bm25

        hf_cases = load_hf_par4pc_cases(splits=("validation",), max_rows_per_split=20)
        if hf_index >= len(hf_cases):
            raise HTTPException(400, f"Case index {hf_index} out of range (max {len(hf_cases)-1})")
        case = hf_cases[hf_index]

        # Run the benchmark pipeline inline (same as graph nodes)
        limitations = decompose_claim_heuristic(case.target_claim)

        if req.method == "local-embedding":
            ranked = rank_candidates_local_embeddings(case, top_k=req.top_k, embedding_model=DEFAULT_EMBEDDING_MODEL)
        else:
            ranked = rank_candidates_bm25(case, top_k=req.top_k)

        chart = build_claim_chart(case, ranked, limitations, top_candidates=req.top_k)
        chart = apply_verification_heuristic(chart)
        report = render_report(case, limitations, ranked, chart)

        result = {
            "case": case,
            "ranked": ranked,
            "limitations": limitations,
            "claim_chart": chart,
            "report": report,
            "warnings": [],
        }
    else:
        result = run_graph(
            case_path=req.case_path,
            top_k=req.top_k,
            use_llm_decompose=req.use_llm_decompose,
            use_llm_verify=req.use_llm_verify,
            retrieval_method=req.method,
            llm_model=req.llm_model,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            reranker_model=DEFAULT_RERANKER_MODEL,
        )

    case = result["case"]
    ranked = result["ranked"]
    limitations = result["limitations"]
    chart = result["claim_chart"]
    report = result["report"]

    predicted = [r.letter for r in ranked]
    gold = set(case.gold_answers)
    recall = len(set(predicted) & gold) / len(gold) if gold else 0.0

    return {
        "application_number": case.application_number,
        "claim_number": case.claim_number,
        "target_claim": case.target_claim,
        "gold_answers": case.gold_answers,
        "candidates_count": len(case.candidates),
        "ranked": [
            {"rank": i + 1, "letter": r.letter, "patent_id": r.patent_id,
             "score": round(float(r.score), 4), "title": r.title}
            for i, r in enumerate(ranked)
        ],
        "limitations": [
            {"label": l.label, "text": l.text} for l in limitations
        ],
        "chart": [
            {
                "limitation": r.limitation_label,
                "candidate": f"{r.candidate_letter} ({r.patent_id})",
                "source": r.source,
                "score": round(float(r.score), 3),
                "verification": r.verification,
                "reason": r.verification_reason,
                "evidence": r.evidence[:500],
            }
            for r in chart
        ],
        "report": report,
        "metrics": {
            "hit_at_1": bool(set(predicted[:1]) & gold),
            "hit_at_k": bool(set(predicted[:req.top_k]) & gold),
            "recall_at_k": round(recall, 3),
            "predicted_top": predicted[:req.top_k],
        },
        "warnings": result.get("warnings", []),
    }


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899, log_level="info", timeout_keep_alive=600)
