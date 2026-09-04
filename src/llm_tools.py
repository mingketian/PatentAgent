from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.claim_analysis import ClaimLimitation, EvidenceMatch, VerificationResult
from src.free_text_qa import QueryEvidenceSnippet, build_rag_context
from src.query_planner import TurnAction, TurnIntent, TurnPlan
from src.prompts import (
    CLAIM_DECOMPOSE_SYSTEM,
    CLAIM_DECOMPOSE_USER,
    PRIOR_ART_RERANK_SYSTEM,
    PRIOR_ART_RERANK_USER,
    QUERY_EXPANSION_SYSTEM,
    QUERY_EXPANSION_USER,
    RAG_ANSWER_SYSTEM,
    RAG_ANSWER_USER,
    RAG_VERIFY_SYSTEM,
    RAG_VERIFY_USER,
    TURN_PLAN_SYSTEM,
    TURN_PLAN_USER,
    VERIFY_EVIDENCE_SYSTEM,
    VERIFY_EVIDENCE_USER,
)


class ClaimDecompositionOutput(BaseModel):
    limitations: list[str] = Field(description="Ordered claim limitations.")


class EvidenceVerificationOutput(BaseModel):
    status: Literal["supported", "partially_supported", "unsupported"]
    reason: str = Field(description="One concise sentence explaining the decision.")


class PriorArtRerankOutput(BaseModel):
    ordered_letters: list[str] = Field(description="Candidate letters ranked from most to least relevant.")
    reason: str = Field(description="Concise explanation for the top-ranked candidates.")


class RagAnswerOutput(BaseModel):
    answer: str = Field(description="Grounded answer with inline patent citations.")
    citations: list[str] = Field(description="Patent/source citations used in the answer.")
    insufficiency_note: str = Field(
        default="",
        description="Optional note when evidence is weak or incomplete.",
    )


class TurnPlanOutput(BaseModel):
    intent: Literal[
        "new_search",
        "follow_up_on_previous_results",
        "compare_previous_results",
        "aspect_filter",
        "similar_patent_search",
        "combination_exploration",
    ]
    action: Literal["retrieve_new", "rerank_existing", "reuse_context"]
    reason: str = Field(description="Short explanation for the decision.")


class QueryExpansionOutput(BaseModel):
    variants: list[str] = Field(description="Patent-oriented search query variants.")


class RagVerificationOutput(BaseModel):
    status: Literal["supported", "partially_supported", "unsupported"]
    reason: str = Field(description="Short explanation of support quality.")


def openai_available() -> bool:
    load_dotenv(Path.cwd() / ".env")
    return bool(os.getenv("OPENAI_API_KEY"))


def _chat_model(model: str | None = None) -> ChatOpenAI:
    load_dotenv(Path.cwd() / ".env")
    return ChatOpenAI(
        model=model or os.getenv("PATENT_AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def decompose_claim_llm(claim_text: str, model: str | None = None) -> list[ClaimLimitation]:
    llm = _chat_model(model).with_structured_output(ClaimDecompositionOutput)
    response = llm.invoke(
        [
            ("system", CLAIM_DECOMPOSE_SYSTEM),
            ("user", CLAIM_DECOMPOSE_USER.format(claim_text=claim_text)),
        ]
    )
    return [
        ClaimLimitation(label=f"L{i + 1}", text=limitation.strip())
        for i, limitation in enumerate(response.limitations)
        if limitation.strip()
    ]


def verify_evidence_llm(row: EvidenceMatch, model: str | None = None) -> VerificationResult:
    llm = _chat_model(model).with_structured_output(EvidenceVerificationOutput)
    response = llm.invoke(
        [
            ("system", VERIFY_EVIDENCE_SYSTEM),
            (
                "user",
                VERIFY_EVIDENCE_USER.format(
                    limitation_text=row.limitation_text,
                    candidate_letter=row.candidate_letter,
                    patent_id=row.patent_id,
                    source=row.source,
                    evidence=row.evidence,
                ),
            ),
        ]
    )
    return VerificationResult(status=response.status, reason=response.reason)


def rerank_prior_art_llm(
    target_claim: str,
    candidates: dict[str, str],
    model: str | None = None,
) -> PriorArtRerankOutput:
    candidate_block = "\n\n".join(
        f"{letter}: {text[:3000]}" for letter, text in sorted(candidates.items())
    )
    llm = _chat_model(model).with_structured_output(PriorArtRerankOutput)
    return llm.invoke(
        [
            ("system", PRIOR_ART_RERANK_SYSTEM),
            (
                "user",
                PRIOR_ART_RERANK_USER.format(
                    target_claim=target_claim,
                    candidates=candidate_block,
                ),
            ),
        ]
    )


def answer_query_with_rag(
    query_text: str,
    snippets: list[QueryEvidenceSnippet],
    model: str | None = None,
) -> RagAnswerOutput:
    evidence_block = build_rag_context(snippets)
    llm = _chat_model(model).with_structured_output(RagAnswerOutput)
    return llm.invoke(
        [
            ("system", RAG_ANSWER_SYSTEM),
            ("user", RAG_ANSWER_USER.format(query_text=query_text, evidence_block=evidence_block)),
        ]
    )


def plan_turn_llm(
    query_text: str,
    has_context: bool,
    previous_titles: list[str],
    model: str | None = None,
) -> TurnPlan:
    llm = _chat_model(model).with_structured_output(TurnPlanOutput)
    response = llm.invoke(
        [
            ("system", TURN_PLAN_SYSTEM),
            (
                "user",
                TURN_PLAN_USER.format(
                    query_text=query_text,
                    has_context="yes" if has_context else "no",
                    previous_titles="; ".join(previous_titles) if previous_titles else "none",
                ),
            ),
        ]
    )
    return TurnPlan(
        intent=response.intent,
        action=response.action,
        reason=response.reason,
        query_text=query_text,
    )


def expand_query_llm(query_text: str, model: str | None = None) -> list[str]:
    llm = _chat_model(model).with_structured_output(QueryExpansionOutput)
    response = llm.invoke(
        [
            ("system", QUERY_EXPANSION_SYSTEM),
            ("user", QUERY_EXPANSION_USER.format(query_text=query_text)),
        ]
    )
    variants = [query_text]
    seen = {query_text.strip().lower()}
    for item in response.variants:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            variants.append(normalized)
            seen.add(normalized.lower())
    return variants


def verify_rag_answer_llm(
    answer_text: str,
    snippets: list[QueryEvidenceSnippet],
    model: str | None = None,
) -> VerificationResult:
    evidence_block = build_rag_context(snippets)
    llm = _chat_model(model).with_structured_output(RagVerificationOutput)
    response = llm.invoke(
        [
            ("system", RAG_VERIFY_SYSTEM),
            ("user", RAG_VERIFY_USER.format(answer_text=answer_text, evidence_block=evidence_block)),
        ]
    )
    return VerificationResult(status=response.status, reason=response.reason)
