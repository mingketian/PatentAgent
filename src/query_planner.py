from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TurnIntent = Literal[
    "new_search",
    "follow_up_on_previous_results",
    "compare_previous_results",
    "aspect_filter",
    "similar_patent_search",
    "combination_exploration",
]

TurnAction = Literal["retrieve_new", "rerank_existing", "reuse_context"]


@dataclass(frozen=True)
class TurnPlan:
    intent: TurnIntent
    action: TurnAction
    reason: str
    query_text: str


FOLLOW_UP_MARKERS = [
    "which of those",
    "which of them",
    "among those",
    "among them",
    "which one",
    "what about",
    "does it",
    "do they",
    "that one",
    "those patents",
    "these patents",
]

COMPARE_MARKERS = [
    "compare",
    "top two",
    "difference",
    "stronger match",
    "strongest match",
    "which is better",
]

ASPECT_MARKERS = [
    "also includes",
    "also include",
    "also has",
    "mentions",
    "handles",
    "focuses on",
    "covers access control",
    "covers",
]

SIMILAR_MARKERS = [
    "similar patent",
    "similar patents",
    "similar to this",
    "similar to that",
    "like this patent",
    "related to this patent",
]

COMBINATION_MARKERS = [
    "combine",
    "combination",
    "plus",
    "together with",
    "if i add",
    "if i combine",
]


def classify_turn(query_text: str, has_context: bool) -> TurnPlan:
    query = " ".join(query_text.lower().split())

    if has_context and any(marker in query for marker in COMPARE_MARKERS):
        return TurnPlan(
            intent="compare_previous_results",
            action="rerank_existing",
            reason="Comparison language points to the current working patent set.",
            query_text=query_text,
        )

    if has_context and any(marker in query for marker in ASPECT_MARKERS):
        return TurnPlan(
            intent="aspect_filter",
            action="rerank_existing",
            reason="The user is filtering previous results by a more specific aspect.",
            query_text=query_text,
        )

    if has_context and any(marker in query for marker in FOLLOW_UP_MARKERS):
        return TurnPlan(
            intent="follow_up_on_previous_results",
            action="rerank_existing",
            reason="The question refers back to previously retrieved patents.",
            query_text=query_text,
        )

    if has_context and any(marker in query for marker in SIMILAR_MARKERS):
        return TurnPlan(
            intent="similar_patent_search",
            action="retrieve_new",
            reason="The user is asking for a fresh similarity search anchored in prior context.",
            query_text=query_text,
        )

    if has_context and any(marker in query for marker in COMBINATION_MARKERS):
        return TurnPlan(
            intent="combination_exploration",
            action="retrieve_new",
            reason="Combination language suggests a broader search over the corpus, not only the current set.",
            query_text=query_text,
        )

    return TurnPlan(
        intent="new_search",
        action="retrieve_new",
        reason="Treating the turn as a fresh search request.",
        query_text=query_text,
    )


def enrich_query_with_context(query_text: str, plan: TurnPlan, last_ranked) -> str:
    if not last_ranked:
        return query_text

    if plan.intent == "combination_exploration":
        top_titles = "; ".join(result.title for result in last_ranked[:2])
        return f"{query_text}\n\nPrior retrieved patents for context: {top_titles}"

    if plan.intent == "similar_patent_search":
        lead = last_ranked[0]
        return f"{query_text}\n\nReference patent: {lead.patent_id} {lead.title}"

    return query_text
