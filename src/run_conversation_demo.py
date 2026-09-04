from __future__ import annotations

from pathlib import Path

from src.free_text_qa import gather_query_evidence, heuristic_rag_answer
from src.persistent_index import index_exists, load_persistent_candidates, search_persistent_index
from src.query_planner import classify_turn, enrich_query_with_context
from src.retrieval import rank_patent_pool_bm25, rank_patent_pool_local_embeddings


DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_demo")
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEMO_TURNS = [
    (
        "1. A method for leveraging social networks in physical gatherings, the method comprising: "
        "generating a profile for each participant at a physical gathering; receiving participant data; "
        "receiving a request for information from a participant; determining access to requested information; "
        "identifying data to fulfill the request; and providing the identified data."
    ),
    "Which of those also includes access control for the requested information?",
    "Compare the top two patents for event participant context and profile handling.",
    "If I combine that with smart invitations, what related patents should I inspect next?",
]


def main() -> None:
    index_dir = str(DEFAULT_INDEX_DIR)
    if not index_exists(index_dir):
        raise SystemExit(
            f"Persistent index not found at {index_dir}. Build it first with python -m src.build_patent_index."
        )

    agent_state = {
        "last_ranked": [],
        "working_patents": [],
    }

    for turn_number, query in enumerate(DEMO_TURNS, start=1):
        plan = classify_turn(query, has_context=bool(agent_state["working_patents"]))
        effective_query = enrich_query_with_context(query, plan, agent_state["last_ranked"])

        if plan.action == "retrieve_new":
            ranked = search_persistent_index(
                effective_query,
                index_dir=index_dir,
                top_k=3,
            )
            working_patents = [result.candidate for result in ranked]
        else:
            working_patents = agent_state["working_patents"] or load_persistent_candidates(index_dir)
            ranked = rank_patent_pool_local_embeddings(
                query,
                working_patents,
                top_k=min(3, len(working_patents)),
                embedding_model=DEFAULT_EMBEDDING_MODEL,
            )

        snippets = gather_query_evidence(query, ranked, snippets_per_patent=2)
        answer = heuristic_rag_answer(query, ranked, snippets, plan=plan)

        print(f"\nTurn {turn_number}")
        print("-------")
        print(f"User: {query}")
        print(f"Intent: {plan.intent}")
        print(f"Action: {plan.action}")
        print(f"Reason: {plan.reason}")
        if effective_query != query:
            print(f"Context query: {effective_query}")
        print("\nAnswer:")
        print(answer)
        print("\nTop patents:")
        for index, result in enumerate(ranked, start=1):
            print(f"{index}. {result.patent_id} | {result.score:.3f} | {result.title}")

        agent_state["last_ranked"] = ranked
        agent_state["working_patents"] = working_patents


if __name__ == "__main__":
    main()
