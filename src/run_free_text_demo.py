from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import combine_patent_pools, load_hf_par4pc_patent_pool, load_unique_patent_pool
from src.free_text_qa import gather_query_evidence, heuristic_rag_answer, verify_rag_answer_heuristic
from src.llm_tools import answer_query_with_rag, openai_available, verify_rag_answer_llm
from src.patent_rerank import rank_patent_pool_hybrid_coverage, rank_patent_pool_patent_specialized
from src.persistent_index import index_exists, load_persistent_candidates, search_persistent_index
from src.retrieval import rank_patent_pool_bm25, rank_patent_pool_local_embeddings


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")
DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_demo")
DEFAULT_QUERY = (
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


def _load_pool(data_dir: str, pool_source: str, hub_rows_per_split: int):
    local_pool = load_unique_patent_pool(data_dir)
    if pool_source == "local":
        return local_pool
    hf_limit = None if hub_rows_per_split <= 0 else hub_rows_per_split
    hub_pool = load_hf_par4pc_patent_pool(max_rows_per_split=hf_limit)
    if pool_source == "hub":
        return hub_pool
    return combine_patent_pools(local_pool, hub_pool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run free-text patent retrieval with optional grounded QA.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--pool-source", choices=["persistent", "local", "hub", "combined"], default="persistent")
    parser.add_argument("--hub-rows-per-split", type=int, default=500)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument(
        "--retrieval-method",
        choices=["bm25", "local-embedding", "hybrid-coverage", "patent-specialized"],
        default="patent-specialized",
    )
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-answer", action="store_true")
    parser.add_argument("--llm-retrieval-decompose", action="store_true")
    parser.add_argument("--llm-query-expansion", action="store_true")
    parser.add_argument("--llm-answer-verification", action="store_true")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()

    if args.pool_source == "persistent":
        if not index_exists(args.index_dir):
            raise SystemExit(
                f"Persistent index not found at {args.index_dir}. "
                "Build it first with python -m src.build_patent_index."
            )
        if args.retrieval_method == "local-embedding":
            ranked = search_persistent_index(
                args.query,
                args.index_dir,
                top_k=args.top_k,
                embedding_model=args.embedding_model,
            )
        elif args.retrieval_method == "hybrid-coverage":
            ranked = rank_patent_pool_hybrid_coverage(
                args.query,
                load_persistent_candidates(args.index_dir),
                top_k=args.top_k,
                embedding_model=args.embedding_model or "AI-Growth-Lab/PatentSBERTa",
            )
        elif args.retrieval_method == "patent-specialized":
            initial = search_persistent_index(
                args.query,
                args.index_dir,
                top_k=max(args.top_k * 4, 12),
                embedding_model=args.embedding_model,
            )
            ranked = rank_patent_pool_patent_specialized(
                args.query,
                [item.candidate for item in initial],
                top_k=args.top_k,
                embedding_model=args.embedding_model or "AI-Growth-Lab/PatentSBERTa",
                use_query_expansion=True,
                use_llm_expansion=args.llm_query_expansion,
                use_llm_decompose=args.llm_retrieval_decompose,
                llm_model=args.llm_model,
            )
        else:
            ranked = rank_patent_pool_bm25(
                args.query,
                load_persistent_candidates(args.index_dir),
                top_k=args.top_k,
            )
    else:
        pool = _load_pool(args.data_dir, args.pool_source, args.hub_rows_per_split)
        if args.retrieval_method == "hybrid-coverage":
            ranked = rank_patent_pool_hybrid_coverage(
                args.query,
                pool,
                top_k=args.top_k,
                embedding_model=args.embedding_model or "AI-Growth-Lab/PatentSBERTa",
            )
        elif args.retrieval_method == "patent-specialized":
            ranked = rank_patent_pool_patent_specialized(
                args.query,
                pool,
                top_k=args.top_k,
                embedding_model=args.embedding_model or "AI-Growth-Lab/PatentSBERTa",
                use_query_expansion=True,
                use_llm_expansion=args.llm_query_expansion,
                use_llm_decompose=args.llm_retrieval_decompose,
                llm_model=args.llm_model,
            )
        elif args.retrieval_method == "local-embedding":
            ranked = rank_patent_pool_local_embeddings(
                args.query,
                pool,
                top_k=args.top_k,
                embedding_model=args.embedding_model or "AI-Growth-Lab/PatentSBERTa",
            )
        else:
            ranked = rank_patent_pool_bm25(args.query, pool, top_k=args.top_k)

    snippets = gather_query_evidence(args.query, ranked, snippets_per_patent=2)
    if args.llm_answer and openai_available():
        response = answer_query_with_rag(args.query, snippets, model=args.llm_model)
        answer = response.answer
        if response.insufficiency_note:
            answer += f"\n\nNote: {response.insufficiency_note}"
    else:
        answer = heuristic_rag_answer(args.query, ranked, snippets)
    if args.llm_answer_verification and openai_available():
        verification = verify_rag_answer_llm(answer, snippets, model=args.llm_model)
    else:
        verification = verify_rag_answer_heuristic(answer, snippets)

    print("Answer\n======")
    print(answer)
    print("\nAnswer Verification\n===================")
    print(f"{verification.status}: {verification.reason}")
    print("\nRanked Patents\n==============")
    for index, result in enumerate(ranked, start=1):
        print(f"{index}. {result.patent_id} | {result.score:.3f} | {result.title}")
    print("\nEvidence\n========")
    for snippet in snippets[: min(6, len(snippets))]:
        print(f"- [{snippet.citation}] retrieval={snippet.retrieval_score:.3f} segment={snippet.segment_score:.3f}")
        print(f"  {snippet.evidence[:240]}")


if __name__ == "__main__":
    main()
