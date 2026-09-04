# Patent Prior-Art Agent: Structured Prior-Art Search and Evidence Synthesis via LLM Agent with Tool Use

Adapted from the annotated CS 329 proposal PDF supplied by the user. This repo version keeps the original proposal structure, but replaces the arXiv literature-survey setting with the patent prior-art workflow implemented in this repository.

Course

CS/QTM/LING-329: Computational Linguistics

Authors

Jonathan Wang, BS in Computer Science, jonathan.wang@emory.edu
Mingke Tian, BS in Computer Science, mingke.tian@emory.edu
Nicole Chen, BS in Computer Science, nicole.chen@emory.edu
Eric Zou, BS in Applied Mathematics and Statistics, eric.zou@emory.edu

## Abstract

Patent search is difficult because relevant prior art is often expressed in specialized, indirect, and highly variable legal language. Keyword search misses semantically relevant patents that describe the same invention differently, while general-purpose LLM assistants can produce unsupported patent mappings or grounded answers without clear evidence trails. We propose the Patent Prior-Art Agent, a tool-using LLM system that converts a target claim, invention description, or patent-related question into a structured, verifiable prior-art analysis: a ranked set of candidate patents, a limitation-level evidence chart, and a grounded answer linked to retrieved patent text.

Built on LangGraph, the system uses five explicit steps for claim decomposition, patent retrieval, evidence extraction, answer synthesis, and verification. Unlike a single-prompt chatbot response, the agent separates retrieval, evidence alignment, synthesis, and validation so each substantive statement can be traced back to patent title, abstract, or claim text. Our implementation builds on the PANORAMA `PAR4PC` benchmark, persistent FAISS indexes, PatentSBERTa embeddings, patent-specific reranking, and optional LLM-backed decomposition and verification.

We evaluate the system on labeled `PAR4PC` benchmark cases and free-text patent-search scenarios. The main success criteria are retrieval hit rate, evidence support quality, and hallucination reduction in grounded answers. Intellectually, the project studies whether a source-traceable multi-step agent outperforms keyword retrieval or single-prompt LLM analysis for prior-art search. More broadly, it aims to help inventors, students, and practitioners explore patent landscapes faster while reducing unsupported legal or technical claims.

## 1. Introduction

### 1.1 Objectives

The goal of this project is to build an LLM-powered agent that supports prior-art search and evidence-grounded patent analysis. Given a target patent claim, an invention description, or a free-text patent question, the system should produce:

1. A ranked list of candidate prior-art patents retrieved from a local or benchmark patent corpus.
2. A decomposition of the target claim into discrete limitations suitable for evidence matching.
3. A source-linked claim chart in which each limitation is mapped to candidate patent evidence.
4. A grounded natural-language answer that cites retrieved patent evidence inline.
5. A verification pass that marks evidence mappings and answer content as supported, partially supported, or unsupported.

The technical scope is a single LangGraph-based agent with retrieval, reranking, evidence extraction, answer synthesis, and verification steps. The system is entirely inference-time and runs over the PANORAMA `PAR4PC` benchmark plus a persistent local patent index built with FAISS.

### 1.2 Motivation

Prior-art search is one of the most time-consuming and error-prone parts of patent analysis. Relevant patents may use different terminology, emphasize different claim structure, or bury key details deep in long-form claim text. A useful system must do more than return a list of documents: it must explain why a patent is relevant, map evidence to claim limitations, and expose uncertainty clearly.

The societal value is broad. Students learning patent analysis, inventors exploring novelty, and practitioners conducting first-pass searches all benefit from faster and more transparent prior-art review. A system that produces grounded evidence instead of unsupported summaries can reduce wasted time and improve trust in AI-assisted patent workflows.

### 1.3 Problem Statement

We identify three concrete failure modes in current patent-search workflows:

Failure 1: LLM hallucination and unsupported mapping. General-purpose assistants can produce plausible-sounding patent analyses that are not faithfully tied to actual patent text. In a legal and technical workflow, unsupported claim-to-evidence mappings are unacceptable.

Failure 2: Keyword mismatch in patent retrieval. Patent language is intentionally broad and stylistically idiosyncratic. A user searching for a concept in plain language may miss highly relevant prior art whose wording differs but whose technical content overlaps substantially.

Failure 3: Single-prompt prior-art analysis quality. A one-shot LLM answer may provide a fluent summary, but it typically does not separate claim decomposition, retrieval, evidence alignment, and verification. The result is weak traceability, incomplete limitation coverage, and poor calibration about what the retrieved patents actually support.

These three failure modes define the gap our system addresses: from an ambiguous patent claim or invention description to a structured, evidence-grounded prior-art analysis.

### 1.4 Innovation Component

Our contribution is not simply using an LLM to summarize patent text. The key idea is a source-traceable workflow in which retrieval, evidence alignment, synthesis, and verification are separated into explicit steps.

1. Structured claim-chart output rather than free-text only. Instead of returning only a narrative paragraph, the system produces a ranked candidate list and a limitation-level evidence chart whose rows are tied to specific patent text segments.
2. Explicit multi-step reasoning over tools rather than single-prompt generation. The agent decomposes the target claim, retrieves prior art, aligns candidate evidence, synthesizes a grounded answer, and verifies selected outputs. This workflow is materially different from asking a chatbot for a one-shot opinion.
3. Patent-specific retrieval and reranking. The system supports PatentSBERTa-based dense retrieval, hybrid coverage reranking, and patent-specialized ranking strategies designed for long-form claims and terminology mismatch.
4. Claim-level verification for evidence grounding. Instead of trusting generated prose, the system checks whether retrieved evidence actually supports a target limitation or answer claim and surfaces unsupported content clearly.

## 2. Background

### 2.1 Related Work

Retrieval-augmented generation. RAG systems combine retrieval with generation to ground model outputs in external evidence. Our system extends this idea by operating over patent corpora and by producing structured claim charts rather than unstructured text alone.

Patent retrieval and semantic search. Patent search requires domain-sensitive representations because title and claim wording often differ from the user query. Patent-oriented embedding models such as PatentSBERTa are especially relevant because they encode patent text more appropriately than generic sentence embeddings.

Benchmark-based prior-art evaluation. The PANORAMA `PAR4PC` benchmark provides labeled cases with target claims, candidate patents, and gold relevance labels. This allows controlled evaluation of retrieval quality and evidence alignment instead of relying only on subjective demos.

LLM-based research and legal assistants. Recent systems use LLMs to support complex knowledge workflows, but many still rely on single-prompt generation with limited source traceability. Our system uses an explicit agent graph to make retrieval and verification part of the method rather than a hidden internal behavior.

Attributed and verified generation. A central concern in grounded generation is whether output claims are actually supported by the source material. This project applies that concern directly to patent evidence mapping and grounded answers.

### 2.2 Preliminary Work

This repository already implements an end-to-end prototype of the Patent Prior-Art Agent. The current codebase supports:

- `PAR4PC` benchmark evaluation
- persistent local FAISS indexing over patent pools
- free-text patent search
- grounded QA over retrieved evidence
- multi-turn follow-up handling with working-set reuse
- limitation-level evidence extraction and verification

The current implementation also exposes the main research opportunity. Retrieval quality improves when patent-specific embeddings and reranking are used, but the system still needs stronger evidence-grounded evaluation and clearer justification for each agent step. The project therefore focuses on making the workflow explicitly source-traceable and measurable rather than treating the system as a general chatbot over patent data.

## 3. Proposed Approach

### 3.1 System Architecture

The system is a LangGraph-based agent that separates the prior-art workflow into explicit nodes. In benchmark mode, the graph loads a `PAR4PC` case, decomposes the target claim into limitations, retrieves and reranks candidate patents, extracts limitation-level evidence, verifies support, and renders a final report. In free-text mode, the same retrieval and grounding components support open-ended patent questions while reusing retrieved working sets across follow-up turns.

### 3.2 Tool Descriptions

#### Tool 1: claim decompose

- Input: target patent claim or focused invention description
- Output: ordered list of claim limitations
- Rationale: patent claims are long and compositional; decomposing them makes later evidence alignment more interpretable
- Implementation: heuristic decomposition by default, with optional LLM decomposition when an API key is available

#### Tool 2: retrieve prior art

- Input: decomposed claim text or free-text patent query plus top-k parameter
- Output: ranked candidate patents with title, abstract, claims, and retrieval score
- Implementation: configurable retrieval methods including BM25, PatentSBERTa local embeddings, hybrid coverage reranking, cross-encoder reranking, patent-specialized reranking, and optional LLM reranking
- Rationale: retrieval should balance semantic similarity, patent-specific terminology, and claim coverage

#### Tool 3: extract evidence

- Input: top-ranked candidate patents plus claim limitations
- Output: limitation-level evidence matches between the target claim and candidate patent segments
- Implementation: build a claim chart by aligning each limitation to patent abstract or claim snippets
- Rationale: a ranked document list alone is insufficient for prior-art analysis; users need evidence anchored to specific text

#### Tool 4: synthesize grounded answer

- Input: ranked patents, extracted evidence, and user query context
- Output: a grounded report or natural-language answer with inline patent citations
- Implementation: render a structured benchmark report or conversational answer over the retrieved working set
- Rationale: the answer layer should summarize the evidence without severing the link to source text

#### Tool 5: verify evidence

- Input: claim-chart rows or grounded answer claims
- Output: supported, partially supported, or unsupported plus a reason
- Implementation: heuristic verification by default, with optional LLM-backed verification when configured
- Rationale: verification reduces unsupported mappings and exposes uncertainty directly to the user

### 3.3 Example Agent Execution Trace

Below is a condensed trace of one benchmark-style run:

1. Thought: "The user wants prior art for a target patent claim. I should break the claim into limitations."
   Action: `claim_decompose(target_claim)`
   Observation: 5 ordered limitations describing system components, signal flow, and control logic.

2. Thought: "I should retrieve candidate patents that overlap semantically and structurally."
   Action: `retrieve_prior_art(limitations, retrieval_method="patent-specialized", top_k=3)`
   Observation: 3 top candidates returned with PatentSBERTa scores and patent metadata.

3. Thought: "A ranked list is not enough. I need evidence for each limitation."
   Action: `extract_evidence(limitations, candidates)`
   Observation: limitation-level chart built with abstract and claim snippets for each candidate patent.

4. Thought: "Now I should verify whether those snippets really support the claimed limitations."
   Action: `verify_evidence(claim_chart)`
   Observation: most rows marked supported, a smaller number marked partially supported or unsupported.

5. Final output: ranked candidate patents, verified claim chart, and a grounded answer summarizing the strongest prior-art matches.

This workflow cannot be reduced cleanly to a single prompt without losing traceability. The agent structure is part of the method.

### 3.4 Dataset

We use the PANORAMA `PAR4PC` benchmark and associated patent pools as the main dataset for development and evaluation. Each benchmark case contains:

- application number
- claim number
- target patent title and abstract
- full target-claim list
- candidate patents with patent ID, title, abstract, and claims
- labeled gold, silver, and negative answer sets

For retrieval experiments, the system can build a persistent FAISS index over:

1. a local `PAR4PC` candidate pool
2. Hugging Face `PAR4PC` parquet splits
3. a combined local-plus-Hub patent pool

Each patent is represented for retrieval by concatenating title, abstract, and claims. The default dense retrieval model is `AI-Growth-Lab/PatentSBERTa`, and the index is stored locally for reuse across experiments and demos.

### 3.5 Evaluation Plan

Our evaluation is designed to answer a concrete question: does a multi-step, source-traceable patent agent produce more reliable prior-art analysis than simpler retrieval or generation baselines?

#### 3.5.1 Retrieval Evaluation

- Dataset: labeled `PAR4PC` cases from local JSON files and larger Hugging Face splits
- Metrics: `hit@1`, `hit@3`, `recall@3`, and `exact@|gold|`
- Comparisons: BM25, PatentSBERTa local embeddings, hybrid coverage, patent-specialized reranking, linear patent reranking, cross-encoder reranking, and optional LLM reranking

#### 3.5.2 Evidence and Answer Quality Evaluation

- Limitation coverage: what fraction of target claim limitations receive non-empty evidence alignments?
- Evidence support quality: what fraction of claim-chart rows are marked supported after verification?
- Grounded answer quality: does the answer cite relevant patents and avoid unsupported statements?
- Hallucination rate: unsupported evidence mappings or unsupported answer claims divided by total checked claims

#### 3.5.3 Baselines

- Baseline 1: BM25 retrieval only
- Baseline 2: PatentSBERTa dense retrieval without patent-specialized reranking
- Baseline 3: single-prompt LLM analysis over the target claim without explicit claim-chart construction

These baselines address the core question of whether the extra agent structure adds measurable value beyond standard retrieval or one-shot generation.

#### 3.5.4 Ablation Study

- Remove claim decomposition: retrieve directly from the raw target claim and measure drop in relevance and coverage
- Remove verification: skip evidence verification and measure increase in unsupported mappings
- Remove patent-specialized reranking: compare generic dense retrieval against the specialized ranking path
- Remove grounded report rendering: output only ranked patents and compare usefulness against the full report

## 4. Timeline

### 4.1 Weekly Schedule

| Dates | Owner | Deliverable |
| --- | --- | --- |
| 3/22-3/28 | Nicole | Clean patent pools, inspect `PAR4PC` data quality, and validate persistent index builds |
| 3/22-3/28 | Mingke | Set up LangGraph workflow and implement baseline retrieval paths |
| 3/29-4/4 | Jonathan | Implement evidence extraction and benchmark report rendering |
| 4/5-4/11 | Eric | Implement verification, claim-chart evaluation, and grounded answer checks |
| 4/12-4/20 | Mingke, Eric | Run retrieval comparisons, ablations, and benchmark analysis |
| 4/20-end | All | Final report writing, presentation preparation, and demo polishing |

### 4.2 Risk Mitigation

- Embedding and indexing cost: if full patent-pool indexing is too expensive, use the bundled demo pool or a capped Hub slice while preserving the experimental setup.
- API limits: the system defaults to heuristic decomposition and verification, so it remains functional without LLM access.
- Retrieval instability: if one advanced reranker underperforms, retain the strongest available baseline such as PatentSBERTa local embeddings.
- Scope control: the core deliverables are claim decomposition, prior-art retrieval, evidence extraction, grounded reporting, and verification. Additional conversational polish remains secondary.

### 4.3 Team Responsibilities

| Member | Primary Responsibility |
| --- | --- |
| Nicole | Dataset preparation, patent-pool cleaning, embedding setup, and FAISS index management |
| Mingke | Agent architecture, LangGraph workflow, retrieval integration, and benchmark orchestration |
| Jonathan | Evidence extraction, grounded reporting, UI and demo interface |
| Eric | Verification logic, evaluation framework, and ablation studies |

All members collaborate on the final report and presentation. Code is managed in a shared GitHub repository using branch-based development.

## A. Contributions

- Jonathan Wang (25%): project framing, grounded report design, evidence extraction logic, demo interface
- Mingke Tian (25%): agent workflow design, retrieval integration, patent-specialized ranking experiments, proposal adaptation
- Nicole Chen (25%): dataset preparation, patent-pool indexing, embedding and FAISS configuration
- Eric Zou (25%): verification design, evaluation metrics, benchmark analysis, ablation study
