# Methods Overview

This document is for teammates who need the system logic, method purpose, and component split without reading the entire README.

## 1. System split

The system has two top-level modes:

### Product mode: `Our Patent Agent`

Purpose:
- user-facing patent search
- vague idea exploration
- claim-style patent search
- follow-up patent QA

### Benchmark mode: `Benchmark`

Purpose:
- evaluate retrieval on PANORAMA `PAR4PC`
- compare baseline and improved reranking
- justify the learned reranker before using it in product QA

---

## 2. Product methods

### A. Normal RAG baseline

Pipeline:

```text
persistent local index
-> retrieve top patents
-> optional evidence snippets
```

Characteristics:
- simple retrieval-only path
- no planner
- no second-stage reranker
- no grounded answer synthesis
- no verification

Use:
- comparison against the full system
- show what a simpler retrieval baseline returns

### B. Our optimized patent agent

Pipeline:

```text
persistent local index coarse recall
-> linear-patent-reranker second-stage rerank
-> conversational planner / working-set reuse
-> evidence extraction
-> grounded answer
-> answer verification
```

Use:
- primary product demo
- follow-up question handling
- evidence-backed patent QA

---

## 3. Why the optimized pipeline has these steps

### Persistent local index

What it is:
- a local FAISS-backed patent store

Why it exists:
- product queries need a reusable patent pool
- avoids rebuilding everything at runtime

### Linear patent reranker

What it is:
- a learned reranking model trained from benchmark-derived patent features

Current feature set:
- `dense_score`
- `bm25_score`
- `field_lexical_score`

Why it exists:
- the first-stage recall step is broad
- reranking helps reorder the top candidate patents using patent-aware signals

### Conversational planner / working-set reuse

What it is:
- a turn classifier that decides whether the current query is:
  - a new search
  - a follow-up
  - a comparison
  - an aspect filter
  - a combination exploration request

Why it exists:
- follow-up patent questions should not always restart from scratch

### Evidence extraction

What it is:
- selecting relevant title / abstract / claim snippets

Why it exists:
- patent retrieval is not enough by itself
- users need to see why a result looks relevant

### Grounded answer

What it is:
- answer generation from retrieved evidence

Why it exists:
- product mode should do more than show ranked patent IDs

### Answer verification

What it is:
- a support check over the produced answer

Why it exists:
- keep the answer tied to retrieved evidence

---

## 4. Benchmark methods

### Benchmark baseline

Method:
- `local-embedding`

Meaning:
- PatentSBERTa over the fixed A-H candidate set

### Benchmark improved method

Method:
- `linear-patent-reranker`

Meaning:
- use benchmark-derived features to rerank the A-H candidates

Why benchmark exists:
- product questions are open-ended
- benchmark gives labeled evaluation
- benchmark results justify moving the reranker into product mode

---

## 5. Most important distinction

Do not confuse these:

### `Persistent local index`

- not a model
- not a reranker
- just the product patent store / recall backend

### `local-embedding`

- a retrieval method
- PatentSBERTa baseline

### `linear-patent-reranker`

- a learned reranking method
- used directly in benchmark mode
- used as second-stage reranking inside the optimized product path

---

## 6. What to say in a demo

Short version:

1. Baseline:
   - retrieve top patents from the local patent index
   - optionally show snippets

2. Optimized:
   - recall candidates from the same patent index
   - rerank them with the learned patent reranker
   - reuse prior results for follow-up questions
   - extract evidence
   - answer from evidence
   - verify the answer

3. Benchmark:
   - validates that the learned reranker improves prior-art ranking on labeled PANORAMA cases
