# Evaluation Protocol

This document explains exactly what dataset was used, how many cases were tested, what each case represents, and how the reported retrieval metrics were produced.

## Dataset Used

The retrieval benchmark is based on **PANORAMA PAR4PC** from:

- `LG-AI-Research/PANORAMA`

We evaluated on the `PAR4PC` split, which is a **patent prior-art retrieval benchmark**.

## What One Case Looks Like

Each PAR4PC case is a **retrieval/reranking task**, not a free-form QA pair.

Each case contains:

1. a **target claim**
   - the patent claim text we want to analyze
2. **8 candidate prior-art patents**
   - labeled `A-H`
3. **gold answers**
   - the subset of candidate letters that are truly relevant prior art

So one case is:

```text
target claim
+ 8 candidate patents (A-H)
+ gold relevant letters
```

Example scenario:

- input: a patent claim about some invention
- candidates: 8 patents that may or may not be relevant
- task: rank the 8 candidates so that true prior-art candidates appear near the top

## Dataset Size

Across all PAR4PC splits:

- `train`: `54,028` cases
- `validation`: `3,029` cases
- `test`: `2,896` cases

Total:

- `59,953` retrieval cases

Each case has exactly `8` candidates, so total claim-candidate pairs are:

- `479,624` claim-candidate pairs

After deduplicating all candidates across the splits:

- `17,877` unique candidate patents

## What We Evaluated

For the final reported retrieval comparison, we used:

- split: `validation`
- cases evaluated: **3,029**

This is the full validation set, not just a small subset.

## What This Evaluation Does Not Include

The reported metrics in this document are for **retrieval / reranking only**.

They do **not** include:

- claim reformulation
- LLM claim decomposition
- heuristic claim decomposition
- query expansion
- grounded answer generation
- answer verification

Those features belong to:

- product-mode QA behavior
- benchmark analysis tooling
- optional ablation / engineering paths

They are not part of the final retrieval metrics reported here unless a separate experiment explicitly turns them on and evaluates them.

## Methods Compared

We evaluated these three main retrieval methods:

### 1. `bm25`

Lexical baseline.

Interpretation:

- uses term overlap / lexical relevance
- simplest baseline

### 2. `local-embedding`

Pretrained semantic baseline using:

- `AI-Growth-Lab/PatentSBERTa`

Interpretation:

- semantic retrieval using pretrained patent embeddings
- this is the main pretrained baseline

### 3. `linear-patent-reranker`

Our optimized method.

Interpretation:

- starts from candidate-level retrieval features
- applies a learned linear reranker
- current best feature set:
  - `dense_score`
  - `bm25_score`
  - `field_lexical_score`

## Evaluation Scenario Per Case

The evaluation scenario is the same structure for every case:

1. read the target claim
2. score the 8 candidates
3. rank the candidates
4. compare the ranked letters with gold answers

This is therefore a **candidate reranking benchmark**, not full open-corpus search.

## Metrics Reported

For each method we report:

### `Hit@1`

Whether the top-1 ranked candidate contains a gold answer.

### `Hit@3`

Whether the top-3 ranked candidates contain at least one gold answer.

### `Recall@3`

How much of the gold answer set is recovered in the top 3.

### `Exact@|gold|`

Whether the top-`|gold|` predictions exactly match the gold answer set.

## Final Full-Validation Results

Validation split: `3,029` cases

- `bm25`
  - `Hit@1 = 0.718`
  - `Hit@3 = 0.926`
  - `Recall@3 = 0.843`
  - `Exact@|gold| = 0.560`

- `local-embedding`
  - `Hit@1 = 0.714`
  - `Hit@3 = 0.950`
  - `Recall@3 = 0.885`
  - `Exact@|gold| = 0.570`

- `linear-patent-reranker`
  - `Hit@1 = 0.754`
  - `Hit@3 = 0.959`
  - `Recall@3 = 0.901`
  - `Exact@|gold| = 0.622`

## Interpretation

From the full validation evaluation:

- `bm25` is a strong lexical baseline
- `local-embedding` improves semantic retrieval behavior over the pretrained patent encoder baseline
- `linear-patent-reranker` is the best overall method among the three compared methods

It is best on all four reported metrics:

- `Hit@1`
- `Hit@3`
- `Recall@3`
- `Exact@|gold|`

## Reproduction Commands

### Full validation lexical baseline

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods bm25
```

### Full validation pretrained baseline + optimized method

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods local-embedding linear-patent-reranker
```

## Related Files

- [`docs/EVALUATION_RESULTS.md`]
- [`outputs/retrieval_comparison.csv`]
- [`outputs/linear_reranker_scan.csv`]
