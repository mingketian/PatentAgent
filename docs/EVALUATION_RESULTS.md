# Evaluation Results

This document summarizes the retrieval evaluation artifacts currently available in the repository and the latest metrics produced for the main methods.

## Methods Covered

The main retrieval methods under comparison are:

1. `bm25`
   - lexical baseline
2. `local-embedding`
   - pretrained semantic baseline using `AI-Growth-Lab/PatentSBERTa`
3. `linear-patent-reranker`
   - our optimized learned reranker

## Current Artifact Locations

The reproduction commands below write result files into `outputs/`, which is git-ignored because it is regenerated on every run:

- `outputs/retrieval_comparison.csv`
- `outputs/linear_reranker_scan.csv`
- `outputs/linear_reranker_forward_selection_30.csv`
- `outputs/linear_reranker_forward_selection_100.csv`
- `outputs/patent_specialized_ablation.csv`

## Full PANORAMA Validation Run

Dataset:

- split: `validation`
- cases: `3029`

### Completed

#### `bm25`

Full validation metrics:

- `Hit@1 = 0.718`
- `Hit@3 = 0.926`
- `Recall@3 = 0.843`
- `Exact@|gold| = 0.560`

Command used:

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods bm25
```

#### `local-embedding`

Full validation metrics:

- `Hit@1 = 0.714`
- `Hit@3 = 0.950`
- `Recall@3 = 0.885`
- `Exact@|gold| = 0.570`

#### `linear-patent-reranker`

Full validation metrics:

- `Hit@1 = 0.754`
- `Hit@3 = 0.959`
- `Recall@3 = 0.901`
- `Exact@|gold| = 0.622`

The full-validation run for the pretrained baseline plus our optimized method was:

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods local-embedding linear-patent-reranker
```

This run covers:

- `local-embedding` = pretrained `PatentSBERTa` baseline
- `linear-patent-reranker` = our optimized learned reranker

## Stable Subset Results Already Available

These are the most relevant completed results already in the repo.

### Validation-100

The strongest completed comparison between the pretrained baseline and our optimized reranker is on `validation-100`.

| Method | Hit@1 | Hit@3 | Recall@3 | Exact@|gold| |
|---|---:|---:|---:|---:|
| `local-embedding` | 0.590 | 0.860 | 0.802 | 0.470 |
| `linear-patent-reranker` | 0.600 | 0.910 | 0.850 | 0.510 |

Interpretation:

- `linear-patent-reranker` is slightly better on `Hit@1`
- it is also better on `Hit@3`, `Recall@3`, and `Exact@|gold|`

### Validation Scan for Learned Reranker

From `outputs/linear_reranker_scan.csv`, the best current learned configuration on `validation-100` is:

- train rows: `200`
- features: `dense_score + bm25_score + field_lexical_score`
- solver: `liblinear`
- `C = 4.0`

Metrics:

| Train Rows | Feature Set | Hit@1 | Hit@3 | Recall@3 | Exact@|gold| |
|---:|---|---:|---:|---:|---:|
| 200 | `dense+bm25+lexical` | 0.600 | 0.910 | 0.850 | 0.510 |

### Local 10-Case Benchmark

From `outputs/retrieval_comparison.csv`:

| Method | Hit@1 | Hit@3 | Recall@3 | Exact@|gold| |
|---|---:|---:|---:|---:|
| `bm25` | 0.400 | 1.000 | 1.000 | 0.400 |
| `local-embedding` | 0.800 | 1.000 | 1.000 | 0.700 |
| `patent-specialized` | 1.000 | 1.000 | 1.000 | 1.000 |

Note:

- `patent-specialized` was an earlier hand-tuned reranker path.
- The current optimized path we are keeping is `linear-patent-reranker`, because it is cleaner and more defensible than the hand-tuned path.

## Recommended Numbers to Report

If you need one clean comparison for the current project story, use:

### Baselines / Product Story

- `bm25` as lexical baseline
- `local-embedding` as pretrained `PatentSBERTa` baseline
- `linear-patent-reranker` as our optimized method

### Full validation comparison

Use the full `validation` split table:

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

Interpretation:

- `linear-patent-reranker` is best on all four reported metrics
- `local-embedding` is stronger than `bm25` on `Hit@3`, `Recall@3`, and `Exact@|gold|`
- `bm25` remains a strong lexical baseline given its simplicity

### Validation-100 reference comparison

For the smaller completed `validation-100` comparison:

- `local-embedding`
  - `Hit@1 = 0.590`
  - `Hit@3 = 0.860`
  - `Recall@3 = 0.802`
  - `Exact@|gold| = 0.470`

- `linear-patent-reranker`
  - `Hit@1 = 0.600`
  - `Hit@3 = 0.910`
  - `Recall@3 = 0.850`
  - `Exact@|gold| = 0.510`

## Reproduction Commands

### Full validation lexical baseline

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods bm25
```

### Full validation pretrained baseline + optimized reranker

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 3029 \
  --methods local-embedding linear-patent-reranker
```

### Validation-100 comparison

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 100 \
  --methods local-embedding linear-patent-reranker
```
