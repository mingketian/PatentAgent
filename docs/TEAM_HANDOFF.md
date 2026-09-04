# Patent Prior-Art Agent: Team Handoff

This file is for teammates working on:

- UI polish
- benchmark evaluation
- demo preparation

## 1. Current System Status

The project UI is now intentionally simplified to two main workflows.

### Our Patent Agent

Input:

- claim text
- invention description
- vague patent search query
- follow-up question over previous results

Output:

- ranked patents
- grounded answer
- supporting evidence snippets
- follow-up over current working set

Default UI behavior:

- coarse recall: `Persistent local index`
- rerank: `linear-patent-reranker`

Product comparison views:

- `Normal RAG baseline`
  - persistent local index retrieval
  - top patents only
  - optional evidence snippets
- `Our optimized patent agent`
  - persistent-index recall
  - learned linear patent reranking
  - conversational planner / working-set reuse
  - evidence extraction
  - grounded answer
  - answer verification
- `Side-by-side comparison`
  - runs both on the same query

### Benchmark

Input:

- one PANORAMA `PAR4PC` case

Output:

- ranked A-H prior art
- claim decomposition
- evidence chart
- verification
- metrics against gold answers

Top-level benchmark choices:

- `PatentSBERTa baseline` -> `local-embedding`
- `Our learned reranker` -> `linear-patent-reranker`

## 2. Important Concept Split

Do not confuse these two:

### Retrieval method

Examples:

- `bm25`
- `local-embedding`
- `linear-patent-reranker`
- `patent-specialized`

This is the ranking algorithm.

### Patent pool / source

Examples:

- `Persistent local index`
- `Local sample pool`
- `Hub PAR4PC pool`
- `Combined`

This is where the patents are loaded from.

`Persistent local index` is not a model. It is a storage/search backend used in free-text mode.

## 3. Recommended UI Settings

### Best product demo settings

- `Mode = Our Patent Agent`
- leave defaults unchanged

Why:

- this is now the actual product pipeline
- it brings the learned benchmark reranker into product QA as a second-stage reranker

### Best stable benchmark settings

- `Mode = Benchmark`
- `Benchmark method = PatentSBERTa baseline`

Why:

- this is the stable PatentSBERTa baseline

### Best benchmark experimental settings

- `Mode = Benchmark`
- `Benchmark method = Our learned reranker`

Why:

- this is the current learned patent-aware reranker
- it is the best experimental benchmark method we have right now

## 4. Current Best Experimental Result

On HF `validation-100`:

### `local-embedding`

- `hit@1 = 0.590`
- `hit@3 = 0.860`
- `recall@3 = 0.802`
- `exact@|gold| = 0.470`

### `linear-patent-reranker`

- `hit@1 = 0.600`
- `hit@3 = 0.910`
- `recall@3 = 0.850`
- `exact@|gold| = 0.510`

Interpretation:

- `local-embedding` remains the stable baseline
- `linear-patent-reranker` is the current best learned benchmark experiment

## 5. What the Learned Reranker Actually Uses

The current default linear reranker is trained on:

- HF `train` slice
- `train_rows = 200`

Feature set:

- `dense_score`
- `bm25_score`
- `field_lexical_score`

Model:

- linear logistic model
- solver: `liblinear`
- `C = 4.0`
- `class_weight = None`

Model files:

- `data/models/linear_patent_reranker_patentsberta_train200_3feat.joblib`
- `data/models/linear_patent_reranker_patentsberta_train200_3feat.json`

## 6. Feature Cache

Feature cache now exists for repeated experiments.

Location:

- `data/cache/features/`

Purpose:

- avoid recomputing patent-specialized feature vectors every time
- speed up reranker training and scan scripts

If experiments feel too slow, prebuild the cache first.

## 7. Commands Teammates Should Use

### Build / refresh benchmark reranker

```bash
python -m src.train_linear_patent_reranker \
  --mode train-default-model \
  --splits train \
  --max-rows-per-split 200
```

### Build feature cache

```bash
python -m src.feature_cache \
  --source hf \
  --splits train \
  --max-rows-per-split 200 \
  --namespace linear_train_200cases

python -m src.feature_cache \
  --source hf \
  --splits validation \
  --max-rows-per-split 100 \
  --namespace scan_eval_100cases
```

### Stable benchmark baseline

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 100 \
  --methods local-embedding
```

### Experimental benchmark reranker

```bash
python -m src.evaluate_par4pc_hf \
  --splits validation \
  --max-rows-per-split 100 \
  --methods linear-patent-reranker
```

### Scan train size / feature sets

```bash
python -m src.scan_linear_reranker_configs \
  --train-rows 50 100 200 \
  --eval-rows 100 \
  --output outputs/linear_reranker_scan.csv
```

### Launch UI

```bash
./scripts/run_app.sh
```

## 8. UI Guidance for the UI Teammate

The UI should present:

1. Our Patent Agent as the user-facing search mode
2. Benchmark as the evaluation mode

The UI should not imply that:

- `Persistent local index` is a model
- benchmark and product retrieval are the same thing

Recommended wording:

- Our Patent Agent: use the product defaults
- Benchmark: `PatentSBERTa baseline`
- Benchmark experimental: `Our learned reranker`

## 9. What Not To Spend Time On

Do not spend more time on:

- adding more retrieval models
- extending hand-tuned `patent-specialized`

Those are not the current bottlenecks.

## 10. What Still Matters

The highest-value remaining work is:

1. better demo / presentation polish
2. maybe one more benchmark figure/table
3. optional UI polish for readability

The system itself is already close to complete for the course project.
