# Data

## What is committed

| Path | Size | What it is |
|---|---:|---|
| `indexes/par4pc_patentsberta_demo/` | ~900 KB | A 112-patent FAISS index, committed so both UIs and the CLI demos run straight after clone with no build step. |
| `models/linear_patent_reranker_patentsberta_train200_3feat.{joblib,json}` | ~2 KB | The shipped reranker: logistic regression over `dense_score`, `bm25_score`, `field_lexical_score`, trained on 200 PAR4PC `train` cases. |

Patent text in the demo index is derived from
[PANORAMA](https://github.com/LGAI-Research/PANORAMA) (LG AI Research); the underlying patent
documents are public records.

## What is not committed

The full index — 17,877 unique patents, ~133 MB — is git-ignored. `web/server.py` expects it at
`data/indexes/par4pc_patentsberta_full/`, so build it before running the web demo against the
full corpus:

```bash
python -m src.build_patent_index \
  --pool-source hub \
  --hub-rows-per-split 2000 \
  --index-dir data/indexes/par4pc_patentsberta_full
```

To rebuild the small demo index instead:

```bash
python -m src.build_patent_index \
  --pool-source combined \
  --hub-rows-per-split 50 \
  --index-dir data/indexes/par4pc_patentsberta_demo
```

`--pool-source combined` mixes the local PANORAMA sample patents with a small Hub slice;
`--pool-source hub` pulls only from the Hugging Face dataset.

Each index directory contains:

```text
index.faiss        IndexFlatIP over L2-normalised 768-d PatentSBERTa vectors
metadata.parquet   letter, patent_id, title, abstract, claims_json, retrieval_text
manifest.json      embedding_model, patent_count, dimension
```

The manifest's `embedding_model` is checked at query time — searching an index with a different
encoder than it was built with raises rather than returning wrong neighbours.

## Retraining the reranker

```bash
python -m src.train_linear_patent_reranker \
  --mode train-default-model --splits train --max-rows-per-split 200
```

Takes seconds. Optionally prebuild the feature cache first:

```bash
python -m src.feature_cache --source hf --splits train \
  --max-rows-per-split 200 --namespace linear_train_200cases
python -m src.feature_cache --source hf --splits validation \
  --max-rows-per-split 100 --namespace scan_eval_100cases
```

Caches are written to `data/cache/`, which is git-ignored.

## Choosing an index at runtime

`web/server.py` resolves its index in this order:

1. `PATENT_AGENT_INDEX_DIR`, if set
2. `data/indexes/par4pc_patentsberta_full/`, if it exists
3. `data/indexes/par4pc_patentsberta_demo/` — the committed fallback

So a fresh clone serves the 112-patent demo index, and building the full index is all it takes
to switch over. `app.py` uses the demo index by default and exposes the index directory as a
UI setting.
