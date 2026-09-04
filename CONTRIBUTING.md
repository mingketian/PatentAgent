# Contributing

## Setup

```bash
conda env create -f environment.yml
conda activate patent-agent
git clone https://github.com/LGAI-Research/PANORAMA.git ../PANORAMA   # benchmark mode only
```

Always run inside the `patent-agent` environment. Most "module not found" reports
(`torchvision`, `streamlit`, `faiss`) are a wrong-environment problem, not a missing
dependency — `scripts/run_app.sh` refuses to start outside it for exactly this reason.

## Ground rules

**Never commit secrets.** `.env` is git-ignored; `.env.example` is the template. If a key is
ever committed, rotate it before doing anything else — rewriting history does not un-leak it.

**Never commit large indexes.** `data/indexes/` is ignored apart from the 112-patent demo
index. The full index is ~133 MB and is rebuilt with one command; see
[`data/README.md`](data/README.md).

**Benchmark before shipping.** A retrieval or reranking change goes through
`src/evaluate_par4pc_hf.py` on the validation split before it becomes a product default. That
is the whole point of keeping benchmark mode in the same repo as the product path.

## Adding a retrieval method

1. Implement `rank_candidates_<name>()` (fixed A–H candidates) and/or
   `rank_patent_pool_<name>()` (open pool) in `src/retrieval.py` or `src/patent_rerank.py`.
2. Register it in the `retrieve_prior_art_node()` switch in `src/graph.py`, with a
   BM25 fallback and a `state["warnings"]` entry for the unavailable-dependency case — every
   node in the graph degrades rather than raising.
3. Add it to the method lists in `app.py` and `web/server.py`.
4. Evaluate:
   ```bash
   python -m src.evaluate_par4pc_hf --splits validation \
     --max-rows-per-split 3029 --methods local-embedding <name>
   ```
5. Record the numbers in `docs/EVALUATION_RESULTS.md` with the command that produced them.

## Refreshing results

Figures and tables under `results/` are generated, not hand-edited:

```bash
python tools/make_results_tables.py
python tools/make_figures.py
```

Both are deterministic — commit the regenerated files alongside the change that moved the
numbers. `make check` (also run in CI) compiles every module and verifies that the committed
tables still match the underlying evaluation data.

## Style

Match the surrounding code: `from __future__ import annotations`, frozen dataclasses for
result types, `lru_cache` on anything that loads a model or an index, and module-private
helpers prefixed with `_`.
