.PHONY: help web app demo bench train figures clean

help:
	@echo "Patent Prior-Art Agent"
	@echo ""
	@echo "  make web       start the FastAPI demo on http://localhost:8899"
	@echo "  make app       start the Streamlit app on http://localhost:8501"
	@echo "  make demo      run the free-text CLI demo"
	@echo "  make bench     full PAR4PC validation run, all three methods"
	@echo "  make train     retrain the shipped linear reranker"
	@echo "  make figures   regenerate results/tables and results/figures"
	@echo "  make clean     remove caches and generated outputs"

web:
	cd web && ./start.sh

app:
	./scripts/run_app.sh

demo:
	python -m src.run_free_text_demo

bench:
	python -m src.evaluate_par4pc_hf --splits validation --max-rows-per-split 3029 \
	  --methods bm25 local-embedding linear-patent-reranker

train:
	python -m src.train_linear_patent_reranker --mode train-default-model \
	  --splits train --max-rows-per-split 200

figures:
	python tools/make_results_tables.py
	python tools/make_figures.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs data/cache
