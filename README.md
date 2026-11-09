# Project INSIGHT

Learning to retrieve research inspiration beyond semantic similarity (CMSC673 capstone).
Uses the author labels in [ScholarCatalyst](https://huggingface.co/datasets/ScholarCatalyst/ScholarCatalyst)
(Kim et al., 2026) to study where similarity search fails and whether a reranker can learn what makes a paper inspiring.

## Setup

```
uv sync
uv run python scripts/download_data.py   # writes data/*.parquet (CC BY-NC 4.0, not committed)
```

## Code

- `src/insight/data.py`: loads the four subsets, parses the mixed date formats, and builds the per-question candidate date filter (settings in `configs/data.yaml`).
- `src/insight/metrics.py`: Recall@K, nDCG@K, MRR and pairwise accuracy (checked against `ranx`).
- `src/insight/folds.py`: 5 project-level folds with an inner validation split (settings in `configs/folds.yaml`); `uv run python scripts/make_folds.py` writes `results/folds.csv`.
- `src/insight/retrieve_bm25.py`: BM25 over title + abstract with the per-question date filter (bm25s; settings in `configs/A.yaml`). Condition A: `uv run python scripts/run_bm25.py` writes `results/run_A.parquet` (top 100 per question) and `results/metrics_A.csv`.
- `src/insight/retrieve_dense.py`: Qwen3-Embedding bi-encoder (query instruction from the ScholarCatalyst paper in the model card's `Instruct: ...\nQuery:...` format, left padding, last-token pooling, L2 normalization) and FAISS exact inner-product search with the per-question date filter (settings in `configs/B.yaml`). Condition B needs a GPU: `uv run python scripts/encode_corpus.py` saves the vectors to `outputs/` and writes `results/shortlist_B.parquet` (top 100 per question, the shortlist the rerankers rescore) and `results/metrics_B.csv`. A CPU smoke test: `uv run python scripts/encode_corpus.py --model Qwen/Qwen3-Embedding-0.6B --dtype float32 --limit 1000 --query-limit 20 --tag smoke --out-dir outputs/smoke --emb-dir outputs/smoke`.
- `src/insight/rerank.py`: Qwen3-Reranker cross-encoder with the model card's prompt and an inspiration instruction; the score is $s = z_{yes} - z_{no}$, the difference of the next-token logits for "yes" and "no" (settings in `configs/C.yaml`). Condition C needs a GPU: `uv run python scripts/run_rerank.py` rescores `results/shortlist_B.parquet` and writes `results/run_C.parquet`, `results/pairs_C.parquet` (scores for every credited and rejected paper) and `results/metrics_C.csv`. A CPU smoke test: `uv run python scripts/run_rerank.py --shortlist results/run_A.parquet --dtype float32 --query-limit 2 --depth 10 --tag smoke --out-dir outputs/smoke`.
- Tests: `uv run pytest` (also run on every push by GitHub Actions) (tests that need the full dataset are skipped when `data/` is missing, and the model card checks are skipped when the Qwen3-Embedding or Qwen3-Reranker weights are not in the local Hugging Face cache).

## Notebooks

- `notebooks/01_explore_dataset.ipynb`: schema, label counts, ID resolution, date and category quirks.
- `notebooks/02_bm25_baseline.ipynb`: condition A vs the published BM25 numbers, the reproduction gate, and a preprocessing sensitivity table.
