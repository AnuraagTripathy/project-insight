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
- Tests: `uv run pytest` (tests that need the full dataset are skipped when `data/` is missing).

## Notebooks

- `notebooks/01_explore_dataset.ipynb`: schema, label counts, ID resolution, date and category quirks.
