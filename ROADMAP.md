# Roadmap

One commit every Monday, Wednesday and Friday. Everything, including RQ3, is done by **Fri Dec 4, 2026**.
Each item is one commit. Check it off (`[x]`, with the short commit description) in the same commit that does the work.

## Rules for the scheduled agent

1. Read the full project context you were given and this file first. Run `git log --oneline` to see what is done.
2. Do the **first unchecked item**. If it is blocked (for example it needs an H100 output that is not in `results/` yet), do the next unblocked item instead, and say so under "Notes for Anuraag".
3. Get the data with `uv sync && uv run python scripts/download_data.py`. Never commit `data/` (CC BY-NC 4.0).
4. Run `uv run pytest` before committing. Do not commit failing tests.
5. **Never invent numbers.** Every number in a notebook, table or this file must come from code that ran in this repo. If a step could not run, say so.
6. Exploration and analysis go in `notebooks/NN_name.ipynb`, executed so the outputs are saved. Reusable code goes in `src/insight/`, entry points in `scripts/`, tests in `tests/`.
7. Small artifacts the next steps need (shortlists, metric tables, label files under ~20 MB) go in `results/`, which is committed. Large ones (embeddings, model checkpoints) stay in `outputs/`, which is gitignored.
8. **GPU jobs:** the agent has no GPU. For an item marked **[H100]**, write the script plus a config, test it on a tiny CPU slice (for example 1,000 papers or 20 pairs), and add the exact command to "H100 queue" below. Anuraag runs it and commits the outputs to `results/`.
9. **Judgment calls** (splits and leakage, label validation, mined negatives, error analysis, spurious gains, claims): do not block. Pick the recommended default, make it a config option, and write it in the decision log with a one-line reason, marked `NEEDS REVIEW`.
10. Writing rules: no em dashes anywhere, math in LaTeX, small figure captions that define every symbol.
11. Exactly one commit per run, pushed to `main`. End the commit message with `Co-Authored-By: Claude <noreply@anthropic.com>`.
12. If every item is checked, make no commit.

## Schedule

### Week of Oct 26: data, metrics, BM25
- [ ] **Mon Oct 26**: `src/insight/data.py` (loaders for all four subsets, ISO8601 date parsing, date filter per the decision log) + `tests/test_data.py` (sizes 190,896 / 894 / 207 / 687, every labeled ID resolves, date filter keeps labeled papers).
- [ ] **Wed Oct 28**: `src/insight/metrics.py` (Recall@K, nDCG@K, MRR, pairwise accuracy) + `src/insight/folds.py` (5 project-level folds, 20% inner validation split, fixed seed) + tests: hand-computed toy cases, `ranx` agreement to 1e-6, edge cases, no project in two folds, no query split from its project.
- [ ] **Fri Oct 30**: `src/insight/retrieve_bm25.py` + `scripts/run_bm25.py` (condition A) + `notebooks/02_bm25_baseline.ipynb` comparing core and subfield Recall@20 with the paper (gate: within 0.03 of 0.23) + `.github/workflows/tests.yml` (pytest on push; tests that need full data are skipped in CI).

### Week of Nov 2: dense baseline and RQ1 coverage
- [ ] **Mon Nov 2** [H100]: `src/insight/retrieve_dense.py` + `scripts/encode_corpus.py` (Qwen3-Embedding-4B, query instruction copied from the model card, last-token pooling, FAISS exact inner product, top 100 per query saved to `results/shortlist_B.parquet`).
- [ ] **Wed Nov 4**: `notebooks/03_dense_baseline.ipynb`: condition B metrics vs the paper (gate: within 0.03 of 0.39), Recall@100 coverage ceiling. If the shortlist is not in `results/` yet, do Fri Nov 6 on BM25's ranking instead and come back.
- [ ] **Fri Nov 6**: `src/insight/diagnose.py` + `notebooks/04_rq1_coverage.ipynb`: rank bins (1 to 20, 21 to 100, beyond 100) for credited papers, cross-field vs in-field (category mapping decided in the decision log), recall by bin and field.

### Week of Nov 9: RQ1 types, reranker C
- [ ] **Mon Nov 9**: `scripts/label_inspiration_types.py`: LLM labels for the 1,348 rationales with the paper's nine types (names and definitions copied from arXiv:2610.02202, never invented). If no API key is available in the agent environment, commit the script, prompt and a 20-rationale dry run, and add the full run to the H100 queue (it is an API job, not GPU).
- [ ] **Wed Nov 11**: `notebooks/05_rq1_types.ipynb`: recall by inspiration type, plus a 50-row sample in `results/handcheck_50.csv` with an empty column for Anuraag's labels. RQ1 summary paragraph.
- [ ] **Fri Nov 13** [H100]: `src/insight/rerank.py` (yes/no logit scorer, $s = z_{yes} - z_{no}$, base template from the Qwen3-Reranker model card, inspiration instruction) + test reproducing the model card's example scores (CPU, 0.6B fits) + `scripts/run_rerank.py` for condition C over `shortlist_B`.

### Week of Nov 16: RQ2 training
- [ ] **Mon Nov 16**: `src/insight/train_reranker.py`: pairwise RankNet loss, LoRA (r 16, alpha 32, dropout 0.05), bf16, early stopping on validation pairwise accuracy. Tests: loss equals $\ln 2$ at margin 0 and decreases with margin, only adapter weights change after one step, 20-pair overfit reaches accuracy 1.0 (CPU).
- [ ] **Wed Nov 18** [H100]: condition D: `configs/D.yaml` + `scripts/train_rerankers.py` over 5 folds. Notebook `06_rq2_results.ipynb` with C vs D once outputs exist.
- [ ] **Fri Nov 20** [H100]: condition E (mined negatives from ranks 30 to 100 of B, weight 0.3) and the core-only vs core+subfield ablation configs. Update notebook 06 with whatever results are in `results/`.

### Week of Nov 23 (Thanksgiving): RQ3
- [ ] **Mon Nov 23**: `src/insight/train_retriever.py`: InfoNCE with in-batch plus author-rejected negatives, $\tau = 0.05$, Qwen3-Embedding-0.6B. Tests: loss on a toy batch matches a hand computation, overfit on a tiny batch (CPU).
- [ ] **Wed Nov 25** [H100]: `configs/rq3.yaml` + `scripts/train_retrievers.py` (5 folds, re-encode corpus per fold, new top 100 to `results/shortlist_rq3_fold*.parquet`).
- [ ] **Fri Nov 27**: `notebooks/07_rq3_results.ipynb`: Recall@100 vs B, full pipeline with the best reranker from D or E.

### Week of Nov 30: analysis and release
- [ ] **Mon Nov 30**: `src/insight/stats.py` (paired bootstrap over projects, 10k samples; tests: recovers a known synthetic effect, resamples whole projects) + `notebooks/08_results_tables.ipynb` (A to E and RQ3 with intervals, core and subfield separate, memorization check, pre vs post June 2025 check).
- [ ] **Wed Dec 2**: `notebooks/09_error_analysis.ipynb`: misranked examples with rationales side by side, grouped for Anuraag to read and judge.
- [ ] **Fri Dec 4**: final figures, README with full reproduction steps, adapter release script for the Hugging Face Hub, summary of results.

## H100 queue

Commands for Anuraag to run on the H100. The agent appends; Anuraag removes an entry after committing its outputs to `results/`.

(empty)

## Decision log

| Date | Decision | Default chosen | Status |
|---|---|---|---|
| Oct 23 | Date filter vs 27 credited papers published after their source paper | Filter only unlabeled candidates; credited and rejected papers are always kept. Config flag `date_filter_exempt_labeled: true`. | Decided by Anuraag |

## Notes for Anuraag

(The agent writes anything that needs your attention here.)
