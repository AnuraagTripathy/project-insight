"""Ranking metrics: Recall@K, nDCG@K, MRR and pairwise accuracy.

All metrics use binary relevance: a paper is relevant to a question if the authors credited it.
Per-question functions take a ranked list of paper ids (best first) and the set of credited ids.
`evaluate` scores a whole run and returns one row per question; average the rows to report.

Conventions (tested in tests/test_metrics.py):
- A question with no credited papers gets NaN for every metric, so it drops out of a mean
  instead of counting as zero. ScholarCatalyst has none, but subsets after filtering might.
- Ranked lists are taken as given. To turn scores into a ranking use `rank`, which breaks score
  ties by paper id so the result is deterministic.
- Pairwise accuracy counts a tie between a credited and a rejected paper as half correct.
"""

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def rank(scores: Mapping[str, float]) -> list[str]:
    """Paper ids sorted by descending score; equal scores are ordered by id (ascending)."""
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def recall_at_k(ranked: Sequence[str], positives: Iterable[str], k: int) -> float:
    """(credited papers in the top k) / (all credited papers)."""
    pos = set(positives)
    if not pos:
        return math.nan
    return len(pos.intersection(ranked[:k])) / len(pos)


def ndcg_at_k(ranked: Sequence[str], positives: Iterable[str], k: int) -> float:
    """Binary nDCG@k: each hit at rank r adds 1 / log2(r + 1), divided by the best possible DCG."""
    pos = set(positives)
    if not pos:
        return math.nan
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:k], start=1) if d in pos)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(pos), k) + 1))
    return dcg / idcg


def mrr(ranked: Sequence[str], positives: Iterable[str], k: int | None = None) -> float:
    """1 / (rank of the first credited paper), 0 if none is in the (top k of the) list."""
    pos = set(positives)
    if not pos:
        return math.nan
    for r, d in enumerate(ranked[:k] if k else ranked, start=1):
        if d in pos:
            return 1.0 / r
    return 0.0


def pairwise_accuracy(scores: Mapping[str, float], positives: Iterable[str],
                      negatives: Iterable[str]) -> float:
    """Fraction of (credited, rejected) pairs where the credited paper scores higher.

    Ties count 0.5. Every credited and rejected paper must have a score. NaN if there are no pairs.
    """
    p = np.array([scores[d] for d in positives], dtype=float)
    n = np.array([scores[d] for d in negatives], dtype=float)
    if len(p) == 0 or len(n) == 0:
        return math.nan
    diff = p[:, None] - n[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def evaluate(run: Mapping[str, Sequence[str]], positives: Mapping[str, Iterable[str]],
             ks: Sequence[int] = (5, 20, 100), ndcg_ks: Sequence[int] = (20,),
             mrr_k: int | None = None) -> pd.DataFrame:
    """One row per question in `positives`, with recall@k, ndcg@k and mrr columns.

    run: {query id: ranked paper ids}. A question missing from the run is scored as an empty
    ranking (all zeros), so a retriever cannot look better by skipping questions.
    """
    rows = []
    for qid, pos in positives.items():
        pos = set(pos)
        ranked = list(run.get(qid, []))
        row = {"query_id": qid}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, pos, k)
        for k in ndcg_ks:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, pos, k)
        row["mrr"] = mrr(ranked, pos, mrr_k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("query_id")
