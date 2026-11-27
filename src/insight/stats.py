"""Paired bootstrap over projects.

Why projects and not questions: the 894 questions come from 207 projects, and the questions of
one project share credited papers and a source paper, so their scores move together. Resampling
questions one by one would treat them as independent and give intervals that are too narrow.
Here every bootstrap sample draws 207 projects with replacement and keeps all questions of each
drawn project together (a project drawn twice counts twice).

The statistic of a sample is the mean over its questions (a project with more questions weighs
more), which is the same mean the point estimate uses, so the bootstrap is centered on it.

Implementation: per project we keep the sum of the per-question values and the number of
questions. One bootstrap sample is a vector of draw counts $c_j$ (how often project $j$ was
drawn, multinomial with $n$ trials and equal probabilities), and its mean is
$\\sum_j c_j S_j / \\sum_j c_j n_j$. All samples are one matrix product.

Conventions (tested in tests/test_stats.py):
- A question where either system is NaN (for example pairwise accuracy with no rejected paper)
  is dropped from both sides before anything else, so the comparison stays paired.
- Intervals are percentile intervals; the two-sided p value is
  $2 \\min(P^*(\\Delta \\le 0), P^*(\\Delta \\ge 0))$ over bootstrap samples, capped at 1.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class StatsConfig:
    n_boot: int = 10_000
    seed: int = 0
    ci: float = 0.95
    # Qwen3 Embedding report, arXiv:2506.05176 v1, submitted Thu 5 Jun 2025.
    release_date: str = "2025-06-05"
    # Roles of a fold whose credited papers count as "seen" by that fold's trained model.
    seen_roles: tuple[str, ...] = ("train", "val")

    @classmethod
    def from_yaml(cls, path):
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        if "seen_roles" in cfg:
            cfg["seen_roles"] = tuple(cfg["seen_roles"])
        return cls(**cfg)


def project_counts(n_projects: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_projects) matrix: how often each project is drawn in each bootstrap sample."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_projects, np.full(n_projects, 1.0 / n_projects), size=n_boot)


def _project_sums(values: np.ndarray, groups: Sequence) -> tuple[np.ndarray, np.ndarray]:
    codes, _ = pd.factorize(pd.Series(groups), sort=True)
    sums = np.bincount(codes, weights=values)
    ns = np.bincount(codes).astype(float)
    return sums, ns


def bootstrap_means(values, groups, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the question mean of `values`, resampling whole projects.

    values: one number per question (NaN not allowed, drop first); groups: project id per question.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("drop NaN values before bootstrapping")
    sums, ns = _project_sums(values, groups)
    counts = project_counts(len(sums), n_boot, seed)
    return (counts @ sums) / (counts @ ns)


def _interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    a = (1 - ci) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)


def bootstrap_ci(values, groups, n_boot: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean with a project-level percentile interval. NaN values are dropped."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    keep = ~np.isnan(values)
    values, groups = values[keep], groups[keep]
    s = bootstrap_means(values, groups, n_boot, seed)
    lo, hi = _interval(s, ci)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi,
            "n_questions": int(len(values)), "n_projects": int(len(set(groups)))}


def paired_bootstrap(a, b, groups, n_boot: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """Paired bootstrap of mean(b) - mean(a) over projects.

    a, b: per-question values of two systems on the same questions, in the same order.
    Returns the two means, the difference, its percentile interval and a two-sided p value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    groups = np.asarray(groups)
    if not (len(a) == len(b) == len(groups)):
        raise ValueError("a, b and groups must have the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b, groups = a[keep], b[keep], groups[keep]
    d = bootstrap_means(b - a, groups, n_boot, seed)
    lo, hi = _interval(d, ci)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float((b - a).mean()),
            "lo": lo, "hi": hi, "p": float(p),
            "n_questions": int(len(a)), "n_projects": int(len(set(groups)))}


def compare(per_q: Mapping[str, pd.DataFrame], pairs: Sequence[tuple[str, str]], metric: str,
            project_of: pd.Series, kinds=("core", "subfield"), **kw) -> pd.DataFrame:
    """Paired bootstraps for (a, b) condition pairs on one metric, per question kind.

    per_q: {condition: per-question frame indexed by query_id with `kind` and `metric` columns}.
    Questions are aligned on query_id; only questions present in both frames are used.
    """
    rows = []
    for a, b in pairs:
        A, B = per_q[a], per_q[b]
        qids = A.index.intersection(B.index)
        for kind in kinds:
            q = qids[A.kind.loc[qids].to_numpy() == kind]
            r = paired_bootstrap(A.loc[q, metric], B.loc[q, metric], project_of.loc[q], **kw)
            rows.append({"a": a, "b": b, "kind": kind, "metric": metric, **r})
    return pd.DataFrame(rows)


def summarize(per_q: Mapping[str, pd.DataFrame], metrics: Sequence[str], project_of: pd.Series,
              kinds=("core", "subfield"), **kw) -> pd.DataFrame:
    """Mean and project-level interval for every condition, metric and question kind present."""
    rows = []
    for name, df in per_q.items():
        for m in metrics:
            if m not in df:
                continue
            for kind in kinds:
                g = df[df.kind == kind]
                r = bootstrap_ci(g[m], project_of.loc[g.index], **kw)
                rows.append({"condition": name, "kind": kind, "metric": m, **r})
    return pd.DataFrame(rows)


def seen_positive_flags(table: pd.DataFrame, folds: pd.DataFrame,
                        seen_roles=("train", "val")) -> pd.DataFrame:
    """One row per held-out credited (question, paper) pair, flagged `seen` when the same paper
    is credited for a question with a role in `seen_roles` in that question's test fold.

    table: insight.data label table (index query_id, `positives`); folds: results/folds.csv.
    Used for the memorization check: a trained model may rank a paper high because it was a
    training positive, not because it learned what makes a paper inspiring.
    """
    rows = []
    for f, g in folds.groupby("fold"):
        seen_q = g.query_id[g.role.isin(seen_roles)]
        seen = set().union(*(set(table.positives.loc[q]) for q in seen_q))
        for q in g.query_id[g.role == "test"]:
            for d in table.positives.loc[q]:
                rows.append({"query_id": q, "doc_id": d, "fold": int(f), "seen": d in seen})
    return pd.DataFrame(rows)


def pair_hits(pairs: pd.DataFrame, run: pd.DataFrame, k: int = 20) -> pd.Series:
    """For (query_id, doc_id) credited pairs, whether the paper is in the run's top k."""
    top = run[run["rank"] <= k]
    hit = set(zip(top.query_id, top.doc_id))
    return pd.Series([(q, d) in hit for q, d in zip(pairs.query_id, pairs.doc_id)], index=pairs.index)


def recall_by_flag(flagged: pd.DataFrame, run: pd.DataFrame, flag: str, k: int = 20) -> pd.DataFrame:
    """Per-question Recall@k computed separately over the credited papers with flag True and False.

    flagged: credited pairs (query_id, doc_id, flag). Returns one row per question with columns
    `True` and `False` (NaN when the question has no credited paper in that group).
    """
    f = flagged.assign(hit=pair_hits(flagged, run, k).to_numpy())
    out = f.groupby(["query_id", flag]).hit.mean().unstack(flag)
    return out.reindex(columns=[True, False])
