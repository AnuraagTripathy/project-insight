"""Tests for insight.stats: the paired bootstrap over projects."""

import math

import numpy as np
import pandas as pd
import pytest

from insight import stats


def _synthetic(n_projects=200, effect=0.1, sd_project=0.2, sd_question=0.05, seed=1):
    """Projects with 2 to 6 questions; system b = a + effect + noise; a project effect is shared
    by all its questions, so questions within a project are strongly correlated."""
    rng = np.random.default_rng(seed)
    groups, a, b = [], [], []
    for p in range(n_projects):
        n = rng.integers(2, 7)
        base = rng.normal(0.5, sd_project)
        shift = effect + rng.normal(0, sd_project)
        for _ in range(n):
            x = base + rng.normal(0, sd_question)
            groups.append(f"p{p}")
            a.append(x)
            b.append(x + shift + rng.normal(0, sd_question))
    return np.array(a), np.array(b), np.array(groups)


def test_project_counts_shape_and_totals():
    c = stats.project_counts(7, 50, seed=0)
    assert c.shape == (50, 7)
    assert (c.sum(axis=1) == 7).all()
    assert (c == stats.project_counts(7, 50, seed=0)).all()


def test_recovers_known_synthetic_effect():
    a, b, g = _synthetic(effect=0.1)
    r = stats.paired_bootstrap(a, b, g, n_boot=10_000, seed=0)
    assert r["lo"] < 0.1 < r["hi"]
    assert r["lo"] > 0
    assert r["p"] < 0.01
    assert r["diff"] == pytest.approx((b - a).mean())
    assert r["n_projects"] == 200


def test_no_effect_interval_covers_zero():
    a, b, g = _synthetic(effect=0.0, seed=3)
    r = stats.paired_bootstrap(a, b, g, n_boot=5_000, seed=0)
    assert r["lo"] < 0 < r["hi"]
    assert r["p"] > 0.05


def test_null_coverage_close_to_nominal():
    """Over 200 null datasets the 95% interval should contain the true effect about 95% of the time."""
    hits = 0
    for s in range(200):
        a, b, g = _synthetic(n_projects=60, effect=0.0, seed=100 + s)
        r = stats.paired_bootstrap(a, b, g, n_boot=1_000, seed=s)
        hits += r["lo"] <= 0 <= r["hi"]
    assert 0.88 <= hits / 200 <= 0.99


def test_resamples_whole_projects():
    """Every value in a project equals the project id's number, so any bootstrap mean is a
    weighted mean of whole projects: with 2 projects of sizes 1 and 3 and values 0 and 1, the
    only reachable means are 0 (p0 twice), 1 (p1 twice) and 3/4 (once each)."""
    values = np.array([0.0, 1.0, 1.0, 1.0])
    groups = np.array(["p0", "p1", "p1", "p1"])
    m = stats.bootstrap_means(values, groups, n_boot=2_000, seed=0)
    assert set(np.round(m, 12)) == {0.0, 0.75, 1.0}


def test_project_resampling_wider_than_question_resampling():
    """With correlated questions, resampling projects must give a wider interval than treating
    every question as its own project."""
    a, b, g = _synthetic(effect=0.1, sd_project=0.2, sd_question=0.01)
    by_project = stats.paired_bootstrap(a, b, g, n_boot=5_000, seed=0)
    by_question = stats.paired_bootstrap(a, b, np.arange(len(a)), n_boot=5_000, seed=0)
    w_p = by_project["hi"] - by_project["lo"]
    w_q = by_question["hi"] - by_question["lo"]
    assert w_p > 1.5 * w_q


def test_nan_dropped_pairwise():
    a = np.array([0.0, 1.0, math.nan, 0.5])
    b = np.array([1.0, math.nan, 1.0, 0.5])
    g = np.array(["x", "x", "y", "z"])
    r = stats.paired_bootstrap(a, b, g, n_boot=200)
    assert r["n_questions"] == 2
    assert r["diff"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        stats.bootstrap_means(a, g)


def test_bootstrap_ci_and_summarize():
    df = pd.DataFrame({"kind": ["core", "subfield"] * 3, "m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                      index=[f"q{i}" for i in range(6)])
    proj = pd.Series(["a", "a", "b", "b", "c", "c"], index=df.index)
    s = stats.summarize({"X": df}, ["m", "missing"], proj, n_boot=500)
    assert list(s.kind) == ["core", "subfield"]
    assert s.loc[s.kind == "core", "mean"].item() == pytest.approx(0.3)
    assert (s["lo"] <= s["mean"]).all() and (s["mean"] <= s["hi"]).all()


def test_compare_aligns_on_query_id():
    idx = ["q1", "q2", "q3", "q4"]
    A = pd.DataFrame({"kind": ["core"] * 4, "m": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    B = A.assign(m=[1.0, 1.0, 1.0, 1.0]).iloc[::-1]  # reversed order must not matter
    proj = pd.Series(["a", "b", "c", "d"], index=idx)
    r = stats.compare({"A": A, "B": B}, [("A", "B")], "m", proj, kinds=("core",), n_boot=200)
    assert r["diff"].item() == 1.0 and r["lo"].item() == 1.0


def test_seen_positive_flags_and_recall_by_flag():
    table = pd.DataFrame({"positives": [["d1", "d2"], ["d1"], ["d3", "d4"]]}, index=["q1", "q2", "q3"])
    folds = pd.DataFrame({"query_id": ["q1", "q2", "q3", "q1", "q2", "q3"],
                          "fold": [0, 0, 0, 1, 1, 1],
                          "role": ["train", "test", "test", "test", "val", "train"]})
    f = stats.seen_positive_flags(table, folds)
    got = {(r.query_id, r.doc_id): r.seen for r in f.itertuples()}
    # fold 0: seen = {d1, d2}; fold 1: seen = {d1} (val) | {d3, d4} (train)
    assert got == {("q2", "d1"): True, ("q3", "d3"): False, ("q3", "d4"): False,
                   ("q1", "d1"): True, ("q1", "d2"): False}
    assert len(stats.seen_positive_flags(table, folds, seen_roles=("train",))) == 5
    run = pd.DataFrame({"query_id": ["q1", "q1", "q3"], "rank": [1, 30, 2], "doc_id": ["d2", "d1", "d4"]})
    r = stats.recall_by_flag(f, run, "seen", k=20)
    assert r.loc["q1", True] == 0.0 and r.loc["q1", False] == 1.0
    assert r.loc["q3", False] == 0.5 and math.isnan(r.loc["q3", True])


def test_config_defaults_and_yaml(tmp_path):
    c = stats.StatsConfig()
    assert c.n_boot == 10_000 and c.release_date == "2025-06-05"
    p = tmp_path / "s.yaml"
    p.write_text("n_boot: 100\nseen_roles: [train]\n")
    c = stats.StatsConfig.from_yaml(p)
    assert c.n_boot == 100 and c.seen_roles == ("train",)
