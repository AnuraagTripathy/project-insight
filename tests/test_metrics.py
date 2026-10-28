import math

import numpy as np
import pytest

from insight import metrics as M

RANKED = ["a", "b", "c", "d", "e"]
POS = {"b", "d", "x"}  # "x" is credited but never retrieved


# Hand-computed toy cases.

def test_recall_toy():
    assert M.recall_at_k(RANKED, POS, 1) == 0.0
    assert M.recall_at_k(RANKED, POS, 2) == pytest.approx(1 / 3)
    assert M.recall_at_k(RANKED, POS, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k(RANKED, POS, 100) == pytest.approx(2 / 3)  # k beyond the list


def test_ndcg_toy():
    # Hits at ranks 2 and 4; ideal puts the 3 credited papers at ranks 1, 2, 3.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert M.ndcg_at_k(RANKED, POS, 5) == pytest.approx(dcg / idcg, abs=1e-12)
    # With k = 2 the ideal DCG only has 2 slots.
    assert M.ndcg_at_k(RANKED, POS, 2) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert M.ndcg_at_k(["b", "d", "x"], POS, 3) == pytest.approx(1.0)


def test_mrr_toy():
    assert M.mrr(RANKED, POS) == 0.5
    assert M.mrr(RANKED, POS, k=1) == 0.0
    assert M.mrr(["x"], POS) == 1.0


def test_no_positive_in_top_k():
    assert M.recall_at_k(RANKED, {"z"}, 5) == 0.0
    assert M.ndcg_at_k(RANKED, {"z"}, 5) == 0.0
    assert M.mrr(RANKED, {"z"}) == 0.0
    assert M.recall_at_k([], {"z"}, 5) == 0.0


def test_no_positives_at_all_is_nan():
    for f in (lambda: M.recall_at_k(RANKED, set(), 5), lambda: M.ndcg_at_k(RANKED, set(), 5),
              lambda: M.mrr(RANKED, set())):
        assert math.isnan(f())


def test_rank_breaks_ties_by_id():
    assert M.rank({"c": 1.0, "a": 1.0, "b": 2.0, "d": 0.5}) == ["b", "a", "c", "d"]
    # Same input in a different order gives the same ranking.
    assert M.rank({"d": 0.5, "b": 2.0, "a": 1.0, "c": 1.0}) == ["b", "a", "c", "d"]


def test_pairwise_accuracy():
    s = {"p1": 3.0, "p2": 1.0, "n1": 2.0, "n2": 0.0, "n3": 1.0}
    # Pairs: p1 beats all 3; p2 loses to n1, beats n2, ties n3 (0.5). Total 4.5 / 6.
    assert M.pairwise_accuracy(s, ["p1", "p2"], ["n1", "n2", "n3"]) == pytest.approx(4.5 / 6)
    assert math.isnan(M.pairwise_accuracy(s, ["p1"], []))
    with pytest.raises(KeyError):
        M.pairwise_accuracy(s, ["missing"], ["n1"])


def test_evaluate_missing_query_scores_zero():
    df = M.evaluate({"q1": RANKED}, {"q1": POS, "q2": {"a"}}, ks=(2, 5))
    assert df.loc["q1", "recall@5"] == pytest.approx(2 / 3)
    assert df.loc["q2"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert list(df.columns) == ["recall@2", "recall@5", "ndcg@20", "mrr"]


# Agreement with ranx on random runs (scores without ties, so both sides rank identically).

def _random_case(seed, n_q=30, n_docs=200, depth=100):
    rng = np.random.default_rng(seed)
    run, qrels = {}, {}
    for q in range(n_q):
        docs = [f"d{i}" for i in rng.choice(n_docs, size=depth, replace=False)]
        scores = rng.permutation(depth).astype(float) + rng.random(depth) * 0.1
        run[f"q{q}"] = dict(zip(docs, scores))
        n_pos = int(rng.integers(1, 15))
        qrels[f"q{q}"] = {f"d{i}": 1 for i in rng.choice(n_docs, size=n_pos, replace=False)}
    return run, qrels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_ranx(seed):
    ranx = pytest.importorskip("ranx")
    run, qrels = _random_case(seed)
    names = ["recall@5", "recall@20", "recall@100", "ndcg@20", "mrr"]
    ref = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), names, return_mean=False, make_comparable=True)
    ours = M.evaluate({q: M.rank(s) for q, s in run.items()}, qrels)
    qids = sorted(qrels)  # ranx returns per-query arrays in sorted query order
    for name in names:
        np.testing.assert_allclose(ours.loc[qids, name].to_numpy(), np.asarray(ref[name]), atol=1e-6)
