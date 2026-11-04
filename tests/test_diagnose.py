import math

import numpy as np
import pandas as pd
import pytest

from insight import data, diagnose as D, metrics

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "corpus.parquet").exists(),
    reason="full dataset not downloaded (run scripts/download_data.py)",
)

TABLE = pd.DataFrame({
    "paper_id": ["p1", "p1", "p2"],
    "kind": ["core", "subfield", "core"],
    "positives": [["a", "b", "c"], ["a"], ["d", "e"]],
}, index=pd.Index(["q1", "q2", "q3"], name="id"))
# q1: a at rank 3, b at rank 50, c missing. q2: a at rank 25. q3: d at 1, e missing; x is unlabeled.
RUN = pd.DataFrame({
    "query_id": ["q1", "q1", "q1", "q2", "q3", "q3"],
    "rank": [1, 3, 50, 25, 1, 2],
    "doc_id": ["x", "a", "b", "a", "d", "x"],
})
CORPUS = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e", "x"],
    "primary_category": ["cs.LG", "cs.CV", "Computer Science", "stat.ML", "", "cs.LG"],
})
QUERIES = pd.DataFrame({"paper_id": ["p1", "p1", "p2"], "paper_domain": ["cs.LG", "cs.LG", "cs.LG"]})


def test_paper_field_levels():
    assert D.paper_field("cs.LG") == "cs.LG"
    assert D.paper_field("q-bio.NC") == "q-bio.NC"
    assert D.paper_field("cond-mat.mtrl-sci", D.FieldConfig(level="archive")) == "cond-mat"
    assert D.paper_field("Computer Science") is None
    assert D.paper_field("") is None and D.paper_field(None) is None
    cfg = D.FieldConfig(level="archive", name_map={"Computer Science": "cs"})
    assert D.paper_field("Computer Science", cfg) == "cs"
    assert D.paper_field("Medicine", cfg) is None
    with pytest.raises(ValueError):
        D.paper_field("cs.LG", D.FieldConfig(level="nope"))


def test_credited_ranks_and_bins():
    pairs = D.credited_ranks(RUN, TABLE)
    assert len(pairs) == 6  # 3 + 1 + 2 credited pairs, unlabeled x is not included
    r = pairs.set_index(["query_id", "doc_id"])["rank"]
    assert r["q1", "a"] == 3 and r["q1", "b"] == 50 and math.isnan(r["q1", "c"])
    assert r["q2", "a"] == 25 and r["q3", "d"] == 1 and math.isnan(r["q3", "e"])
    bins = D.rank_bin(pd.Series([1, 20, 21, 100, 101, np.nan]))
    assert list(bins) == ["1 to 20", "1 to 20", "21 to 100", "21 to 100", "beyond 100", "beyond 100"]


def test_bin_shares_rows_sum_to_one():
    s = D.bin_shares(D.credited_ranks(RUN, TABLE))
    assert np.allclose(s.drop(columns="n").sum(axis=1), 1.0)
    assert s.loc["core", "n"] == 5 and s.loc["subfield", "n"] == 1
    assert s.loc["core", "1 to 20"] == pytest.approx(2 / 5)
    assert s.loc["core", "beyond 100"] == pytest.approx(2 / 5)


def test_add_fields_policies():
    pairs = D.credited_ranks(RUN, TABLE)
    ex = D.add_fields(pairs, CORPUS, QUERIES).set_index(["query_id", "doc_id"]).split
    assert ex["q1", "a"] == "in-field" and ex["q1", "b"] == "cross-field"
    assert ex["q1", "c"] == "unknown" and ex["q3", "e"] == "unknown"
    assert ex["q3", "d"] == "cross-field"
    cr = D.add_fields(pairs, CORPUS, QUERIES, D.FieldConfig(unknown_policy="cross")).set_index(["query_id", "doc_id"]).split
    assert cr["q1", "c"] == "cross-field" and cr["q1", "a"] == "in-field"
    ar = D.add_fields(pairs, CORPUS, QUERIES, D.FieldConfig(level="archive", name_map={"Computer Science": "cs"}))
    ar = ar.set_index(["query_id", "doc_id"]).split
    assert ar["q1", "b"] == "in-field" and ar["q1", "c"] == "in-field" and ar["q3", "d"] == "cross-field"


def test_recall_by_group_matches_metrics_on_all_pairs():
    pairs = D.credited_ranks(RUN, TABLE)
    pairs["all"] = "all"
    rec = D.recall_by_group(pairs, "all", ks=(20, 100))
    # Ranks in RUN have gaps, so fill the unlisted positions with dummy ids.
    ranked = {}
    for q, g in RUN.groupby("query_id"):
        lst = [f"_{i}" for i in range(int(g["rank"].max()))]
        for d, r in zip(g.doc_id, g["rank"]):
            lst[r - 1] = d
        ranked[q] = lst
    ev = metrics.evaluate(ranked, TABLE.positives.to_dict(), ks=(20, 100))
    for kind in ["core", "subfield"]:
        qs = TABLE.index[TABLE.kind == kind]
        assert rec.loc[(kind, "all"), "recall@20"] == pytest.approx(ev.loc[qs, "recall@20"].mean())
        assert rec.loc[(kind, "all"), "recall@100"] == pytest.approx(ev.loc[qs, "recall@100"].mean())


def test_recall_by_group_skips_questions_without_group():
    pairs = D.add_fields(D.credited_ranks(RUN, TABLE), CORPUS, QUERIES)
    rec = D.recall_by_group(pairs, "split", ks=(20, 100))
    # core in-field: only q1's a (rank 3). core cross-field: q1's b (50) and q3's d (1).
    assert rec.loc[("core", "in-field"), "n_questions"] == 1
    assert rec.loc[("core", "in-field"), "recall@20"] == 1.0
    assert rec.loc[("core", "cross-field"), "n_questions"] == 2
    assert rec.loc[("core", "cross-field"), "recall@20"] == pytest.approx((0 + 1) / 2)
    assert rec.loc[("core", "cross-field"), "recall@100"] == 1.0


@needs_data
def test_full_data_bins_match_bm25_metrics():
    ds = data.load()
    run = pd.read_parquet(data.ROOT / "results" / "run_A.parquet")
    pairs = D.credited_ranks(run, ds.table)
    assert len(pairs) == ds.table.positives.map(len).sum()
    pairs["all"] = "all"
    rec = D.recall_by_group(pairs, "all")
    per_q = pd.read_csv(data.ROOT / "results" / "metrics_A.csv", index_col="query_id")
    for kind in ["core", "subfield"]:
        assert rec.loc[(kind, "all"), "recall@20"] == pytest.approx(per_q[per_q.kind == kind]["recall@20"].mean())
