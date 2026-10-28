import pandas as pd
import pytest

from insight import data, folds

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "queries.parquet").exists(),
    reason="full dataset not downloaded (run scripts/download_data.py)",
)


def toy_queries(n_projects=23, per_project=4):
    rows = []
    for p in range(n_projects):
        for j in range(per_project):
            kind = "core_query" if j == 0 else "subfield_query"
            rows.append({"id": f"q_{p}_{j}", "paper_id": f"p{p}", "type": kind})
    return pd.DataFrame(rows)


def check_splits(q, cfg):
    project_of = dict(zip(q.id, q.paper_id))
    test_seen = []
    for f in range(cfg.n_folds):
        s = folds.split(q, f, cfg)
        tr, va, te = set(s.train), set(s.val), set(s.test)
        # Every question in exactly one role.
        assert tr | va | te == set(q.id)
        assert not (tr & va or tr & te or va & te)
        # No project shared between roles.
        ptr, pva, pte = ({project_of[x] for x in xs} for xs in (tr, va, te))
        assert not (ptr & pva or ptr & pte or pva & pte)
        test_seen += s.test
        # Validation is about val_frac of the non-test projects.
        assert len(pva) == round(cfg.val_frac * (len(ptr) + len(pva)))
    # Each question is tested exactly once over the folds.
    assert sorted(test_seen) == sorted(q.id)


def test_toy_splits():
    q = toy_queries()
    check_splits(q, folds.FoldConfig())


def test_fold_sizes_balanced_and_deterministic():
    q = toy_queries()
    f1 = folds.assign_folds(q.paper_id, 5, seed=0)
    f2 = folds.assign_folds(q.paper_id.sample(frac=1, random_state=3), 5, seed=0)
    pd.testing.assert_series_equal(f1, f2)  # input order does not matter
    sizes = f1.value_counts()
    assert sizes.max() - sizes.min() <= 1 and len(sizes) == 5
    assert not f1.equals(folds.assign_folds(q.paper_id, 5, seed=1))


def test_config_from_yaml():
    cfg = folds.FoldConfig.from_yaml(folds.ROOT / "configs" / "folds.yaml")
    assert cfg == folds.FoldConfig(n_folds=5, val_frac=0.2, seed=0)


@needs_data
def test_real_splits_no_project_in_two_folds():
    q = data.load_queries()
    cfg = folds.FoldConfig()
    check_splits(q, cfg)
    qf = folds.query_folds(q, cfg)
    # No question split from its project: all questions of a project share one fold.
    assert q.assign(fold=qf.to_numpy()).groupby("paper_id").fold.nunique().max() == 1
    t = folds.fold_table(q, cfg)
    assert len(t) == 5 * 894
    assert (t[t.role == "test"].groupby("query_id").size() == 1).all()


@needs_data
def test_saved_folds_match_code():
    path = folds.ROOT / "results" / "folds.csv"
    if not path.exists():
        pytest.skip("results/folds.csv not written yet")
    saved = pd.read_csv(path)
    fresh = folds.fold_table(data.load_queries(), folds.FoldConfig.from_yaml(folds.ROOT / "configs" / "folds.yaml"))
    pd.testing.assert_frame_equal(saved, fresh)
