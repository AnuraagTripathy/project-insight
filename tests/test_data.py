import numpy as np
import pandas as pd
import pytest

from insight import data

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "corpus.parquet").exists(),
    reason="full dataset not downloaded (run scripts/download_data.py)",
)


# Toy tests: no data needed.

def test_parse_dates_handles_all_formats():
    s = pd.Series(["2024-03-05 17:30:00", "2024-03-05", "2024-03", "2024", "1899-12-30", ""])
    out = data.parse_dates(s)
    assert list(out[:4]) == [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-05"),
                             pd.Timestamp("2024-03-01"), pd.Timestamp("2024-01-01")]
    assert out[4:].isna().all()


def test_candidate_mask_toy():
    pub = pd.to_datetime(pd.Series(["2020-01-01", "2025-07-09", "2025-07-10", "2026-01-01", None]))
    src = pd.Timestamp("2025-07-09")
    # Same calendar day is allowed, later days are not, undated kept by default.
    assert data.candidate_mask(pub, src).tolist() == [True, True, False, False, True]
    assert data.candidate_mask(pub, src, keep_undated=False).tolist() == [True, True, False, False, False]
    # A labeled late paper is kept only when exempt.
    assert data.candidate_mask(pub, src, [3]).tolist() == [True, True, False, True, True]
    assert data.candidate_mask(pub, src, [3], exempt_labeled=False)[3] == False  # noqa: E712


def test_config_from_yaml():
    cfg = data.DataConfig.from_yaml(data.ROOT / "configs" / "data.yaml")
    assert cfg.date_filter and cfg.date_filter_exempt_labeled and cfg.keep_undated
    assert cfg.data_dir == data.DATA_DIR


# Full-data tests.

@pytest.fixture(scope="module")
def ds():
    return data.load()


@needs_data
def test_sizes(ds):
    assert len(ds.corpus) == 190_896
    assert len(ds.queries) == 894
    assert (ds.labels.kind == "core").sum() == 207
    assert (ds.labels.kind == "subfield").sum() == 687
    assert ds.corpus.id.is_unique and ds.queries.id.is_unique
    assert set(ds.labels.query_id) == set(ds.queries.id)


@needs_data
def test_table_join(ds):
    t = ds.table
    assert len(t) == 894 and t.index.is_unique
    assert {"kind", "paper_id", "src_pub", "positives", "negatives"} <= set(t.columns)


@needs_data
def test_label_counts(ds):
    n_pos = ds.labels.positives.map(len)
    n_neg = ds.labels.negatives.map(len)
    core = ds.labels.kind == "core"
    assert n_pos[core].min() >= 1 and n_pos[~core].min() >= 1
    assert n_neg[core].min() >= 4
    assert n_pos.sum() == 4845
    assert ds.labels[core].rationales.map(len).sum() + (~core).sum() == 1348


@needs_data
def test_every_labeled_id_resolves(ds):
    ids = set(ds.corpus.id)
    for col in ["positives", "negatives"]:
        labeled = {i for xs in ds.labels[col] for i in xs}
        assert labeled <= ids, f"{len(labeled - ids)} {col} missing from corpus"


@needs_data
def test_every_project_has_one_core_query(ds):
    counts = ds.queries.groupby("paper_id").kind.value_counts().unstack(fill_value=0)
    assert (counts["core"] == 1).all()


@needs_data
def test_dates_parse(ds):
    assert ds.corpus.pub.isna().sum() == 1  # only the 1899-12-30 placeholder
    assert ds.queries.src_pub.notna().all()


@pytest.fixture(scope="module")
def masks(ds):
    return data.candidate_masks(ds)


@needs_data
def test_date_filter_keeps_every_labeled_paper(ds, masks):
    pos_of = pd.Series(np.arange(len(ds.corpus)), index=ds.corpus.id)
    for qid, row in ds.table.iterrows():
        labeled = pos_of[list(row.positives) + list(row.negatives)].to_numpy()
        assert masks[qid][labeled].all(), qid


@needs_data
def test_date_filter_drops_late_unlabeled(ds, masks):
    pos_of = pd.Series(np.arange(len(ds.corpus)), index=ds.corpus.id)
    pub = ds.corpus.pub.to_numpy()
    for qid, row in ds.table.iterrows():
        labeled = np.zeros(len(pub), dtype=bool)
        labeled[pos_of[list(row.positives) + list(row.negatives)].to_numpy()] = True
        late = (pub > np.datetime64(row.src_pub)) & ~labeled
        assert not masks[qid][late].any(), qid


@needs_data
def test_strict_filter_loses_exactly_the_late_credited_pairs(ds):
    strict = data.Dataset(ds.corpus, ds.queries, ds.labels,
                          data.DataConfig(date_filter_exempt_labeled=False))
    m = data.candidate_masks(strict)
    pos_of = pd.Series(np.arange(len(ds.corpus)), index=ds.corpus.id)
    lost = sum((~m[qid][pos_of[list(row.positives)].to_numpy()]).sum()
               for qid, row in ds.table.iterrows())
    assert lost == 27
