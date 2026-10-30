import numpy as np
import pandas as pd
import pytest

from insight import data, metrics, retrieve_bm25 as R

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "corpus.parquet").exists(),
    reason="full dataset not downloaded (run scripts/download_data.py)",
)

# Toy corpus: "d" and "e" have identical text, so they tie on every query.
CORPUS = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e"],
    "title": ["Spiking neural networks", "Reinforcement learning", "Graph kernels", "Protein folding", "Protein folding"],
    "text": ["Energy efficient spiking neurons.", "Jump start policies for reinforcement learning agents.",
             "Kernels on graphs.", "Folding proteins with learning.", "Folding proteins with learning."],
})
INDEX = R.BM25Index(R.doc_texts(CORPUS), CORPUS.id)


def test_doc_texts_joins_fields_and_handles_missing():
    df = pd.DataFrame({"title": ["T", None], "text": ["abs", "only abstract"]})
    assert R.doc_texts(df) == ["T abs", " only abstract"]
    assert R.doc_texts(df, ["text"]) == ["abs", "only abstract"]


def test_best_match_first_and_zero_scores_dropped():
    hits = INDEX.search("spiking neurons", k=5)
    assert [d for d, _ in hits] == ["a"]  # no other paper shares a term
    assert hits[0][1] > 0


def test_scores_unknown_terms_are_zero():
    assert (INDEX.scores("zzzz qqqq") == 0).all()
    assert INDEX.search("zzzz") == []


def test_ties_broken_by_id_and_top_k_respected():
    hits = INDEX.search("protein folding", k=1)
    assert [d for d, _ in hits] == ["d"]  # d and e tie, d < e
    hits = INDEX.search("protein folding", k=5)
    assert [d for d, _ in hits][:2] == ["d", "e"] and hits[0][1] == hits[1][1]
    assert [d for d, _ in hits] == metrics.rank(dict(hits))


def test_mask_excludes_papers():
    mask = np.array([True, True, True, False, True])
    assert [d for d, _ in INDEX.search("protein folding", k=5, mask=mask)][0] == "e"


def test_stemming_option_matches_inflections():
    plain = R.BM25Index(R.doc_texts(CORPUS), CORPUS.id, R.BM25Config(stemmer=None))
    stem = R.BM25Index(R.doc_texts(CORPUS), CORPUS.id, R.BM25Config(stemmer="english"))
    assert plain.search("kernel") == []  # only "kernels" is in the corpus
    assert [d for d, _ in stem.search("kernel")] == ["c"]


def test_run_table_and_ranked_lists():
    q = pd.DataFrame({"id": ["q1", "q2"], "question": ["protein folding", "graph kernels"]})
    df = R.run(INDEX, q, masks={"q1": np.array([True, True, True, False, True]), "q2": None}, k=2)
    assert list(df.columns) == ["query_id", "rank", "doc_id", "score"]
    lists = R.as_ranked_lists(df)
    assert lists["q1"][0] == "e" and lists["q2"] == ["c"]


def test_config_from_yaml():
    cfg = R.BM25Config.from_yaml(data.ROOT / "configs" / "A.yaml")
    assert cfg.top_k == 100 and cfg.fields == ["title", "text"]
    assert (cfg.k1, cfg.b, cfg.stemmer) == (1.5, 0.75, None)


# Full-data test: the committed run must match its own metrics file and the published gate.

@needs_data
def test_committed_run_A_reproduces_gate():
    run_path = data.ROOT / "results" / "run_A.parquet"
    if not run_path.exists():
        pytest.skip("results/run_A.parquet not generated")
    ds = data.load()
    run = pd.read_parquet(run_path)
    assert run.query_id.nunique() == 894 and run.groupby("query_id").size().max() == 100
    per_q = metrics.evaluate(R.as_ranked_lists(run), ds.table.positives.to_dict())
    core = per_q.loc[ds.table.index[ds.table.kind == "core"], "recall@20"].mean()
    assert abs(core - 0.23) <= 0.03
