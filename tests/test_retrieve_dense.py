import numpy as np
import pandas as pd
import pytest
import torch

from insight import data, metrics, retrieve_dense as R


def _cached(model_id: str) -> bool:
    """True if the model's weights are already in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    # Small checkpoints ship one safetensors file, larger ones a sharded index.
    return any(isinstance(try_to_load_from_cache(model_id, f), str)
               for f in ("model.safetensors", "model.safetensors.index.json"))


SMALL = "Qwen/Qwen3-Embedding-0.6B"
needs_small_model = pytest.mark.skipif(not _cached(SMALL), reason=f"{SMALL} not in the local HF cache")


def test_format_query_matches_model_card():
    # get_detailed_instruct in the Qwen3-Embedding card: no space after "Query:".
    assert R.format_query("Explain gravity", "Do X") == "Instruct: Do X\nQuery:Explain gravity"
    assert R.format_query("q").startswith("Instruct: Given a scientific research question, retrieve papers")


def test_doc_texts_title_newline_abstract_and_missing_fields():
    df = pd.DataFrame({"title": ["T", None, " "], "text": ["abs", "only abstract", "x"]})
    assert R.doc_texts(df) == ["T\nabs", "only abstract", "x"]
    assert R.doc_texts(df, ["text"]) == ["abs", "only abstract", "x"]


def test_last_token_pool_left_and_right_padding():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    assert torch.equal(R.last_token_pool(h, left), h[:, -1])
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert torch.equal(R.last_token_pool(h, right), torch.stack([h[0, 1], h[1, 2]]))


def _unit(rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


DOCS = _unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
IDS = ["a", "b", "c", "d", "e"]
INDEX = R.DenseIndex(DOCS, IDS)


def test_search_exact_order_and_scores():
    q = _unit([[1, 0.05, 0]])[0]
    hits = INDEX.search(q, k=5)
    assert [d for d, _ in hits][:3] == ["a", "b", "c"]
    np.testing.assert_allclose([s for _, s in hits], DOCS[[IDS.index(d) for d, _ in hits]] @ q, rtol=1e-6)


def test_search_ties_broken_by_id():
    q = np.array([0, 0, 1], dtype=np.float32)
    hits = INDEX.search(q, k=2)
    assert [d for d, _ in hits] == ["d", "e"] and hits[0][1] == hits[1][1]
    assert [d for d, _ in hits] == metrics.rank(dict(hits))[:2]


def test_mask_removes_papers_but_still_returns_k():
    q = _unit([[1, 0.05, 0]])[0]
    mask = np.array([False, False, True, True, True])
    hits = INDEX.search(q, k=3, mask=mask)
    assert len(hits) == 3 and {d for d, _ in hits} == {"c", "d", "e"}


def test_masked_search_equals_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    docs = _unit(rng.normal(size=(500, 16)))
    ids = [f"p{i:03d}" for i in range(500)]
    index = R.DenseIndex(docs, ids)
    for _ in range(10):
        q = _unit(rng.normal(size=(1, 16)))[0]
        mask = rng.random(500) > 0.3
        s = docs @ q
        allowed = np.flatnonzero(mask)
        expect = [ids[i] for i in allowed[np.argsort(-s[allowed], kind="stable")][:20]]
        assert [d for d, _ in index.search(q, 20, mask)] == expect


def test_run_table_shape():
    q_emb = _unit([[1, 0, 0], [0, 1, 0]])
    df = R.run(INDEX, ["q1", "q2"], q_emb, masks={"q1": None, "q2": np.array([1, 1, 0, 1, 1], bool)}, k=2)
    assert list(df.columns) == ["query_id", "rank", "doc_id", "score"]
    assert df[df.query_id == "q1"].doc_id.tolist() == ["a", "b"]
    assert "c" not in df[df.query_id == "q2"].doc_id.tolist()


def test_config_from_yaml():
    cfg = R.DenseConfig.from_yaml(data.ROOT / "configs" / "B.yaml")
    assert cfg.model == "Qwen/Qwen3-Embedding-4B" and cfg.top_k == 100
    assert cfg.instruction == R.PAPER_INSTRUCTION and cfg.fields == ["title", "text"]


@needs_small_model
def test_encoder_reproduces_model_card_scores():
    # Reference scores printed in the Qwen3-Embedding-0.6B model card (Transformers example).
    cfg = R.DenseConfig(model=SMALL, dtype="float32", instruction="Given a web search query, retrieve relevant passages that answer the query")
    enc = R.Encoder(cfg, device="cpu")
    q = enc.encode_queries(["What is the capital of China?", "Explain gravity"])
    d = enc.encode_docs(["The capital of China is Beijing.",
                         "Gravity is a force that attracts two bodies towards each other. It gives weight to "
                         "physical objects and is responsible for the movement of planets around the sun."])
    expected = [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
    np.testing.assert_allclose(q @ d.T, expected, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
    # Deterministic, and independent of batch composition (length sorting, padding).
    np.testing.assert_allclose(enc.encode_queries(["Explain gravity"]), q[1:], atol=1e-5)


LARGE = "Qwen/Qwen3-Embedding-4B"


@pytest.mark.skipif(not _cached(LARGE), reason=f"{LARGE} not in the local HF cache")
def test_encoder_4b_close_to_model_card_scores_in_bf16():
    # Reference scores printed in the Qwen3-Embedding-4B model card. bf16 on CPU, so a looser tolerance.
    cfg = R.DenseConfig(model=LARGE, dtype="bfloat16", instruction="Given a web search query, retrieve relevant passages that answer the query")
    enc = R.Encoder(cfg, device="cpu")
    q = enc.encode_queries(["What is the capital of China?", "Explain gravity"])
    d = enc.encode_docs(["The capital of China is Beijing.",
                         "Gravity is a force that attracts two bodies towards each other. It gives weight to "
                         "physical objects and is responsible for the movement of planets around the sun."])
    expected = [[0.7534257769584656, 0.1146894246339798], [0.03198453038930893, 0.6258305311203003]]
    assert q.shape[1] == 2560
    np.testing.assert_allclose(q @ d.T, expected, atol=5e-3)
