import math

import numpy as np
import pandas as pd
import pytest

from insight import data, metrics, rerank as R

MODEL = "Qwen/Qwen3-Reranker-0.6B"


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)


needs_model = pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} not in the local HF cache")

CARD_QUERY = "What is the capital of China?"
CARD_DOCS = ["The capital of China is Beijing.",
             "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
             "objects and is responsible for the movement of planets around the sun."]


def test_format_pair_matches_model_card():
    # format_instruction in the Qwen3-Reranker card (Transformers usage).
    assert R.format_pair("q", "d", "I") == "<Instruct>: I\n<Query>: q\n<Document>: d"
    assert R.PREFIX.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert R.PREFIX.endswith("<|im_end|>\n<|im_start|>user\n")
    assert R.SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_config_from_yaml():
    cfg = R.RerankConfig.from_yaml(data.ROOT / "configs" / "C.yaml")
    assert cfg.model == MODEL and cfg.depth == 100 and cfg.instruction == R.INSPIRATION_INSTRUCTION
    assert cfg.fields == ["title", "text"] and cfg.max_length == 8192


def _fake_scores(queries, docs):
    # Score = number of 'x' in the doc; lets the test control the order exactly.
    return np.array([d.count("x") for d in docs], dtype=float)


def test_rerank_run_resorts_with_id_tiebreak_and_keeps_first_rank():
    sl = pd.DataFrame({"query_id": ["q1"] * 4 + ["q2"] * 2, "rank": [1, 2, 3, 4, 1, 2],
                       "doc_id": ["a", "b", "c", "d", "a", "e"], "score": 0.0})
    text = {"a": "x", "b": "xxx", "c": "xx", "d": "xx", "e": "xxxx"}
    out = R.rerank_run(sl, {"q1": "Q1", "q2": "Q2"}, text, _fake_scores, depth=100, top_k=100)
    q1 = out[out.query_id == "q1"]
    assert q1.doc_id.tolist() == ["b", "c", "d", "a"]  # c and d tie at 2: id order
    assert q1["rank"].tolist() == [1, 2, 3, 4] and q1.first_rank.tolist() == [2, 3, 4, 1]
    assert out[out.query_id == "q2"].doc_id.tolist() == ["e", "a"]
    assert q1.doc_id.tolist() == metrics.rank(dict(zip(q1.doc_id, q1.score)))


def test_rerank_run_depth_and_top_k():
    sl = pd.DataFrame({"query_id": "q", "rank": [1, 2, 3], "doc_id": ["a", "b", "c"], "score": 0.0})
    text = {"a": "", "b": "x", "c": "xx"}
    out = R.rerank_run(sl, {"q": "Q"}, text, _fake_scores, depth=2, top_k=1)
    assert out.doc_id.tolist() == ["b"]  # c is beyond depth 2, so it is never scored


def test_labeled_pairs():
    table = pd.DataFrame({"positives": [["a"], ["b", "c"]], "negatives": [["d", "e"], []]},
                         index=["q1", "q2"])
    lp = R.labeled_pairs(table, ["q1", "q2"])
    assert list(zip(lp.query_id, lp.doc_id, lp.label)) == [
        ("q1", "a", 1), ("q1", "d", 0), ("q1", "e", 0), ("q2", "b", 1), ("q2", "c", 1)]


@pytest.fixture(scope="module")
def reranker():
    return R.Reranker(R.RerankConfig(model=MODEL, dtype="float32", instruction=R.CARD_INSTRUCTION), device="cpu")


@needs_model
def test_prompt_tokens_equal_the_repo_chat_template(reranker):
    # The model repo's chat_template.jinja builds the same prompt the card's code does.
    tok = reranker.tokenizer
    for d in CARD_DOCS:
        text = tok.apply_chat_template([{"role": "query", "content": CARD_QUERY},
                                        {"role": "document", "content": d}], tokenize=False)
        assert reranker.token_ids(CARD_QUERY, d) == tok(text, add_special_tokens=False)["input_ids"]
    assert (reranker.yes_id, reranker.no_id) == (9693, 2152)  # 1_LogitScore/config.json in the repo


@needs_model
def test_scores_close_to_model_card_reference(reranker):
    # The card's Sentence Transformers example prints raw logit differences [7.625, -11.375].
    # Measured here (float32, CPU, same prompt as the repo's chat template): 7.597 and -12.192.
    # The first matches; the second is 0.82 lower in float32, bf16 and fp16 alike (see the
    # decision log, Nov 9), so it is checked with a looser tolerance. The order and signs agree.
    s = reranker.score([CARD_QUERY] * 2, CARD_DOCS)
    assert abs(s[0] - 7.625) < 0.05
    assert abs(s[1] - (-11.375)) < 1.0
    assert s[0] > 0 > s[1]


@needs_model
def test_score_is_log_odds_of_the_cards_probability(reranker):
    # The card's compute_logits returns softmax([z_no, z_yes])[1]; sigmoid(s) must equal it.
    import torch
    inputs = reranker.collate([reranker.token_ids(CARD_QUERY, d) for d in CARD_DOCS])
    with torch.inference_mode():
        logits = reranker.model(**inputs).logits[:, -1, :]
        card = torch.log_softmax(torch.stack([logits[:, reranker.no_id], logits[:, reranker.yes_id]], 1), 1)[:, 1].exp()
        s = reranker.score_inputs(inputs)
    np.testing.assert_allclose(torch.sigmoid(s).numpy(), card.numpy(), atol=1e-6)


@needs_model
def test_scores_deterministic_and_independent_of_batching(reranker):
    q = [CARD_QUERY, "Explain gravity", CARD_QUERY]
    d = [CARD_DOCS[0], CARD_DOCS[1], CARD_DOCS[1]]
    together = reranker.score(q, d, batch_size=3)
    np.testing.assert_allclose(reranker.score(q, d, batch_size=3), together, atol=0)
    alone = np.array([reranker.score([a], [b])[0] for a, b in zip(q, d)])
    np.testing.assert_allclose(together, alone, atol=1e-3)
    assert not reranker.model.training


@needs_model
def test_score_inputs_keeps_gradients_for_training(reranker):
    inputs = reranker.collate([reranker.token_ids(CARD_QUERY, CARD_DOCS[0])])
    s = reranker.score_inputs(inputs)
    assert s.requires_grad and s.shape == (1,) and math.isfinite(s.item())
