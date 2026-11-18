import math

import numpy as np
import pandas as pd
import pytest
import torch

from insight import data, folds, train_retriever as T
from insight.retrieve_dense import format_query

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)


needs_model = pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} not in the local HF cache")
needs_data = pytest.mark.skipif(not (data.DATA_DIR / "corpus.parquet").exists(), reason="data/ not downloaded")


# ---------- loss ----------

def test_info_nce_matches_hand_computation():
    # One question q = (1, 0); papers: positive (1, 0), negatives (0, 1) and (0.6, 0.8); tau 0.5.
    # Cosines 1, 0, 0.6 -> logits 2, 0, 1.2 -> L = -log(e^2 / (e^2 + e^0 + e^1.2)).
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    expected = -math.log(math.exp(2) / (math.exp(2) + math.exp(0) + math.exp(1.2)))
    assert T.info_nce(q, d, [0], tau=0.5).item() == pytest.approx(expected, abs=1e-6)


def test_info_nce_toy_batch_with_mask_and_mean():
    # Two questions over three papers. Question 0 has paper 2 masked out (a false negative).
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    tau = 0.1
    l0 = -math.log(math.exp(10) / (math.exp(10) + math.exp(0)))
    l1 = -math.log(math.exp(10) / (math.exp(0) + math.exp(10) + math.exp(8)))
    got = T.info_nce(q, d, [0, 1], mask, tau).item()
    assert got == pytest.approx((l0 + l1) / 2, abs=1e-5)


def test_info_nce_equal_scores_is_log_n_and_falls_with_gap():
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[0.0, 1.0]] * 4)
    assert T.info_nce(q, d, [0]).item() == pytest.approx(math.log(4), abs=1e-6)
    losses = []
    for c in [0.0, 0.3, 0.6, 0.9, 1.0]:
        pos = torch.tensor([[c, math.sqrt(1 - c * c)]])
        losses.append(T.info_nce(q, torch.cat([pos, torch.tensor([[0.0, 1.0]] * 3)]), [0], tau=0.5).item())
    assert all(a > b for a, b in zip(losses, losses[1:]))


# ---------- examples, batches, masks, leakage ----------

def _toy_table():
    return pd.DataFrame({
        "paper_id": ["P1", "P1", "P2", "P3", "P3"],
        "kind": ["core", "subfield", "core", "core", "subfield"],
        "positives": [["a", "b"], ["c"], ["d"], ["e"], ["f", "g"]],
        "negatives": [["x", "c"], ["y"], ["z", "a"], ["w"], []],
    }, index=["q1", "q2", "q3", "q4", "q5"])


def test_build_examples_one_row_per_credited_paper():
    ex = T.build_examples(_toy_table(), ["q1", "q2"])
    assert list(zip(ex.query_id, ex.pos_id)) == [("q1", "a"), ("q1", "b"), ("q2", "c")]


def test_fold_examples_never_reads_test_labels():
    t = _toy_table()
    sp = folds.Split(fold=0, train=["q1", "q2"], val=["q3"], test=["q4", "q5"])
    ex, val_q = T.fold_examples(t, sp)
    assert set(ex.query_id) == {"q1", "q2"} and val_q == ["q3"]
    ex2, val_q2 = T.fold_examples(t.drop(index=["q4", "q5"]), sp)
    assert ex.equals(ex2) and val_q == val_q2
    core_only, _ = T.fold_examples(t, sp, T.RetrieverTrainConfig(kinds=["core"]))
    assert set(core_only.query_id) == {"q1"}


def test_make_batch_masks_false_negatives():
    t = _toy_table()
    ex = pd.DataFrame({"query_id": ["q1", "q1", "q2", "q3"], "pos_id": ["a", "b", "c", "d"]})
    rng = np.random.default_rng(0)
    qids, docs, target, mask = T.make_batch(ex, t, n_hard=5, rng=rng)
    assert docs[:4] == ["a", "b", "c", "d"] and list(target) == [0, 1, 2, 3]
    assert sorted(docs[4:]) == sorted(["x", "c", "x", "c", "y", "z", "a"])
    col = {j: d for j, d in enumerate(docs)}
    for i, q in enumerate(qids):
        assert mask[i, i]  # the target is always usable
        for j, d in col.items():
            if j == i:
                continue
            if q == "q1":
                # a and b are credited for q1; c is credited for sibling q2 but rejected for q1,
                # so it stays a negative.
                assert mask[i, j] == (d not in {"a", "b"})
            if q == "q2":
                # sibling q1's credited a and b are masked; c (its own positive) is masked elsewhere.
                assert mask[i, j] == (d not in {"a", "b", "c"})
            if q == "q3":
                # a is credited for q1 (another project) but rejected for q3: a true negative.
                assert mask[i, j] == (d != "d")
    # Without project masking, only the question's own credited papers are masked for q2.
    _, docs2, _, mask2 = T.make_batch(ex, t, n_hard=0, rng=rng, mask_project_positives=False)
    assert docs2 == ["a", "b", "c", "d"] and mask2[2].all()


def test_val_pool_holds_labeled_papers_and_is_seeded():
    t = _toy_table()
    corpus = list("abcdefgxyzw") + [f"o{i}" for i in range(50)]
    p1 = T.val_pool(t, ["q3"], corpus, pool_size=10, seed=1)
    assert {"d", "z", "a"} <= set(p1) and len(p1) == 13
    assert p1 == T.val_pool(t, ["q3"], corpus, pool_size=10, seed=1)
    assert p1 == [c for c in corpus if c in set(p1)]  # corpus order
    assert T.pool_k(100, 20000, 190896) == 10 and T.pool_k(100, 50, 190896) == 1


# ---------- training on a tiny random Qwen3 (fast, no download) ----------

class TinyEncoder:
    """Same interface as insight.retrieve_dense.Encoder, with a tiny random Qwen3 and characters as tokens."""

    def __init__(self, seed=0):
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(seed)
        cfg = Qwen3Config(vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=512)
        self.torch = torch
        self.model = Qwen3Model(cfg).eval()

    def tokenize(self, texts, max_length):
        ids = [[3 + (ord(c) % 256) for c in t][-max_length:] for t in texts]
        n = max(map(len, ids))
        return {"input_ids": torch.tensor([[0] * (n - len(x)) + x for x in ids]),
                "attention_mask": torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids])}


def _toy_retrieval(n=6):
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    rng = np.random.default_rng(0)
    table = pd.DataFrame({
        "paper_id": [f"P{i}" for i in range(n)],
        "kind": ["core"] * n,
        "positives": [[f"p{i}"] for i in range(n)],
        "negatives": [[f"n{i}"] for i in range(n)],
    }, index=[f"q{i}" for i in range(n)])
    # Every text ends with the same characters, so a random model's last-token vectors do not
    # already pair questions with their papers; the pairing has to be learned.
    question_of = {f"q{i}": f"{words[i % 8]} {i} end" for i in range(n)}
    text_of = {}
    for i in range(n):
        text_of[f"p{i}"] = " ".join(rng.choice(words, 2)) + " end"
        text_of[f"n{i}"] = " ".join(rng.choice(words, 2)) + " end"
    return table, question_of, text_of


def test_overfit_tiny_batch_reaches_recall_1():
    table, question_of, text_of = _toy_retrieval()
    enc = TinyEncoder()
    ex = T.build_examples(table, table.index)
    pool = sorted(text_of)
    cfg = T.RetrieverTrainConfig(lr=3e-3, weight_decay=0.0, epochs=60, batch_size=6, hard_negatives=1,
                                 warmup_frac=0.0, patience=1000, val_k=1, eval_every=20,
                                 max_length_query=64, max_length_doc=64, instruction="find")
    out = T.train(enc, ex, table, question_of, text_of, cfg, val_q=list(table.index), pool_ids=pool, log=lambda *_: None)
    h = out["history"]
    assert out["val_k"] == 1
    assert h.train_loss.iloc[-1] < h.train_loss.iloc[1]
    assert out["best_val_recall"] == 1.0  # each question's credited paper ranks first among all 12
    assert h.val_recall.iloc[0] < 1.0  # it did not start there
    assert not enc.model.training


def test_one_step_changes_weights_and_lora_option_freezes_base():
    table, question_of, text_of = _toy_retrieval()
    ex = T.build_examples(table, table.index)
    cfg = T.RetrieverTrainConfig(epochs=1, batch_size=6, lr=1e-3, max_length_query=64, max_length_doc=64)
    enc = TinyEncoder()
    before = {n: p.detach().clone() for n, p in enc.model.named_parameters()}
    T.train(enc, ex, table, question_of, text_of, cfg, log=lambda *_: None)
    changed = [n for n, p in enc.model.named_parameters() if not torch.equal(p, before[n])]
    assert len(changed) > len(before) // 2  # full fine-tuning updates the base weights

    enc = TinyEncoder()
    base = {n: p.detach().clone() for n, p in enc.model.named_parameters()}
    T.train(enc, ex, table, question_of, text_of,
            T.RetrieverTrainConfig(epochs=1, batch_size=6, lr=1e-3, lora_r=4, max_length_query=64,
                                   max_length_doc=64), log=lambda *_: None)
    for n, p in enc.model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad
        else:
            assert not p.requires_grad
            assert torch.equal(p, base[n.replace("base_model.model.", "").replace(".base_layer", "")])


def test_training_is_seeded():
    table, question_of, text_of = _toy_retrieval()
    ex = T.build_examples(table, table.index)
    cfg = T.RetrieverTrainConfig(epochs=2, batch_size=3, lr=1e-3, max_length_query=64, max_length_doc=64)
    runs = []
    for _ in range(2):
        enc = TinyEncoder()
        T.train(enc, ex, table, question_of, text_of, cfg, val_q=["q0", "q1"], pool_ids=sorted(text_of),
                log=lambda *_: None)
        runs.append(torch.cat([p.detach().flatten() for p in enc.model.parameters()]))
    assert torch.equal(runs[0], runs[1])


# ---------- real data ----------

@needs_data
def test_real_folds_examples_come_from_training_questions():
    ds = data.load()
    table = ds.table
    for sp in T_splits():
        ex, val_q = T.fold_examples(table, sp)
        assert set(ex.query_id) <= set(sp.train)
        assert not set(val_q) & set(sp.test)
        assert len(ex) == sum(len(table.at[q, "positives"]) for q in sp.train)


def T_splits():
    from insight.train_reranker import splits_from_csv
    return splits_from_csv("results/folds.csv")


# ---------- real model (only if cached) ----------

@needs_model
def test_real_embedder_one_full_step_lowers_loss_on_its_batch():
    import gc

    from insight.retrieve_dense import DenseConfig, Encoder

    gc.collect()
    enc = Encoder(DenseConfig(model=MODEL, dtype="float32"), device="cpu")
    table, question_of, text_of = _toy_retrieval(4)
    ex = T.build_examples(table, table.index)
    rng = np.random.default_rng(0)
    qids, docs, target, mask = T.make_batch(ex, table, 1, rng)
    qt = [format_query(question_of[q]) for q in qids]
    dt = [text_of[d] for d in docs]

    def loss():
        return T.info_nce(T.embed(enc, qt, 64), T.embed(enc, dt, 64), torch.as_tensor(target),
                          torch.as_tensor(mask), 0.05)

    trainable = T.prepare_model(enc, T.RetrieverTrainConfig())
    # Plain SGD keeps no optimizer state, so full fine-tuning of 0.6B fits the CI and CPU memory.
    opt = torch.optim.SGD(trainable, lr=1e-3)
    enc.model.train()
    l0 = loss()
    l0.backward()
    opt.step()
    with torch.no_grad():
        l1 = loss()
    assert l1.item() < l0.item()
