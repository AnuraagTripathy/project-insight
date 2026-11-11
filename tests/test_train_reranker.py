import math

import numpy as np
import pandas as pd
import pytest
import torch

from insight import data, folds, rerank as R, train_reranker as T

MODEL = "Qwen/Qwen3-Reranker-0.6B"


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_id, "model.safetensors"), str)


needs_model = pytest.mark.skipif(not _cached(MODEL), reason=f"{MODEL} not in the local HF cache")
needs_data = pytest.mark.skipif(not (data.DATA_DIR / "corpus.parquet").exists(), reason="data/ not downloaded")


# ---------- loss ----------

def test_loss_is_ln2_at_zero_margin():
    s = torch.tensor([0.3, -2.0, 5.0])
    assert T.pairwise_loss(s, s).item() == pytest.approx(math.log(2), abs=1e-7)


def test_loss_decreases_with_margin_and_matches_formula():
    margins = torch.linspace(-6, 6, 25)
    loss = torch.stack([T.pairwise_loss(m.view(1), torch.zeros(1)) for m in margins])
    assert torch.all(loss[1:] < loss[:-1])
    expected = -torch.log(torch.sigmoid(margins))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert T.pairwise_loss(torch.tensor([200.0]), torch.tensor([0.0])).item() == pytest.approx(0, abs=1e-12)
    assert math.isfinite(T.pairwise_loss(torch.tensor([-200.0]), torch.tensor([0.0])).item())


def test_loss_weights():
    sp, sn = torch.tensor([1.0, -1.0]), torch.zeros(2)
    per = -torch.log(torch.sigmoid(sp - sn))
    assert T.pairwise_loss(sp, sn, torch.tensor([1.0, 0.0])).item() == pytest.approx(per[0].item(), abs=1e-6)
    w = torch.tensor([1.0, 0.3])
    assert T.pairwise_loss(sp, sn, w).item() == pytest.approx(((w * per).sum() / w.sum()).item(), abs=1e-6)


# ---------- pairs and leakage ----------

def _toy_table():
    return pd.DataFrame({
        "kind": ["core", "subfield", "core", "subfield", "core"],
        "positives": [["a", "b"], ["a"], ["c"], ["d", "e"], ["f"]],
        "negatives": [["x", "y", "z"], [], ["w"], ["v"], ["u", "t"]],
    }, index=["q1", "q2", "q3", "q4", "q5"])


def test_build_pairs_all_and_capped():
    t = _toy_table()
    p = T.build_pairs(t, ["q1", "q2", "q3"])
    assert len(p) == 2 * 3 + 0 + 1  # q2 has no rejected papers, so no pairs
    assert set(zip(p.query_id, p.pos_id, p.neg_id)) >= {("q1", "a", "x"), ("q1", "b", "z"), ("q3", "c", "w")}
    assert (p.weight == 1.0).all()
    capped = T.build_pairs(t, ["q1"], max_per_query=4, seed=1)
    assert len(capped) == 4 and capped.duplicated().sum() == 0
    assert capped.equals(T.build_pairs(t, ["q1"], max_per_query=4, seed=1))  # seeded


def test_fold_pairs_never_reads_test_labels():
    t = _toy_table()
    sp = folds.Split(fold=0, train=["q1", "q2", "q3"], val=["q4"], test=["q5"])
    pairs, val = T.fold_pairs(t, sp)
    assert set(pairs.query_id) <= {"q1", "q2", "q3"} and set(val.query_id) == {"q4"}
    assert not {"u", "t", "f"} & (set(pairs.pos_id) | set(pairs.neg_id) | set(val.doc_id))
    # Same output with the test rows deleted, or replaced by garbage: test labels are never read.
    pairs2, val2 = T.fold_pairs(t.drop(index="q5"), sp)
    assert pairs.equals(pairs2) and val.equals(val2)
    bad = t.copy()
    bad.at["q5", "positives"] = None
    pairs3, _ = T.fold_pairs(bad, sp)
    assert pairs.equals(pairs3)


def test_fold_pairs_rejects_overlapping_split_and_filters_kinds():
    t = _toy_table()
    with pytest.raises(AssertionError):
        T.fold_pairs(t, folds.Split(fold=0, train=["q1", "q5"], val=[], test=["q5"]))
    sp = folds.Split(fold=0, train=["q1", "q2", "q3", "q4"], val=[], test=["q5"])
    pairs, _ = T.fold_pairs(t, sp, T.TrainConfig(kinds=["core"]))
    assert set(pairs.query_id) == {"q1", "q3"}


@needs_data
def test_real_fold_pairs_stay_inside_train_and_val():
    ds = data.load()
    table, cfg = ds.table, folds.FoldConfig.from_yaml(data.ROOT / "configs" / "folds.yaml")
    for f in range(cfg.n_folds):
        sp = folds.split(ds.queries, f, cfg)
        pairs, val = T.fold_pairs(table, sp)
        assert set(pairs.query_id) <= set(sp.train) and set(val.query_id) <= set(sp.val)
        assert len(pairs) == sum(len(table.at[q, "positives"]) * len(table.at[q, "negatives"]) for q in sp.train)


def test_config_defaults_match_the_plan():
    c = T.TrainConfig()
    assert (c.lora_r, c.lora_alpha, c.lora_dropout, c.epochs) == (16, 32, 0.05, 3)
    assert c.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------- training loop on a tiny random Qwen3 (fast, no download) ----------

class TinyReranker:
    """Same interface as insight.rerank.Reranker, with a tiny random Qwen3 and a character tokenizer."""

    def __init__(self, seed=0):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(seed)
        cfg = Qwen3Config(vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=256, tie_word_embeddings=False)
        self.torch = torch
        self.model = Qwen3ForCausalLM(cfg).eval()
        self.yes_id, self.no_id = 1, 2

    def token_ids(self, query, doc):
        return [3 + (ord(c) % 256) for c in f"{query}|{doc}"]

    def collate(self, ids):
        n = max(map(len, ids))
        inp = torch.tensor([[0] * (n - len(x)) + x for x in ids])
        mask = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids])
        return {"input_ids": inp, "attention_mask": mask}

    def score_inputs(self, inputs):
        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
        return (logits[:, self.yes_id] - logits[:, self.no_id]).float()


def _toy_pairs(n=20):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    question_of = {f"q{i}": f"question {i}" for i in range(n)}
    text_of = {}
    rows = []
    for i in range(n):
        p, m = f"p{i}", f"n{i}"
        text_of[p] = " ".join(rng.choice(words, 3)) + f" {i}"
        text_of[m] = " ".join(rng.choice(words, 3)) + f" {i}!"
        rows.append((f"q{i}", p, m))
    pairs = pd.DataFrame(rows, columns=["query_id", "pos_id", "neg_id"]).assign(weight=1.0)
    val = pd.DataFrame([(q, d, l) for q, p, m in rows for d, l in [(p, 1), (m, 0)]],
                       columns=["query_id", "doc_id", "label"])
    return pairs, val, question_of, text_of


def test_one_step_changes_only_adapter_weights():
    rr = TinyReranker()
    before = {n: p.detach().clone() for n, p in rr.model.named_parameters()}
    pairs, _, qo, to = _toy_pairs(4)
    cfg = T.TrainConfig(epochs=1, batch_pairs=4, grad_accum=1, lr=1e-2, warmup_frac=0.0, lora_dropout=0.0)
    T.train(rr, pairs, None, qo, to, cfg, log=lambda *_: None)
    changed, frozen_changed = [], []
    for n, p in rr.model.named_parameters():
        base = n.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" in n:
            changed.append(n)
        elif not torch.equal(p.detach(), before[base]):
            frozen_changed.append(n)
    assert changed and not frozen_changed
    assert all(not p.requires_grad for n, p in rr.model.named_parameters() if "lora_" not in n)
    # lora_B starts at zero; after one step it is non-zero, so the adapters really moved.
    assert any(p.abs().sum() > 0 for n, p in rr.model.named_parameters() if "lora_B" in n)


def test_lora_starts_as_identity():
    rr = TinyReranker()
    pairs, val, qo, to = _toy_pairs(5)
    ids = [rr.token_ids(qo[q], to[d]) for q, d in zip(val.query_id, val.doc_id)]
    base = T.score_ids(rr, ids, 8)
    rr.model = T.add_lora(rr.model, T.TrainConfig())
    rr.model.eval()
    assert np.allclose(T.score_ids(rr, ids, 8), base, atol=1e-6)


def test_overfit_20_pairs_reaches_accuracy_1():
    rr = TinyReranker()
    pairs, val, qo, to = _toy_pairs(20)
    cfg = T.TrainConfig(epochs=40, batch_pairs=10, grad_accum=1, lr=3e-3, warmup_frac=0.0,
                        lora_dropout=0.0, eval_every=10, patience=100)
    out = T.train(rr, pairs, val, qo, to, cfg, log=lambda *_: None)
    assert out["best_val_pairwise_acc"] == 1.0
    # The best adapters are the ones left in the model, in eval mode.
    ids = [rr.token_ids(qo[q], to[d]) for q, d in zip(val.query_id, val.doc_id)]
    assert T.mean_pairwise_accuracy(val, T.score_ids(rr, ids, 16)) == 1.0
    assert out["history"].train_loss.iloc[-1] < out["history"].train_loss.iloc[1]


def test_early_stopping_stops_and_restores_best():
    rr = TinyReranker()
    pairs, val, qo, to = _toy_pairs(6)
    # lr 0: nothing changes, so validation never improves on step 0 and training stops early.
    cfg = T.TrainConfig(epochs=10, batch_pairs=2, grad_accum=1, lr=0.0, eval_every=1, patience=2,
                        lora_dropout=0.0)
    out = T.train(rr, pairs, val, qo, to, cfg, log=lambda *_: None)
    assert out["best_step"] == 0 and out["steps"] == 2 and len(out["history"]) == 3


def test_mean_pairwise_accuracy_averages_questions():
    val = pd.DataFrame({"query_id": ["a", "a", "a", "b", "b", "c"], "doc_id": ["1", "2", "3", "4", "5", "6"],
                        "label": [1, 0, 0, 1, 0, 1]})
    # a: 1 beats 2 but not 3 -> 0.5; b: 4 beats 5 -> 1.0; c has no rejected papers -> skipped.
    assert T.mean_pairwise_accuracy(val, [2, 1, 3, 1, 0, 9]) == pytest.approx(0.75)


# ---------- the real 0.6B reranker (CPU, short texts) ----------

@needs_model
def test_real_reranker_one_lora_step_only_adapters_change():
    rr = R.Reranker(R.RerankConfig(model=MODEL, dtype="float32"), device="cpu")
    qo = {"q": "Can spiking neural networks learn control policies with reinforcement learning?"}
    to = {"p": "Jump-Start Reinforcement Learning", "n": "Deep Reinforcement Learning with Spiking Q-learning"}
    pairs = pd.DataFrame({"query_id": ["q"], "pos_id": ["p"], "neg_id": ["n"], "weight": [1.0]})
    ids = [rr.token_ids(qo["q"], to[d]) for d in ["p", "n"]]
    s0 = T.score_ids(rr, ids, 2)
    before = {n: p.detach().clone() for n, p in rr.model.named_parameters()}
    cfg = T.TrainConfig(epochs=1, batch_pairs=1, grad_accum=1, lr=1e-3, warmup_frac=0.0, lora_dropout=0.0)
    T.train(rr, pairs, None, qo, to, cfg, log=lambda *_: None)
    n_lora = 0
    for n, p in rr.model.named_parameters():
        if "lora_" in n:
            n_lora += 1
            continue
        base = n.replace("base_model.model.", "").replace(".base_layer", "")
        assert torch.equal(p.detach(), before[base]), n
    assert n_lora == 28 * 4 * 2  # 28 layers x 4 attention projections x (A, B)
    s1 = T.score_ids(rr, ids, 2)
    assert (s1[0] - s1[1]) > (s0[0] - s0[1])  # the step widened the margin on its own pair
    trainable = sum(p.numel() for p in rr.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in rr.model.parameters())
    assert trainable / total < 0.01


def test_gradient_checkpointing_gives_the_same_training():
    pairs, val, qo, to = _toy_pairs(6)
    outs = []
    for gc in [False, True]:
        rr = TinyReranker()
        cfg = T.TrainConfig(epochs=2, batch_pairs=3, grad_accum=1, lr=1e-2, warmup_frac=0.0,
                            lora_dropout=0.0, eval_every=1, patience=100, gradient_checkpointing=gc)
        outs.append(T.train(rr, pairs, val, qo, to, cfg, log=lambda *_: None)["history"])
    assert np.allclose(outs[0].train_loss.fillna(0), outs[1].train_loss.fillna(0), atol=1e-5)
