"""Training the reranker on author labels (conditions D and E).

The reranker scores a (question, paper) pair as s(q, d) = z_yes - z_no (see insight.rerank).
Training pushes a credited paper p+ above a rejected paper p- of the *same* question q with the
pairwise logistic (RankNet) loss

    L = -log sigmoid(s(q, p+) - s(q, p-)) = softplus(-(s(q, p+) - s(q, p-))).

At margin 0 the loss is ln 2; it falls towards 0 as the margin grows and rises as it turns
negative. A per-pair weight w scales each term (the weighted mean is sum(w L) / sum(w)), so
condition E can give its mined, noisy negatives a lower weight.

Only LoRA adapters are trained: each targeted weight matrix W0 (d x k) is frozen and a low-rank
correction B A (B is d x r, A is r x k, r << d) is learned, W = W0 + (alpha / r) B A. B starts at
zero, so before the first step the model scores exactly like the frozen one (condition C).

Early stopping: after each evaluation the mean per-question pairwise accuracy on the inner
validation questions is computed (insight.metrics.pairwise_accuracy, ties count 0.5); the
adapter weights from the best evaluation are kept, and training stops after `patience`
evaluations without improvement.

Leakage: `fold_pairs` reads only the train and validation questions of a split, never the test
fold's labels; a test checks its output does not change when the test rows are removed.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from insight import metrics


@dataclass
class TrainConfig:
    # LoRA, the roadmap's starting values: r 16, alpha 32, dropout 0.05 on the attention projections
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.05
    epochs: int = 3
    batch_pairs: int = 8  # (credited, rejected) pairs per optimizer micro-batch
    grad_accum: int = 4  # micro-batches per optimizer step
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    # Early stopping on validation pairwise accuracy
    eval_every: int | None = None  # optimizer steps between evaluations; None = once per epoch
    patience: int = 2  # evaluations without improvement before stopping
    eval_batch_size: int = 32
    # Data
    max_pairs_per_query: int | None = None  # None = every credited x rejected pair
    kinds: list[str] = field(default_factory=lambda: ["core", "subfield"])  # ablation: ["core"]
    seed: int = 0

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("train", {}))


def pairwise_loss(s_pos, s_neg, weight=None):
    """Weighted mean of -log sigmoid(s_pos - s_neg) over pairs (torch tensors)."""
    import torch.nn.functional as F

    per_pair = F.softplus(-(s_pos - s_neg))  # equals -log sigmoid(margin), numerically stable
    if weight is None:
        return per_pair.mean()
    weight = weight.to(per_pair)
    return (weight * per_pair).sum() / weight.sum()


def build_pairs(table: pd.DataFrame, query_ids, max_per_query: int | None = None,
                seed: int = 0) -> pd.DataFrame:
    """Every (question, credited, rejected) triple for the given questions.

    table: indexed by query id with list columns `positives` and `negatives`.
    Returns columns query_id, pos_id, neg_id, weight (1.0). A question with no rejected papers
    gives no pairs. With max_per_query, a seeded random subset of each question's pairs is kept.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for qid in query_ids:
        r = table.loc[qid]
        qp = [(qid, p, n) for p in r.positives for n in r.negatives]
        if max_per_query is not None and len(qp) > max_per_query:
            keep = np.sort(rng.choice(len(qp), size=max_per_query, replace=False))
            qp = [qp[i] for i in keep]
        rows += qp
    df = pd.DataFrame(rows, columns=["query_id", "pos_id", "neg_id"])
    df["weight"] = 1.0
    return df


def fold_pairs(table: pd.DataFrame, split, cfg: TrainConfig = TrainConfig()):
    """Training pairs and validation labels for one fold, read from train and val questions only.

    split: an insight.folds.Split. Questions are restricted to cfg.kinds (the core-only
    ablation drops subfield questions from training and validation alike).
    Returns (train_pairs, val_labels) where val_labels has columns query_id, doc_id, label.
    """
    test = set(split.test)
    assert not test & set(split.train) and not test & set(split.val), "test question in train or val"
    sub = table.loc[list(split.train) + list(split.val)]  # the only rows this function reads
    kinds = set(cfg.kinds)
    train_q = [q for q in split.train if sub.at[q, "kind"] in kinds]
    val_q = [q for q in split.val if sub.at[q, "kind"] in kinds]
    pairs = build_pairs(sub, train_q, cfg.max_pairs_per_query, cfg.seed)
    rows = []
    for q in val_q:
        r = sub.loc[q]
        rows += [(q, d, 1) for d in r.positives] + [(q, d, 0) for d in r.negatives]
    val = pd.DataFrame(rows, columns=["query_id", "doc_id", "label"])
    return pairs, val


def add_lora(model, cfg: TrainConfig):
    """Wrap a causal LM with LoRA adapters; every other weight is frozen."""
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def adapter_state(model) -> dict:
    """A CPU copy of the trainable (adapter) weights."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model, state: dict):
    params = dict(model.named_parameters())
    for n, v in state.items():
        params[n].data.copy_(v.to(params[n].device, params[n].dtype))


def mean_pairwise_accuracy(val: pd.DataFrame, scores) -> float:
    """Mean over questions of pairwise accuracy (questions without pairs are skipped)."""
    v = val.assign(score=np.asarray(scores, dtype=float))
    accs = []
    for _, g in v.groupby("query_id", sort=False):
        a = metrics.pairwise_accuracy(dict(zip(g.doc_id, g.score)), g.doc_id[g.label == 1],
                                      g.doc_id[g.label == 0])
        if not math.isnan(a):
            accs.append(a)
    return float(np.mean(accs)) if accs else math.nan


def score_ids(reranker, ids: list[list[int]], batch_size: int) -> np.ndarray:
    """Scores for pre-tokenized prompts, no gradients, longest first for less padding."""
    torch = reranker.torch
    order = np.argsort([len(x) for x in ids], kind="stable")[::-1]
    out = np.empty(len(ids), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            idx = order[start:start + batch_size]
            out[idx] = reranker.score_inputs(reranker.collate([ids[i] for i in idx])).cpu().numpy()
    return out


def train(reranker, pairs: pd.DataFrame, val: pd.DataFrame | None, question_of: dict, text_of: dict,
          cfg: TrainConfig = TrainConfig(), log=print):
    """Train LoRA adapters on `pairs` in place, with early stopping on `val`.

    reranker: an insight.rerank.Reranker (or any object with token_ids, collate, score_inputs,
        model and torch). Its model is wrapped with LoRA here if it is not already a PEFT model.
    pairs: columns query_id, pos_id, neg_id and optional weight.
    val: columns query_id, doc_id, label, or None to skip evaluation (no early stopping).
    Returns a dict with the history (one row per evaluation), the best step and best accuracy.
    The best adapter weights are loaded into the model before returning, and it is left in
    eval mode.
    """
    torch = reranker.torch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    from peft import PeftModel

    if not isinstance(reranker.model, PeftModel):
        reranker.model = add_lora(reranker.model, cfg)
    model = reranker.model
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # Tokenize once: every distinct (question, paper) prompt.
    cache = {}

    def tok(q, d):
        key = (q, d)
        if key not in cache:
            cache[key] = reranker.token_ids(question_of[q], text_of[d])
        return cache[key]

    pos_ids = [tok(q, d) for q, d in zip(pairs.query_id, pairs.pos_id)]
    neg_ids = [tok(q, d) for q, d in zip(pairs.query_id, pairs.neg_id)]
    weights = pairs["weight"].to_numpy(dtype=np.float32) if "weight" in pairs else np.ones(len(pairs), np.float32)
    val_ids = [tok(q, d) for q, d in zip(val.query_id, val.doc_id)] if val is not None and len(val) else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(pairs) / (cfg.batch_pairs * cfg.grad_accum))
    total = max(1, steps_per_epoch * cfg.epochs)
    warm = int(cfg.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    eval_every = cfg.eval_every or steps_per_epoch

    def evaluate():
        model.eval()
        acc = mean_pairwise_accuracy(val, score_ids(reranker, val_ids, cfg.eval_batch_size))
        model.train()
        return acc

    history = []
    best = {"step": 0, "val_pairwise_acc": evaluate() if val_ids else math.nan,
            "state": adapter_state(model)}
    history.append({"step": 0, "epoch": 0.0, "train_loss": math.nan, "val_pairwise_acc": best["val_pairwise_acc"]})
    log(f"step 0: val pairwise acc {best['val_pairwise_acc']:.4f}, {len(pairs)} train pairs, "
        f"{steps_per_epoch} steps per epoch")

    rng = np.random.default_rng(cfg.seed)
    model.train()
    step, since_best, running, t0 = 0, 0, [], time.time()
    stop = False
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(pairs))
        micro = [order[i:i + cfg.batch_pairs] for i in range(0, len(order), cfg.batch_pairs)]
        for m, idx in enumerate(micro):
            batch = reranker.collate([pos_ids[i] for i in idx] + [neg_ids[i] for i in idx])
            s = reranker.score_inputs(batch)
            n = len(idx)
            w = torch.as_tensor(weights[idx], device=s.device)
            # Scale by the micro-batch's share of the step's pairs, so accumulation averages over the step.
            loss = pairwise_loss(s[:n], s[n:], w)
            (loss * len(idx) / (cfg.batch_pairs * cfg.grad_accum)).backward()
            running.append(loss.item())
            if (m + 1) % cfg.grad_accum and m + 1 != len(micro):
                continue
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or (m + 1 == len(micro) and epoch + 1 == cfg.epochs):
                acc = evaluate() if val_ids else math.nan
                row = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                       "train_loss": float(np.mean(running)), "val_pairwise_acc": acc}
                history.append(row)
                log(f"step {step} (epoch {row['epoch']}): train loss {row['train_loss']:.4f}, "
                    f"val pairwise acc {acc:.4f}, {time.time() - t0:.0f}s")
                running = []
                if val_ids is None or acc > best["val_pairwise_acc"]:
                    best = {"step": step, "val_pairwise_acc": acc, "state": adapter_state(model)}
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= cfg.patience:
                        log(f"early stop: no improvement for {cfg.patience} evaluations")
                        stop = True
                        break
        if stop:
            break

    load_adapter_state(model, best["state"])
    model.eval()
    return {"history": pd.DataFrame(history), "best_step": best["step"],
            "best_val_pairwise_acc": best["val_pairwise_acc"], "steps": step,
            "config": asdict(cfg)}
