"""Training the first-stage retriever on author labels (RQ3).

The retriever is a bi-encoder (insight.retrieve_dense): a question q and a paper d are embedded
separately into unit vectors, and their score is the inner product s(q, d) = <e_q, e_d>, which
equals cosine similarity because both vectors have length 1. We fine-tune Qwen3-Embedding-0.6B so
that more credited papers reach a question's top 100.

Loss (InfoNCE). One training example is a question q and one paper p+ credited for it. In a batch
of B examples, each question is compared with every paper in the batch: the B credited papers
(its own and the other examples', the "in-batch negatives") plus up to `hard_negatives` papers
the authors rejected for each question (the "hard negatives"). With temperature tau,

    L = -log( exp(s(q, p+) / tau) / ( exp(s(q, p+) / tau) + sum_j exp(s(q, p_j-) / tau) ) ),

a softmax over one positive and many negatives, i.e. cross entropy with the positive as target.
At tau = 0.05 a cosine gap of 0.05 between two papers becomes a logit gap of 1.

Masking (false negatives). Some in-batch papers must not count as negatives for a question:
another paper credited for the same question (two examples of one question can share a batch),
a duplicate of its own positive, and, by default, a paper credited for another question of the
same project, unless the authors rejected that paper for this question (then it is a true
negative). Masked papers are left out of the sum. Papers are compared by id, so duplicates are
handled the same way.

Validation (early stopping). Encoding all 190,896 papers after every epoch is too slow, so the
model is checked on a fixed pool: every credited and rejected paper of the inner validation
questions plus a seeded random sample of `val_pool_size` other corpus papers. The metric is
Recall@k' in that pool with k' = round(val_k * pool / corpus size), so the top k' keeps the same
share of the pool as the top 100 keeps of the corpus. The date filter is not applied in the pool.
This is a proxy for full-corpus Recall@100 (decision log, Nov 18).

Leakage: `fold_examples` reads only the train and validation questions of a split, never the
test fold's labels; a test checks its output does not change when the test rows are removed.
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
from insight.retrieve_dense import PAPER_INSTRUCTION, format_query, last_token_pool


@dataclass
class RetrieverTrainConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    instruction: str = PAPER_INSTRUCTION  # same query instruction as condition B
    max_length_query: int = 512
    max_length_doc: int = 1024
    # Loss
    tau: float = 0.05
    hard_negatives: int = 3  # author-rejected papers per example, resampled every epoch
    mask_project_positives: bool = True  # never use a paper credited for a sibling question as a negative
    # Fine-tuning: lora_r None = full fine-tuning; an int = LoRA of that rank on target_modules.
    lora_r: int | None = None
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Optimization
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    epochs: int = 3
    batch_size: int = 32  # examples (question, credited paper) per step
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    autocast: str | None = None  # "bfloat16" on the H100 (weights stay float32); None on CPU
    # Early stopping on validation pool recall
    eval_every: int | None = None  # steps between evaluations; None = once per epoch
    patience: int = 2
    val_pool_size: int = 20000
    val_k: int = 100
    eval_batch_size: int = 64
    # Data
    kinds: list[str] = field(default_factory=lambda: ["core", "subfield"])
    seed: int = 0

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("train", {}))


# ---------- loss ----------

def info_nce(q_emb, d_emb, target, mask=None, tau: float = 0.05):
    """Mean InfoNCE loss over questions.

    q_emb: (B, h) question vectors; d_emb: (N, h) paper vectors (unit length, so q_emb @ d_emb.T
    is cosine similarity); target: (B,) column of each question's positive in d_emb;
    mask: optional (B, N) bool, True where the paper may be used (the target must be True).
    """
    import torch
    import torch.nn.functional as F

    logits = (q_emb @ d_emb.T).float() / tau
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return F.cross_entropy(logits, torch.as_tensor(target, device=logits.device))


# ---------- examples, batches and masks ----------

def build_examples(table: pd.DataFrame, query_ids) -> pd.DataFrame:
    """One row per (question, credited paper): query_id, pos_id."""
    rows = [(q, p) for q in query_ids for p in table.at[q, "positives"]]
    return pd.DataFrame(rows, columns=["query_id", "pos_id"])


def fold_examples(table: pd.DataFrame, split, cfg: RetrieverTrainConfig = RetrieverTrainConfig()):
    """Training examples and validation questions for one fold, from train and val rows only.

    Returns (examples, val_q): examples has columns query_id, pos_id; val_q is the list of
    validation question ids (restricted to cfg.kinds, like training).
    """
    test = set(split.test)
    assert not test & set(split.train) and not test & set(split.val), "test question in train or val"
    sub = table.loc[list(split.train) + list(split.val)]  # the only rows this function reads
    kinds = set(cfg.kinds)
    train_q = [q for q in split.train if sub.at[q, "kind"] in kinds]
    val_q = [q for q in split.val if sub.at[q, "kind"] in kinds]
    return build_examples(sub, train_q), val_q


def project_positives(table: pd.DataFrame) -> dict:
    """{project id: every paper credited for any of its questions in `table`}."""
    return {pid: {d for ps in g.positives for d in ps} for pid, g in table.groupby("paper_id")}


def make_batch(examples: pd.DataFrame, table: pd.DataFrame, n_hard: int, rng,
               mask_project_positives: bool = True, proj_pos: dict | None = None):
    """Papers and mask for one batch of examples.

    Returns (query_ids, doc_ids, target, mask): doc_ids lists the B positives first (example i's
    positive is column i), then the sampled hard negatives of every example; mask[i, j] is False
    when paper j is credited for question i (other than column i itself) or, with
    mask_project_positives, credited for another question of i's project and not rejected for i.
    proj_pos: precomputed project_positives (computed from `table` if None).
    """
    qids = examples.query_id.tolist()
    docs = examples.pos_id.tolist()
    for q in qids:
        neg = list(table.at[q, "negatives"])
        if n_hard and neg:
            take = rng.choice(len(neg), size=min(n_hard, len(neg)), replace=False)
            docs += [neg[i] for i in sorted(take)]
    use_proj = mask_project_positives and "paper_id" in table
    if use_proj and proj_pos is None:
        proj_pos = project_positives(table)
    mask = np.ones((len(qids), len(docs)), dtype=bool)
    for i, q in enumerate(qids):
        r = table.loc[q]
        bad = set(r.positives)
        if use_proj:
            bad |= proj_pos.get(r.paper_id, set()) - set(r.negatives)
        for j, d in enumerate(docs):
            if j != i and d in bad:
                mask[i, j] = False
    return qids, docs, np.arange(len(qids)), mask


# ---------- encoding ----------

def embed(encoder, texts: list[str], max_length: int):
    """Unit vectors for texts, with gradients (encoder: insight.retrieve_dense.Encoder or alike)."""
    import torch.nn.functional as F

    batch = encoder.tokenize(texts, max_length)
    hidden = encoder.model(**batch).last_hidden_state
    return F.normalize(last_token_pool(hidden, batch["attention_mask"]).float(), p=2, dim=1)


def encode_texts(encoder, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    """Unit vectors for texts without gradients, longest first, returned in input order."""
    torch = encoder.torch
    order = np.argsort([len(t) for t in texts], kind="stable")[::-1]
    out = None
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            e = embed(encoder, [texts[i] for i in idx], max_length).cpu().numpy()
            if out is None:
                out = np.empty((len(texts), e.shape[1]), dtype=np.float32)
            out[idx] = e
    return out


# ---------- validation proxy ----------

def val_pool(table: pd.DataFrame, val_q, corpus_ids, pool_size: int, seed: int = 0) -> list:
    """Every credited and rejected paper of the validation questions plus a seeded sample of
    `pool_size` other corpus papers (fewer if the corpus is smaller), in corpus order."""
    labeled = {d for q in val_q for d in list(table.at[q, "positives"]) + list(table.at[q, "negatives"])}
    others = [d for d in corpus_ids if d not in labeled]
    rng = np.random.default_rng(seed)
    take = set(rng.choice(len(others), size=min(pool_size, len(others)), replace=False).tolist())
    keep = labeled | {others[i] for i in take}
    return [d for d in corpus_ids if d in keep]


def pool_k(val_k: int, pool: int, corpus: int) -> int:
    """k' = round(val_k * pool / corpus), at least 1: the same share of the pool as val_k of the corpus."""
    return max(1, round(val_k * pool / corpus))


def evaluate_pool(encoder, table: pd.DataFrame, val_q, pool_ids, question_of: dict, text_of: dict,
                  k: int, cfg: RetrieverTrainConfig) -> dict:
    """Mean Recall@k over val_q in the pool, and mean pairwise accuracy (credited vs rejected)."""
    model = encoder.model
    was_training = model.training
    model.eval()
    qv = encode_texts(encoder, [format_query(question_of[q], cfg.instruction) for q in val_q],
                      cfg.max_length_query, cfg.eval_batch_size)
    dv = encode_texts(encoder, [text_of[d] for d in pool_ids], cfg.max_length_doc, cfg.eval_batch_size)
    if was_training:
        model.train()
    sims = qv @ dv.T
    pool_arr = np.asarray(pool_ids)
    rec, acc = [], []
    for i, q in enumerate(val_q):
        pos, neg = list(table.at[q, "positives"]), list(table.at[q, "negatives"])
        top = np.lexsort((pool_arr, -sims[i]))[:k]  # ties by paper id, as in insight.metrics.rank
        rec.append(metrics.recall_at_k(pool_arr[top].tolist(), pos, k))
        sc = dict(zip(pool_ids, sims[i].tolist()))
        a = metrics.pairwise_accuracy(sc, pos, neg)
        if not math.isnan(a):
            acc.append(a)
    rec = [r for r in rec if not math.isnan(r)]
    return {"val_recall": float(np.mean(rec)) if rec else math.nan,
            "val_pairwise_acc": float(np.mean(acc)) if acc else math.nan}


# ---------- training ----------

def prepare_model(encoder, cfg: RetrieverTrainConfig):
    """Full fine-tuning (cfg.lora_r None) or LoRA adapters; returns the trainable parameters."""
    if cfg.lora_r is not None:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not isinstance(encoder.model, PeftModel):
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=list(cfg.target_modules), bias="none",
                              task_type="FEATURE_EXTRACTION")
            encoder.model = get_peft_model(encoder.model, lcfg)
    else:
        for p in encoder.model.parameters():
            p.requires_grad_(True)
    if cfg.gradient_checkpointing:
        encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        encoder.model.enable_input_require_grads()
    return [p for p in encoder.model.parameters() if p.requires_grad]


def _state(model) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def _load_state(model, state: dict):
    params = dict(model.named_parameters())
    for n, v in state.items():
        params[n].data.copy_(v.to(params[n].device, params[n].dtype))


def train(encoder, examples: pd.DataFrame, table: pd.DataFrame, question_of: dict, text_of: dict,
          cfg: RetrieverTrainConfig = RetrieverTrainConfig(), val_q=None, pool_ids=None,
          corpus_size: int | None = None, log=print) -> dict:
    """Fine-tune encoder in place with InfoNCE, early-stopping on validation pool recall.

    examples: columns query_id, pos_id (training questions only). table: indexed by query id,
    columns positives, negatives and (for project masking) paper_id; only rows of the examples'
    questions and val_q are read (project positives come from the examples' questions, which hold
    every question of their projects since folds never split a project). val_q / pool_ids None: no evaluation, the last weights are kept.
    corpus_size: for k' (defaults to the pool size, so k' = val_k).
    Returns the history (one row per evaluation), best step and best validation recall; the best
    weights are loaded into the model and it is left in eval mode.
    """
    torch = encoder.torch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainable = prepare_model(encoder, cfg)
    model = encoder.model
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total = max(1, steps_per_epoch * cfg.epochs)
    warm = int(cfg.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    eval_every = cfg.eval_every or steps_per_epoch
    do_eval = bool(val_q) and pool_ids is not None
    k = pool_k(cfg.val_k, len(pool_ids), corpus_size or len(pool_ids)) if do_eval else None

    def evaluate():
        return evaluate_pool(encoder, table, val_q, pool_ids, question_of, text_of, k, cfg) if do_eval \
            else {"val_recall": math.nan, "val_pairwise_acc": math.nan}

    ev = evaluate()
    best = {"step": 0, **ev, "state": _state(model)}
    history = [{"step": 0, "epoch": 0.0, "train_loss": math.nan, **ev}]
    log(f"step 0: val recall@{k} {ev['val_recall']:.4f}, val pairwise acc {ev['val_pairwise_acc']:.4f}, "
        f"{len(examples)} examples, {steps_per_epoch} steps per epoch")

    ac = (torch.autocast(device_type=device.type, dtype=getattr(torch, cfg.autocast))
          if cfg.autocast else None)
    train_rows = table.loc[list(dict.fromkeys(examples.query_id))]
    proj_pos = project_positives(train_rows) if "paper_id" in train_rows else None
    rng = np.random.default_rng(cfg.seed)
    model.train()
    step, since_best, running, t0, stop = 0, 0, [], time.time(), False
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), cfg.batch_size):
            ex = examples.iloc[order[start:start + cfg.batch_size]]
            qids, docs, target, mask = make_batch(ex, train_rows, cfg.hard_negatives, rng,
                                                 cfg.mask_project_positives, proj_pos)
            qtexts = [format_query(question_of[q], cfg.instruction) for q in qids]
            dtexts = [text_of[d] for d in docs]
            if ac is not None:
                with ac:
                    qe, de = embed(encoder, qtexts, cfg.max_length_query), embed(encoder, dtexts, cfg.max_length_doc)
            else:
                qe, de = embed(encoder, qtexts, cfg.max_length_query), embed(encoder, dtexts, cfg.max_length_doc)
            loss = info_nce(qe, de, torch.as_tensor(target, device=qe.device),
                            torch.as_tensor(mask, device=qe.device), cfg.tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running.append(loss.item())
            step += 1
            last = start + cfg.batch_size >= len(order) and epoch + 1 == cfg.epochs
            if step % eval_every == 0 or last:
                ev = evaluate()
                row = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                       "train_loss": float(np.mean(running)), **ev}
                history.append(row)
                log(f"step {step} (epoch {row['epoch']}): train loss {row['train_loss']:.4f}, "
                    f"val recall@{k} {ev['val_recall']:.4f}, val pairwise acc {ev['val_pairwise_acc']:.4f}, "
                    f"{time.time() - t0:.0f}s")
                running = []
                if not do_eval or ev["val_recall"] > best["val_recall"]:
                    best = {"step": step, **ev, "state": _state(model)}
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= cfg.patience:
                        log(f"early stop: no improvement for {cfg.patience} evaluations")
                        stop = True
                        break
        if stop:
            break

    _load_state(model, best["state"])
    model.eval()
    return {"history": pd.DataFrame(history), "best_step": best["step"],
            "best_val_recall": best["val_recall"], "val_k": k, "steps": step, "config": asdict(cfg)}


def save_encoder(encoder, out_dir):
    """Save the fine-tuned encoder as a plain Transformers model (LoRA merged), plus its tokenizer,
    so insight.retrieve_dense.Encoder can load it by path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = encoder.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    encoder.tokenizer.save_pretrained(out_dir)
