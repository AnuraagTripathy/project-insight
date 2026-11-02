"""Dense retrieval with a Qwen3-Embedding bi-encoder and FAISS (condition B).

The encoder follows the Qwen3-Embedding model card (Transformers usage):

- queries are written as ``"Instruct: {task}\\nQuery:{question}"`` (no space after "Query:",
  as in the card's ``get_detailed_instruct``); documents get no instruction;
- the tokenizer pads on the left and the embedding is the hidden state of the last token
  (``last_token_pool``), then L2-normalized, so inner product equals cosine similarity.

The task instruction is the one ScholarCatalyst (arXiv:2610.02202, Appendix C) prepends for its
Qwen3-Embedding runs. Each paper is embedded as title and abstract joined by a newline; the paper
does not say how documents were formatted (decision log, Nov 2).

Search: a FAISS ``IndexFlatIP`` (exact inner product over every paper). The date filter removes
papers per question, which a flat index cannot do natively, so each question asks FAISS for
``k + (number of papers its mask removes)`` hits and then drops the masked ones. That is exact:
at most that many papers can be removed, so at least k allowed papers remain among the hits.
Equal scores are ordered by paper id, the same rule as insight.metrics.rank.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ScholarCatalyst, arXiv:2610.02202, Appendix C: the instruction prepended to every query.
PAPER_INSTRUCTION = (
    "Given a scientific research question, retrieve papers that could inspire or contribute to "
    "answering the question, including relevant background knowledge, methods, and ideas."
)


@dataclass
class DenseConfig:
    model: str = "Qwen/Qwen3-Embedding-4B"
    instruction: str = PAPER_INSTRUCTION
    fields: list[str] = field(default_factory=lambda: ["title", "text"])
    field_sep: str = "\n"
    # Token caps. The model accepts 32K; these only cut pathological texts (decision log, Nov 2).
    max_length_query: int = 512
    max_length_doc: int = 1024
    batch_size: int = 32
    dtype: str = "bfloat16"  # "float32" on CPU for exact checks
    attn_implementation: str | None = "sdpa"
    top_k: int = 100

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("dense", {}))


def format_query(question: str, instruction: str = PAPER_INSTRUCTION) -> str:
    """Query string exactly as the model card's get_detailed_instruct builds it."""
    return f"Instruct: {instruction}\nQuery:{question}"


def doc_texts(corpus: pd.DataFrame, fields=("title", "text"), sep="\n") -> list[str]:
    """One string per paper: the chosen fields joined by `sep`, empty fields left out."""
    cols = corpus[list(fields)].fillna("").astype(str)
    return [sep.join(v.strip() for v in row if v.strip()) for row in cols.itertuples(index=False)]


def last_token_pool(last_hidden_states, attention_mask):
    """Hidden state of each sequence's last real token (model card's last_token_pool).

    Works for left padding (last column is real for every row) and right padding.
    """
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[rows, lengths]


class Encoder:
    """Qwen3-Embedding wrapper: texts in, L2-normalized float32 vectors out."""

    def __init__(self, cfg: DenseConfig = DenseConfig(), device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
        kwargs = {"dtype": getattr(torch, cfg.dtype)}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        self.model = AutoModel.from_pretrained(cfg.model, **kwargs).to(self.device).eval()

    def encode(self, texts: list[str], max_length: int, batch_size: int | None = None,
               show_progress: bool = False) -> np.ndarray:
        """Embed texts in batches of similar length (sorted), returned in the input order."""
        torch = self.torch
        bs = batch_size or self.cfg.batch_size
        lengths = [len(t) for t in texts]
        order = np.argsort(lengths)[::-1]  # longest first: an out-of-memory error shows up at once
        out = None
        it = range(0, len(texts), bs)
        if show_progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="encode", unit="batch")
        with torch.inference_mode():
            for start in it:
                idx = order[start:start + bs]
                batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                       max_length=max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                emb = torch.nn.functional.normalize(last_token_pool(hidden, batch["attention_mask"]), p=2, dim=1)
                emb = emb.float().cpu().numpy()
                if out is None:
                    out = np.empty((len(texts), emb.shape[1]), dtype=np.float32)
                out[idx] = emb
        return out if out is not None else np.empty((0, 0), dtype=np.float32)

    def encode_queries(self, questions: list[str], **kw) -> np.ndarray:
        texts = [format_query(q, self.cfg.instruction) for q in questions]
        return self.encode(texts, self.cfg.max_length_query, **kw)

    def encode_docs(self, texts: list[str], **kw) -> np.ndarray:
        return self.encode(texts, self.cfg.max_length_doc, **kw)


class DenseIndex:
    """Exact inner-product search over fixed document vectors with per-question masks."""

    def __init__(self, doc_emb: np.ndarray, ids):
        import faiss

        self.ids = np.asarray(ids)
        emb = np.ascontiguousarray(doc_emb, dtype=np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, q_emb: np.ndarray, k: int = 100, mask: np.ndarray | None = None):
        """Top k (paper id, score) pairs for one query vector, best first, within `mask`."""
        n = len(self.ids)
        removed = 0 if mask is None else int(n - mask.sum())
        want = min(n, k + removed)
        scores, idx = self.index.search(np.ascontiguousarray(q_emb[None, :], dtype=np.float32), want)
        scores, idx = scores[0], idx[0]
        keep = idx >= 0
        if mask is not None:
            keep &= mask[np.clip(idx, 0, n - 1)]
        scores, idx = scores[keep].astype(np.float64), idx[keep]
        order = np.lexsort((self.ids[idx], -scores))[:k]
        return list(zip(self.ids[idx[order]].tolist(), scores[order].tolist()))


def run(index: DenseIndex, query_ids, q_emb: np.ndarray, masks=None, k: int = 100) -> pd.DataFrame:
    """Long run table (query_id, rank, doc_id, score), same shape as retrieve_bm25.run.

    masks: {query id: candidate mask}, a function query id -> mask, or None for no date filter.
    """
    rows = []
    for qid, v in zip(query_ids, q_emb):
        m = masks(qid) if callable(masks) else (masks[qid] if masks is not None else None)
        for r, (d, s) in enumerate(index.search(v, k, m), start=1):
            rows.append((qid, r, d, s))
    return pd.DataFrame(rows, columns=["query_id", "rank", "doc_id", "score"])
