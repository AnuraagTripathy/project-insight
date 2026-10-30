"""BM25 retrieval over the full corpus (condition A).

Uses the `bm25s` library, the BM25 implementation linked from the ScholarCatalyst paper's
retriever table. Each paper is indexed as "title + abstract". The paper does not state its
tokenizer, stemmer or k1 / b, so the defaults below are bm25s's own (Lucene-style scoring,
k1 = 1.5, b = 0.75, English stopwords, no stemming, the default of bm25s.tokenize); all are
configurable in configs/A.yaml, including a Snowball stemmer (decision log, Oct 30).

The date filter is applied at query time: BM25 scores every paper, then papers outside the
question's candidate mask (insight.data.candidate_mask) are dropped before taking the top k.
Papers with score 0 share no term with the question and are never returned. Equal scores are
ordered by paper id, the same rule as insight.metrics.rank.
"""

from dataclasses import dataclass, field
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
import yaml


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    stopwords: str | None = "en"
    stemmer: str | None = None  # a PyStemmer language such as "english", or None for no stemming
    fields: list[str] = field(default_factory=lambda: ["title", "text"])
    top_k: int = 100

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("bm25", {}))


def doc_texts(corpus: pd.DataFrame, fields=("title", "text")) -> list[str]:
    """One string per paper: the chosen fields joined by a space (missing values become "")."""
    return corpus[list(fields)].fillna("").astype(str).agg(" ".join, axis=1).tolist()


class BM25Index:
    def __init__(self, texts: list[str], ids, cfg: BM25Config = BM25Config(), show_progress=False):
        self.cfg = cfg
        self.ids = np.asarray(ids)
        self.stemmer = Stemmer.Stemmer(cfg.stemmer) if cfg.stemmer else None
        tokens = self._tokenize(texts, show_progress)
        self.model = bm25s.BM25(k1=cfg.k1, b=cfg.b, method=cfg.method)
        self.model.index(tokens, show_progress=show_progress)

    def _tokenize(self, texts, show_progress=False):
        return bm25s.tokenize(texts, stopwords=self.cfg.stopwords or [], stemmer=self.stemmer,
                              return_ids=False, show_progress=show_progress)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every indexed paper for one question (0 if no shared term)."""
        tokens = [t for t in self._tokenize([query])[0] if t in self.model.vocab_dict]
        if not tokens:
            return np.zeros(len(self.ids), dtype=np.float32)
        return self.model.get_scores(tokens)

    def search(self, query: str, k: int | None = None, mask: np.ndarray | None = None):
        """Top k (paper id, score) pairs, best first, restricted to `mask` if given."""
        k = k or self.cfg.top_k
        s = self.scores(query).astype(np.float64)
        keep = s > 0
        if mask is not None:
            keep &= mask
        idx = np.flatnonzero(keep)
        if len(idx) > k:
            # Keep everything tied with the k-th score so the id tie-break is exact.
            kth = np.partition(s[idx], len(idx) - k)[len(idx) - k]
            idx = idx[s[idx] >= kth]
        order = np.lexsort((self.ids[idx], -s[idx]))[:k]
        idx = idx[order]
        return list(zip(self.ids[idx].tolist(), s[idx].tolist()))


def run(index: BM25Index, queries: pd.DataFrame, masks=None, k: int | None = None) -> pd.DataFrame:
    """Long run table (query_id, rank, doc_id, score) for every question.

    queries: needs columns `id` and `question`. masks: {query id: candidate mask} or a function
    query id -> mask, or None for no date filter.
    """
    rows = []
    for qid, text in zip(queries.id, queries.question):
        m = masks(qid) if callable(masks) else (masks[qid] if masks is not None else None)
        for r, (d, s) in enumerate(index.search(text, k, m), start=1):
            rows.append((qid, r, d, s))
    return pd.DataFrame(rows, columns=["query_id", "rank", "doc_id", "score"])


def as_ranked_lists(run_df: pd.DataFrame) -> dict[str, list[str]]:
    """{query id: ranked paper ids} from a long run table, the input to insight.metrics.evaluate."""
    run_df = run_df.sort_values(["query_id", "rank"])
    return run_df.groupby("query_id").doc_id.agg(list).to_dict()
