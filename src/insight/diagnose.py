"""RQ1 diagnosis: where does similarity search lose the credited papers?

Everything here works on a run table (query_id, rank, doc_id, score), the format written by
scripts/run_bm25.py and scripts/encode_corpus.py, so the same code serves condition A now and
condition B once its shortlist exists.

Three steps:

1. `credited_ranks`: one row per credited (question, paper) pair with the rank the retriever gave
   that paper, NaN if it is not in the saved top list (beyond rank 100).
2. `rank_bin`: 1 to 20 (found), 21 to 100 (on the shortlist but too low, a reranker can fix it),
   beyond 100 (only better retrieval can fix it).
3. `add_fields`: in-field vs cross-field. A credited paper is cross-field when its field differs
   from the source paper's field (`queries.paper_domain`). How a corpus category becomes a field
   is set by `FieldConfig` (decision log, Nov 4, NEEDS REVIEW).

Recall by group (`recall_by_group`) is per question: among the credited papers of one question
that fall in a group, the share ranked in the top k; then averaged over questions that have at
least one credited paper in that group.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# An arXiv category code: an archive (cs, stat, q-bio, cond-mat, ...) with an optional subject.
ARXIV_CODE = re.compile(r"^[a-z][a-z\-]*(\.[A-Za-z\-]+)?$")

BIN_LABELS = ("1 to 20", "21 to 100", "beyond 100")


@dataclass
class FieldConfig:
    # "primary": the full arXiv primary category (cs.LG), the granularity of the paper's
    # Tables 8 and 9. "archive": the part before the dot (cs, stat, math, ...).
    level: str = "primary"
    # Papers whose category is not an arXiv code (most Semantic Scholar and OpenAlex papers, and
    # blanks): "exclude" leaves them out of the in/cross split (reported as "unknown"), "cross"
    # counts them as cross-field.
    unknown_policy: str = "exclude"
    # Used only at level "archive": non-arXiv field names mapped to an archive.
    name_map: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("fields", {}))


def paper_field(category, cfg: FieldConfig | None = None):
    """Coarse field of one corpus category string, or None when it cannot be mapped."""
    cfg = cfg or FieldConfig()
    if cfg.level not in ("primary", "archive"):
        raise ValueError(f"unknown level {cfg.level!r}")
    cat = (category or "").strip()
    if ARXIV_CODE.match(cat):
        return cat if cfg.level == "primary" else cat.split(".")[0]
    if cfg.level == "archive":
        return cfg.name_map.get(cat)
    return None


def credited_ranks(run: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """One row per credited (question, paper) pair: query_id, project, kind, doc_id, rank.

    run: query_id, rank (1 = best), doc_id. table: indexed by query id, with columns paper_id,
    kind and positives (as in `insight.data.Dataset.table`). rank is NaN when the paper is not
    in the run (beyond the saved top list).
    """
    pairs = (table[["paper_id", "kind", "positives"]].explode("positives")
             .rename(columns={"paper_id": "project", "positives": "doc_id"})
             .rename_axis("query_id").reset_index())
    pairs = pairs.dropna(subset=["doc_id"])
    ranks = run[["query_id", "doc_id", "rank"]].drop_duplicates(["query_id", "doc_id"])
    out = pairs.merge(ranks, on=["query_id", "doc_id"], how="left")
    out["rank"] = out["rank"].astype(float)
    return out


def rank_bin(rank: pd.Series, edges=(20, 100), labels=BIN_LABELS) -> pd.Series:
    """Bin ranks into (0, 20], (20, 100], beyond. NaN (not retrieved) goes to the last bin."""
    r = pd.Series(rank, dtype=float).fillna(np.inf)
    return pd.cut(r, [0, *edges, np.inf], labels=list(labels))


def add_fields(pairs: pd.DataFrame, corpus: pd.DataFrame, queries: pd.DataFrame,
               cfg: FieldConfig | None = None) -> pd.DataFrame:
    """Add doc_field, src_field and split ("in-field", "cross-field" or "unknown")."""
    cfg = cfg or FieldConfig()
    cat = corpus.set_index("id").primary_category
    src = queries.drop_duplicates("paper_id").set_index("paper_id").paper_domain
    out = pairs.copy()
    out["doc_field"] = [paper_field(c, cfg) for c in cat.reindex(out.doc_id)]
    out["src_field"] = [paper_field(c, cfg) for c in src.reindex(out.project)]
    known = out.doc_field.notna() & out.src_field.notna()
    split = np.where(out.doc_field == out.src_field, "in-field", "cross-field")
    if cfg.unknown_policy == "exclude":
        split = np.where(known, split, "unknown")
    elif cfg.unknown_policy != "cross":
        raise ValueError(f"unknown unknown_policy {cfg.unknown_policy!r}")
    out["split"] = split
    return out


def recall_by_group(pairs: pd.DataFrame, group: str, ks=(20, 100)) -> pd.DataFrame:
    """Mean per-question Recall@k within each (kind, group).

    For one question and group g: (credited papers in g ranked in the top k) / (credited papers
    in g). Questions with no credited paper in g do not count for g. Column n_questions gives
    how many questions each mean is over, n_pairs how many credited pairs.
    """
    p = pairs.copy()
    for k in ks:
        p[f"recall@{k}"] = (p["rank"] <= k).astype(float)
    per_q = p.groupby(["kind", group, "query_id"], observed=True)[[f"recall@{k}" for k in ks]].mean()
    out = per_q.groupby(["kind", group], observed=True).mean()
    out.insert(0, "n_questions", per_q.groupby(["kind", group], observed=True).size())
    out.insert(1, "n_pairs", p.groupby(["kind", group], observed=True).size())
    return out


def bin_shares(pairs: pd.DataFrame, by=("kind",)) -> pd.DataFrame:
    """Share of credited pairs in each rank bin, per group (rows sum to 1), plus a count n."""
    bins = rank_bin(pairs["rank"])
    keys = [pairs[b] for b in by]
    shares = pd.crosstab(keys, bins, normalize="index")
    shares["n"] = pd.crosstab(keys, bins).sum(axis=1)
    return shares
