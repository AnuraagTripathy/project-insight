"""Load ScholarCatalyst and apply the candidate date filter.

The four parquet files written by scripts/download_data.py are turned into three tables:

- corpus: one row per paper, with a parsed publication date `pub`.
- queries: one row per question, with the parsed source paper date `src_pub` and a short
  `kind` ("core" or "subfield").
- labels: one row per question with its credited ids (`positives`), author-rejected ids
  (`negatives`) and rationales, in the same shape for core and subfield questions.

Date filter: a question may only retrieve papers published no later than its source paper,
except that papers labeled for that question (credited or rejected) are always kept when
`exempt_labeled` is true (decision log, Oct 23).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# The corpus uses 1899-12-30 (the spreadsheet epoch) for one paper with no date.
PLACEHOLDER_DATES = {"1899-12-30"}


@dataclass
class DataConfig:
    data_dir: Path = DATA_DIR
    date_filter: bool = True
    date_filter_exempt_labeled: bool = True
    # Papers with no usable date: keep them as candidates (True) or drop them (False).
    keep_undated: bool = True

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "data_dir" in raw:
            p = Path(raw["data_dir"])
            raw["data_dir"] = p if p.is_absolute() else ROOT / p
        return cls(**raw)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse the mixed ISO 8601 date strings to day precision.

    Handles `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD`, `YYYY-MM` and `YYYY`. Partial dates map to the
    first day of the period. Placeholder and empty dates become NaT. The time of day is dropped
    so that "no later than" compares calendar days.
    """
    s = s.where(~s.isin(PLACEHOLDER_DATES) & (s.str.strip() != ""))
    return pd.to_datetime(s, format="ISO8601").dt.normalize()


def load_corpus(data_dir=DATA_DIR) -> pd.DataFrame:
    corpus = pd.read_parquet(Path(data_dir) / "corpus.parquet")
    corpus["pub"] = parse_dates(corpus.published)
    return corpus


def load_queries(data_dir=DATA_DIR) -> pd.DataFrame:
    queries = pd.read_parquet(Path(data_dir) / "queries.parquet")
    queries["src_pub"] = parse_dates(queries.paper_published)
    queries["kind"] = queries.type.str.removesuffix("_query")
    return queries


def load_labels(data_dir=DATA_DIR) -> pd.DataFrame:
    """One row per question: query_id, kind, positives, negatives, rationales.

    `rationales` is a dict {credited id: rationale} for core questions. Subfield questions have
    a single rationale describing all their credited papers together, stored under the key None.
    """
    d = Path(data_dir)
    core = pd.read_parquet(d / "core_query.parquet")
    sub = pd.read_parquet(d / "subfield_query.parquet")
    core_rows = pd.DataFrame({
        "query_id": core.query_id,
        "kind": "core",
        "positives": core.positive_docs.map(lambda xs: [x["id"] for x in xs]),
        "negatives": core.hard_negatives.map(list),
        "rationales": core.positive_docs.map(lambda xs: {x["id"]: x["author_rationale"] for x in xs}),
    })
    sub_rows = pd.DataFrame({
        "query_id": sub.query_id,
        "kind": "subfield",
        "positives": sub.positive_docs.map(list),
        "negatives": sub.hard_negatives.map(list),
        "rationales": sub.author_rationale.map(lambda r: {None: r}),
    })
    return pd.concat([core_rows, sub_rows], ignore_index=True)


@dataclass
class Dataset:
    corpus: pd.DataFrame
    queries: pd.DataFrame
    labels: pd.DataFrame
    config: DataConfig

    @property
    def table(self) -> pd.DataFrame:
        """Queries joined with their labels, indexed by query id."""
        labels = self.labels.drop(columns="kind")
        return self.queries.merge(labels, left_on="id", right_on="query_id").set_index("id")


def load(config: DataConfig | None = None) -> Dataset:
    config = config or DataConfig()
    return Dataset(
        corpus=load_corpus(config.data_dir),
        queries=load_queries(config.data_dir),
        labels=load_labels(config.data_dir),
        config=config,
    )


def candidate_mask(pub, src_pub, labeled_idx=(), *, exempt_labeled=True, keep_undated=True) -> np.ndarray:
    """Boolean mask over the corpus: which papers one question may retrieve.

    pub: publication dates of all corpus papers (datetime64 array or Series, NaT if unknown).
    src_pub: the question's source paper date.
    labeled_idx: corpus row positions of the papers labeled (credited or rejected) for this
        question. They are always allowed when exempt_labeled is true.
    """
    pub = pd.to_datetime(pd.Series(pub)).to_numpy()
    undated = np.isnat(pub)
    with np.errstate(invalid="ignore"):
        mask = pub <= np.datetime64(pd.Timestamp(src_pub).normalize())
    mask[undated] = keep_undated
    if exempt_labeled:
        mask[np.asarray(labeled_idx, dtype=int)] = True
    return mask


def candidate_masks(ds: Dataset) -> dict[str, np.ndarray]:
    """candidate_mask for every question, keyed by query id (894 x 190,896 booleans)."""
    cfg = ds.config
    pos_of = pd.Series(np.arange(len(ds.corpus)), index=ds.corpus.id)
    pub = ds.corpus.pub.to_numpy()
    masks = {}
    for qid, row in ds.table.iterrows():
        if not cfg.date_filter:
            masks[qid] = np.ones(len(pub), dtype=bool)
            continue
        labeled = pos_of[list(row.positives) + list(row.negatives)].to_numpy()
        masks[qid] = candidate_mask(pub, row.src_pub, labeled,
                                    exempt_labeled=cfg.date_filter_exempt_labeled,
                                    keep_undated=cfg.keep_undated)
    return masks
