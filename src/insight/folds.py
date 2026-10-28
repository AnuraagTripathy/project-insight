"""Project-level cross-validation folds.

A project is one source paper (`queries.paper_id`) with all of its questions (one core plus its
subfield questions). Questions of the same project share credited papers, so a project is never
split: every question of a project lands in the same fold.

`assign_folds` shuffles the project ids with a fixed seed and deals them round-robin into
`n_folds` folds (sizes differ by at most one). For test fold f, `split` returns the test
questions (fold f) and divides the remaining projects into train and an inner validation set
(`val_frac` of the training projects, also chosen with the seed) for tuning and early stopping.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FoldConfig:
    n_folds: int = 5
    val_frac: float = 0.2
    seed: int = 0

    @classmethod
    def from_yaml(cls, path):
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))


@dataclass
class Split:
    fold: int
    train: list[str]  # query ids
    val: list[str]
    test: list[str]


def assign_folds(project_ids, n_folds: int = 5, seed: int = 0) -> pd.Series:
    """Fold number (0 to n_folds - 1) for each unique project id, as a Series indexed by project."""
    projects = np.array(sorted(set(project_ids)))
    order = np.random.default_rng(seed).permutation(len(projects))
    folds = np.empty(len(projects), dtype=int)
    folds[order] = np.arange(len(projects)) % n_folds
    return pd.Series(folds, index=pd.Index(projects, name="paper_id"), name="fold")


def query_folds(queries: pd.DataFrame, cfg: FoldConfig = FoldConfig()) -> pd.Series:
    """Fold number for each question (indexed by query id), inherited from its project."""
    f = assign_folds(queries.paper_id, cfg.n_folds, cfg.seed)
    return pd.Series(f.loc[queries.paper_id].to_numpy(), index=pd.Index(queries.id, name="query_id"),
                     name="fold")


def split(queries: pd.DataFrame, fold: int, cfg: FoldConfig = FoldConfig()) -> Split:
    """Train, inner validation and test question ids for one test fold."""
    pf = assign_folds(queries.paper_id, cfg.n_folds, cfg.seed)
    test_p = set(pf.index[pf == fold])
    rest = np.array(sorted(pf.index[pf != fold]))
    # A different seed per fold, so each fold's validation projects are drawn independently.
    rng = np.random.default_rng([cfg.seed, fold])
    n_val = int(round(cfg.val_frac * len(rest)))
    val_p = set(rng.choice(rest, size=n_val, replace=False))
    in_test = queries.paper_id.isin(test_p)
    in_val = queries.paper_id.isin(val_p)
    return Split(
        fold=fold,
        train=queries.id[~in_test & ~in_val].tolist(),
        val=queries.id[in_val].tolist(),
        test=queries.id[in_test].tolist(),
    )


def fold_table(queries: pd.DataFrame, cfg: FoldConfig = FoldConfig()) -> pd.DataFrame:
    """Long table (query_id, paper_id, kind, fold, role) with role in train / val / test.

    One row per (question, test fold): every question appears n_folds times, once as test.
    This is the table saved to results/folds.csv so every condition uses the same splits.
    """
    parts = []
    for f in range(cfg.n_folds):
        s = split(queries, f, cfg)
        role = {**{q: "train" for q in s.train}, **{q: "val" for q in s.val}, **{q: "test" for q in s.test}}
        parts.append(pd.DataFrame({
            "query_id": queries.id, "paper_id": queries.paper_id,
            "type": queries.type, "fold": f, "role": queries.id.map(role),
        }))
    return pd.concat(parts, ignore_index=True)
