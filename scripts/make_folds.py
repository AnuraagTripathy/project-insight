"""Write results/folds.csv: the project-level folds every condition uses.

Usage: uv run python scripts/make_folds.py [--config configs/folds.yaml]
"""

import argparse

from insight import data, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(folds.ROOT / "configs" / "folds.yaml"))
    ap.add_argument("--out", default=str(folds.ROOT / "results" / "folds.csv"))
    args = ap.parse_args()
    cfg = folds.FoldConfig.from_yaml(args.config)
    ds = data.load()
    t = folds.fold_table(ds.queries, cfg)
    folds.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"wrote {len(t)} rows to {args.out}")
    # Projects and questions per role, for each test fold.
    print(t.groupby(["fold", "role"]).agg(projects=("paper_id", "nunique"), questions=("query_id", "size"),
                                         core=("type", lambda s: (s == "core_query").sum())).unstack("role"))
    # Not leakage, but the input to the memorization check: credited (question, paper) pairs in the
    # test fold whose paper is also credited, or rejected, for some training question.
    lab = ds.table
    for f in range(cfg.n_folds):
        s = folds.split(ds.queries, f, cfg)
        tr_pos = {d for q in s.train for d in lab.loc[q, "positives"]}
        tr_neg = {d for q in s.train for d in lab.loc[q, "negatives"]}
        pairs = [d for q in s.test for d in lab.loc[q, "positives"]]
        n_pos = sum(d in tr_pos for d in pairs)
        n_neg = sum(d in tr_neg for d in pairs)
        print(f"fold {f}: {len(pairs)} test credited pairs; paper credited in train {n_pos} "
              f"({n_pos / len(pairs):.1%}), rejected in train {n_neg} ({n_neg / len(pairs):.1%})")


if __name__ == "__main__":
    main()
