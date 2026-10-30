"""Condition A: BM25 top 100 for every question, with metrics.

Writes results/run_A.parquet (query_id, rank, doc_id, score) and results/metrics_A.csv (one row
per question), then prints core and subfield means and the reproduction gate.

Usage: uv run python scripts/run_bm25.py [--config configs/A.yaml] [--limit N]
"""

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from insight import data, metrics, retrieve_bm25 as R

# Published core and subfield values (ScholarCatalyst, arXiv:2610.02202, Table 3, BM25 rows).
PAPER = {
    "core": {"ndcg@20": 0.16, "recall@5": 0.12, "recall@20": 0.23, "recall@100": 0.39},
    "subfield": {"ndcg@20": 0.21, "recall@5": 0.12, "recall@20": 0.26, "recall@100": 0.45},
}
GATE_TOL = 0.03


def summarize(per_q: pd.DataFrame, kinds: pd.Series) -> pd.DataFrame:
    cols = ["ndcg@20", "recall@5", "recall@20", "recall@100", "mrr"]
    out = per_q[cols].groupby(kinds.loc[per_q.index]).mean()
    out.insert(0, "n", per_q.groupby(kinds.loc[per_q.index]).size())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "A.yaml"))
    ap.add_argument("--out-dir", default=str(data.ROOT / "results"))
    ap.add_argument("--tag", default="A", help="suffix for the output files")
    ap.add_argument("--limit", type=int, default=None, help="only the first N corpus papers (smoke test)")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = R.BM25Config.from_yaml(args.config)
    dcfg = data.DataConfig.from_yaml(data.ROOT / raw.get("data_config", "configs/data.yaml"))
    ds = data.load(dcfg)
    corpus = ds.corpus if args.limit is None else ds.corpus.iloc[: args.limit]

    t0 = time.time()
    index = R.BM25Index(R.doc_texts(corpus, cfg.fields), corpus.id, cfg, show_progress=True)
    print(f"indexed {len(corpus)} papers in {time.time() - t0:.0f}s")

    table = ds.table
    pos_of = pd.Series(np.arange(len(corpus)), index=corpus.id)
    pub = corpus.pub.to_numpy()

    def mask(qid):
        if not dcfg.date_filter:
            return None
        row = table.loc[qid]
        labeled = pos_of.reindex(list(row.positives) + list(row.negatives)).dropna().astype(int)
        return data.candidate_mask(pub, row.src_pub, labeled.to_numpy(),
                                   exempt_labeled=dcfg.date_filter_exempt_labeled,
                                   keep_undated=dcfg.keep_undated)

    t0 = time.time()
    run_df = R.run(index, ds.queries, mask, cfg.top_k)
    print(f"searched {ds.queries.id.nunique()} questions in {time.time() - t0:.0f}s")

    per_q = metrics.evaluate(R.as_ranked_lists(run_df), table.positives.to_dict())
    per_q.insert(0, "kind", table.kind.loc[per_q.index])
    out = data.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(out / f"run_{args.tag}.parquet", index=False)
    per_q.to_csv(out / f"metrics_{args.tag}.csv")
    print(f"wrote {out}/run_{args.tag}.parquet ({len(run_df)} rows) and metrics_{args.tag}.csv")

    summary = summarize(per_q.drop(columns="kind"), table.kind)
    print(summary.round(4).to_string())
    for kind in ["core", "subfield"]:
        ours, paper = summary.loc[kind, "recall@20"], PAPER[kind]["recall@20"]
        status = "PASS" if abs(ours - paper) <= GATE_TOL else "FAIL"
        print(f"{kind} Recall@20: ours {ours:.4f} vs paper {paper:.2f} (diff {ours - paper:+.4f}) {status}")


if __name__ == "__main__":
    main()
