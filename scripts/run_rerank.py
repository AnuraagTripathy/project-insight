"""Condition C: rerank a retriever's shortlist with Qwen3-Reranker.

Steps:
1. Read the shortlist (default results/shortlist_B.parquet) and keep each question's top `depth`.
2. Score every (question, paper) pair with s = z_yes - z_no and re-sort; write
   results/run_<tag>.parquet (query_id, rank, doc_id, score, first_rank).
3. Score every labeled (question, credited or rejected paper) pair, shortlisted or not, and write
   results/pairs_<tag>.parquet (query_id, doc_id, label, score) for pairwise accuracy.
4. Write results/metrics_<tag>.csv (one row per question: recall@5/20/100, ndcg@20, mrr,
   pairwise accuracy) and print core and subfield means next to the shortlist's own metrics.

Full run (H100): uv run python scripts/run_rerank.py --config configs/C.yaml
CPU smoke test:  uv run python scripts/run_rerank.py --shortlist results/run_A.parquet \
                     --dtype float32 --query-limit 2 --depth 10 --tag smoke --out-dir outputs/smoke
"""

import argparse
import time
from dataclasses import replace

import pandas as pd
import yaml

from insight import data, metrics, retrieve_bm25, rerank as R

COLS = ["ndcg@20", "recall@5", "recall@20", "recall@100", "mrr", "pairwise_acc"]


def per_question(run_df: pd.DataFrame, pairs: pd.DataFrame, table: pd.DataFrame, qids) -> pd.DataFrame:
    """Ranking metrics from run_df plus pairwise accuracy from the labeled-pair scores."""
    per_q = metrics.evaluate(retrieve_bm25.as_ranked_lists(run_df), table.positives.loc[qids].to_dict())
    acc = {}
    for qid, g in pairs.groupby("query_id"):
        s = dict(zip(g.doc_id, g.score))
        acc[qid] = metrics.pairwise_accuracy(s, g.doc_id[g.label == 1], g.doc_id[g.label == 0])
    per_q["pairwise_acc"] = pd.Series(acc).reindex(per_q.index)
    per_q.insert(0, "kind", table.kind.loc[per_q.index])
    return per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "C.yaml"))
    ap.add_argument("--shortlist", default=None, help="override the config's shortlist")
    ap.add_argument("--out-dir", default=str(data.ROOT / "results"))
    ap.add_argument("--tag", default="C", help="suffix for the output files")
    ap.add_argument("--model", default=None, help="override the config's model")
    ap.add_argument("--dtype", default=None, help="override the config's dtype")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None, help="override the config's depth")
    ap.add_argument("--query-limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--no-pairs", action="store_true", help="skip scoring the labeled pairs")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = R.RerankConfig.from_yaml(args.config)
    cfg = replace(cfg, **{k: v for k, v in {"model": args.model, "dtype": args.dtype,
                                             "batch_size": args.batch_size, "depth": args.depth}.items()
                          if v is not None})
    dcfg = data.DataConfig.from_yaml(data.ROOT / raw.get("data_config", "configs/data.yaml"))
    ds = data.load(dcfg)
    table = ds.table
    shortlist = pd.read_parquet(data.ROOT / (args.shortlist or raw["shortlist"]))
    qids = list(dict.fromkeys(shortlist.query_id))
    if args.query_limit is not None:
        qids = qids[: args.query_limit]
        shortlist = shortlist[shortlist.query_id.isin(qids)]
    full = args.query_limit is None and len(qids) == len(table)

    text_of = dict(zip(ds.corpus.id, R.doc_texts(ds.corpus, cfg.fields, cfg.field_sep)))
    question_of = table.question.to_dict()

    rr = R.Reranker(cfg)
    print(f"model {cfg.model} on {rr.device}, "
          f"dtype {cfg.dtype}, depth {cfg.depth}, {len(qids)} questions")

    t0 = time.time()
    run_df = R.rerank_run(shortlist, question_of, text_of, rr.score, cfg.depth, cfg.top_k,
                          show_progress=True)
    n_pairs = len(run_df)
    print(f"scored {n_pairs} shortlist pairs in {time.time() - t0:.0f}s")

    pairs = R.labeled_pairs(table, qids)
    if args.no_pairs:
        pairs = pairs.iloc[:0].assign(score=[])
    else:
        t0 = time.time()
        pairs["score"] = rr.score(pairs.query_id.map(question_of).tolist(),
                                  pairs.doc_id.map(text_of).tolist(), show_progress=True)
        print(f"scored {len(pairs)} labeled pairs in {time.time() - t0:.0f}s")

    out = data.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(out / f"run_{args.tag}.parquet", index=False)
    pairs.to_parquet(out / f"pairs_{args.tag}.parquet", index=False)
    per_q = per_question(run_df, pairs, table, qids)
    per_q.to_csv(out / f"metrics_{args.tag}.csv")
    print(f"wrote {out}/run_{args.tag}.parquet, pairs_{args.tag}.parquet, metrics_{args.tag}.csv")

    # The shortlist's own metrics on the same questions and depth, for the B to C comparison.
    base = metrics.evaluate(retrieve_bm25.as_ranked_lists(shortlist[shortlist["rank"] <= cfg.depth]),
                            table.positives.loc[qids].to_dict())
    base.insert(0, "kind", table.kind.loc[base.index])
    for name, df in [("shortlist", base), (args.tag, per_q)]:
        cols = [c for c in COLS if c in df]
        summary = df.groupby("kind")[cols].mean()
        summary.insert(0, "n", df.groupby("kind").size())
        print(f"\n{name}:\n{summary.round(4).to_string()}")
    if not full:
        print("\npartial run (--query-limit): results are a smoke test only")


if __name__ == "__main__":
    main()
