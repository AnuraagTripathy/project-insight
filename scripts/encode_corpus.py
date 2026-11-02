"""Condition B: encode the corpus and questions with Qwen3-Embedding, then FAISS top 100.

Steps:
1. Embed every paper (title + abstract) and every question (with the paper's instruction).
   Vectors go to outputs/emb_<tag>_{corpus,queries}.npy (gitignored, about 2 GB for the 4B model).
2. Exact inner-product search with the date filter of configs/data.yaml.
3. Write results/shortlist_<tag>.parquet (query_id, rank, doc_id, score), the shortlist that
   conditions C, D and E rescore, and results/metrics_<tag>.csv (one row per question), then
   print core and subfield means and the reproduction gate (full corpus only).

Full run (H100): uv run python scripts/encode_corpus.py --config configs/B.yaml
CPU smoke test:  uv run python scripts/encode_corpus.py --model Qwen/Qwen3-Embedding-0.6B \
                     --dtype float32 --limit 1000 --query-limit 20 --tag smoke --out-dir outputs/smoke
Re-run only the search from saved vectors: add --reuse.
"""

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml

from insight import data, metrics, retrieve_bm25, retrieve_dense as R

# Published Qwen3-Emb-4B values (ScholarCatalyst, arXiv:2610.02202, Table 3 / Table 19).
PAPER = {
    "core": {"ndcg@20": 0.25, "recall@5": 0.18, "recall@20": 0.39, "recall@100": 0.59},
    "subfield": {"ndcg@20": 0.35, "recall@5": 0.21, "recall@20": 0.44, "recall@100": 0.69},
}
GATE_TOL = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "B.yaml"))
    ap.add_argument("--out-dir", default=str(data.ROOT / "results"))
    ap.add_argument("--emb-dir", default=str(data.ROOT / "outputs"))
    ap.add_argument("--tag", default="B", help="suffix for the output files")
    ap.add_argument("--model", default=None, help="override the config's model")
    ap.add_argument("--dtype", default=None, help="override the config's dtype")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N corpus papers (smoke test)")
    ap.add_argument("--query-limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--reuse", action="store_true", help="load saved vectors instead of encoding")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    cfg = R.DenseConfig.from_yaml(args.config)
    cfg = replace(cfg, **{k: v for k, v in {"model": args.model, "dtype": args.dtype,
                                             "batch_size": args.batch_size}.items() if v is not None})
    dcfg = data.DataConfig.from_yaml(data.ROOT / raw.get("data_config", "configs/data.yaml"))
    ds = data.load(dcfg)
    corpus = ds.corpus if args.limit is None else ds.corpus.iloc[: args.limit]
    queries = ds.queries if args.query_limit is None else ds.queries.iloc[: args.query_limit]
    full = args.limit is None and args.query_limit is None

    emb_dir = data.Path(args.emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    c_path, q_path = emb_dir / f"emb_{args.tag}_corpus.npy", emb_dir / f"emb_{args.tag}_queries.npy"
    if args.reuse:
        doc_emb, q_emb = np.load(c_path), np.load(q_path)
        assert len(doc_emb) == len(corpus) and len(q_emb) == len(queries), "saved vectors do not match"
    else:
        enc = R.Encoder(cfg)
        print(f"model {cfg.model} on {enc.device}, dtype {cfg.dtype}")
        t0 = time.time()
        q_emb = enc.encode_queries(queries.question.tolist(), show_progress=True)
        print(f"encoded {len(queries)} questions in {time.time() - t0:.0f}s")
        t0 = time.time()
        doc_emb = enc.encode_docs(R.doc_texts(corpus, cfg.fields, cfg.field_sep), show_progress=True)
        print(f"encoded {len(corpus)} papers in {time.time() - t0:.0f}s")
        np.save(q_path, q_emb)
        np.save(c_path, doc_emb)
        (emb_dir / f"emb_{args.tag}_meta.json").write_text(json.dumps({
            "config": vars(cfg), "n_corpus": len(corpus), "n_queries": len(queries),
            "corpus_ids_first_last": [corpus.id.iloc[0], corpus.id.iloc[-1]]}, indent=1))

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
    index = R.DenseIndex(doc_emb, corpus.id)
    run_df = R.run(index, queries.id, q_emb, mask, cfg.top_k)
    print(f"searched {len(queries)} questions in {time.time() - t0:.0f}s")

    per_q = metrics.evaluate(retrieve_bm25.as_ranked_lists(run_df),
                             table.positives.loc[queries.id].to_dict())
    per_q.insert(0, "kind", table.kind.loc[per_q.index])
    out = data.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(out / f"shortlist_{args.tag}.parquet", index=False)
    per_q.to_csv(out / f"metrics_{args.tag}.csv")
    print(f"wrote {out}/shortlist_{args.tag}.parquet ({len(run_df)} rows) and metrics_{args.tag}.csv")

    cols = ["ndcg@20", "recall@5", "recall@20", "recall@100", "mrr"]
    summary = per_q.groupby("kind")[cols].mean()
    summary.insert(0, "n", per_q.groupby("kind").size())
    print(summary.round(4).to_string())
    if not full:
        print("partial run (--limit or --query-limit): no reproduction gate")
        return
    for kind in ["core", "subfield"]:
        ours, paper = summary.loc[kind, "recall@20"], PAPER[kind]["recall@20"]
        status = "PASS" if abs(ours - paper) <= GATE_TOL else "FAIL"
        print(f"{kind} Recall@20: ours {ours:.4f} vs paper {paper:.2f} (diff {ours - paper:+.4f}) {status}")


if __name__ == "__main__":
    main()
