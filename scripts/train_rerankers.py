"""Conditions D and E: train one LoRA reranker per fold and score that fold's held-out questions.

For each test fold f (folds from results/folds.csv, the same splits every condition uses):
1. Build training pairs from fold f's training questions and validation labels from its inner
   validation questions (insight.train_reranker.fold_pairs never reads test rows).
2. Load a fresh Qwen3-Reranker, add LoRA, train with the pairwise loss and early stopping on
   validation pairwise accuracy; keep the best adapters (outputs/<tag>/fold<f>/adapter).
3. With those adapters, rescore fold f's test questions: their shortlist top `depth` (from
   condition B, the same shortlist C rescores) and all their labeled pairs.
Every question is a test question in exactly one fold, so concatenating the 5 folds gives one
held-out score per question. When all folds are done the script writes
results/run_<tag>.parquet, pairs_<tag>.parquet, metrics_<tag>.csv and train_history_<tag>.csv,
and prints core and subfield means.

Full run (H100): uv run python scripts/train_rerankers.py --config configs/D.yaml
Resume after an interruption: add --resume (finished folds are reused; a fold whose adapter
exists but whose held-out scores do not, for example because the shortlist was missing, is
rescored from the saved adapter without retraining).
CPU smoke test:  uv run python scripts/train_rerankers.py --folds 0 --dtype float32 --fields title \
    --max-train-pairs 20 --max-val-queries 2 --test-limit 2 --epochs 1 --depth 10 \
    --shortlist results/run_A.parquet --tag smoke --work-dir outputs/smoke_D --out-dir outputs/smoke_D
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import yaml

from insight import data, rerank as R, train_reranker as T


def load_reranker(cfg: R.RerankConfig, adapter_dir: Path | None = None):
    rr = R.Reranker(cfg)
    if adapter_dir is not None:
        from peft import PeftModel

        rr.model = PeftModel.from_pretrained(rr.model, str(adapter_dir)).to(rr.device).eval()
    return rr


def summarize(per_q: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in R.METRIC_COLS if c in per_q]
    s = per_q.groupby("kind")[cols].mean()
    s.insert(0, "n", per_q.groupby("kind").size())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "D.yaml"))
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="test folds to run (default all)")
    ap.add_argument("--tag", default=None, help="output suffix (default: the config's tag)")
    ap.add_argument("--work-dir", default=None, help="per-fold outputs (default: the config's work_dir)")
    ap.add_argument("--out-dir", default=str(data.ROOT / "results"))
    ap.add_argument("--shortlist", default=None, help="override the config's shortlist")
    ap.add_argument("--resume", action="store_true", help="reuse finished folds and saved adapters")
    # Overrides, mostly for the CPU smoke test
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--fields", nargs="*", default=None, help="paper fields shown to the reranker")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-pairs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-train-pairs", type=int, default=None, help="seeded subset of training pairs (smoke)")
    ap.add_argument("--max-val-queries", type=int, default=None, help="first N validation questions (smoke)")
    ap.add_argument("--test-limit", type=int, default=None, help="first N test questions per fold (smoke)")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    rcfg = R.RerankConfig.from_yaml(args.config)
    rcfg = replace(rcfg, **{k: v for k, v in {"model": args.model, "dtype": args.dtype, "fields": args.fields,
                                               "depth": args.depth}.items() if v is not None})
    tcfg = T.TrainConfig.from_yaml(args.config)
    tcfg = replace(tcfg, **{k: v for k, v in {"epochs": args.epochs, "batch_pairs": args.batch_pairs,
                                               "grad_accum": args.grad_accum}.items() if v is not None})
    if args.gradient_checkpointing:
        tcfg = replace(tcfg, gradient_checkpointing=True)
    tag = args.tag or raw.get("tag", "D")
    work = data.ROOT / (args.work_dir or raw.get("work_dir", f"outputs/{tag}"))
    smoke = any(v is not None for v in [args.max_train_pairs, args.max_val_queries, args.test_limit])

    ds = data.load(data.DataConfig.from_yaml(data.ROOT / raw.get("data_config", "configs/data.yaml")))
    table = ds.table
    text_of = dict(zip(ds.corpus.id, R.doc_texts(ds.corpus, rcfg.fields, rcfg.field_sep)))
    question_of = table.question.to_dict()
    splits = T.splits_from_csv(data.ROOT / raw.get("folds", "results/folds.csv"))
    run_folds = args.folds if args.folds is not None else [s.fold for s in splits]

    sl_path = data.ROOT / (args.shortlist or raw["shortlist"])
    shortlist = pd.read_parquet(sl_path) if sl_path.exists() else None
    if shortlist is None:
        print(f"WARNING: {sl_path} not found: training and labeled-pair scores only, no reranked run. "
              f"Rerun with --resume once it exists to rescore from the saved adapters.")

    for sp in splits:
        if sp.fold not in run_folds:
            continue
        fdir = work / f"fold{sp.fold}"
        adapter = fdir / "adapter"
        finished = (fdir / "pairs.parquet").exists() and ((fdir / "run.parquet").exists() or shortlist is None)
        if args.resume and finished:
            print(f"fold {sp.fold}: finished, reusing {fdir}")
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if args.resume and (adapter / "adapter_config.json").exists():
            print(f"fold {sp.fold}: rescoring held-out questions from {adapter}")
            rr = load_reranker(rcfg, adapter)
            test_q = list(sp.test)[:args.test_limit] if args.test_limit is not None else list(sp.test)
            run_df, pairs = T.score_test_fold(rr, table, test_q, question_of, text_of, shortlist,
                                              rcfg.depth, rcfg.top_k, tcfg.eval_batch_size)
            run_df, pairs = run_df.assign(fold=sp.fold), pairs.assign(fold=sp.fold)
        else:
            rr = load_reranker(rcfg)
            print(f"fold {sp.fold}: model {rcfg.model} on {rr.device}, dtype {rcfg.dtype}")
            res = T.run_fold(rr, table, sp, question_of, text_of, tcfg, shortlist, rcfg.depth, rcfg.top_k,
                             args.max_train_pairs, args.max_val_queries, args.test_limit)
            rr.model.save_pretrained(str(adapter))
            res["train"]["history"].to_csv(fdir / "history.csv", index=False)
            info = {k: res["train"][k] for k in ["best_step", "best_val_pairwise_acc", "steps"]}
            info.update(n_train_pairs=res["n_train_pairs"], n_val_questions=res["n_val_questions"],
                        train_config=asdict(tcfg), rerank_config=asdict(rcfg), smoke=smoke)
            (fdir / "summary.json").write_text(json.dumps(info, indent=2))
            run_df, pairs = res["run"], res["pairs"]
        if shortlist is not None:
            run_df.to_parquet(fdir / "run.parquet", index=False)
        pairs.to_parquet(fdir / "pairs.parquet", index=False)
        print(f"fold {sp.fold}: done in {time.time() - t0:.0f}s, wrote {fdir}")
        del rr

    # Combine once every fold has held-out scores.
    fdirs = [work / f"fold{s.fold}" for s in splits]
    if not all((d / "pairs.parquet").exists() for d in fdirs):
        missing = [s.fold for s, d in zip(splits, fdirs) if not (d / "pairs.parquet").exists()]
        print(f"folds still to run: {missing}; not combining yet")
        return
    pairs = pd.concat([pd.read_parquet(d / "pairs.parquet") for d in fdirs], ignore_index=True)
    have_run = all((d / "run.parquet").exists() for d in fdirs)
    run_df = (pd.concat([pd.read_parquet(d / "run.parquet") for d in fdirs], ignore_index=True)
              if have_run else pd.DataFrame(columns=["query_id", "rank", "doc_id", "score", "first_rank", "fold"]))
    hist = pd.concat([pd.read_csv(d / "history.csv") for d in fdirs if (d / "history.csv").exists()],
                     ignore_index=True)
    qids = list(dict.fromkeys(pairs.query_id))
    assert pairs.groupby("query_id").fold.nunique().max() == 1, "a question scored in two folds"
    per_q = R.per_question(run_df, pairs, table, qids)
    per_q.insert(1, "fold", pairs.groupby("query_id").fold.first().reindex(per_q.index))
    if not have_run:  # ranking metrics need the shortlist; keep pairwise accuracy only
        per_q = per_q[["kind", "fold", "pairwise_acc"]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if have_run:
        run_df.to_parquet(out / f"run_{tag}.parquet", index=False)
    pairs.to_parquet(out / f"pairs_{tag}.parquet", index=False)
    per_q.to_csv(out / f"metrics_{tag}.csv")
    hist.to_csv(out / f"train_history_{tag}.csv", index=False)
    print(f"wrote {out}: {'run, ' if have_run else ''}pairs, metrics and train_history for {tag}")
    print(f"\n{tag} (held out, {len(qids)} questions):\n{summarize(per_q).round(4).to_string()}")
    if smoke:
        print("\nsmoke run (subsampled pairs or questions): numbers are not results")


if __name__ == "__main__":
    main()
