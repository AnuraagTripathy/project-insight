"""RQ3: train one retriever per fold, re-encode the corpus with it, and keep each test question's top 100.

For each test fold f (folds from results/folds.csv, the same splits every condition uses):
1. Build InfoNCE examples from fold f's training questions and the early-stopping pool from its
   inner validation questions (insight.train_retriever.fold_examples never reads test rows).
2. Fine-tune a fresh Qwen3-Embedding-0.6B (configs/rq3.yaml `train:`), keep the best epoch and
   save it as a plain Transformers model (outputs/rq3/fold<f>/model).
3. Reload that model in the encoding precision, encode all corpus papers and fold f's test
   questions, and search with FAISS exact inner product and the date filter, as condition B does.
   The top 100 goes to results/shortlist_rq3_fold<f>.parquet.
Every question is a test question in exactly one fold, so when all 5 folds are done the script
also writes results/shortlist_rq3.parquet (all 894 questions, each from the model that never
trained on it, the input for the reranker), metrics_rq3.csv (one row per question, with its
fold) and train_history_rq3.csv, and prints core and subfield means next to condition B's.

Full run (H100): uv run python scripts/train_retrievers.py --config configs/rq3.yaml
Resume after an interruption: add --resume (a fold with a saved model is not retrained; a fold
with saved corpus vectors is not re-encoded).
CPU smoke test:  uv run python scripts/train_retrievers.py --dtype float32 --fields title \
    --lora-r 16 --lr 1e-4 --gradient-checkpointing --epochs 1 --batch-size 4 --hard-negatives 1 \
    --max-train-examples 8 --max-val-queries 2 --test-limit 2 --corpus-limit 300 --val-pool-size 100 \
    --tag smoke --work-dir outputs/smoke_rq3 --out-dir outputs/smoke_rq3
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from insight import data, metrics, retrieve_bm25, retrieve_dense as R, train_retriever as T
from insight.train_reranker import splits_from_csv

COLS = ["ndcg@20", "recall@5", "recall@20", "recall@100", "mrr"]


def summarize(per_q: pd.DataFrame) -> pd.DataFrame:
    s = per_q.groupby("kind")[[c for c in COLS if c in per_q]].mean()
    s.insert(0, "n", per_q.groupby("kind").size())
    return s


def free_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "rq3.yaml"))
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="test folds to run (default all)")
    ap.add_argument("--tag", default=None, help="output suffix (default: the config's tag)")
    ap.add_argument("--work-dir", default=None, help="per-fold outputs (default: the config's work_dir)")
    ap.add_argument("--out-dir", default=str(data.ROOT / "results"))
    ap.add_argument("--resume", action="store_true", help="reuse saved models and corpus vectors")
    # Overrides, mostly for the CPU smoke test
    ap.add_argument("--dtype", default=None, help="encoding precision (and training weights)")
    ap.add_argument("--fields", nargs="*", default=None, help="paper fields embedded")
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="training examples per step")
    ap.add_argument("--hard-negatives", type=int, default=None)
    ap.add_argument("--encode-batch-size", type=int, default=None)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--val-pool-size", type=int, default=None)
    ap.add_argument("--max-train-examples", type=int, default=None, help="seeded subset of examples (smoke)")
    ap.add_argument("--max-val-queries", type=int, default=None, help="first N validation questions (smoke)")
    ap.add_argument("--test-limit", type=int, default=None, help="first N test questions per fold (smoke)")
    ap.add_argument("--corpus-limit", type=int, default=None,
                    help="first N papers plus the used questions' labeled papers (smoke)")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    dcfg = R.DenseConfig.from_yaml(args.config)
    dcfg = replace(dcfg, **{k: v for k, v in {"dtype": args.dtype, "fields": args.fields,
                                               "batch_size": args.encode_batch_size}.items() if v is not None})
    tcfg = T.RetrieverTrainConfig.from_yaml(args.config)
    over = {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
            "hard_negatives": args.hard_negatives, "val_pool_size": args.val_pool_size}
    tcfg = replace(tcfg, **{k: v for k, v in over.items() if v is not None})
    if args.lora_r is not None:
        tcfg = replace(tcfg, lora_r=args.lora_r)
    if args.gradient_checkpointing:
        tcfg = replace(tcfg, gradient_checkpointing=True)
    train_dtype = raw.get("train_dtype", "float32")
    if args.dtype is not None:  # one precision for everything, no autocast (CPU smoke runs)
        train_dtype = args.dtype
        tcfg = replace(tcfg, autocast=None)
    tag = args.tag or raw.get("tag", "rq3")
    work = data.ROOT / (args.work_dir or raw.get("work_dir", f"outputs/{tag}"))
    smoke = any(v is not None for v in [args.max_train_examples, args.max_val_queries, args.test_limit,
                                        args.corpus_limit])

    datacfg = data.DataConfig.from_yaml(data.ROOT / raw.get("data_config", "configs/data.yaml"))
    ds = data.load(datacfg)
    table = ds.table
    question_of = table.question.to_dict()
    # Training reads paper text by id (credited and rejected papers may lie outside a smoke subset).
    text_of = dict(zip(ds.corpus.id, R.doc_texts(ds.corpus, dcfg.fields, dcfg.field_sep)))
    splits = splits_from_csv(data.ROOT / raw.get("folds", "results/folds.csv"))
    run_folds = args.folds if args.folds is not None else [s.fold for s in splits]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for sp in splits:
        if sp.fold not in run_folds:
            continue
        fdir = work / f"fold{sp.fold}"
        model_dir, emb_path = fdir / "model", fdir / "corpus_emb.npy"
        if args.resume and (fdir / "run.parquet").exists():
            print(f"fold {sp.fold}: finished, reusing {fdir}")
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        examples, val_q = T.fold_examples(table, sp, tcfg)
        if args.max_train_examples is not None and len(examples) > args.max_train_examples:
            examples = examples.sample(args.max_train_examples, random_state=tcfg.seed).reset_index(drop=True)
        if args.max_val_queries is not None:
            val_q = val_q[:args.max_val_queries]
        test_q = list(sp.test)[:args.test_limit] if args.test_limit is not None else list(sp.test)
        corpus = ds.corpus if args.corpus_limit is None else \
            T.smoke_corpus(ds.corpus, table, list(val_q) + test_q, args.corpus_limit)
        texts = [text_of[d] for d in corpus.id]

        # 1-2. train (or reuse the saved model)
        if args.resume and (model_dir / "config.json").exists():
            print(f"fold {sp.fold}: reusing trained model {model_dir}")
        else:
            enc = R.Encoder(replace(dcfg, model=tcfg.model, dtype=train_dtype))
            print(f"fold {sp.fold}: training {tcfg.model} on {enc.device}, weights {train_dtype}, "
                  f"autocast {tcfg.autocast}, {'full fine-tuning' if tcfg.lora_r is None else f'LoRA r {tcfg.lora_r}'}")
            pool = T.val_pool(table, val_q, corpus.id.tolist(), tcfg.val_pool_size, tcfg.seed)
            res = T.train(enc, examples, table, question_of, text_of, tcfg, val_q=val_q, pool_ids=pool,
                          corpus_size=len(corpus))
            T.save_encoder(enc, model_dir)
            res["history"].assign(fold=sp.fold).to_csv(fdir / "history.csv", index=False)
            info = {k: res[k] for k in ["best_step", "best_val_recall", "val_k", "steps"]}
            info.update(n_examples=len(examples), n_val_questions=len(val_q), val_pool=len(pool),
                        train_seconds=round(time.time() - t0), train_dtype=train_dtype,
                        train_config=asdict(tcfg), smoke=smoke)
            (fdir / "summary.json").write_text(json.dumps(info, indent=2))
            del enc
            free_memory()

        # 3. encode the corpus with the fold's model and search for its test questions
        t1 = time.time()
        enc = R.Encoder(replace(dcfg, model=str(model_dir)))
        doc_emb = None
        if args.resume and emb_path.exists():
            doc_emb = np.load(emb_path)
            if len(doc_emb) != len(corpus):
                doc_emb = None
            else:
                print(f"fold {sp.fold}: reusing corpus vectors {emb_path}")
        print(f"fold {sp.fold}: {'searching' if doc_emb is not None else 'encoding'} {len(corpus)} papers "
              f"for {len(test_q)} test questions "
              f"on {enc.device}, dtype {dcfg.dtype}")
        run_df, doc_emb, _ = T.retrieve_fold(
            enc, corpus.id.tolist(), texts, test_q, question_of, data.mask_fn(corpus, table, datacfg),
            dcfg.instruction, dcfg.max_length_query, dcfg.max_length_doc, dcfg.top_k, dcfg.batch_size,
            doc_emb=doc_emb, show_progress=True)
        if raw.get("keep_corpus_emb", True):
            np.save(emb_path, doc_emb)
        run_df = run_df.assign(fold=sp.fold)
        run_df.to_parquet(fdir / "run.parquet", index=False)
        run_df.to_parquet(out / f"shortlist_{tag}_fold{sp.fold}.parquet", index=False)
        print(f"fold {sp.fold}: encoded and searched in {time.time() - t1:.0f}s; done in {time.time() - t0:.0f}s, "
              f"wrote {out}/shortlist_{tag}_fold{sp.fold}.parquet ({len(run_df)} rows)")
        del enc, doc_emb
        free_memory()

    # Combine once every fold has a run.
    fdirs = [work / f"fold{s.fold}" for s in splits]
    if not all((d / "run.parquet").exists() for d in fdirs):
        missing = [s.fold for s, d in zip(splits, fdirs) if not (d / "run.parquet").exists()]
        print(f"folds still to run: {missing}; not combining yet")
        return
    run_df = pd.concat([pd.read_parquet(d / "run.parquet") for d in fdirs], ignore_index=True)
    assert run_df.groupby("query_id").fold.nunique().max() == 1, "a question searched in two folds"
    hist = pd.concat([pd.read_csv(d / "history.csv") for d in fdirs if (d / "history.csv").exists()],
                     ignore_index=True)
    qids = list(dict.fromkeys(run_df.query_id))
    per_q = metrics.evaluate(retrieve_bm25.as_ranked_lists(run_df), table.positives.loc[qids].to_dict())
    per_q.insert(0, "kind", table.kind.loc[per_q.index])
    per_q.insert(1, "fold", run_df.groupby("query_id").fold.first().reindex(per_q.index))
    run_df.to_parquet(out / f"shortlist_{tag}.parquet", index=False)
    per_q.to_csv(out / f"metrics_{tag}.csv")
    hist.to_csv(out / f"train_history_{tag}.csv", index=False)
    print(f"wrote {out}: shortlist_{tag}.parquet ({len(run_df)} rows), metrics_{tag}.csv, train_history_{tag}.csv")
    print(f"\n{tag} (held out, {len(qids)} questions):\n{summarize(per_q).round(4).to_string()}")
    for b in raw.get("baselines", []):
        bp = data.ROOT / b
        if bp.exists():
            base = pd.read_csv(bp, index_col=0)
            base = base.loc[base.index.intersection(per_q.index)]
            print(f"\n{bp.name} on the same questions:\n{summarize(base).round(4).to_string()}")
    if smoke:
        print("\nsmoke run (subsampled examples, questions or corpus): numbers are not results")


if __name__ == "__main__":
    main()
