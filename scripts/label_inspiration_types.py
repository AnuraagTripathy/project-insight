"""RQ1: label every author rationale with the paper's inspiration types using an LLM.

1,348 units: 661 per-paper core rationales and 687 per-question subfield rationales. The prompt,
taxonomy and reply parsing live in insight.inspiration_types; this script only builds units,
calls the Anthropic Messages API (structured JSON output) and writes the label file.

Modes:
  --dry-run   Build the prompts for a fixed sample of `dry_run_n` units (default 20), write them to
              outputs/rq1_types/dryrun_prompts.jsonl (gitignored: they contain dataset text), and
              print sizes and a cost estimate. Makes no API call and needs no key.
  (default)   Label all units (or the first --limit). Replies are cached one JSON line per unit in
              outputs/rq1_types/replies.jsonl, so a rerun only calls the API for missing units.
              Writes results/inspiration_types.csv (ids and 0/1 type columns, no dataset text).

Usage:
  uv run python scripts/label_inspiration_types.py --dry-run
  uv run --group llm python scripts/label_inspiration_types.py            # needs ANTHROPIC_API_KEY
"""

import argparse
import json
import time
from pathlib import Path

from insight import data, inspiration_types as T

OUT_DIR = data.ROOT / "outputs" / "rq1_types"


def build_units():
    ds = data.load()
    return T.rationale_units(ds.labels, ds.queries, ds.corpus)


def estimate(units, cfg):
    sys_chars = len(T.system_prompt(cfg))
    user_chars = [len(T.user_prompt(u, cfg)) for _, u in units.iterrows()]
    tok_in = (sys_chars * len(units) + sum(user_chars)) / cfg.chars_per_token
    return sys_chars, user_chars, tok_in


def call_model(client, unit, cfg):
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=[{"type": "text", "text": T.system_prompt(cfg), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": T.user_prompt(unit, cfg)}],
        output_config={"effort": cfg.effort,
                       "format": {"type": "json_schema", "schema": T.output_schema(cfg)}},
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal on {unit['unit_id']}")
    text = next(b.text for b in resp.content if b.type == "text")
    return text, resp.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(data.ROOT / "configs" / "rq1_types.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N units")
    ap.add_argument("--out", default=str(data.ROOT / "results" / "inspiration_types.csv"))
    args = ap.parse_args()
    cfg = T.TypeConfig.from_yaml(args.config)

    units = build_units()
    print(f"units: {len(units)} ({(units.kind == 'core').sum()} core, {(units.kind == 'subfield').sum()} subfield)")
    sys_chars, user_chars, tok_in = estimate(units, cfg)
    print(f"system prompt: {sys_chars} chars; user prompts: median {sorted(user_chars)[len(user_chars) // 2]}, "
          f"max {max(user_chars)} chars")
    print(f"estimated input tokens for all units (chars / {cfg.chars_per_token:g}, no caching): {tok_in:,.0f} "
          f"-> about ${tok_in * cfg.price_in / 1e6:.2f} at ${cfg.price_in:g}/M; output and thinking tokens "
          f"come on top at ${cfg.price_out:g}/M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        sample = T.dry_run_sample(units, cfg.dry_run_n, cfg.seed)
        path = OUT_DIR / "dryrun_prompts.jsonl"
        with path.open("w") as f:
            for _, u in sample.iterrows():
                f.write(json.dumps({"unit_id": u.unit_id, "kind": u.kind, "system": T.system_prompt(cfg),
                                    "user": T.user_prompt(u, cfg), "schema": T.output_schema(cfg)}) + "\n")
        print(f"dry run: {len(sample)} prompts ({(sample.kind == 'core').sum()} core, "
              f"{(sample.kind == 'subfield').sum()} subfield) -> {path}; no API call made")
        return

    import anthropic  # optional dependency group "llm"

    client = anthropic.Anthropic()
    cache_path = OUT_DIR / "replies.jsonl"
    replies = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            replies[r["unit_id"]] = r
    todo = units if args.limit is None else units.head(args.limit)
    todo = todo[~todo.unit_id.isin(replies)]
    print(f"cached replies: {len(replies)}; to label now: {len(todo)}")
    used_in = used_out = 0
    t0 = time.time()
    with cache_path.open("a") as f:
        for i, (_, u) in enumerate(todo.iterrows(), 1):
            text, usage = call_model(client, u, cfg)
            parsed = T.parse_reply(text, cfg)  # raises on an invalid reply, so nothing bad is cached
            rec = {"unit_id": u.unit_id, **parsed, "model": cfg.model}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            replies[u.unit_id] = rec
            used_in += usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
            used_out += usage.output_tokens
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  tokens in {used_in:,} out {used_out:,}")

    table = T.label_table(units, replies, cfg)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} labeled units -> {args.out}")
    print(T.type_shares(table, cfg).round(3).to_string())


if __name__ == "__main__":
    main()
