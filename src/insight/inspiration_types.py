"""RQ1: inspiration-type labels for author rationales.

The taxonomy is copied word for word from ScholarCatalyst (Kim et al., 2026, arXiv:2610.02202),
Appendix B.3, Table 13 ("Taxonomy of inspiration types. Definitions describe how a source work
shaped the target work."). The paper's text calls it a "nine-type taxonomy", but Table 13 defines
eight types; the ninth bar in its Figure 7 is "Others" (1.8% of rationales) and has no definition.
Our wording for "other" is therefore ours, not the paper's (decision log, Nov 6).

The paper labels only the core-query rationales ("663 key-inspiration rationales"), multi-label,
"with every type it clearly expresses". We label both kinds of rationale:

* core: one rationale per credited (question, paper) pair (661 units in the released data);
* subfield: one rationale per question, describing all its credited papers together (687 units),
  so its types apply to the whole group of credited papers.

This module holds everything that does not need the network: the taxonomy, turning the dataset
into labeling units, the prompt, the JSON schema, parsing and validation of a model reply, and
turning replies into a label table. The API call lives in scripts/label_inspiration_types.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

# (key, name in the paper, definition copied from Table 13 of arXiv:2610.02202). Order as in Table 13.
PAPER_TYPES: list[tuple[str, str, str]] = [
    ("method", "Method",
     "A specific method, mechanism, component, formulation, or principle from the source work is "
     "adopted or adapted."),
    ("generalization", "Generalization",
     "An idea, finding, or method from the source work is extended to a new domain, modality, task, "
     "setting, or broader scope."),
    ("limitation", "Limitation",
     "A limitation, failure mode, or unresolved issue in the source work motivates the target work "
     "to address it."),
    ("framework", "Framework",
     "The target work builds on the source work's overall framework, model, or system rather than a "
     "specific component."),
    ("phenomenon", "Phenomenon",
     "An empirical phenomenon or observation in the source work motivates the target work to "
     "explain, characterize, or formalize it."),
    ("reframing", "Reframing",
     "An insight from the source work provides a new perspective, abstraction, or interpretation of "
     "the target problem."),
    ("motivation", "Motivation",
     "The source work motivates the research problem or direction pursued in the target work. This "
     "label is used only when the motivation is not better captured by limitation, phenomenon, or "
     "evidence."),
    ("evidence", "Evidence",
     "The source work provides evidence or an independent finding that supports the plausibility or "
     "direction of the target work without being directly transferred."),
]

# The paper's Figure 7 has a ninth bar, "Others", without a definition. This wording is ours.
OTHER_TYPE = ("other", "Other", "None of the types above is clearly expressed by the rationale.")

SYSTEM_PROMPT = (
    "You classify why a researcher says a prior paper inspired their project. You are given a "
    "research question written by the project's author, the prior paper(s) they credited, and the "
    "author's rationale. Using only the taxonomy below, label the rationale with every type it "
    "clearly expresses. A rationale can have more than one type. In the definitions, the source "
    "work is the credited prior paper and the target work is the author's project.\n\n"
    "Taxonomy of inspiration types:\n{taxonomy}\n\n"
    "Answer with JSON only: {{\"types\": [...], \"reason\": \"...\"}}, where types is a list of "
    "type keys from the taxonomy (most important first) and reason is one short sentence."
)

USER_TEMPLATE = (
    "Research question:\n{question}\n\n"
    "{credited_heading}:\n{credited}\n\n"
    "Author's rationale:\n{rationale}"
)


@dataclass
class TypeConfig:
    include_other: bool = True       # offer the paper's ninth label ("Others"), with our definition
    include_titles: bool = True      # show the credited paper titles in the prompt
    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 1024
    dry_run_n: int = 20
    seed: int = 0
    # Prices in USD per million tokens, used only for the printed cost estimate.
    price_in: float = 4.0
    price_out: float = 20.0
    chars_per_token: float = 4.0     # rough estimate; the real count comes back in the API usage
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path) -> "TypeConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {k: raw.pop(k) for k in list(raw) if k in cls.__dataclass_fields__}
        return cls(**known, extra=raw)


def taxonomy(cfg: TypeConfig | None = None) -> list[tuple[str, str, str]]:
    cfg = cfg or TypeConfig()
    return PAPER_TYPES + ([OTHER_TYPE] if cfg.include_other else [])


def type_keys(cfg: TypeConfig | None = None) -> list[str]:
    return [k for k, _, _ in taxonomy(cfg)]


def system_prompt(cfg: TypeConfig | None = None) -> str:
    lines = [f"- {key} ({name}): {definition}" for key, name, definition in taxonomy(cfg)]
    return SYSTEM_PROMPT.format(taxonomy="\n".join(lines))


def output_schema(cfg: TypeConfig | None = None) -> dict:
    """JSON schema for the reply (used as the API's structured-output format)."""
    return {
        "type": "object",
        "properties": {
            "types": {"type": "array", "items": {"type": "string", "enum": type_keys(cfg)}},
            "reason": {"type": "string"},
        },
        "required": ["types", "reason"],
        "additionalProperties": False,
    }


def rationale_units(labels: pd.DataFrame, queries: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per rationale to label.

    Columns: unit_id, query_id, paper_id (the project), kind, doc_ids (credited papers the
    rationale covers), question, rationale, titles. Core units have one doc id each; a subfield
    unit covers all credited papers of its question.
    """
    q = queries.set_index("id")
    title = corpus.set_index("id").title
    rows = []
    for lab in labels.itertuples(index=False):
        for doc_id, text in lab.rationales.items():
            docs = [doc_id] if doc_id is not None else list(lab.positives)
            rows.append({
                "unit_id": f"{lab.query_id}::{doc_id}" if doc_id is not None else lab.query_id,
                "query_id": lab.query_id,
                "paper_id": q.at[lab.query_id, "paper_id"],
                "kind": lab.kind,
                "doc_ids": docs,
                "question": q.at[lab.query_id, "question"],
                "rationale": text,
                "titles": [title.get(d, "") for d in docs],
            })
    return pd.DataFrame(rows)


def user_prompt(unit, cfg: TypeConfig | None = None) -> str:
    cfg = cfg or TypeConfig()
    titles = [t for t in unit["titles"] if t] if cfg.include_titles else []
    if titles:
        heading = "Credited paper" if len(titles) == 1 else "Credited papers (the rationale covers all of them)"
        credited = "\n".join(f"- {t}" for t in titles)
    else:
        heading, credited = "Credited paper", "(title not shown)"
    return USER_TEMPLATE.format(question=unit["question"], credited_heading=heading,
                                credited=credited, rationale=unit["rationale"])


def parse_reply(text: str, cfg: TypeConfig | None = None) -> dict:
    """Parse and validate one model reply. Returns {"types": [...], "reason": str}.

    Accepts the JSON object alone or wrapped in a ```json fence. Type keys are lowercased and
    matched against the taxonomy (paper names like "Method" are accepted too); duplicates are
    dropped keeping the first. Raises ValueError on anything else: unknown types, an empty list,
    or "other" combined with a real type.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"reply is not JSON: {text[:80]!r}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("types"), list):
        raise ValueError("reply must be an object with a 'types' list")
    keys = type_keys(cfg)
    out: list[str] = []
    for t in obj["types"]:
        k = str(t).strip().lower()
        if k == "others":
            k = "other"
        if k not in keys:
            raise ValueError(f"unknown inspiration type {t!r}")
        if k not in out:
            out.append(k)
    if not out:
        raise ValueError("empty type list")
    if "other" in out and len(out) > 1:
        raise ValueError("'other' cannot be combined with another type")
    return {"types": out, "reason": str(obj.get("reason", ""))}


def label_table(units: pd.DataFrame, replies: dict[str, dict], cfg: TypeConfig | None = None) -> pd.DataFrame:
    """Turn parsed replies {unit_id: {"types": [...]}} into the committed label file.

    One row per unit with ids only (no dataset text, since ScholarCatalyst is CC BY-NC): unit_id,
    query_id, paper_id, kind, doc_ids (";"-joined), n_docs, types (";"-joined, most important
    first), n_types, then one 0/1 column per type. Units without a reply are left out.
    """
    keys = type_keys(cfg)
    rows = []
    for u in units.itertuples(index=False):
        r = replies.get(u.unit_id)
        if r is None:
            continue
        row = {"unit_id": u.unit_id, "query_id": u.query_id, "paper_id": u.paper_id, "kind": u.kind,
               "doc_ids": ";".join(u.doc_ids), "n_docs": len(u.doc_ids),
               "types": ";".join(r["types"]), "n_types": len(r["types"])}
        row.update({k: int(k in r["types"]) for k in keys})
        rows.append(row)
    return pd.DataFrame(rows, columns=["unit_id", "query_id", "paper_id", "kind", "doc_ids", "n_docs",
                                       "types", "n_types", *keys])


def type_shares(table: pd.DataFrame, cfg: TypeConfig | None = None) -> pd.DataFrame:
    """Share of units carrying each type, per kind (comparable to the paper's Figure 7 for core)."""
    keys = type_keys(cfg)
    return table.groupby("kind")[keys].mean().T


def dry_run_sample(units: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """A fixed sample of n units, half core and half subfield (core gets the odd one)."""
    n_core = (n + 1) // 2
    core = units[units.kind == "core"].sample(n=min(n_core, (units.kind == "core").sum()), random_state=seed)
    sub = units[units.kind == "subfield"].sample(n=min(n - len(core), (units.kind == "subfield").sum()),
                                                 random_state=seed)
    return pd.concat([core, sub], ignore_index=True)
