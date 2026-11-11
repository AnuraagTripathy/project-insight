"""Cross-encoder reranking with Qwen3-Reranker (condition C; D and E add LoRA on top).

The reranker is a decoder-only LLM. It reads one (question, paper) pair inside a fixed chat
prompt and is asked whether the paper meets the instruction; the answer can only be "yes" or
"no". The prompt is copied from the Qwen3-Reranker model card (Transformers usage), which is the
same text as the repo's ``chat_template.jinja``:

    <|im_start|>system
    Judge whether the Document meets the requirements based on the Query and the Instruct
    provided. Note that the answer can only be "yes" or "no".<|im_end|>
    <|im_start|>user
    <Instruct>: {instruction}
    <Query>: {query}
    <Document>: {doc}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

(the system line is one line in the real prompt). The score is the difference of the next-token
logits for "yes" and "no",

    s(q, d) = z_yes - z_no,

which is the log-odds of "yes": sigmoid(s) equals the card's probability
softmax([z_no, z_yes])[1]. It is also the raw score the card's Sentence Transformers example
prints. Training (D, E) uses the same score, so ``score_inputs`` keeps gradients.

The instruction asks about inspiration rather than relevance. Neither the model card nor
ScholarCatalyst gives a pointwise inspiration instruction (the paper's Appendix G.2 prompt is
listwise), so ours is condensed from that listwise prompt (decision log, Nov 9).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Model card, Transformers usage: the text before and after "<Instruct>: ...<Document>: ...".
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CARD_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Ours (decision log, Nov 9), condensed from ScholarCatalyst's listwise reranking prompt
# (arXiv:2610.02202, Appendix G.2): "Rank a paper higher when its scientific contribution would
# meaningfully inform how a researcher thinks about or approaches this question. Rank it lower
# when it is merely topically related and does not provide insight for this research."
INSPIRATION_INSTRUCTION = (
    "Given a research question, judge whether the paper could inspire the research: its scientific "
    "contribution would meaningfully inform how a researcher thinks about or approaches the "
    "question, rather than being merely topically related"
)


@dataclass
class RerankConfig:
    model: str = "Qwen/Qwen3-Reranker-0.6B"
    instruction: str = INSPIRATION_INSTRUCTION
    fields: list[str] = field(default_factory=lambda: ["title", "text"])
    field_sep: str = "\n"
    max_length: int = 8192  # whole prompt, in tokens (the card's value; nothing here comes close)
    batch_size: int = 16
    dtype: str = "bfloat16"  # "float32" on CPU for exact checks
    attn_implementation: str | None = "sdpa"
    depth: int = 100  # shortlist papers rescored per question
    top_k: int = 100  # reranked papers written per question

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw.get("rerank", {}))


def format_pair(query: str, doc: str, instruction: str = INSPIRATION_INSTRUCTION) -> str:
    """The user turn exactly as the card's format_instruction builds it."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def doc_texts(corpus: pd.DataFrame, fields=("title", "text"), sep="\n") -> list[str]:
    """One string per paper: the chosen fields joined by `sep`, empty fields left out."""
    cols = corpus[list(fields)].fillna("").astype(str)
    return [sep.join(v.strip() for v in row if v.strip()) for row in cols.itertuples(index=False)]


class Reranker:
    """Qwen3-Reranker wrapper: (question, paper text) pairs in, scores z_yes - z_no out."""

    def __init__(self, cfg: RerankConfig = RerankConfig(), device: str | None = None, model=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
        if model is None:
            kwargs = {"dtype": getattr(torch, cfg.dtype)}
            if cfg.attn_implementation:
                kwargs["attn_implementation"] = cfg.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs).to(self.device).eval()
        self.model = model
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def token_ids(self, query: str, doc: str) -> list[int]:
        """Prompt token ids for one pair, as the card's process_inputs builds them.

        The middle part is truncated so the whole prompt fits in cfg.max_length (the card's
        truncation, applied before the fixed prefix and suffix are added).
        """
        room = self.cfg.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        mid = self.tokenizer(format_pair(query, doc, self.cfg.instruction), add_special_tokens=False,
                             truncation=True, max_length=room)["input_ids"]
        return self.prefix_ids + mid + self.suffix_ids

    def collate(self, ids: list[list[int]]):
        """Left-pad a batch of token id lists into model inputs on the device."""
        batch = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def score_inputs(self, inputs):
        """s = z_yes - z_no at the last position, as a float tensor (gradients kept).

        logits_to_keep=1 makes the model project only the last position onto the vocabulary
        (151,669 tokens) instead of every position, which is what dominates memory in training.
        """
        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
        return (logits[:, self.yes_id] - logits[:, self.no_id]).float()

    def score(self, queries: list[str], docs: list[str], batch_size: int | None = None,
              show_progress: bool = False) -> np.ndarray:
        """Scores for pairs (queries[i], docs[i]), batched by similar length, in input order."""
        torch = self.torch
        assert len(queries) == len(docs)
        bs = batch_size or self.cfg.batch_size
        ids = [self.token_ids(q, d) for q, d in zip(queries, docs)]
        order = np.argsort([len(x) for x in ids], kind="stable")[::-1]  # longest first
        out = np.empty(len(ids), dtype=np.float64)
        it = range(0, len(ids), bs)
        if show_progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="rerank", unit="batch")
        with torch.inference_mode():
            for start in it:
                idx = order[start:start + bs]
                out[idx] = self.score_inputs(self.collate([ids[i] for i in idx])).cpu().numpy()
        return out


def rerank_run(shortlist: pd.DataFrame, question_of: dict, text_of: dict, score_fn,
               depth: int = 100, top_k: int = 100, **score_kw) -> pd.DataFrame:
    """Rescore each question's top `depth` shortlist papers and re-sort them.

    shortlist: long table (query_id, rank, doc_id, score) from a retriever.
    question_of: {query id: question text}; text_of: {paper id: paper text}.
    score_fn(queries, docs, **score_kw) -> array of scores, e.g. Reranker.score.
    Returns a long table (query_id, rank, doc_id, score, first_rank), best first, where
    first_rank is the paper's rank in the shortlist. Equal scores are ordered by paper id,
    the same rule as insight.metrics.rank.
    """
    sl = shortlist[shortlist["rank"] <= depth].sort_values(["query_id", "rank"])
    scores = score_fn([question_of[q] for q in sl.query_id], [text_of[d] for d in sl.doc_id], **score_kw)
    df = pd.DataFrame({"query_id": sl.query_id.to_numpy(), "doc_id": sl.doc_id.to_numpy(),
                       "score": np.asarray(scores, dtype=np.float64), "first_rank": sl["rank"].to_numpy()})
    df["neg"] = -df.score
    df = df.sort_values(["query_id", "neg", "doc_id"]).drop(columns="neg")
    df["rank"] = df.groupby("query_id").cumcount() + 1
    df = df[df["rank"] <= top_k]
    return df[["query_id", "rank", "doc_id", "score", "first_rank"]].reset_index(drop=True)


def labeled_pairs(table: pd.DataFrame, query_ids) -> pd.DataFrame:
    """Every (question, credited or rejected paper) pair: query_id, doc_id, label (1 or 0).

    These are scored separately from the shortlist so pairwise accuracy covers all labeled
    papers, not only those the retriever happened to shortlist.
    """
    rows = []
    for qid in query_ids:
        r = table.loc[qid]
        rows += [(qid, d, 1) for d in r.positives] + [(qid, d, 0) for d in r.negatives]
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "label"])
