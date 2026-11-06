import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from insight import data, inspiration_types as T

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "corpus.parquet").exists(),
    reason="full dataset not downloaded (run scripts/download_data.py)",
)

LABELS = pd.DataFrame({
    "query_id": ["q1", "q2"],
    "kind": ["core", "subfield"],
    "positives": [["a", "b"], ["a", "c"]],
    "rationales": [{"a": "A gave us the trick.", "b": "B showed it fails."}, {None: "Both shaped our view."}],
})
QUERIES = pd.DataFrame({"id": ["q1", "q2"], "paper_id": ["p1", "p1"], "question": ["How?", "Why?"]})
CORPUS = pd.DataFrame({"id": ["a", "b", "c"], "title": ["Paper A", "Paper B", ""]})


def test_taxonomy_matches_paper_table_13():
    keys = [k for k, _, _ in T.PAPER_TYPES]
    assert keys == ["method", "generalization", "limitation", "framework", "phenomenon",
                    "reframing", "motivation", "evidence"]
    # Spot-check verbatim definitions against arXiv:2610.02202 Table 13.
    d = {k: v for k, _, v in T.PAPER_TYPES}
    assert d["method"].startswith("A specific method, mechanism, component, formulation, or principle")
    assert d["motivation"].endswith("not better captured by limitation, phenomenon, or evidence.")
    assert T.type_keys() == keys + ["other"]                      # nine with the Figure 7 "Others" bar
    assert T.type_keys(T.TypeConfig(include_other=False)) == keys
    assert "other (Other)" in T.system_prompt() and "other" not in T.system_prompt(T.TypeConfig(include_other=False)).split("Taxonomy")[1].split("Answer")[0]


def test_units_core_per_paper_subfield_per_question():
    u = T.rationale_units(LABELS, QUERIES, CORPUS)
    assert list(u.unit_id) == ["q1::a", "q1::b", "q2"]
    assert u.doc_ids.tolist() == [["a"], ["b"], ["a", "c"]]
    assert u.titles.tolist() == [["Paper A"], ["Paper B"], ["Paper A", ""]]
    assert (u.paper_id == "p1").all()


def test_user_prompt_contents():
    u = T.rationale_units(LABELS, QUERIES, CORPUS)
    p = T.user_prompt(u.iloc[0])
    assert "How?" in p and "- Paper A" in p and "A gave us the trick." in p
    p2 = T.user_prompt(u.iloc[2])
    assert "Credited paper:" in p2 and "- Paper A" in p2      # blank title dropped, so one title left
    p3 = T.user_prompt(u.iloc[0], T.TypeConfig(include_titles=False))
    assert "Paper A" not in p3


def test_parse_reply_valid_and_normalized():
    assert T.parse_reply('{"types": ["method", "evidence"], "reason": "r"}') == {"types": ["method", "evidence"], "reason": "r"}
    assert T.parse_reply('```json\n{"types": ["Method", "method", "Limitation"], "reason": ""}\n```')["types"] == ["method", "limitation"]
    assert T.parse_reply('{"types": ["Others"], "reason": ""}')["types"] == ["other"]


@pytest.mark.parametrize("bad", [
    "not json", "[]", '{"types": "method"}', '{"types": []}', '{"types": ["inspiration"]}',
    '{"types": ["other", "method"]}',
])
def test_parse_reply_rejects(bad):
    with pytest.raises(ValueError):
        T.parse_reply(bad)


def test_parse_reply_other_disabled():
    with pytest.raises(ValueError):
        T.parse_reply('{"types": ["other"]}', T.TypeConfig(include_other=False))


def test_schema_enum_matches_taxonomy():
    s = T.output_schema()
    assert s["properties"]["types"]["items"]["enum"] == T.type_keys()
    assert s["additionalProperties"] is False


def test_label_table_and_shares():
    u = T.rationale_units(LABELS, QUERIES, CORPUS)
    replies = {"q1::a": {"types": ["method"]}, "q2": {"types": ["evidence", "reframing"]}}
    t = T.label_table(u, replies)
    assert list(t.unit_id) == ["q1::a", "q2"]                    # q1::b has no reply
    assert t.loc[1, "doc_ids"] == "a;c" and t.loc[1, "n_types"] == 2
    assert t.loc[0, "method"] == 1 and t.loc[0, "evidence"] == 0
    assert "rationale" not in t.columns and "question" not in t.columns   # no dataset text
    sh = T.type_shares(t)
    assert sh.loc["method", "core"] == 1.0 and sh.loc["reframing", "subfield"] == 1.0


def test_dry_run_sample_is_fixed_and_balanced():
    units = pd.DataFrame({"unit_id": [f"u{i}" for i in range(40)], "kind": ["core"] * 20 + ["subfield"] * 20})
    a, b = T.dry_run_sample(units, 7, seed=0), T.dry_run_sample(units, 7, seed=0)
    assert a.unit_id.tolist() == b.unit_id.tolist()
    assert (a.kind == "core").sum() == 4 and (a.kind == "subfield").sum() == 3


@pytest.mark.skipif(importlib.util.find_spec("anthropic") is None, reason="anthropic not installed (group llm)")
def test_call_model_request_and_reply_against_local_mock():
    """Runs the script's real SDK call against a local HTTP server returning a canned reply."""
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            body = json.dumps({
                "id": "msg_1", "type": "message", "role": "assistant", "model": seen["body"]["model"],
                "content": [{"type": "text", "text": '{"types": ["method"], "reason": "adapted"}'}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import anthropic
        spec = importlib.util.spec_from_file_location("lit", data.ROOT / "scripts" / "label_inspiration_types.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
        unit = T.rationale_units(LABELS, QUERIES, CORPUS).iloc[0]
        cfg = T.TypeConfig()
        text, usage = mod.call_model(client, unit, cfg)
    finally:
        srv.shutdown()
    assert T.parse_reply(text, cfg)["types"] == ["method"]
    assert seen["path"] == "/v1/messages"
    b = seen["body"]
    assert b["model"] == "claude-opus-5-5"
    assert b["output_config"]["format"]["schema"] == T.output_schema(cfg)
    assert b["output_config"]["effort"] == "low"
    assert "A gave us the trick." in b["messages"][0]["content"]


@needs_data
def test_real_units_counts():
    ds = data.load()
    u = T.rationale_units(ds.labels, ds.queries, ds.corpus)
    assert len(u) == 1348
    assert (u.kind == "core").sum() == 661 and (u.kind == "subfield").sum() == 687
    assert u.unit_id.is_unique
    assert u.rationale.str.strip().str.len().gt(0).all()
    assert (u[u.kind == "core"].doc_ids.str.len() == 1).all()
