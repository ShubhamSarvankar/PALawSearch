"""
CI leakage check: no eval-split case may appear as a training query.

Run with: pytest tests/test_leakage.py -v
"""

import json
from pathlib import Path

MANIFEST    = Path("data/train/mine_manifest.json")
PAIRS_JSONL = Path("data/train/pairs.jsonl")
QRELS_JSONL = Path("data/eval/qrels_citation.jsonl")


def test_leakage_zero_via_manifest():
    """Fast check: mine_pairs.py already computed the overlap count."""
    assert MANIFEST.exists(), "Run 'just mine' first to generate the manifest."
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    lc = manifest["leakage_check"]
    assert lc["overlap_count"] == 0, (
        f"LEAKAGE: {lc['overlap_count']} eval query id(s) appear in training pair src_ids"
    )


def test_leakage_full_scan():
    """Independent re-scan of pairs.jsonl and qrels.jsonl — does not trust the manifest."""
    assert PAIRS_JSONL.exists(), "Run 'just mine' first."
    assert QRELS_JSONL.exists(), "Run 'just mine' first."

    train_src_ids: set[str] = set()
    with PAIRS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            train_src_ids.add(json.loads(line)["src_id"])

    eval_query_ids: set[str] = set()
    with QRELS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eval_query_ids.add(json.loads(line)["query_id"])

    overlap = train_src_ids & eval_query_ids
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} eval query id(s) found in training pair src_ids. "
        f"First 5: {sorted(overlap)[:5]}"
    )
