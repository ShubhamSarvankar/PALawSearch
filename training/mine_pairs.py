"""
Training pair mining and eval qrel generation.

Three positive sources (train split only):
  direct_cite   — train case A cites in-corpus train case B
  co_cite       — two train cases A, B both cite the same authority (any split)

One negative source:
  random        — random train case that is not the positive (no BM25 required
                  random negatives; hard retrieval-based negatives can be added later)

Eval qrels built from in-corpus citations of eval-split cases.

Leakage invariant enforced:
  No eval-split case_id may appear as a training-pair src_id or pos_id.
  Checked by assertion here and by tests/test_leakage.py.

Usage: python -m training.mine_pairs   (or: just mine)

Reads:
  data/parsed/cases.jsonl
  data/graph/split.json
  data/graph/id_to_idx.json
  data/graph/adj_in_corpus.npz
  data/graph/inv_index.json
Writes:
  data/train/pairs.jsonl        {src_id, pos_id, query, positive}
  data/train/triplets.jsonl     {src_id, pos_id, neg_id, query, positive, hard_neg}
  data/eval/qrels_citation.jsonl {query_id, query_text, relevant_doc_ids}
  data/train/mine_manifest.json
"""

import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse

PARSED_JSONL   = Path("data/parsed/cases.jsonl")
GRAPH_DIR      = Path("data/graph")
TRAIN_DIR      = Path("data/train")
EVAL_DIR       = Path("data/eval")

SPLIT_JSON     = GRAPH_DIR / "split.json"
ID_TO_IDX_JSON = GRAPH_DIR / "id_to_idx.json"
ADJ_FILE       = GRAPH_DIR / "adj_in_corpus.npz"
INV_INDEX_JSON = GRAPH_DIR / "inv_index.json"

PAIRS_JSONL    = TRAIN_DIR / "pairs.jsonl"
TRIPLETS_JSONL = TRAIN_DIR / "triplets.jsonl"
QRELS_JSONL    = EVAL_DIR  / "qrels_citation.jsonl"
MANIFEST       = TRAIN_DIR / "mine_manifest.json"

# Mining caps
MAX_DIRECT_PER_CASE      = 10   # max (query, pos) pairs from one train source case
MAX_COCITE_PER_ANCHOR    = 5    # max pairs per co-citation anchor
COCITE_BUDGET            = 300_000   # global cap on co-citation pairs
TEXT_QUERY_CAP           = 1024      # chars
TEXT_DOC_CAP             = 4096      # chars
SEED                     = 42


def _query_text(head_matter: str, full_text: str) -> str:
    """Use head_matter if non-empty, else first TEXT_QUERY_CAP chars of full_text."""
    q = head_matter.strip()[:TEXT_QUERY_CAP]
    return q if q else full_text[:TEXT_QUERY_CAP]


def main() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    # ── Load graph artifacts ─────────────────────────────────────────────────
    print("Loading split ...")
    split: dict[str, str] = json.loads(SPLIT_JSON.read_text("utf-8"))

    print("Loading id_to_idx ...")
    id_to_idx: dict[str, int] = json.loads(ID_TO_IDX_JSON.read_text("utf-8"))
    n_corpus = max(id_to_idx.values()) + 1
    idx_to_id: list[str] = [""] * n_corpus
    for cid, idx in id_to_idx.items():
        idx_to_id[idx] = cid

    print("Loading adjacency matrix ...")
    adj: scipy.sparse.csr_matrix = scipy.sparse.load_npz(str(ADJ_FILE))

    print("Loading inverted index ...")
    inv_index: dict[str, list[str]] = json.loads(INV_INDEX_JSON.read_text("utf-8"))
    print(f"  {len(inv_index):,} unique citation targets")

    # ── Build case text lookup ───────────────────────────────────────────────
    print("\nBuilding case text lookup from cases.jsonl ...")
    # Only store what we need for mining and qrel generation.
    train_q: dict[str, str] = {}   # train case_id → query text
    train_d: dict[str, str] = {}   # train case_id → doc text
    eval_q:  dict[str, str] = {}   # eval case_id  → query text
    n = 0
    with PARSED_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d   = json.loads(line)
            cid = d["id"]
            sp  = split.get(cid)
            if sp == "train":
                train_q[cid] = _query_text(d.get("head_matter", ""), d.get("full_text", ""))
                train_d[cid] = d.get("full_text", "")[:TEXT_DOC_CAP]
            elif sp == "eval":
                eval_q[cid]  = _query_text(d.get("head_matter", ""), d.get("full_text", ""))
            n += 1
            if n % 50_000 == 0:
                print(f"  ... {n:,} cases loaded", flush=True)

    train_ids_list = sorted(train_q.keys())   # stable order for deterministic sampling
    print(f"  Train cases loaded : {len(train_ids_list):,}")
    print(f"  Eval cases loaded  : {len(eval_q):,}")

    # ── Mining ───────────────────────────────────────────────────────────────
    seen_pairs: set[tuple[str, str]] = set()   # (src_id, pos_id) dedup

    def _random_neg(src_id: str, pos_id: str) -> str:
        """Sample a random train case that is not src or pos."""
        while True:
            cand = rng.choice(train_ids_list)
            if cand != src_id and cand != pos_id:
                return cand

    counts = {
        "direct_cite_pairs"  : 0,
        "co_cite_pairs"      : 0,
        "total_pairs"        : 0,
        "eval_queries"       : 0,
        "eval_with_relevant" : 0,
    }

    print("\nPass 1 — direct citation pairs ...")
    with (
        PAIRS_JSONL.open("w",    encoding="utf-8") as pf,
        TRIPLETS_JSONL.open("w", encoding="utf-8") as tf,
    ):
        for src_id in train_ids_list:
            si = id_to_idx.get(src_id)
            if si is None:
                continue
            out_nbrs = adj[si].indices   # in-corpus cases this case cites
            pos_targets = [
                idx_to_id[di]
                for di in out_nbrs
                if idx_to_id[di] and split.get(idx_to_id[di]) == "train"
                   and idx_to_id[di] != src_id
            ]
            if len(pos_targets) > MAX_DIRECT_PER_CASE:
                pos_targets = rng.sample(pos_targets, MAX_DIRECT_PER_CASE)

            for pos_id in pos_targets:
                if (src_id, pos_id) in seen_pairs:
                    continue
                seen_pairs.add((src_id, pos_id))
                q_text  = train_q[src_id]
                d_text  = train_d[pos_id]
                neg_id  = _random_neg(src_id, pos_id)
                pf.write(json.dumps({"src_id": src_id, "pos_id": pos_id,
                                     "query": q_text, "positive": d_text},
                                    ensure_ascii=False) + "\n")
                tf.write(json.dumps({"src_id": src_id, "pos_id": pos_id, "neg_id": neg_id,
                                     "query": q_text, "positive": d_text,
                                     "hard_neg": train_d.get(neg_id, "")},
                                    ensure_ascii=False) + "\n")
                counts["direct_cite_pairs"] += 1

        print(f"  Direct citation pairs: {counts['direct_cite_pairs']:,}")

        # ── Co-citation pairs ────────────────────────────────────────────────
        print("\nPass 2 — co-citation pairs ...")
        cocite_total = 0
        n_anchors    = 0
        for authority, src_ids in inv_index.items():
            if cocite_total >= COCITE_BUDGET:
                break
            # only train citers as both query and positive
            train_citers = [s for s in src_ids if split.get(s) == "train"]
            if len(train_citers) < 2:
                continue
            n_anchors += 1
            # build candidate pairs
            if len(train_citers) > 20:
                train_citers = rng.sample(train_citers, 20)
            pairs_here: list[tuple[str, str]] = []
            for i, a in enumerate(train_citers):
                for b in train_citers[i + 1 :]:
                    pairs_here.append((a, b))
            if len(pairs_here) > MAX_COCITE_PER_ANCHOR:
                pairs_here = rng.sample(pairs_here, MAX_COCITE_PER_ANCHOR)

            for src_id, pos_id in pairs_here:
                if cocite_total >= COCITE_BUDGET:
                    break
                if (src_id, pos_id) in seen_pairs:
                    continue
                seen_pairs.add((src_id, pos_id))
                q_text = train_q[src_id]
                d_text = train_d[pos_id]
                neg_id = _random_neg(src_id, pos_id)
                pf.write(json.dumps({"src_id": src_id, "pos_id": pos_id,
                                     "query": q_text, "positive": d_text},
                                    ensure_ascii=False) + "\n")
                tf.write(json.dumps({"src_id": src_id, "pos_id": pos_id, "neg_id": neg_id,
                                     "query": q_text, "positive": d_text,
                                     "hard_neg": train_d.get(neg_id, "")},
                                    ensure_ascii=False) + "\n")
                cocite_total += 1

        counts["co_cite_pairs"] = cocite_total
        print(f"  Co-citation anchors used : {n_anchors:,}")
        print(f"  Co-citation pairs        : {cocite_total:,}")

    counts["total_pairs"] = counts["direct_cite_pairs"] + counts["co_cite_pairs"]

    # ── Eval qrels ───────────────────────────────────────────────────────────
    print("\nPass 3 — eval qrels ...")
    with QRELS_JSONL.open("w", encoding="utf-8") as ef:
        for eval_id, q_text in eval_q.items():
            counts["eval_queries"] += 1
            si = id_to_idx.get(eval_id)
            if si is None:
                continue
            relevant = [
                idx_to_id[di]
                for di in adj[si].indices
                if idx_to_id[di]
            ]
            if not relevant:
                continue
            ef.write(json.dumps({
                "query_id"        : eval_id,
                "query_text"      : q_text,
                "relevant_doc_ids": relevant,
            }, ensure_ascii=False) + "\n")
            counts["eval_with_relevant"] += 1

    print(f"  Eval queries               : {counts['eval_queries']:,}")
    print(f"  Eval queries with ≥1 rel   : {counts['eval_with_relevant']:,}")

    # ── Leakage check ────────────────────────────────────────────────────────
    print("\nLeakage check ...")
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
    overlap_count = len(overlap)

    # Hard assertion — must be zero
    assert overlap_count == 0, (
        f"LEAKAGE DETECTED: {overlap_count} eval query id(s) appear in training pair src_ids. "
        f"First 5: {sorted(overlap)[:5]}"
    )
    print(f"  Training pair src_ids : {len(train_src_ids):,}")
    print(f"  Eval query ids        : {len(eval_query_ids):,}")
    print(f"  Overlap (must be 0)   : {overlap_count}  ✓")

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        **counts,
        "caps": {
            "max_direct_per_case"   : MAX_DIRECT_PER_CASE,
            "max_cocite_per_anchor" : MAX_COCITE_PER_ANCHOR,
            "cocite_budget"         : COCITE_BUDGET,
            "text_query_cap"        : TEXT_QUERY_CAP,
            "text_doc_cap"          : TEXT_DOC_CAP,
        },
        "seed": SEED,
        "leakage_check": {
            "train_src_ids_count" : len(train_src_ids),
            "eval_query_ids_count": len(eval_query_ids),
            "overlap_count"       : overlap_count,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")

    sep = "─" * 62
    print(f"\n{sep}")
    print("  MINING MANIFEST")
    print(sep)
    print(f"  Direct citation pairs        : {counts['direct_cite_pairs']:>10,}")
    print(f"  Co-citation pairs            : {counts['co_cite_pairs']:>10,}")
    print(f"  Total training pairs         : {counts['total_pairs']:>10,}")
    print(f"  Eval queries (total)         : {counts['eval_queries']:>10,}")
    print(f"  Eval queries with ≥1 rel doc : {counts['eval_with_relevant']:>10,}")
    print(f"  Leakage overlap              : {overlap_count:>10,}  ✓")
    print(sep)
    print(f"\n  pairs.jsonl    → {PAIRS_JSONL}")
    print(f"  triplets.jsonl → {TRIPLETS_JSONL}")
    print(f"  qrels          → {QRELS_JSONL}")
    print(f"  Manifest       → {MANIFEST}")


if __name__ == "__main__":
    main()
