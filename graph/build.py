"""
Build the citation graph from edges.jsonl.

Uses scipy.sparse (CSR) rather than networkx: at PA scale (197K nodes, ~1.4M
edges) a networkx DiGraph consumes ~400 MB, while a scipy CSR matrix with the
same edges uses ~15 MB. Degree statistics are computed directly from the matrix.

Usage: python -m graph.build   (or: just graph, which runs resolve first)

Reads:  data/parsed/cases.jsonl  — for stable id ordering
        data/graph/edges.jsonl
Writes: data/graph/graph_manifest.json
        data/graph/id_to_idx.json
        data/graph/adj_in_corpus.npz   (scipy CSR, int8, deduplicated)
        data/graph/inv_index.json      {dst_id: [src_id, ...]} for all edges
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse

PARSED_JSONL = Path("data/parsed/cases.jsonl")
GRAPH_DIR    = Path("data/graph")
EDGES_JSONL  = GRAPH_DIR / "edges.jsonl"
MANIFEST     = GRAPH_DIR / "graph_manifest.json"
ID_TO_IDX    = GRAPH_DIR / "id_to_idx.json"
ADJ_FILE     = GRAPH_DIR / "adj_in_corpus.npz"
INV_INDEX    = GRAPH_DIR / "inv_index.json"


def _load_id_order(jsonl: Path) -> list[str]:
    ids: list[str] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
    return ids


def _deg_stats(arr: np.ndarray) -> dict:
    nonzero = int((arr > 0).sum())
    return {
        "mean_all"    : round(float(arr.mean()), 2),
        "mean_nonzero": round(float(arr[arr > 0].mean()), 2) if nonzero else 0.0,
        "p50"         : round(float(np.percentile(arr, 50)), 1),
        "p95"         : round(float(np.percentile(arr, 95)), 1),
        "p99"         : round(float(np.percentile(arr, 99)), 1),
        "max"         : int(arr.max()),
        "cases_nonzero": nonzero,
    }


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: stable id ordering ──────────────────────────────────────────
    print("Pass 1 — loading case id ordering from cases.jsonl ...")
    case_id_list = _load_id_order(PARSED_JSONL)
    n_corpus = len(case_id_list)
    id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(case_id_list)}
    print(f"  Corpus: {n_corpus:,} cases")

    # ── Pass 2: read edges ───────────────────────────────────────────────────
    print("\nPass 2 — reading edges.jsonl ...")
    rows_raw: list[int] = []
    cols_raw: list[int] = []
    # inv_index maps dst_id → set of in-corpus src_ids (all edges, including
    # out-of-corpus targets, because co-citation mining uses both)
    inv_index: dict[str, set[str]] = defaultdict(set)
    n_in_corpus = 0
    n_out_corpus = 0

    with EDGES_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            src, dst = e["src_id"], e["dst_id"]
            inv_index[dst].add(src)
            if e["in_corpus"]:
                n_in_corpus += 1
                si = id_to_idx.get(src)
                di = id_to_idx.get(dst)
                if si is not None and di is not None:
                    rows_raw.append(si)
                    cols_raw.append(di)
            else:
                n_out_corpus += 1
            if (i + 1) % 500_000 == 0:
                print(f"  ... {i+1:,} edges processed", flush=True)

    print(f"  In-corpus edges (raw)  : {n_in_corpus:,}")
    print(f"  Out-of-corpus edges    : {n_out_corpus:,}")

    # ── Deduplicate in-corpus edges ──────────────────────────────────────────
    print("\nDeduplicating in-corpus edges ...")
    rc_raw = np.column_stack([
        np.array(rows_raw, dtype=np.int32),
        np.array(cols_raw, dtype=np.int32),
    ])
    del rows_raw, cols_raw
    rc = np.unique(rc_raw, axis=0)
    del rc_raw
    n_unique = len(rc)
    print(f"  Unique in-corpus edges : {n_unique:,}")

    # ── Build CSR adjacency matrix ───────────────────────────────────────────
    print("\nBuilding CSR adjacency matrix ...")
    adj = scipy.sparse.csr_matrix(
        (np.ones(n_unique, dtype=np.int8), (rc[:, 0], rc[:, 1])),
        shape=(n_corpus, n_corpus),
        dtype=np.int8,
    )
    scipy.sparse.save_npz(str(ADJ_FILE), adj)
    print(f"  Saved: {ADJ_FILE}  shape={adj.shape}, nnz={adj.nnz:,}")

    # ── Degree statistics ────────────────────────────────────────────────────
    out_deg = np.array(adj.sum(axis=1), dtype=np.int32).flatten()
    in_deg  = np.array(adj.sum(axis=0), dtype=np.int32).flatten()

    # ── Co-citation statistics ───────────────────────────────────────────────
    # co-citation anchor = a cited authority cited by ≥2 PA corpus cases
    cocite_anchors = sum(1 for s in inv_index.values() if len(s) >= 2)
    cocite_pairs   = int(sum(
        len(s) * (len(s) - 1) // 2 for s in inv_index.values() if len(s) >= 2
    ))

    # ── Persist artifacts ────────────────────────────────────────────────────
    ID_TO_IDX.write_text(json.dumps(id_to_idx), "utf-8")

    print("\nSaving inverted index ...")
    inv_ser = {k: sorted(v) for k, v in inv_index.items()}
    INV_INDEX.write_text(json.dumps(inv_ser, ensure_ascii=False), "utf-8")
    print(f"  {len(inv_ser):,} unique citation targets")

    manifest = {
        "corpus_nodes"           : n_corpus,
        "in_corpus_edges_raw"    : n_in_corpus,
        "in_corpus_edges_unique" : n_unique,
        "out_of_corpus_edges"    : n_out_corpus,
        "total_edges"            : n_in_corpus + n_out_corpus,
        "out_degree"             : _deg_stats(out_deg),
        "in_degree"              : _deg_stats(in_deg),
        "cocite_anchors_ge2"     : cocite_anchors,
        "cocite_pairs_estimate"  : cocite_pairs,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    sep = "─" * 62
    print(f"\n{sep}")
    print("  GRAPH MANIFEST")
    print(sep)
    print(f"  Corpus nodes                    : {n_corpus:>10,}")
    print(f"  Unique in-corpus edges          : {n_unique:>10,}")
    print(f"  Out-of-corpus edges             : {n_out_corpus:>10,}")
    od = manifest["out_degree"]
    id_ = manifest["in_degree"]
    print(f"  Cases with out-degree ≥1        : {od['cases_nonzero']:>10,}  (candidate eval queries)")
    print(f"  Cases with in-degree ≥1         : {id_['cases_nonzero']:>10,}  (candidate targets)")
    print(f"  Out-degree: mean={od['mean_all']:.1f}, p95={od['p95']:.0f}, max={od['max']}")
    print(f"  In-degree:  mean={id_['mean_all']:.1f}, p95={id_['p95']:.0f}, max={id_['max']}")
    print(f"  Co-citation anchors (≥2 citers) : {cocite_anchors:>10,}")
    print(f"  Co-citation pairs (estimated)   : {cocite_pairs:>10,}")
    print(sep)
    print(f"\n  adj_in_corpus.npz → {ADJ_FILE}")
    print(f"  id_to_idx.json    → {ID_TO_IDX}")
    print(f"  inv_index.json    → {INV_INDEX}")
    print(f"  graph_manifest    → {MANIFEST}")


if __name__ == "__main__":
    main()
