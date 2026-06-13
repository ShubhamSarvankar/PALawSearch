"""
Map each cites_to entry to a canonical case ID and emit edges.jsonl.

Two resolution strategies (in priority order):
  cap_resolved   — CAP provided target_case_ids directly
  string_matched — raw_cite normalizes to a citation string in our corpus

Unresolvable entries (no case_ids, no string match) are counted but not emitted;
we have no reliable target ID to put in the graph.

Usage: python -m graph.resolve   (or: just resolve)

Reads:  data/parsed/cases.jsonl
Writes: data/graph/edges.jsonl
        data/graph/resolve_manifest.json
"""

import json
import re
from pathlib import Path

PARSED_JSONL = Path("data/parsed/cases.jsonl")
GRAPH_DIR    = Path("data/graph")
EDGES_JSONL  = GRAPH_DIR / "edges.jsonl"
MANIFEST     = GRAPH_DIR / "resolve_manifest.json"


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace — stable key for citation string matching."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def _build_lookup_and_ids(jsonl: Path) -> tuple[dict[str, str], set[str]]:
    """
    Single pass: build string lookup and collect corpus ids.

    Returns:
        lookup     — {normalized_cite_string: case_id}
        corpus_ids — set of all canonical case ids in the corpus
    """
    lookup: dict[str, str] = {}
    corpus_ids: set[str] = set()
    n = 0
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["id"]
            corpus_ids.add(cid)
            for cite_str in d.get("citations") or []:
                key = _norm(cite_str)
                if key and key not in lookup:
                    lookup[key] = cid
            n += 1
            if n % 50_000 == 0:
                print(f"  ... {n:,} cases scanned", flush=True)
    return lookup, corpus_ids


def _resolve_and_emit(
    jsonl: Path,
    lookup: dict[str, str],
    corpus_ids: set[str],
    out_path: Path,
) -> dict:
    counts = {
        "total_cites_to": 0,
        "cap_resolved_edges": 0,
        "string_matched_edges": 0,
        "unresolved": 0,
        "in_corpus_edges": 0,
        "out_of_corpus_edges": 0,
    }
    n = 0
    with jsonl.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            src = d["id"]
            for c in d.get("cites_to") or []:
                counts["total_cites_to"] += 1
                tids = c.get("target_case_ids") or []
                if tids:
                    for dst in tids:
                        in_corpus = dst in corpus_ids
                        counts["cap_resolved_edges"] += 1
                        if in_corpus:
                            counts["in_corpus_edges"] += 1
                        else:
                            counts["out_of_corpus_edges"] += 1
                        g.write(json.dumps({
                            "src_id": src,
                            "dst_id": dst,
                            "in_corpus": in_corpus,
                            "confidence": "cap_resolved",
                        }, ensure_ascii=False) + "\n")
                else:
                    key = _norm(c.get("raw_cite", ""))
                    if key and key in lookup:
                        dst = lookup[key]
                        counts["string_matched_edges"] += 1
                        counts["in_corpus_edges"] += 1
                        g.write(json.dumps({
                            "src_id": src,
                            "dst_id": dst,
                            "in_corpus": True,
                            "confidence": "string_matched",
                        }, ensure_ascii=False) + "\n")
                    else:
                        counts["unresolved"] += 1
            n += 1
            if n % 50_000 == 0:
                print(f"  ... {n:,} cases resolved", flush=True)
    return counts


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1 — building citation string lookup + collecting corpus ids ...")
    lookup, corpus_ids = _build_lookup_and_ids(PARSED_JSONL)
    print(f"  Corpus size    : {len(corpus_ids):,}")
    print(f"  Lookup entries : {len(lookup):,}")

    print("\nPass 2 — resolving citations → edges.jsonl ...")
    counts = _resolve_and_emit(PARSED_JSONL, lookup, corpus_ids, EDGES_JSONL)

    total_edges = counts["cap_resolved_edges"] + counts["string_matched_edges"]
    sep = "─" * 60
    print(f"\n{sep}")
    print("  RESOLUTION SUMMARY")
    print(sep)
    print(f"  Total cites_to entries  : {counts['total_cites_to']:>10,}")
    print(f"  CAP-resolved edges      : {counts['cap_resolved_edges']:>10,}")
    print(f"  String-matched edges    : {counts['string_matched_edges']:>10,}")
    print(f"  Unresolved (skipped)    : {counts['unresolved']:>10,}")
    print(f"  Total edges written     : {total_edges:>10,}")
    print(f"  In-corpus edges         : {counts['in_corpus_edges']:>10,}")
    print(f"  Out-of-corpus edges     : {counts['out_of_corpus_edges']:>10,}")
    print(sep)

    manifest = {
        "corpus_size": len(corpus_ids),
        "lookup_size": len(lookup),
        **counts,
        "total_edges": total_edges,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"\n  Manifest → {MANIFEST}")
    print(f"  Edges    → {EDGES_JSONL}")


if __name__ == "__main__":
    main()
