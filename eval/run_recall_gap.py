"""Recall-gap estimation via the validated LLM judge (κ=0.773, rubric v2).

Quantifies how much citation qrels undercount true topical relevance.

Method
------
1. Sample N_QUERIES eval queries at random (seed 42).
2. For each, compute trained-legalbert hybrid_rrf top-10 from cached embeddings + BM25 run.
3. Keep only docs NOT in the citation qrels for that query (the non-cited top-10).
4. Skip content-free opinions (extracted text < CONTENT_FREE_MIN_CHARS chars on either side).
5. Judge remaining pairs with rubric v2; compute fraction the judge calls RELEVANT.
6. Report rate + 95% bootstrap CI.

Usage
-----
    python -m eval.run_recall_gap --dry-run   # count API calls, do not call judge
    python -m eval.run_recall_gap --run        # count + run + write recall_gap.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from eval.judge import (
    RUBRIC_VERSION,
    JUDGE_MODEL,
    QUERY_TEXT_MAX,
    DOC_TEXT_MAX,
    _extract_text,
    call_judge,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent

QRELS_PATH    = ROOT / "data/eval/qrels_citation.jsonl"
BM25_RUN_PATH = ROOT / "data/eval/runs/bm25.jsonl"
CASES_PATH    = ROOT / "data/parsed/cases.jsonl"
EMBED_DIR     = ROOT / "data/eval/embeddings"
OUT_PATH      = ROOT / "data/eval/recall_gap.json"

SAMPLE_SEED  = 42
N_QUERIES    = 80
TOP_K        = 10
RRF_K        = 60    # must match run_eval.py
DENSE_TOP_K  = 200   # must match run_eval.py

# Trained-legalbert slug (mirrors run_eval._embed_slug logic)
_LEGALBERT_PATH = str(
    ROOT / "models/checkpoints/encoder/legalbert-20260608-144233/model"
)
_EMBED_SLUG = _LEGALBERT_PATH.replace("/", "_").replace("\\", "_").replace(":", "_")

# Content-free detection aligns with the manual exclusion criteria from judge validation
# (κ=0.773 frozen on the 76-pair clean subset).  No length threshold — purely content-based.
# Patterns derived from the 4 validated exclusions plus corpus-wide analysis of the 80-query sample.

BOOTSTRAP_N    = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_qrels() -> tuple[dict[str, set[str]], dict[str, str]]:
    """Returns (qrels, head_matters) where head_matters maps qid -> raw head_matter text."""
    qrels: dict[str, set[str]] = {}
    query_text: dict[str, str] = {}
    with open(QRELS_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["query_id"]
            qrels[qid] = set(obj["relevant_doc_ids"])
            query_text[qid] = obj.get("query_text", "")
    return qrels, query_text


def _load_bm25_run(qids: set[str]) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    with open(BM25_RUN_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["qid"]
            if qid in qids:
                run[qid] = [d for d in obj["doc_ids"] if d != qid]
    return run


def _load_embeddings(
    qids: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Load cached trained-legalbert embeddings. Returns (corpus_vecs, corpus_ids,
    sampled_query_vecs, sampled_qid_order)."""
    print("  Loading corpus embeddings ...", flush=True)
    corp = np.load(EMBED_DIR / f"{_EMBED_SLUG}.npz")
    corpus_ids: list[str] = corp["ids"].tolist()
    corpus_vecs: np.ndarray = corp["vecs"]  # (N_corpus, 768) float32, L2-normalised

    print("  Loading query embeddings ...", flush=True)
    qry = np.load(EMBED_DIR / f"{_EMBED_SLUG}_queries.npz")
    all_qids: list[str] = qry["qids"].tolist()
    all_qvecs: np.ndarray = qry["vecs"]  # (N_queries, 768) float32

    # Filter to the sampled qids (preserving order)
    qid_set = set(qids)
    idx_map: dict[str, int] = {qid: i for i, qid in enumerate(all_qids)}
    sampled_idx = [idx_map[q] for q in qids if q in idx_map]
    sampled_qid_order = [qids[i] for i, q in enumerate(qids) if q in idx_map]
    sampled_query_vecs = all_qvecs[sampled_idx]  # (80, 768)

    missing = set(qids) - set(sampled_qid_order)
    if missing:
        print(f"  WARNING: {len(missing)} sampled qids not found in query embedding cache")

    print(
        f"  Loaded: corpus={len(corpus_ids):,} docs, {sampled_query_vecs.shape[1]}d; "
        f"queries={len(sampled_qid_order)}"
    )
    return corpus_vecs, corpus_ids, sampled_query_vecs, sampled_qid_order


def _dense_retrieve(
    corpus_ids: list[str],
    corpus_vecs: np.ndarray,
    query_vecs: np.ndarray,
    qid_order: list[str],
    top_k: int = DENSE_TOP_K,
) -> dict[str, list[str]]:
    """Brute-force top-k cosine retrieval over the 80-query subset. No GPU needed."""
    corpus_arr = np.array(corpus_ids)
    id_to_idx: dict[str, int] = {did: i for i, did in enumerate(corpus_ids)}
    run: dict[str, list[str]] = {}

    # Batch all 80 queries at once — matmul is (80, 197130) which is fine in RAM
    scores = query_vecs @ corpus_vecs.T  # (80, N_corpus)
    for bi, qid in enumerate(qid_order):
        row = scores[bi].copy()
        if qid in id_to_idx:
            row[id_to_idx[qid]] = -2.0  # exclude self
        top_idx = np.argpartition(row, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        run[qid] = corpus_arr[top_idx].tolist()

    return run


def _rrf_fuse(
    bm25_run: dict[str, list[str]],
    dense_run: dict[str, list[str]],
    k: int = RRF_K,
) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for qid in dense_run:
        bm25_ranks = {did: r + 1 for r, did in enumerate(bm25_run.get(qid, []))}
        dense_ranks = {did: r + 1 for r, did in enumerate(dense_run.get(qid, []))}
        all_ids = set(bm25_ranks) | set(dense_ranks)
        scores = {
            did: (1.0 / (k + bm25_ranks[did]) if did in bm25_ranks else 0.0)
                 + (1.0 / (k + dense_ranks[did]) if did in dense_ranks else 0.0)
            for did in all_ids
        }
        run[qid] = sorted(scores, key=scores.__getitem__, reverse=True)
    return run


def _load_case_texts(needed_ids: set[str]) -> dict[str, dict]:
    """Single-pass through cases.jsonl; returns {id: {head_matter, full_text}}."""
    print(f"  Scanning cases.jsonl for {len(needed_ids)} cases ...", flush=True)
    texts: dict[str, dict] = {}
    found = 0
    with open(CASES_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 50_000 == 0 and i > 0:
                print(f"    {i:,} lines scanned, {found}/{len(needed_ids)} found", flush=True)
            obj = json.loads(line)
            if obj["id"] in needed_ids:
                texts[obj["id"]] = {
                    "head_matter": obj.get("head_matter", ""),
                    "full_text":   obj.get("full_text", ""),
                }
                found += 1
                if found == len(needed_ids):
                    break
    print(f"  Done: {found}/{len(needed_ids)} cases found.")
    return texts


# ---------------------------------------------------------------------------
# Content-free detection
# ---------------------------------------------------------------------------

def _is_content_free(text: str) -> bool:
    """Detect per curiam orders, memorandum affirmances, and pointer opinions.

    Criteria are content-based (not length-based) and align with the 4 manually
    validated exclusions from judge validation (κ=0.773, rubric v2).

    Returns True if the extracted text has no judgeable legal content.
    """
    import re as _re
    t = text.strip()

    # 1. Empty — trivially unjudgeable
    if len(t) < 10:
        return True

    first_400 = t[:400]

    # 2. "AND NOW, this" in first 400 chars — canonical PA court order boilerplate.
    #    Exception: a named judge author ("OPINION BY [name]") before this signal
    #    indicates a substantive opinion styled as an order (e.g. Commonwealth Court).
    if _re.search(r"(?i)\bAND\s+NOW,\s+this\b", first_400):
        if not _re.search(r"(?i)\bopinion\s+by\b", first_400):
            return True

    # 3. ORDER opener with no named judge author — pure procedural order.
    #    "OPINION BY" before the disposition signals a substantive opinion.
    if _re.match(r"(?i)^order\b", t) and not _re.search(r"(?i)\bopinion\s+by\b", t[:300]):
        return True

    # 4. Pointer-to-lower-court formula.  "affirmed on the opinion/basis of" near
    #    the start is the whole reason — it IS the disposition, not a passing reference.
    if _re.search(r"(?i)\baffirm(ed)?\s+on\s+the\s+(opinion|basis)\s+of\b", t[:500]):
        return True

    # 5. PER CURIAM immediately followed by a bare disposition (no analysis follows).
    #    Catches "OPINION OF THE COURT\nPER CURIAM:\nOrder affirmed." and
    #    "Opinion\nPer Curiam :\nThe judgment of sentence is vacated and remanded."
    if _re.search(
        r"(?i)\bper\s*curi[ao]m[^a-z\n]{0,10}\n?\s*(the\s+)?(order|judgment[s]?)"
        r"(\s+of\s+sentence)?\s+(is\s+)?(affirmed|reversed|vacated|remanded)",
        t,
    ):
        return True

    # 6. Standalone one-line affirmances / remands (the entire text is the disposition).
    if _re.match(
        r"(?i)^(judgment[s]?\s+of\s+sentence\s+(are\s+)?affirmed"
        r"|the\s+order\s+of\s+the\s+(lower|trial|court)\b.*?\bis\s+affirmed"
        r"|order\s+affirmed"
        r"|remanded\s+for\b)",
        t,
    ):
        return True

    # 7. PA court determination / pointer-quash opener.
    #    "It appearing that [...], it is hereby ordered" — court order form.
    #    "It appearing that [...], this appeal is quashed...remanded. See [case]." — pointer.
    if _re.match(r"(?i)^it\s+appearing\s+that\b", t) and _re.search(
        r"(?i)(\bis\s+hereby\s+(ordered|adjudged|decreed)\b"
        r"|\bthis\s+appeal\s+is\s+(quashed|dismissed)\b)",
        t[:500],
    ):
        return True

    return False


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_proportion_ci(
    hits: int,
    n: int,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> tuple[float, float]:
    """95% bootstrap CI on a proportion via resampling a Bernoulli vector."""
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, hits / n, size=n_resamples) / n
    lo = float(np.percentile(draws, (1 - ci) / 2 * 100))
    hi = float(np.percentile(draws, (1 + ci) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_pairs(
    qid_sample: list[str],
    qrels: dict[str, set[str]],
    hybrid_run: dict[str, list[str]],
    case_texts: dict[str, dict],
) -> tuple[list[dict], dict[str, int], int]:
    """
    Build (query, non-cited-doc) pairs, applying content-free exclusion.

    Returns (pairs, per_query_counts, n_excluded_content_free).
    Each pair: {qid, doc_id, query_text, doc_text}.
    per_query_counts: {qid: n_pairs_after_exclusion}
    """
    pairs: list[dict] = []
    per_query: dict[str, int] = {}
    n_excluded = 0

    for qid in qid_sample:
        top10 = hybrid_run.get(qid, [])[:TOP_K]
        rel_set = qrels.get(qid, set())
        non_cited = [d for d in top10 if d not in rel_set]

        q_case = case_texts.get(qid, {})
        q_extracted = _extract_text(
            q_case.get("head_matter", ""),
            q_case.get("full_text", ""),
            QUERY_TEXT_MAX,
        )

        accepted = 0
        for doc_id in non_cited:
            d_case = case_texts.get(doc_id, {})
            d_extracted = _extract_text(
                d_case.get("head_matter", ""),
                d_case.get("full_text", ""),
                DOC_TEXT_MAX,
            )

            if _is_content_free(q_extracted) or _is_content_free(d_extracted):
                n_excluded += 1
                continue

            pairs.append({
                "qid":        qid,
                "doc_id":     doc_id,
                "query_text": q_extracted,
                "doc_text":   d_extracted,
            })
            accepted += 1

        per_query[qid] = accepted

    return pairs, per_query, n_excluded


def run_judge(pairs: list[dict]) -> list[dict]:
    """Call the LLM judge on each pair. Returns list of result dicts."""
    results: list[dict] = []
    total = len(pairs)
    for i, pair in enumerate(pairs):
        print(f"  [{i+1:3d}/{total}] qid={pair['qid']} doc={pair['doc_id']} ...",
              end=" ", flush=True)
        result = call_judge(pair["query_text"], pair["doc_text"])
        results.append({
            "qid":       pair["qid"],
            "doc_id":    pair["doc_id"],
            "label":     result["label"],
            "judge_raw": result["raw"],
        })
        print(result["label"])
    return results


def compute_gap(results: list[dict]) -> dict:
    """Compute recall-gap rate + bootstrap CI from judge results."""
    valid = [r for r in results if r["label"] != "PARSE_ERROR"]
    parse_errors = [r for r in results if r["label"] == "PARSE_ERROR"]
    n = len(valid)
    n_relevant = sum(1 for r in valid if r["label"] == "RELEVANT")
    rate = n_relevant / n if n > 0 else 0.0
    ci_lo, ci_hi = _bootstrap_proportion_ci(n_relevant, n)
    return {
        "recall_gap_rate":        round(rate, 4),
        "ci_95_lo":               round(ci_lo, 4),
        "ci_95_hi":               round(ci_hi, 4),
        "n_judged":               n,
        "n_relevant":             n_relevant,
        "n_not_relevant":         n - n_relevant,
        "n_parse_errors":         len(parse_errors),
        "parse_error_pairs":      [{"qid": r["qid"], "doc_id": r["doc_id"]} for r in parse_errors],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool) -> None:
    t0 = time.time()
    mode = "DRY-RUN" if dry_run else "RUN"
    print(f"\n=== Recall-gap estimation [{mode}] ===\n")

    # 1. Load qrels and sample 80 queries
    print("Loading qrels ...", flush=True)
    qrels, _query_texts = _load_qrels()
    all_qids = sorted(qrels.keys())
    rng = random.Random(SAMPLE_SEED)
    qid_sample = rng.sample(all_qids, N_QUERIES)
    print(f"Sampled {N_QUERIES} queries (seed={SAMPLE_SEED}) from {len(all_qids):,} eval queries.")

    # 2. Load BM25 run (only the 80 sampled)
    print("Loading BM25 run ...", flush=True)
    bm25_run = _load_bm25_run(set(qid_sample))
    print(f"  BM25 run loaded for {len(bm25_run)} queries.")

    # 3. Load embeddings and compute hybrid_rrf
    print("Loading embeddings ...", flush=True)
    corpus_vecs, corpus_ids, query_vecs, qid_order = _load_embeddings(qid_sample)

    print("Computing dense retrieval (brute-force cosine, no GPU needed) ...", flush=True)
    dense_run = _dense_retrieve(corpus_ids, corpus_vecs, query_vecs, qid_order)

    print("Computing hybrid_rrf (RRF fusion) ...", flush=True)
    hybrid_run = _rrf_fuse(bm25_run, dense_run)

    # 4. Find non-cited top-10 docs for each query
    needed_ids: set[str] = set(qid_sample)
    non_cited_per_query: dict[str, list[str]] = {}
    for qid in qid_sample:
        top10 = hybrid_run.get(qid, [])[:TOP_K]
        rel_set = qrels.get(qid, set())
        non_cited = [d for d in top10 if d not in rel_set]
        non_cited_per_query[qid] = non_cited
        needed_ids.update(top10)  # need texts for full top-10 to do content-free check

    n_non_cited_total = sum(len(v) for v in non_cited_per_query.values())
    print(
        f"\nTop-10 non-cited doc counts across {N_QUERIES} queries: "
        f"total={n_non_cited_total}, avg={n_non_cited_total/N_QUERIES:.1f}/query"
    )

    # 5. Load case texts for all involved cases
    print("Loading case texts ...", flush=True)
    case_texts = _load_case_texts(needed_ids)

    # 6. Build pairs with content-free exclusion
    pairs, per_query_counts, n_cf_excluded = build_pairs(
        qid_sample, qrels, hybrid_run, case_texts
    )
    n_total_api_calls = len(pairs)

    # 7. Report per-query call counts
    print("\n--- Per-query API call plan ---")
    print(f"  {'qid':>12}  {'non-cited':>9}  {'after-excl':>10}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*10}")
    for qid in qid_sample:
        nc = len(non_cited_per_query[qid])
        ac = per_query_counts.get(qid, 0)
        print(f"  {qid:>12}  {nc:>9}  {ac:>10}")

    print(f"\n--- Summary ---")
    print(f"  Queries sampled:              {N_QUERIES}")
    print(f"  Non-cited top-10 docs total:  {n_non_cited_total}")
    print(f"  Excluded as content-free:     {n_cf_excluded}  (content-based: orders, pointers, one-liners)")
    print(f"  TOTAL API CALLS PLANNED:      {n_total_api_calls}")
    print(f"  Judge model:                  {JUDGE_MODEL}")
    print(f"  Rubric version:               {RUBRIC_VERSION}")

    if dry_run:
        print("\n[DRY-RUN complete — no API calls made. Re-run with --run to execute.]\n")
        return

    # 8. Run judge
    print(f"\nRunning judge on {n_total_api_calls} pairs ...\n")
    results = run_judge(pairs)

    # 9. Check for PARSE_ERRORs
    parse_errors = [r for r in results if r["label"] == "PARSE_ERROR"]
    if parse_errors:
        print(f"\nWARNING: {len(parse_errors)} PARSE_ERROR labels:")
        for pe in parse_errors:
            print(f"  qid={pe['qid']} doc={pe['doc_id']}")
    else:
        print("\nNo PARSE_ERRORs.")

    # 10. Compute recall gap
    gap = compute_gap(results)
    rate = gap["recall_gap_rate"]
    print(
        f"\n=== Recall-gap result ===\n"
        f"  Rate:     {rate:.3f}  ({gap['n_relevant']}/{gap['n_judged']} "
        f"non-cited top-10 docs judged RELEVANT)\n"
        f"  95% CI:   [{gap['ci_95_lo']:.3f}, {gap['ci_95_hi']:.3f}]  "
        f"(bootstrap, N={BOOTSTRAP_N:,}, seed={BOOTSTRAP_SEED})\n"
        f"  n_judged: {gap['n_judged']}  |  n_excluded_content_free: {n_cf_excluded}"
    )

    # 11. Write output
    output = {
        "method":               "trained-legalbert hybrid_rrf top-10, non-cited docs",
        "judge_model":          JUDGE_MODEL,
        "rubric_version":       RUBRIC_VERSION,
        "sample_seed":          SAMPLE_SEED,
        "n_queries":            N_QUERIES,
        "top_k":                TOP_K,
        "content_free_criteria": "content-based: AND_NOW, ORDER_no_author, pointer_affirmed_on, percuriam_oneliner, oneliner, it_appearing",
        "n_non_cited_total":    n_non_cited_total,
        "n_excluded_content_free": n_cf_excluded,
        "n_judged":             gap["n_judged"],
        "n_relevant":           gap["n_relevant"],
        "n_not_relevant":       gap["n_not_relevant"],
        "n_parse_errors":       gap["n_parse_errors"],
        "recall_gap_rate":      gap["recall_gap_rate"],
        "ci_95_lo":             gap["ci_95_lo"],
        "ci_95_hi":             gap["ci_95_hi"],
        "parse_error_pairs":    gap["parse_error_pairs"],
        "per_query_results":    results,
        "wall_clock_seconds":   round(time.time() - t0, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults written: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge recall-gap estimation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="Count API calls only; do not call the judge")
    group.add_argument("--run", action="store_true",
                       help="Run the judge and write recall_gap.json")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
