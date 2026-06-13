"""
Citation-grounded eval harness.

Systems
-------
  frozen-minilm            sentence-transformers/all-MiniLM-L6-v2
  frozen-legalbert         nlpaueb/legal-bert-base-uncased
  trained-minilm           models/checkpoints/encoder/minilm-20260605-130458/model
  trained-legalbert        models/checkpoints/encoder/legalbert-20260608-144233/model
  trained-legalbert+reranker  trained-legalbert dense + cross-encoder reranker

Methods per run
---------------
  bm25          BM25-only anchor (same ES index for all systems)
  dense         brute-force cosine against offline corpus embeddings
  dense_rerank  top-200 dense candidates -> cross-encoder rerank
  bm25_rerank   top-200 BM25 candidates  -> cross-encoder rerank
  hybrid_rrf    RRF(bm25, dense), k=60

Outputs
-------
  data/eval/results.csv          long-format: system, method, query_set, metric, value
  data/eval/per_query_ndcg.jsonl one JSON line per system+method: {system, method, scores:{qid:f}}

Caches (skipped if present)
----------------------------
  data/eval/runs/bm25.jsonl
  data/eval/embeddings/{slug}.npz        corpus embeddings (corpus_ids, vecs)
  data/eval/embeddings/{slug}_queries.npz query embeddings (qids, vecs)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QRELS_PATH   = ROOT / "data/eval/qrels_citation.jsonl"
CORPUS_PATH  = ROOT / "data/parsed/cases.jsonl"
EMBED_DIR    = ROOT / "data/eval/embeddings"
RUNS_DIR     = ROOT / "data/eval/runs"
OUT_CSV      = ROOT / "data/eval/results.csv"
OUT_NDCG     = ROOT / "data/eval/per_query_ndcg.jsonl"

RERANKER_CKPT = ROOT / "models/checkpoints/reranker/20260609-130602/model"
ENCODERS = {
    "frozen-minilm":    "sentence-transformers/all-MiniLM-L6-v2",
    "frozen-legalbert": "nlpaueb/legal-bert-base-uncased",
    "trained-minilm":   str(ROOT / "models/checkpoints/encoder/minilm-20260605-130458/model"),
    "trained-legalbert": str(ROOT / "models/checkpoints/encoder/legalbert-20260608-144233/model"),
}

RERANKER_SYSTEM = "trained-legalbert+reranker"

BM25_TOP_K    = 200
DENSE_TOP_K   = 200
RERANK_TOP_K  = 200
RRF_K         = 60
EMBED_BATCH   = 256
DENSE_Q_BATCH = 128   # queries per cosine-sim batch
BM25_WORKERS  = 16
BOOTSTRAP_N   = 10_000
BOOTSTRAP_SEED = 42
DOC_EMBED_MAX  = 4096  # chars fed to encoder per corpus doc
DOC_RERANK_MAX = 2000  # chars fed to cross-encoder per candidate doc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(0) / 1e9
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Preflight checks — fail fast before the 197k corpus load
# ---------------------------------------------------------------------------

MIN_FREE_VRAM_GB = 6.0


def preflight() -> None:
    """Assert ES reachable + index exists + CUDA available. Abort loudly on failure."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    from config.settings import settings
    from elasticsearch import Elasticsearch

    errors: list[str] = []

    # --- ES reachability ---
    try:
        es = Elasticsearch(
            settings.es_host,
            basic_auth=(settings.es_user, settings.es_password),
            verify_certs=False,
            request_timeout=10,
        )
        health = es.cluster.health()
        status = health.get("status", "unknown")
        if status not in ("green", "yellow"):
            errors.append(f"ES cluster status is '{status}' (need green or yellow)")
        else:
            _log(f"Preflight: ES cluster status = {status}")
    except Exception as exc:
        errors.append(f"ES unreachable at {settings.es_host}: {exc}")
        # Cannot check index if cluster is unreachable
        _log(f"PREFLIGHT FAIL: {errors[-1]}")
        _abort(errors)

    # --- BM25 index exists and is non-empty ---
    try:
        if not es.indices.exists(index=settings.es_index_bm25):
            errors.append(f"Index '{settings.es_index_bm25}' does not exist — run: just index-bm25")
        else:
            stats = es.indices.stats(index=settings.es_index_bm25)
            doc_count = stats["_all"]["primaries"]["docs"]["count"]
            if doc_count == 0:
                errors.append(f"Index '{settings.es_index_bm25}' exists but has 0 docs — run: just index-bm25")
            else:
                _log(f"Preflight: {settings.es_index_bm25} has {doc_count:,} docs")
    except Exception as exc:
        errors.append(f"Could not check index '{settings.es_index_bm25}': {exc}")

    # --- CUDA + free VRAM ---
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available — GPU required for offline embedding")
        else:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1e9
            reserved_gb = torch.cuda.memory_reserved(0) / 1e9
            free_gb = total_gb - reserved_gb
            _log(f"Preflight: CUDA device={props.name}, total={total_gb:.1f}GB, free={free_gb:.1f}GB")
            if free_gb < MIN_FREE_VRAM_GB:
                errors.append(
                    f"Only {free_gb:.1f}GB VRAM free; need >= {MIN_FREE_VRAM_GB}GB — run: just gpu-free"
                )
    except Exception as exc:
        errors.append(f"Could not check CUDA: {exc}")

    _abort(errors)


def _abort(errors: list[str]) -> None:
    if errors:
        print("\n=== PREFLIGHT FAILED ===", flush=True)
        for e in errors:
            print(f"  ERROR: {e}", flush=True)
        print("========================\n", flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Returns (qrels, queries) where queries maps qid -> query_text."""
    qrels: dict[str, set[str]] = {}
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            qrels[qid] = set(rec["relevant_doc_ids"])
            queries[qid] = rec["query_text"]
    return qrels, queries


def load_corpus(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Returns (corpus_ids, corpus_embed_texts, corpus_rerank_texts).
    corpus_rerank_texts maps id -> truncated text for CE scoring.
    """
    _log(f"Loading corpus from {path} ...")
    corpus_ids: list[str] = []
    embed_texts: list[str] = []
    rerank_lookup: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            did = doc["id"]
            text = doc.get("full_text") or doc.get("head_matter") or ""
            corpus_ids.append(did)
            embed_texts.append(text[:DOC_EMBED_MAX])
            rerank_lookup[did] = text[:DOC_RERANK_MAX]
    _log(f"  {len(corpus_ids):,} corpus docs loaded")
    return corpus_ids, embed_texts, rerank_lookup


# ---------------------------------------------------------------------------
# BM25 via ES (cached)
# ---------------------------------------------------------------------------

def _bm25_one(query_text: str, qid: str, es_client: Any, index: str) -> tuple[str, list[str]]:
    body = {
        "query": {
            "multi_match": {
                "query": query_text,
                "type": "best_fields",
                "fields": ["name^4", "parties^3", "head_matter^3", "full_text^1"],
            }
        },
        "size": BM25_TOP_K,
        "_source": ["id"],
    }
    resp = es_client.search(index=index, body=body, request_timeout=60)
    ids = [h["_source"]["id"] for h in resp["hits"]["hits"]]
    return qid, ids


def build_bm25_run(
    queries: dict[str, str],
    cache_path: Path,
) -> dict[str, list[str]]:
    if cache_path.exists():
        _log(f"Loading cached BM25 run from {cache_path}")
        run: dict[str, list[str]] = {}
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                run[rec["qid"]] = rec["doc_ids"]
        _log(f"  {len(run):,} BM25 queries loaded from cache")
        return run

    _log("Building BM25 run via ES ...")
    from config.settings import settings
    from elasticsearch import Elasticsearch

    es = Elasticsearch(
        settings.es_host,
        basic_auth=(settings.es_user, settings.es_password),
        verify_certs=False,
        request_timeout=60,
    )
    index = settings.es_index_bm25

    run: dict[str, list[str]] = {}
    qid_list = list(queries.keys())
    total = len(qid_list)
    done = 0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=BM25_WORKERS) as pool:
            futures = {
                pool.submit(_bm25_one, queries[qid], qid, es, index): qid
                for qid in qid_list
            }
            for fut in as_completed(futures):
                qid, ids = fut.result()
                run[qid] = ids
                fout.write(json.dumps({"qid": qid, "doc_ids": ids}) + "\n")
                done += 1
                if done % 500 == 0:
                    _log(f"  BM25: {done}/{total} queries done")

    tmp_path.rename(cache_path)
    _log(f"  BM25 run complete ({total} queries), cached to {cache_path}")
    return run


# ---------------------------------------------------------------------------
# Dense embeddings (cached per encoder)
# ---------------------------------------------------------------------------

def _embed_slug(encoder_name: str) -> str:
    return encoder_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def build_embeddings(
    encoder_name: str,
    corpus_ids: list[str],
    corpus_texts: list[str],
    queries: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (corpus_vecs, query_vecs, query_id_order).
    corpus_vecs: (N_corpus, dim) float32, L2-normalized
    query_vecs:  (N_queries, dim) float32, L2-normalized
    """
    import torch
    from sentence_transformers import SentenceTransformer

    slug = _embed_slug(encoder_name)
    corpus_cache = EMBED_DIR / f"{slug}.npz"
    query_cache  = EMBED_DIR / f"{slug}_queries.npz"
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # --- corpus embeddings ---
    if corpus_cache.exists():
        _log(f"Loading cached corpus embeddings: {corpus_cache.name}")
        data = np.load(corpus_cache)
        corpus_vecs = data["vecs"]
    else:
        _log(f"Embedding corpus with {encoder_name} ({len(corpus_texts):,} docs) ...")
        model = SentenceTransformer(encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
        corpus_vecs = model.encode(
            corpus_texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.savez_compressed(corpus_cache, ids=np.array(corpus_ids), vecs=corpus_vecs)
        _log(f"  Corpus embeddings saved: {corpus_cache.name}, shape={corpus_vecs.shape}, VRAM={_vram_gb():.2f}GB")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- query embeddings ---
    qid_order = list(queries.keys())
    query_texts = [queries[q] for q in qid_order]

    if query_cache.exists():
        _log(f"Loading cached query embeddings: {query_cache.name}")
        data = np.load(query_cache)
        query_vecs = data["vecs"]
    else:
        _log(f"Embedding {len(query_texts):,} queries ...")
        model = SentenceTransformer(encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
        query_vecs = model.encode(
            query_texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.savez_compressed(query_cache, qids=np.array(qid_order), vecs=query_vecs)
        _log(f"  Query embeddings saved: {query_cache.name}, shape={query_vecs.shape}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return corpus_vecs, query_vecs, qid_order


# ---------------------------------------------------------------------------
# Dense retrieval (brute-force cosine)
# ---------------------------------------------------------------------------

def dense_retrieve(
    corpus_ids: list[str],
    corpus_vecs: np.ndarray,
    query_vecs: np.ndarray,
    qid_order: list[str],
    top_k: int = DENSE_TOP_K,
    exclude_self: bool = True,
) -> dict[str, list[str]]:
    """Brute-force top-k cosine retrieval. corpus_vecs and query_vecs are pre-normalized."""
    n_queries = len(qid_order)
    run: dict[str, list[str]] = {}
    corpus_arr = np.array(corpus_ids)
    # O(1) self-exclusion lookup
    id_to_idx: dict[str, int] = {did: i for i, did in enumerate(corpus_ids)}

    for start in range(0, n_queries, DENSE_Q_BATCH):
        end = min(start + DENSE_Q_BATCH, n_queries)
        q_batch = query_vecs[start:end]          # (B, dim)
        scores = q_batch @ corpus_vecs.T          # (B, N_corpus)

        for bi in range(end - start):
            qid = qid_order[start + bi]
            row = scores[bi]
            if exclude_self and qid in id_to_idx:
                row = row.copy()
                row[id_to_idx[qid]] = -2.0

            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            run[qid] = corpus_arr[top_idx].tolist()

        if (start // DENSE_Q_BATCH) % 10 == 0:
            _log(f"  dense: {end}/{n_queries} queries")

    return run


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    bm25_run: dict[str, list[str]],
    dense_run: dict[str, list[str]],
    k: int = RRF_K,
) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for qid in bm25_run:
        bm25_ranks = {did: rank + 1 for rank, did in enumerate(bm25_run.get(qid, []))}
        dense_ranks = {did: rank + 1 for rank, did in enumerate(dense_run.get(qid, []))}
        all_ids = set(bm25_ranks) | set(dense_ranks)
        scores = {
            did: (1.0 / (k + bm25_ranks[did]) if did in bm25_ranks else 0.0)
                 + (1.0 / (k + dense_ranks[did]) if did in dense_ranks else 0.0)
            for did in all_ids
        }
        run[qid] = sorted(scores, key=scores.__getitem__, reverse=True)
    return run


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def runs_to_metrics(
    runs: dict[tuple[str, str], dict[str, list[str]]],
    qrels: dict[str, set[str]],
) -> tuple[list[dict], dict[tuple[str, str], dict[str, float]]]:
    """
    Returns (rows_for_csv, per_query_ndcg10_by_system_method).
    rows_for_csv: list of {system, method, query_set, metric, value}
    """
    from eval.metrics import compute_all, per_query_ndcg10

    csv_rows: list[dict] = []
    pq_ndcg: dict[tuple[str, str], dict[str, float]] = {}

    for (system, method), run in runs.items():
        m = compute_all(qrels, run)
        for metric, value in m.items():
            csv_rows.append({
                "system": system,
                "method": method,
                "query_set": "all",
                "metric": metric,
                "value": value,
            })
        pq_ndcg[(system, method)] = per_query_ndcg10(qrels, run)

    return csv_rows, pq_ndcg


# ---------------------------------------------------------------------------
# Paired bootstrap significance test
# ---------------------------------------------------------------------------

def bootstrap_p(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float, float, float, int]:
    """
    Standard paired bootstrap significance test (one-sided, H1: mean_b > mean_a).

    For each resample: draw N query indices with replacement, compute mean(b-a).
    p = fraction of resamples where boot_mean_delta <= 0.
    CI = 95th percentile interval of the bootstrapped mean deltas.

    Returns (mean_a, mean_b, observed_delta, ci_lo, ci_hi, p_value, N).
    """
    assert len(scores_a) == len(scores_b), "score lists must have equal length"
    N = len(scores_a)
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    observed_delta = float(b.mean() - a.mean())

    rng = np.random.default_rng(seed)
    diffs = b - a  # per-query deltas, shape (N,)

    # Vectorised bootstrap in batches to keep memory < ~200 MB peak
    _BATCH = 500
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _BATCH):
        end = min(start + _BATCH, n_resamples)
        idx = rng.integers(0, N, size=(end - start, N), dtype=np.int32)
        boot_means[start:end] = diffs[idx].mean(axis=1)

    p = float((boot_means <= 0.0).mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return float(a.mean()), float(b.mean()), observed_delta, ci_lo, ci_hi, p, N


def run_significance(
    pq_ndcg: dict[tuple[str, str], dict[str, float]],
    qrels: dict[str, set[str]],
) -> None:
    """Paired bootstrap significance for key system comparisons (nDCG@10)."""
    pairs = [
        ("frozen-minilm",     "dense",        "trained-minilm",             "dense",        "trained-minilm vs frozen-minilm"),
        ("frozen-legalbert",  "dense",        "trained-legalbert",          "dense",        "trained-legalbert vs frozen-legalbert"),
        ("trained-minilm",    "dense",        "trained-legalbert",          "dense",        "trained-legalbert vs trained-minilm"),
        ("trained-legalbert", "dense",        "trained-legalbert+reranker", "dense_rerank", "reranker vs trained-legalbert"),
    ]
    qids = sorted(qrels.keys())
    print("\n=== Paired bootstrap significance (nDCG@10, N=10,000 resamples, seed=42) ===")
    print(f"  {'comparison':<43}  {'delta':>7}  {'95% CI':>19}  {'p':>6}")
    print("  " + "-" * 82)
    for sys_a, method_a, sys_b, method_b, label in pairs:
        key_a = (sys_a, method_a)
        key_b = (sys_b, method_b)
        if key_a not in pq_ndcg or key_b not in pq_ndcg:
            print(f"  {label}: MISSING RUN (need {key_a} and/or {key_b})")
            continue
        a_scores = [pq_ndcg[key_a].get(q, 0.0) for q in qids]
        b_scores = [pq_ndcg[key_b].get(q, 0.0) for q in qids]
        mean_a, mean_b, delta, ci_lo, ci_hi, p, N = bootstrap_p(a_scores, b_scores)
        sig = " *" if p < 0.05 else ""
        print(
            f"  {label:<43}  {delta:>+7.4f}  [{ci_lo:>+.4f}, {ci_hi:>+.4f}]  p={p:.4f}{sig}"
        )
        print(
            f"    {sys_a}:{method_a} = {mean_a:.4f}  ->  {sys_b}:{method_b} = {mean_b:.4f}"
            f"  (N={N:,} queries)"
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system", "method", "query_set", "metric", "value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _log(f"Results CSV written: {path} ({len(rows)} rows)")


def write_per_query_ndcg(
    pq_ndcg: dict[tuple[str, str], dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (system, method), scores in pq_ndcg.items():
            f.write(json.dumps({"system": system, "method": method, "scores": scores}) + "\n")
    _log(f"Per-query nDCG JSONL written: {path}")


def print_summary(rows: list[dict]) -> None:
    """Print a compact results table for ndcg@10 and recall@10."""
    from collections import defaultdict
    by_sys_method: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["system"], row["method"])
        by_sys_method[key][row["metric"]] = row["value"]

    print("\n=== Results summary (all queries) ===")
    header = f"{'system':<30} {'method':<14} {'ndcg@10':>8} {'recall@10':>10} {'recall@20':>10} {'mrr':>8} {'map':>8}"
    print(header)
    print("-" * len(header))
    order = [
        ("bm25",                       "bm25"),
        ("frozen-minilm",              "dense"),
        ("frozen-minilm",              "hybrid_rrf"),
        ("frozen-legalbert",           "dense"),
        ("frozen-legalbert",           "hybrid_rrf"),
        ("trained-minilm",             "dense"),
        ("trained-minilm",             "hybrid_rrf"),
        ("trained-legalbert",          "dense"),
        ("trained-legalbert",          "hybrid_rrf"),
        ("trained-legalbert+reranker", "bm25_rerank"),
        ("trained-legalbert+reranker", "dense_rerank"),
    ]
    for key in order:
        m = by_sys_method.get(key, {})
        if not m:
            continue
        sys_label, method_label = key
        print(
            f"{sys_label:<30} {method_label:<14} "
            f"{m.get('ndcg@10', 0):>8.4f} {m.get('recall@10', 0):>10.4f} "
            f"{m.get('recall@20', 0):>10.4f} {m.get('mrr', 0):>8.4f} "
            f"{m.get('map', 0):>8.4f}"
        )


# ---------------------------------------------------------------------------
# Standalone significance recompute (no GPU / no ES needed)
# ---------------------------------------------------------------------------

def sig_from_cache() -> None:
    """Load cached per-query nDCG JSONL and recompute significance tests only."""
    _log("Loading qrels ...")
    qrels, _ = load_qrels(QRELS_PATH)

    _log(f"Loading cached per-query nDCG from {OUT_NDCG} ...")
    pq_ndcg: dict[tuple[str, str], dict[str, float]] = {}
    with open(OUT_NDCG, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pq_ndcg[(rec["system"], rec["method"])] = rec["scores"]
    _log(f"  {len(pq_ndcg)} system+method entries loaded")

    run_significance(pq_ndcg, qrels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    _log("Citation-grounded eval starting")
    preflight()

    # --- Load qrels and corpus ---
    qrels, queries = load_qrels(QRELS_PATH)
    _log(f"Qrels: {len(qrels):,} queries, {sum(len(v) for v in qrels.values()):,} total relevant pairs")

    corpus_ids, corpus_embed_texts, rerank_lookup = load_corpus(CORPUS_PATH)

    # dict for quick BM25 text lookup
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # --- BM25 run (fixed anchor) ---
    bm25_run = build_bm25_run(queries, RUNS_DIR / "bm25.jsonl")

    # Self-filter BM25 (query's own ID may appear)
    for qid in list(bm25_run.keys()):
        bm25_run[qid] = [d for d in bm25_run[qid] if d != qid]

    # --- Collect all runs ---
    all_runs: dict[tuple[str, str], dict[str, list[str]]] = {}

    # BM25 baseline: same for all systems, report under "bm25" / "bm25"
    all_runs[("bm25", "bm25")] = bm25_run

    # --- Encoder systems ---
    for enc_slug, enc_path in ENCODERS.items():
        _log(f"\n--- Encoder: {enc_slug} ({enc_path}) ---")
        corpus_vecs, query_vecs, qid_order = build_embeddings(
            enc_path, corpus_ids, corpus_embed_texts, queries
        )

        _log(f"Dense retrieval for {enc_slug} ...")
        d_run = dense_retrieve(corpus_ids, corpus_vecs, query_vecs, qid_order)
        all_runs[(enc_slug, "dense")] = d_run

        _log(f"RRF for {enc_slug} ...")
        rrf_run = rrf_fuse(bm25_run, d_run)
        all_runs[(enc_slug, "hybrid_rrf")] = rrf_run

        # Free matrix from memory before next encoder
        del corpus_vecs, query_vecs
        gc.collect()

    # --- Reranker runs (trained-legalbert as base) ---
    _log("\n--- Reranker runs (trained-legalbert + reranker) ---")
    import torch as _torch
    from sentence_transformers import CrossEncoder as _CE
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    _log(f"Loading reranker from {RERANKER_CKPT} ...")
    _ce_model = _CE(str(RERANKER_CKPT), device=_device, max_length=384)
    _log(f"  Reranker loaded on {_device}")

    def _rerank_with_loaded(base_run: dict[str, list[str]]) -> dict[str, list[str]]:
        reranked: dict[str, list[str]] = {}
        qids = list(base_run.keys())
        for i, qid in enumerate(qids):
            candidates = base_run[qid][:RERANK_TOP_K]
            if not candidates:
                reranked[qid] = []
                continue
            pairs = [(queries[qid], rerank_lookup.get(did, "")) for did in candidates]
            scores = _ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
            order = np.argsort(scores)[::-1]
            reranked[qid] = [candidates[j] for j in order]
            if (i + 1) % 500 == 0:
                _log(f"  rerank: {i+1}/{len(qids)} queries, VRAM={_vram_gb():.2f}GB")
        return reranked

    base_dense_run = all_runs[("trained-legalbert", "dense")]
    _log("  Reranking dense candidates ...")
    all_runs[(RERANKER_SYSTEM, "dense_rerank")] = _rerank_with_loaded(base_dense_run)

    _log("  Reranking BM25 candidates ...")
    all_runs[(RERANKER_SYSTEM, "bm25_rerank")] = _rerank_with_loaded(bm25_run)

    del _ce_model
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    _log("  Reranking complete")

    # --- Compute metrics ---
    _log("\nComputing metrics ...")
    csv_rows, pq_ndcg = runs_to_metrics(all_runs, qrels)

    # --- Output ---
    write_csv(csv_rows, OUT_CSV)
    write_per_query_ndcg(pq_ndcg, OUT_NDCG)

    # --- Summary table ---
    print_summary(csv_rows)

    # --- Significance ---
    run_significance(pq_ndcg, qrels)

    elapsed = (time.time() - t0) / 3600
    _log(f"\nDone. Total wall-clock: {elapsed:.2f}h")


if __name__ == "__main__":
    if "--sig-only" in sys.argv:
        sig_from_cache()
    else:
        main()
