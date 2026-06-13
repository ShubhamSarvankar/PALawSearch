"""
External benchmark: CLERC (jhu-clsp/CLERC)
Document-level, indirect single-removed test set.

Comparison: frozen nlpaueb/legal-bert-base-uncased vs trained PA-LegalBERT.
Corpus: all qrel-relevant docs + reservoir-sampled distractors → ~150k total (seed=42).
Results → data/eval/clerc_benchmark.json.

CAVEAT: subsampled corpus preserves the frozen→trained delta but absolute scores
are NOT comparable to full-corpus CLERC paper results (paper BM25 nDCG@10=0.054).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

# ── CLERC dataset ────────────────────────────────────────────────────────────
HF_DATASET   = "jhu-clsp/CLERC"
QRELS_FILE   = "qrels/qrels-doc.test.indirect.tsv"
QUERIES_FILE = "queries/test.single-removed.indirect.tsv"
COLLECTION_FILE = "collection/collection.doc.tsv.gz"

# ── Corpus sampling ───────────────────────────────────────────────────────────
CORPUS_SEED   = 42
CORPUS_TARGET = 150_000

# ── Models ────────────────────────────────────────────────────────────────────
FROZEN_MODEL_ID  = "nlpaueb/legal-bert-base-uncased"
TRAINED_MODEL_PATH = str(ROOT / "models/checkpoints/encoder/legalbert-20260608-144233/model")

# ── Encoding / retrieval ──────────────────────────────────────────────────────
ENCODE_BATCH    = 128
DOC_EMBED_MAX   = 4096   # chars; tokenizer truncates to 512 tokens anyway
RETRIEVAL_TOP_K = 100    # enough for R@100

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BOOTSTRAP_N    = 10_000
BOOTSTRAP_SEED = 42

RESULTS_PATH = ROOT / "data/eval/clerc_benchmark.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                qid, _, did, _ = parts
                qrels.setdefault(qid, set()).add(did)
    return qrels


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def _build_corpus(
    relevant_ids: set[str],
    collection_path: Path,
    target: int,
    seed: int,
) -> dict[str, str]:
    """
    Stream collection.doc.tsv.gz (full scan), return {doc_id: text} for `target` docs:
    all relevant docs + reservoir-sampled distractors to reach `target`.
    Reservoir sampling (Algorithm R) gives a uniform random distractor draw.
    """
    rng = random.Random(seed)
    n_distractor_target = target - len(relevant_ids)
    corpus: dict[str, str] = {}           # relevant docs (always kept)
    reservoir: list[tuple[str, str]] = [] # distractor reservoir
    n_non_rel_seen = 0

    _log(f"Streaming {collection_path.name} for {len(relevant_ids)} relevant "
         f"+ {n_distractor_target:,} distractor docs ...")
    t0 = time.time()

    with gzip.open(collection_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            text = text[:DOC_EMBED_MAX]  # truncate to save RAM

            if doc_id in relevant_ids:
                corpus[doc_id] = text
            else:
                n_non_rel_seen += 1
                if len(reservoir) < n_distractor_target:
                    reservoir.append((doc_id, text))
                else:
                    j = rng.randint(0, n_non_rel_seen - 1)
                    if j < n_distractor_target:
                        reservoir[j] = (doc_id, text)

            if (i + 1) % 300_000 == 0:
                _log(f"  {i+1:,} docs scanned | {len(corpus)}/{len(relevant_ids)} "
                     f"relevant found | {time.time()-t0:.0f}s")

    _log(f"Scan done: {i+1:,} docs in {time.time()-t0:.0f}s")

    missing = relevant_ids - set(corpus)
    if missing:
        _log(f"WARNING: {len(missing)} relevant docs not found in collection: "
             f"{sorted(missing)[:5]}")

    for did, text in reservoir:
        corpus[did] = text

    _log(f"Corpus: {len(corpus):,} docs "
         f"({len(corpus) - len(reservoir)} relevant + {len(reservoir):,} distractor)")
    return corpus


def _encode(
    texts: list[str],
    model_path: str,
    label: str,
) -> np.ndarray:
    """Encode with SentenceTransformer; return L2-normalised float32 array."""
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  Loading {label} ({model_path}) on {device}")
    model = SentenceTransformer(model_path, device=device)

    _log(f"  Encoding {len(texts):,} texts ...")
    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _log(f"  Done in {time.time()-t0:.0f}s  shape={vecs.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vecs.astype(np.float32)


def _retrieve(
    q_vecs: np.ndarray,
    d_vecs: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    top_k: int,
) -> dict[str, list[str]]:
    """Brute-force cosine (vecs L2-normalised → dot = cosine). Returns ranked run."""
    scores = q_vecs @ d_vecs.T  # (n_q, n_d)
    run: dict[str, list[str]] = {}
    for i, qid in enumerate(query_ids):
        top_idx = np.argpartition(scores[i], -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[i][top_idx])[::-1]]
        run[qid] = [doc_ids[j] for j in top_idx]
    return run


def _bootstrap_p(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float, float, float, int]:
    """
    Paired bootstrap significance test (one-sided, H1: mean_b > mean_a).
    Returns (mean_a, mean_b, delta, ci_lo, ci_hi, p_value, N).
    """
    assert len(scores_a) == len(scores_b)
    N = len(scores_a)
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diffs = b - a
    obs_delta = float(diffs.mean())

    rng = np.random.default_rng(seed)
    _BATCH = 500
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _BATCH):
        end = min(start + _BATCH, n_resamples)
        idx = rng.integers(0, N, size=(end - start, N), dtype=np.int32)
        boot_means[start:end] = diffs[idx].mean(axis=1)

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    p = float((boot_means <= 0.0).mean())
    return float(a.mean()), float(b.mean()), obs_delta, ci_lo, ci_hi, p, N


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the full benchmark")
    args = parser.parse_args()

    if not args.run:
        print("Usage: python -m eval.external_benchmark --run")
        return

    # Suppress HF Hub symlink warning on Windows
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    warnings.filterwarnings("ignore", message=".*symlinks.*")

    from eval.metrics import ndcg_at_k, recall_at_k, per_query_ndcg10
    from huggingface_hub import hf_hub_download

    print("=== CLERC external benchmark ===\n")
    t_start = time.time()

    # 1. Download small files (cached after first run)
    _log("Fetching qrels + queries from HF Hub ...")
    qrels_path  = Path(hf_hub_download(HF_DATASET, QRELS_FILE,   repo_type="dataset"))
    queries_path = Path(hf_hub_download(HF_DATASET, QUERIES_FILE, repo_type="dataset"))

    qrels   = _load_qrels(qrels_path)
    queries = _load_queries(queries_path)

    eval_qids    = sorted(set(queries) & set(qrels))
    relevant_ids = {did for qid in eval_qids for did in qrels[qid]}

    _log(f"Queries: {len(eval_qids)} | Unique relevant docs: {len(relevant_ids)}")

    # 2. Download collection (7.63 GB, one-time HF cache)
    _log("Fetching collection.doc.tsv.gz (7.63 GB, one-time download, cached after this) ...")
    collection_path = Path(hf_hub_download(
        HF_DATASET, COLLECTION_FILE, repo_type="dataset"
    ))

    # 3. Build subsampled corpus
    corpus = _build_corpus(relevant_ids, collection_path, CORPUS_TARGET, CORPUS_SEED)
    corpus_ids   = sorted(corpus)
    corpus_texts = [corpus[did] for did in corpus_ids]
    n_relevant_in_corpus = len(relevant_ids & set(corpus_ids))

    print(f"\nCorpus size: {len(corpus_ids):,} docs "
          f"({n_relevant_in_corpus} relevant + {len(corpus_ids)-n_relevant_in_corpus:,} distractor)")

    # 4. Encode queries (per model, since weights differ)
    query_texts = [queries[qid] for qid in eval_qids]

    model_configs = [
        ("frozen",  FROZEN_MODEL_ID),
        ("trained", TRAINED_MODEL_PATH),
    ]

    results: dict[str, dict] = {}
    per_query: dict[str, dict[str, float]] = {}

    for label, model_path in model_configs:
        print(f"\n{'='*60}")
        print(f"Model: {label}  ({model_path})")
        print(f"{'='*60}")

        q_vecs = _encode(query_texts, model_path, f"{label} queries")
        d_vecs = _encode(corpus_texts, model_path, f"{label} corpus")

        _log("Retrieving ...")
        run = _retrieve(q_vecs, d_vecs, eval_qids, corpus_ids, RETRIEVAL_TOP_K)

        ndcg10 = ndcg_at_k(qrels, run, 10)
        r10    = recall_at_k(qrels, run, 10)
        r100   = recall_at_k(qrels, run, 100)
        pq     = per_query_ndcg10(qrels, run)

        results[label] = {
            "model":       model_path,
            "ndcg@10":     round(ndcg10, 4),
            "recall@10":   round(r10,    4),
            "recall@100":  round(r100,   4),
        }
        per_query[label] = pq
        _log(f"nDCG@10={ndcg10:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")

    # 5. Paired bootstrap on per-query nDCG@10
    common = sorted(set(per_query["frozen"]) & set(per_query["trained"]))
    a_scores = [per_query["frozen"][q]  for q in common]
    b_scores = [per_query["trained"][q] for q in common]
    mean_a, mean_b, delta, ci_lo, ci_hi, p_val, N = _bootstrap_p(a_scores, b_scores)

    # 6. Print summary
    print("\n" + "="*60)
    print("RESULTS (subsampled CLERC, doc-level, indirect single-removed)")
    print("="*60)
    print(f"Corpus:  {len(corpus_ids):,} docs  "
          f"({n_relevant_in_corpus} relevant + {len(corpus_ids)-n_relevant_in_corpus:,} distractor, seed={CORPUS_SEED})")
    print(f"Queries: {len(eval_qids)}")
    print()
    print(f"{'Model':>10}  {'nDCG@10':>8}  {'R@10':>8}  {'R@100':>8}")
    print(f"{'------':>10}  {'-------':>8}  {'----':>8}  {'-----':>8}")
    for lbl, r in results.items():
        print(f"{lbl:>10}  {r['ndcg@10']:>8.4f}  {r['recall@10']:>8.4f}  {r['recall@100']:>8.4f}")
    print()
    print(f"delta frozen->trained nDCG@10: {delta:+.4f}  "
          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_val:.4f}  (N={N})")
    print()
    print("NOTE: CLERC paper BM25 nDCG@10=0.054 (full 1.84M-doc corpus — NOT comparable here).")
    print(f"Total wall clock: {(time.time()-t_start)/60:.1f} min")

    # 7. Write JSON
    output = {
        "dataset":          HF_DATASET,
        "eval_variant":     "doc-level indirect single-removed",
        "corpus_size":      len(corpus_ids),
        "n_relevant":       n_relevant_in_corpus,
        "n_distractor":     len(corpus_ids) - n_relevant_in_corpus,
        "corpus_seed":      CORPUS_SEED,
        "n_queries":        len(eval_qids),
        "retrieval_top_k":  RETRIEVAL_TOP_K,
        "models":           results,
        "paired_bootstrap_ndcg10": {
            "mean_frozen":    round(mean_a, 4),
            "mean_trained":   round(mean_b, 4),
            "delta":          round(delta,  4),
            "ci_95_lo":       round(ci_lo,  4),
            "ci_95_hi":       round(ci_hi,  4),
            "p_value_one_tailed": round(p_val, 4),
            "n_queries":      N,
            "n_resamples":    BOOTSTRAP_N,
        },
        "caveat": (
            "Subsampled corpus (150k docs). "
            "The frozen→trained delta is interpretable as a domain-transfer signal. "
            "Absolute scores are NOT comparable to CLERC full-corpus paper results "
            "(paper BM25 nDCG@10=0.054, LegalBERT-DPR-ft nDCG@10 higher)."
        ),
        "wall_clock_seconds": round(time.time() - t_start),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
