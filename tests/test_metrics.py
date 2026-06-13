"""
Cross-check hand-written IR metrics against pytrec_eval on a synthetic run.

Run: pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import random
from typing import Any

import pytrec_eval
import pytest

from eval.metrics import (
    compute_all,
    map_score,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

SEED = 42
N_QUERIES = 200
CORPUS_SIZE = 1000
MAX_REL = 10
RUN_DEPTH = 200
TOL = 1e-6   # floating-point agreement tolerance


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    rng = random.Random(SEED)
    corpus = [str(i) for i in range(CORPUS_SIZE)]

    qrels: dict[str, set[str]] = {}
    run: dict[str, list[str]] = {}

    for q in range(N_QUERIES):
        qid = f"q{q}"
        n_rel = rng.randint(1, MAX_REL)
        rel = set(rng.sample(corpus, n_rel))
        qrels[qid] = rel

        # Ranked list: random permutation biased toward relevant docs near top
        ranked = list(corpus)
        rng.shuffle(ranked)
        # Push up to 3 relevant docs into top-20
        for r in list(rel)[:3]:
            if r in ranked:
                ranked.remove(r)
            pos = rng.randint(0, min(19, len(ranked)))
            ranked.insert(pos, r)
        run[qid] = ranked[:RUN_DEPTH]

    return qrels, run


def _to_pytrec(
    qrels: dict[str, set[str]],
    run: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pt_qrels = {qid: {d: 1 for d in rel} for qid, rel in qrels.items()}
    # Scores = 1/(rank+1) so pytrec_eval sees the same ordering as our ranked lists
    pt_run = {
        qid: {doc_id: 1.0 / (i + 1) for i, doc_id in enumerate(docs)}
        for qid, docs in run.items()
    }
    return pt_qrels, pt_run


def _pytrec_mean(results: dict[str, dict[str, float]], key: str) -> float:
    vals = [v[key] for v in results.values()]
    return sum(vals) / len(vals)


class TestMetricsAgainstPytrecEval:

    def test_recall_at_1(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"recall_1"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "recall_1")
        got = recall_at_k(qrels, run, 1)
        assert abs(got - expected) < TOL, f"recall@1: got {got:.8f}, expected {expected:.8f}"

    def test_recall_at_5(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"recall_5"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "recall_5")
        got = recall_at_k(qrels, run, 5)
        assert abs(got - expected) < TOL, f"recall@5: got {got:.8f}, expected {expected:.8f}"

    def test_recall_at_10(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"recall_10"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "recall_10")
        got = recall_at_k(qrels, run, 10)
        assert abs(got - expected) < TOL, f"recall@10: got {got:.8f}, expected {expected:.8f}"

    def test_recall_at_20(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"recall_20"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "recall_20")
        got = recall_at_k(qrels, run, 20)
        assert abs(got - expected) < TOL, f"recall@20: got {got:.8f}, expected {expected:.8f}"

    def test_ndcg_at_10(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"ndcg_cut_10"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "ndcg_cut_10")
        got = ndcg_at_k(qrels, run, 10)
        assert abs(got - expected) < TOL, f"ndcg@10: got {got:.8f}, expected {expected:.8f}"

    def test_mrr(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"recip_rank"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "recip_rank")
        got = mrr(qrels, run)
        assert abs(got - expected) < TOL, f"mrr: got {got:.8f}, expected {expected:.8f}"

    def test_map(self, synthetic_data: Any) -> None:
        qrels, run = synthetic_data
        pt_q, pt_r = _to_pytrec(qrels, run)
        ev = pytrec_eval.RelevanceEvaluator(pt_q, {"map"})
        expected = _pytrec_mean(ev.evaluate(pt_r), "map")
        got = map_score(qrels, run)
        assert abs(got - expected) < TOL, f"map: got {got:.8f}, expected {expected:.8f}"

    def test_compute_all_consistent(self, synthetic_data: Any) -> None:
        """compute_all() must be consistent with individual functions."""
        qrels, run = synthetic_data
        all_m = compute_all(qrels, run)
        assert abs(all_m["recall@1"]  - recall_at_k(qrels, run, 1))  < TOL
        assert abs(all_m["recall@5"]  - recall_at_k(qrels, run, 5))  < TOL
        assert abs(all_m["recall@10"] - recall_at_k(qrels, run, 10)) < TOL
        assert abs(all_m["recall@20"] - recall_at_k(qrels, run, 20)) < TOL
        assert abs(all_m["ndcg@10"]   - ndcg_at_k(qrels, run, 10))   < TOL
        assert abs(all_m["mrr"]       - mrr(qrels, run))              < TOL
        assert abs(all_m["map"]       - map_score(qrels, run))        < TOL

    def test_perfect_run_gives_one(self, synthetic_data: Any) -> None:
        """Exactly 1 relevant doc per query ranked first => all metrics = 1.0."""
        # Use isolated single-relevant queries so all metrics are unambiguously 1.0
        qrels = {f"q{i}": {f"rel_{i}"} for i in range(50)}
        perfect_run = {
            qid: list(rel) + [f"noise_{j}" for j in range(50)]
            for qid, rel in qrels.items()
        }
        m = compute_all(qrels, perfect_run)
        for name, val in m.items():
            assert abs(val - 1.0) < TOL, f"perfect run: {name} = {val:.8f} != 1.0"

    def test_empty_run_gives_zero(self, synthetic_data: Any) -> None:
        """An empty run must score 0.0 on all metrics."""
        qrels, _ = synthetic_data
        empty_run: dict[str, list[str]] = {qid: [] for qid in qrels}
        m = compute_all(qrels, empty_run)
        for name, val in m.items():
            assert abs(val) < TOL, f"empty run: {name} = {val:.8f} != 0.0"
