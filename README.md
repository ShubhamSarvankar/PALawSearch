# PA-LawSearch

Legal case retrieval and RAG system over ~197k Pennsylvania appellate cases, with fine-tuned retrieval encoders and a rigorous, validated evaluation harness.

The trained LegalBERT encoder improves citation-grounded nDCG@10 by **+0.051** over the frozen baseline (paired bootstrap, p < 0.0001, N = 12,035 queries). Full numbers, significance tests, and caveats: [RESULTS.md](RESULTS.md).

---

## What it does

Given a legal query or a case, the system retrieves relevant prior PA cases and can answer natural-language questions over them with grounded citations. Four retrieval methods are available — BM25, dense, dense + rerank, and hybrid RRF — served through a single Flask API with a React frontend.

What distinguishes it from a standard RAG system: the retrieval encoders are **fine-tuned on a citation-derived training set** (not frozen off-the-shelf models), and the performance claims are backed by a validated eval harness with standard IR metrics cross-checked against pytrec_eval, a judge validated at Cohen's κ = 0.773, and an external benchmark on independent public data.

---

## Architecture

```
React frontend
    |
Flask API  (/cases  /cases/<id>  /ask  /health)
    |                |                    |
Redis cache    Search layer          RAG service
               BM25 / dense /        hybrid RRF
               rerank variants    -> Ollama qwen3:8b (local)
                    |
             Elasticsearch
             pa_cases_bm25   (lexical)
             pa_cases_dense  (KNN, trained encoder)
```

**Offline pipeline** (not in request path):

```
download -> parse (+ cites_to extraction) -> citation graph -> train/eval split
                                                  |                   |
                                           eval qrels            training triplets
                                                  |                   |
                                           eval harness       fine-tune encoder x2
                                           (nDCG, Recall,     fine-tune reranker
                                            judge, CLERC)
```

**Models in use**

| Role | Model | Notes |
|---|---|---|
| Dense encoder | LegalBERT (fine-tuned) | nlpaueb/legal-bert-base-uncased base, trained on 859k citation triplets |
| Cross-encoder reranker | BGE-reranker-base (fine-tuned) | Used in dense_rerank and bm25_rerank methods |
| RAG generator | qwen3:8b via Ollama | Frozen, local inference only |
| BM25 | Elasticsearch | Shared anchor across all dense systems |
| Cache | Redis | Caches reranked result lists |

---

## Results summary

See [RESULTS.md](RESULTS.md) for full tables, significance tests, and methodology notes.

**Citation-grounded nDCG@10, hybrid RRF (N = 12,035 eval queries)**

| Encoder | Frozen | Trained | Delta | p |
|---|---|---|---|---|
| LegalBERT | 0.0399 | 0.0910 | +0.051 | < 0.0001 |
| MiniLM | 0.0562 | 0.0838 | +0.028 | < 0.0001 |

Retrieval quality is a lower bound: citation qrels record only what authors cited, not everything topically relevant. The eval harness quantifies this caveat (roughly half of top-10 non-cited docs are judged topically relevant by the validated judge).

---

## Quick start

**Requirements:** Python 3.11, Docker Desktop, Ollama, [`just`](https://github.com/casey/just), CUDA-capable GPU recommended for encoding and training.

### 1. Environment

```bash
just setup          # creates .venv, installs torch (cu128) + all deps
cp .env.example .env
# edit .env: set ES_PASSWORD, ANTHROPIC_API_KEY
```

### 2. Services

```bash
just services-up    # starts Elasticsearch + Redis via Docker Desktop
```

### 3. Data and indexing

```bash
just download       # fetches full PA corpus (~200k cases) from static.case.law
just ingest         # parses cases, extracts citations, deduplicates
just index-bm25     # builds BM25 ES index
just embed          # encodes corpus with trained encoder -> .npz
just index-dense    # builds KNN ES index from embeddings
```

### 4. Serve

```bash
just serve          # Flask API on :5000
just serve-frontend # React dev server (separate terminal)
```

Ollama must be running locally with `qwen3:8b` pulled:

```bash
ollama pull qwen3:8b
```

### 5. Evaluate

```bash
just eval           # citation-grounded eval matrix -> data/eval/results.csv
just kappa          # recompute Cohen's kappa (no API calls)
just recall-gap     # LLM judge recall-gap estimation (~672 Anthropic API calls)
just clerc-bench    # CLERC external benchmark (~60 min, first run downloads 7.6 GB)
```

---

## Training (optional, checkpoints committed)

Trained checkpoints are committed. Re-running training is only needed to reproduce from scratch.

```bash
just gpu-free           # confirm GPU is idle before training
just train-encoder base=legalbert
just train-encoder base=minilm
just train-reranker
just reindex-dense      # re-embeds corpus with updated encoder, rebuilds dense index
```

Training runs within 8 GB VRAM on a single GPU. See [RESULTS.md](RESULTS.md) for peak VRAM and wall-clock per model.

---

## Configuration

All secrets go in `.env` — never in source. The required variables:

```
ES_PASSWORD=...          # Elasticsearch password
ANTHROPIC_API_KEY=...    # Required only for just recall-gap
```

Optional overrides (defaults in `config/settings.py`):

```
ENCODER_MODEL_PATH=...   # defaults to committed trained LegalBERT checkpoint
ES_HOST=http://localhost:9200
OLLAMA_MODEL=qwen3:8b
```

---

## Evaluation methodology in brief

The eval harness measures standard IR metrics (nDCG@10, Recall@k, MRR, MAP) against citation-grounded qrels: cases in the test split that cite each other. The LLM judge (claude-haiku with a committed rubric) is used only for recall-gap estimation, not for headline metrics, and was validated against owner-provided human labels at Cohen's κ = 0.773 before deployment. An external benchmark on CLERC (US federal citation retrieval, jhu-clsp/CLERC) confirms the domain-transfer signal. Full methodology in [RESULTS.md](RESULTS.md).
