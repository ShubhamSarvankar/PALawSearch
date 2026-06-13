# PA-LawSearch v2 — task runner
# Requires: just (https://github.com/casey/just)
# All recipes are OS-agnostic; Windows shell is powershell, POSIX shell elsewhere.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Force UTF-8 output on Windows so Unicode box-drawing / arrows print correctly
export PYTHONIOENCODING := "utf-8"

# Create .venv with Python 3.11 and install all deps
setup:
    py -3.11 -m venv .venv
    .venv/Scripts/pip install --upgrade pip
    .venv/Scripts/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    .venv/Scripts/pip install -e ".[dev]"

# Start Elasticsearch + Redis via Docker Desktop
services-up:
    docker compose up -d

# Stop services (keeps volumes)
services-down:
    docker compose down

# Stop Ollama and confirm GPU is idle before training
gpu-free:
    @echo "Stopping Ollama..."
    -taskkill /IM ollama.exe /F 2>$null
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# --- Data pipeline ---

# Show size estimate and disk check; does NOT download
estimate:
    .venv/Scripts/python -m ingest.download --estimate

# Run the full download (review `just estimate` output first)
download:
    .venv/Scripts/python -m ingest.download --go

ingest:
    .venv/Scripts/python -m ingest.parse

index-bm25:
    .venv/Scripts/python -m indexing.bm25_indexer

embed:
    .venv/Scripts/python -m indexing.embed

index-dense:
    .venv/Scripts/python -m indexing.index_dense_from_file

# --- Graph + training ---

# Resolve citations to canonical case ids (writes edges.jsonl)
resolve:
    .venv/Scripts/python -m graph.resolve

# Stage 1: resolve citations + build graph artifacts (runs resolve first)
graph:
    .venv/Scripts/python -m graph.resolve
    .venv/Scripts/python -m graph.build

split:
    .venv/Scripts/python -m graph.split

mine:
    .venv/Scripts/python -m training.mine_pairs

smoke-encoder base="minilm":
    .venv/Scripts/python -m training.train_encoder --base {{base}} --smoke

train-encoder base="minilm":
    .venv/Scripts/python -m training.train_encoder --base {{base}}

train-reranker:
    .venv/Scripts/python -m training.train_reranker

reindex-dense:
    just embed
    just index-dense

# --- Evaluation ---

eval:
    .venv/Scripts/python -m eval.run_eval

# Reproduce Cohen's kappa from committed judge + human labels (no API calls)
kappa:
    .venv/Scripts/python -m eval.compute_kappa

# Estimate citation-qrels recall gap using the validated LLM judge (rubric v2)
recall-gap:
    .venv/Scripts/python -m eval.run_recall_gap --run

# External benchmark on CLERC (frozen vs trained, ~150k subsampled corpus)
clerc-bench:
    .venv/Scripts/python -m eval.external_benchmark --run

# --- Optimization ---

quantize:
    .venv/Scripts/python -m optimize.quantize

bench:
    .venv/Scripts/python -m optimize.bench_latency

# --- Serving ---

serve:
    .venv/Scripts/python -m api.app

serve-frontend:
    cd frontend && npm run dev
