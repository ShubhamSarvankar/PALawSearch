# PA-LawSearch v2 — task runner
# Requires: just (https://github.com/casey/just)
# All recipes are OS-agnostic; Windows shell is powershell, POSIX shell elsewhere.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

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

# --- Data pipeline (Phase 1) ---

download:
    .venv/Scripts/python -m ingest.download

ingest:
    .venv/Scripts/python -m ingest.parse

index-bm25:
    .venv/Scripts/python -m indexing.index_bm25

embed:
    .venv/Scripts/python -m indexing.embed

index-dense:
    .venv/Scripts/python -m indexing.index_dense

# --- Graph + training (Phases 2-3) ---

graph:
    .venv/Scripts/python -m graph.build

split:
    .venv/Scripts/python -m graph.split

mine:
    .venv/Scripts/python -m training.mine_pairs

train-encoder base="minilm":
    .venv/Scripts/python -m training.train_encoder --base {{base}}

train-reranker:
    .venv/Scripts/python -m training.train_reranker

reindex-dense:
    just embed
    just index-dense

# --- Evaluation (Phase 4) ---

eval:
    .venv/Scripts/python -m eval.run_eval

# --- Optimization (Phase 5) ---

quantize:
    .venv/Scripts/python -m optimize.quantize

bench:
    .venv/Scripts/python -m optimize.bench_latency

# --- Serving ---

serve:
    .venv/Scripts/python -m api.app

serve-frontend:
    cd frontend && npm run dev
