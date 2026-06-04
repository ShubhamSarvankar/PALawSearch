# Environment Notes

Recorded here per architecture §0.2 so the "trained on Blackwell" claim is substantiated.

## Hardware

- GPU: NVIDIA GeForce RTX 5060 (laptop), 8151 MiB VRAM, Blackwell sm_120
- RAM: 32 GB
- OS: Windows 11 Home
- NVIDIA Driver: 591.91
- CUDA Version (driver-reported): 13.1

## PyTorch / CUDA

Validation gate passed 2026-06-04.

| Field | Value |
|---|---|
| torch version | 2.11.0+cu128 |
| torch.version.cuda | 12.8 |
| install channel | cu128 stable (no nightly pivot needed) |
| `get_device_name(0)` | NVIDIA GeForce RTX 5060 Laptop GPU |
| `get_device_capability(0)` | (12, 0) |
| `cuda.is_available()` | True |
| matmul 4096×4096 | pass (real tensor executed on device) |

Install command that produced this environment:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Validation gate script

```python
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))   # expect (12, 0)
x = torch.randn(4096, 4096, device="cuda"); print("matmul", (x@x).sum().item())
```

## Python

| Field | Value |
|---|---|
| version | 3.11.x |
| source | python.org x64 (not Microsoft Store) |
| venv | `.venv/` at repo root |
