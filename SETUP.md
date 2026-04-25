# Setup

## Requirements

- Python 3.13+
- CUDA-capable GPU (tested on T4 / A100 via Google Colab)
- CUDA toolkit with `nvcc` accessible on `PATH`
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

The CUDA kernels are compiled automatically on first use via `nvcc`. No manual build step is needed. Compiled `.so` files are cached in `/tmp/` and reused on subsequent runs.

---

## Install

```bash
git clone https://github.com/RAAHUL-tech/PagedInfer.git
cd PagedInfer
uv sync          # installs all dependencies from pyproject.toml
```

Or with pip:

```bash
pip install torch transformers fastapi uvicorn sentencepiece
```

---

## Checkpoint

The scripts expect a LLaMA checkpoint at `model_checkpoint/llama_ckpt.pt` in the format produced by the training notebooks:

```python
{
    "model_state" : state_dict,
    "config"      : vars(LLaMAConfig),   # n_layers, dim, n_heads, ...
    "step"        : int,
}
```

A bare `state_dict` (no wrapper dict) is also accepted.

If you trained with the notebooks on Google Drive, copy the checkpoint:

```python
# In Colab
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy(
    '/content/drive/MyDrive/PagedInfer/llama_ckpt.pt',
    '/content/PagedInfer/model_checkpoint/llama_ckpt.pt'
)
```

---

## Running the inference scripts

Each script takes `--checkpoint` as a required argument. All scripts default to `--device cuda`.

```bash
# Baseline: full recompute, no KV cache
uv run python inference/generate_no_cache.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --prompt "Once upon a time"

# Flat KV cache
uv run python inference/generate_flat_kv.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --prompt "Once upon a time"

# GPU paged KV cache with CUDA kernels
uv run python inference/generate_paged_gpu.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --prompt "Once upon a time"

# Prefix caching — runs 3 experiments
uv run python inference/generate_prefix_cached.py \
    --checkpoint model_checkpoint/llama_ckpt.pt

# Continuous batching server (8 requests, compare vs static baseline)
uv run python inference/serve_continuous.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --compare_static
```

---

## Running the HTTP inference server

```bash
# Start the server
uv run python server/api_server.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --port 8000 \
    --kv_budget_gb 2.0
```

The server starts a background engine thread and begins accepting requests immediately.

```bash
# In a separate terminal — single streaming request
python server/client.py \
    --prompt "Explain attention mechanisms" \
    --stream

# Non-streaming
python server/client.py \
    --prompt "Explain attention mechanisms"

# Concurrent load test (8 requests in parallel)
python server/client.py --load_test

# Engine health and metrics
python server/client.py --health
python server/client.py --metrics
```

Or with curl:

```bash
# Non-streaming
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_new_tokens": 64}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000/generate/stream \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_new_tokens": 64}'
```

---

## Running in Google Colab

```python
# Install and clone
!pip install -q uv
!git clone https://github.com/RAAHUL-tech/PagedInfer.git
%cd PagedInfer
!uv sync

# Check nvcc is available
!nvcc --version

# Run the prefix-caching experiment
!uv run python inference/generate_prefix_cached.py \
    --checkpoint /content/drive/MyDrive/PagedInfer/llama_ckpt.pt \
    --kv_budget_gb 2.0
```

The kernels compile on the first run (under 30 seconds). The compiled `.so` files are cached in `/tmp/` for the session.

---

## Common issues

**`nvcc: command not found`**  
Install the CUDA toolkit or use a Colab/Kaggle runtime with GPU enabled.

**`RuntimeError: GPU memory budget too small`**  
Reduce `--kv_budget_gb` or free VRAM by restarting the runtime before running.

**`AttributeError: 'GPUPagedKVCache' object has no attribute ...`**  
Make sure you're on the `main` branch — older notebooks contain a simplified version of `GPUPagedKVCache` that differs from the production class.

**Checkpoint key mismatch**  
The loader in `inference/model_loader.py` automatically remaps keys from the notebook training format (`ffn.w1` → `ffn.w_gate`, etc.). If you see unexpected missing keys, check that `_remap_state_dict` covers your checkpoint's naming.
