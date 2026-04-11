"""
inference/model_loader.py — shared checkpoint + tokenizer loading for all inference scripts.

Checkpoint format (written by the notebook):
    {
        'model_state'    : state_dict,
        'optimizer_state': ...,   (ignored at inference)
        'config'         : vars(LLaMAConfig),
        'step'           : int,
        'train_losses'   : [...],
        'val_losses'     : [...],
    }
Also accepts a bare state_dict (no wrapper dict) for simpler checkpoints.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
import torch

# Make sure the repo root is on the path regardless of where the script runs
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, Transformer

# Fields that ModelConfig recognises — anything else in the old LLaMAConfig
# (training hyper-params, grad_checkpoint, …) is silently dropped.
_MODEL_FIELDS = {f.name for f in fields(ModelConfig)}

# ── Key remapping ─────────────────────────────────────────────────────────────
# The notebook's LLaMA class used different attribute names from our refactored
# model.  Map old checkpoint keys → new model keys before load_state_dict.
#
# Notebook → PagedInfer
#   token_emb.weight      → token_emb.embedding.weight   (TokenEmbedding wrapper)
#   ffn.w1                → ffn.w_gate   (gate projection)
#   ffn.w2                → ffn.w_down   (down projection)
#   ffn.w3                → ffn.w_up     (up projection)

def _remap_state_dict(state_dict: dict) -> dict:
    """Rename notebook checkpoint keys to match the PagedInfer model layout."""
    remapped = {}
    for k, v in state_dict.items():
        # token embedding: bare nn.Embedding → wrapped TokenEmbedding
        if k == "token_emb.weight":
            k = "token_emb.embedding.weight"
        # SwiGLU FFN projections: w1/w2/w3 → w_gate/w_down/w_up
        elif k.endswith(".ffn.w1.weight"):
            k = k[:-len("w1.weight")] + "w_gate.weight"
        elif k.endswith(".ffn.w2.weight"):
            k = k[:-len("w2.weight")] + "w_down.weight"
        elif k.endswith(".ffn.w3.weight"):
            k = k[:-len("w3.weight")] + "w_up.weight"
        remapped[k] = v
    return remapped


def load_model(
    checkpoint: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> tuple[Transformer, ModelConfig]:
    """
    Load a Transformer from a checkpoint produced by the notebook.

    Returns:
        model : Transformer in eval mode on `device`
        cfg   : ModelConfig used to build the model
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)

    # ── Unwrap checkpoint dict ────────────────────────────────────────────────
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict  = ckpt["model_state"]
        raw_config  = ckpt.get("config", {})
        step        = ckpt.get("step", "?")
    else:
        # Bare state_dict — no config info, fall back to defaults
        state_dict = ckpt
        raw_config = {}
        step       = "?"

    # ── Build ModelConfig (drop unknown / training-only keys) ─────────────────
    model_kwargs = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
    model_kwargs.setdefault("attention_type", "gqa")   # old checkpoints won't have this
    cfg = ModelConfig(**model_kwargs)

    # ── Instantiate and load weights ──────────────────────────────────────────
    model = Transformer(cfg)
    model.load_state_dict(_remap_state_dict(state_dict))
    model.eval()

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)

    print(f"Loaded checkpoint: {checkpoint}  (step={step}, {cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h)")
    return model, cfg


def load_tokenizer(tokenizer_id: str = "huggyllama/llama-7b"):
    """
    Load the LLaMA SentencePiece tokenizer via HuggingFace Transformers.

    The public mirror 'hf-internal-testing/llama-tokenizer' is identical to the
    official meta-llama tokenizer and does not require gated access.
    """

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok
