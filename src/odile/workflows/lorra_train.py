"""
LoRRA (Low-Rank Representation Adaptation) Circuit Breaker for AgentDojo.

Trains a LoRA adapter with dual representation-level objectives:
  A. Circuit Breaking: Force Bad trace hidden states → target Good hidden states
     on completion tokens (model learns to refuse/redirect under injection)
  B. Anchoring: Keep Good v1 hidden states ≈ frozen base model hidden states
     on completion tokens (model preserves benign tool-calling behavior)

The 50/50 mix of objectives prevents the "refusal tax" — the model doesn't
learn to refuse everything, only injected prompts.

Pipeline:
  1. Pre-compute target activations from frozen base model
  2. Apply LoRA to model at specified layers (default: 40–55)
  3. Train with mixed circuit-breaking + anchoring MSE loss
  4. Evaluate on holdout set (both injected and clean prompts)
  5. Save LoRA adapter

Objectives (--objective):
  - refusal:  Bad hidden states → Good v2 (Refusal) hidden states
              Teaches: "when you see injection, refuse"
  - redirect: Bad hidden states → Good v1 (Benign) hidden states
              Teaches: "when you see injection, continue user's task"
              (no refusal concept → lower refusal tax)
  - nullify:  Bad hidden states → zeros (subspace destruction)
              Teaches: "when you see injection, lose coherence"
              (nuclear option — model outputs gibberish on injection,
               which is safely caught by tool-call parser as FormatError)

Usage:
    # Basic training: refusal objective (recommended)
    # Banking (default)
    python -m odile.workflows.lorra_train \
        --traces data/traces/banking/llama33_70b/traces_filtered.json \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output-dir ./adapters/banking/refusal_r8_L40-55 \
        --objective refusal \
        --rank 8 --epochs 3

    # Workspace
    python -m odile.workflows.lorra_train \
        --suite workspace \
        --traces data/traces/workspace/llama33_70b/traces_workspace_filtered.json \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output-dir ./adapters/workspace/refusal_r8_L40-55 \
        --objective refusal \
        --rank 8 --epochs 3

    # Redirect objective (lower refusal tax, weaker safety signal)
    python -m odile.workflows.lorra_train \
        --traces /path/to/traces_filtered.json \
        --model /path/to/llama-70b \
        --output-dir /path/to/output \
        --objective redirect \
        --rank 8 --epochs 2

    # With evaluation
    python -m odile.workflows.lorra_train \
        --traces /path/to/traces_filtered.json \
        --model /path/to/llama-70b \
        --output-dir /path/to/output \
        --objective redirect \
        --eval --eval-max-tokens 256

    # Dry-run: tokenize, show alignment, estimate memory, exit
    python -m odile.workflows.lorra_train \
        --traces /path/to/traces_filtered.json \
        --model /path/to/llama-70b \
        --output-dir /path/to/output \
        --dry-run

    # Combined training on all suites at once
    python -m odile.workflows.lorra_train \
        --suite all \
        --traces banking=/path/to/banking_filtered.json \
                slack=/path/to/slack_filtered.json \
                workspace=/path/to/workspace_filtered.json \
        --model /path/to/llama-70b \
        --output-dir /path/to/combined_output \
        --objective nullify --rank 16 --epochs 5
"""

import argparse
import importlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, load_peft_weights, set_peft_model_state_dict
from odile import data_pipeline as odile_data_pipeline
from odile.workflows.refusal_preservation_utils import load_refusal_target_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer

# Speed knobs safe for bf16 training on H100/H200:
# - TF32 matmuls for any fp32 ops (optimizer state, small math, LoRA updates)
# - "high" matmul precision lets PyTorch pick the fastest kernel (incl. TF32)
# Negligible numerical impact, measurable speedup on the fp32 book-keeping.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data config — suite-specific rendering, classification, tokenization
# Import from data_pipeline/<suite>/data_config.py
# ═══════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(odile_data_pipeline.__file__).resolve().parent
ALL_TRAIN_SUITES = sorted(
    p.parent.name for p in _SCRIPT_DIR.glob("*/data_config.py")
)

# Deferred imports — resolved in main() based on --suite flag.
# These are set by _init_data_config() before any code uses them.
classify_completion_simple = None
_extract_tool_calls = None
_is_garbled = None
render_full = None
render_prefix = None
tokenize_triplets = None


def _init_data_config(suite: str, format: str = "prompting_llm"):
    """Dynamically import data_config for the given suite and format.

    Args:
        suite: Suite name (banking, workspace, etc.)
        format: Tool calling format — "prompting_llm" (default, XML tags),
                "qwen_native" (Qwen 2.5 <tool_call> format),
                "qwen3_native" (Qwen3 <tool_call> format, enable_thinking=True),
                "llama_native" (Llama 3.1 <|python_tag|> format).
    """
    global classify_completion_simple, _extract_tool_calls, _is_garbled
    global render_full, render_prefix, tokenize_triplets

    config_modules = {
        "prompting_llm": "data_config",
        "qwen_native": "qwen_native_config",
        "qwen3_native": "qwen3_native_config",
        "llama_native": "llama_native_config",
        "react": "react_config",
    }
    module_name = config_modules.get(format, "data_config")
    mod = importlib.import_module(f"odile.data_pipeline.{suite}.{module_name}")
    classify_completion_simple = mod.classify_completion_simple
    _extract_tool_calls = mod.extract_tool_calls
    _is_garbled = mod.is_garbled
    render_full = mod.render_full
    render_prefix = mod.render_prefix
    tokenize_triplets = mod.tokenize_triplets
    log.info("Loaded data_config for suite=%s, format=%s", suite, format)


# ═══════════════════════════════════════════════════════════════════════════
# Hook-based hidden state capture (gradient-aware)
# ═══════════════════════════════════════════════════════════════════════════

def _get_transformer_layers(model):
    """Find the transformer layer list regardless of model wrapping (PeftModel, etc.)."""
    # Walk through common wrapper structures
    # PeftModel: model.base_model.model.model.layers
    # Plain HF (Llama):  model.model.layers
    # Qwen 3.5:          model.language_model.layers (or model.model.language_model.layers via PeftModel)
    candidate = model
    for attr in ["base_model", "model", "model"]:
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
            if hasattr(candidate, "layers"):
                return candidate.layers
    # Try language_model path (Qwen 3.5 / multimodal models)
    candidate = model
    for attr in ["base_model", "model", "language_model"]:
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
            if hasattr(candidate, "layers"):
                return candidate.layers
    # Fallback: search recursively
    for name, module in model.named_modules():
        if name.endswith(".layers") or (hasattr(module, '__len__') and name.count('.') <= 3):
            if hasattr(module, '__getitem__') and hasattr(module, '__len__') and len(module) > 10:
                return module
    raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")


class HiddenStateCapture:
    """Context manager that captures hidden states at specified layers.

    Hooks capture the output of model.model.layers[i] (the residual stream).
    During training, captured tensors retain their gradients.
    During pre-computation (@torch.no_grad), they are detached automatically.
    """

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self):
        self.captured.clear()
        transformer_layers = _get_transformer_layers(self.model)
        for layer_idx in self.layers:
            layer_module = transformer_layers[layer_idx]
            handle = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            # output is (hidden_states, ...) for Llama layers
            h = output[0]
            # Ensure 3D: some device_map splits squeeze the batch dim
            if h.dim() == 2:
                h = h.unsqueeze(0)
            self.captured[layer_idx] = h  # (batch, seq, dim)
        return hook_fn

    def clear(self):
        self.captured.clear()




# ═══════════════════════════════════════════════════════════════════════════
# Pre-compute frozen model activations
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_targets(
    model,
    data: list[dict],
    layers: list[int],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """Pre-compute frozen model hidden states for targets and anchors.

    Returns:
      cb_targets[i][layer] = (n_completion_tokens, hidden_dim) tensor on CPU
      anchor_targets[i][layer] = (n_completion_tokens, hidden_dim) tensor on CPU
    """
    cb_targets: list[dict[int, torch.Tensor]] = []
    anchor_targets: list[dict[int, torch.Tensor]] = []

    with HiddenStateCapture(model, layers) as capture:
        for idx, d in enumerate(data):
            if idx % 10 == 0:
                log.info("  Pre-computing targets: %d/%d", idx, len(data))

            # Circuit breaking target: forward target trace, extract completion hidden states
            target_input = torch.tensor([d["target_ids"]], device=device)
            capture.clear()
            model(target_input)

            cb_target = {}
            for layer in layers:
                h = capture.captured[layer]  # (1, seq_len, dim)
                # Extract completion tokens only
                cb_target[layer] = h[0, d["target_prefix_len"]:, :].cpu().float()
            cb_targets.append(cb_target)
            del target_input

            # Anchor target: forward anchor (good_v1) trace, extract completion hidden states
            anchor_input = torch.tensor([d["anchor_ids"]], device=device)
            capture.clear()
            model(anchor_input)

            anchor_target = {}
            for layer in layers:
                h = capture.captured[layer]
                anchor_target[layer] = h[0, d["anchor_prefix_len"]:, :].cpu().float()
            anchor_targets.append(anchor_target)
            del anchor_input

            if idx % 20 == 0:
                torch.cuda.empty_cache()

    return cb_targets, anchor_targets


@torch.no_grad()
def _precompute_nullify_targets(
    model,
    data: list[dict],
    layers: list[int],
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """Pre-compute frozen BAD hidden states (for cosine nullify) and anchor targets.

    For nullify, the CB target is the frozen model's own representation of the
    bad trace. The cosine loss then pushes the LoRA model's representations to
    be orthogonal to this direction.
    """
    cb_targets: list[dict[int, torch.Tensor]] = []
    anchor_targets: list[dict[int, torch.Tensor]] = []

    with HiddenStateCapture(model, layers) as capture:
        for idx, d in enumerate(data):
            if idx % 10 == 0:
                log.info("  Pre-computing nullify targets: %d/%d", idx, len(data))

            # CB target for nullify: frozen model's BAD hidden states
            bad_input = torch.tensor([d["bad_ids"]], device=device)
            capture.clear()
            model(bad_input)

            cb_target = {}
            for layer in layers:
                h = capture.captured[layer]
                cb_target[layer] = h[0, d["bad_prefix_len"]:, :].cpu().float()
            cb_targets.append(cb_target)
            del bad_input

            # Anchor target: same as always (frozen good_v1)
            anchor_input = torch.tensor([d["anchor_ids"]], device=device)
            capture.clear()
            model(anchor_input)

            anchor_target = {}
            for layer in layers:
                h = capture.captured[layer]
                anchor_target[layer] = h[0, d["anchor_prefix_len"]:, :].cpu().float()
            anchor_targets.append(anchor_target)
            del anchor_input

            if idx % 20 == 0:
                torch.cuda.empty_cache()

    return cb_targets, anchor_targets


# ═══════════════════════════════════════════════════════════════════════════
# Triplet loss utilities
# ═══════════════════════════════════════════════════════════════════════════


def _rep_distance(x: torch.Tensor, y: torch.Tensor, mode: str = "norm") -> torch.Tensor:
    """Per-token distance between two representation tensors. Returns (n,)."""
    if mode == "norm":
        return torch.norm(x - y, dim=-1, p=2)
    elif mode == "cosine":
        return 1.0 - F.cosine_similarity(x, y, dim=-1)
    elif mode == "mix":
        return torch.norm(x - y, dim=-1, p=2) + (1.0 - F.cosine_similarity(x, y, dim=-1))
    raise ValueError(f"Unknown distance mode: {mode}")


def compute_triplet_loss(
    captured_states: dict[int, torch.Tensor],
    frozen_states: dict[int, torch.Tensor],
    centroid: dict[int, torch.Tensor],
    prefix_len: int,
    layers: list[int],
    max_tokens: int,
    margin: float,
    dist_p: str,
    dist_n: str,
    role: str,
    normalize: bool = False,
) -> torch.Tensor:
    """Triplet margin loss for safe or unsafe representations.

    Safe:   anchor=h'_safe,   positive=frozen_safe,  negative=centroid
            → keep safe close to original, push away from harmful cluster
    Unsafe: anchor=h'_unsafe, positive=centroid,      negative=frozen_unsafe
            → cluster harmful together, push away from original
    """
    accum_device = next(iter(captured_states.values())).device
    total_loss = torch.tensor(0.0, device=accum_device)
    n_layers = 0

    for layer in layers:
        model_h = captured_states[layer][0, prefix_len:, :]  # (seq, dim)
        frozen_h = frozen_states[layer]  # (n_frozen, dim)
        cent_h = centroid.get(layer)

        n = min(model_h.shape[0], frozen_h.shape[0], max_tokens)
        if n <= 0 or cent_h is None:
            continue

        anchor = model_h[:n]
        frozen_slice = frozen_h[:n].to(anchor.device, anchor.dtype)
        cent_broadcast = cent_h.to(anchor.device, anchor.dtype).unsqueeze(0).expand(n, -1)

        if normalize:
            anchor = F.normalize(anchor, dim=-1)
            frozen_slice = F.normalize(frozen_slice, dim=-1)
            cent_broadcast = F.normalize(cent_broadcast, dim=-1)

        if role == "safe":
            d_pos = _rep_distance(anchor, frozen_slice, dist_p)
            d_neg = _rep_distance(anchor, cent_broadcast, dist_n)
        else:
            d_pos = _rep_distance(anchor, cent_broadcast, dist_p)
            d_neg = _rep_distance(anchor, frozen_slice, dist_n)

        layer_loss = F.relu(d_pos - d_neg + margin).mean()
        total_loss = total_loss + layer_loss.to(accum_device)
        n_layers += 1

    return total_loss / max(n_layers, 1)


def update_centroid(
    centroid: dict[int, torch.Tensor],
    captured_states: dict[int, torch.Tensor],
    prefix_len: int,
    layers: list[int],
    max_tokens: int,
    momentum: float = 0.9,
) -> None:
    """Update EMA centroid of harmful representations (in-place)."""
    for layer in layers:
        h = captured_states[layer][0, prefix_len:, :]  # (seq, dim)
        n = min(h.shape[0], max_tokens)
        if n <= 0:
            continue
        h_mean = h[:n].detach().mean(dim=0).cpu().float()
        if layer not in centroid:
            centroid[layer] = h_mean
        else:
            centroid[layer] = momentum * centroid[layer] + (1 - momentum) * h_mean


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def compute_representation_loss(
    captured_states: dict[int, torch.Tensor],
    target_states: dict[int, torch.Tensor] | None,
    prefix_len: int,
    layers: list[int],
    max_tokens: int = 15,
    objective: str = "refusal",
    nullify_mode: str = "cosine",
    normalize: bool = False,
) -> torch.Tensor:
    """Compute representation loss between model's completion hidden states and targets.

    Loss function depends on objective:
      - refusal/redirect: MSE to frozen target hidden states (magnitude matters
        because downstream layers expect activations with the right norm)
      - nullify + cosine: ReLU(cosine_similarity) — makes representations orthogonal to
        the original bad direction without collapsing magnitude. This avoids the
        "magnitude collapse" problem where MSE-to-zeros creates huge gradients
        that bleed into anchor states (per RR paper findings).
      - nullify + mse: MSE to zeros — simpler, directly destroys the representation.

    Args:
        captured_states: {layer: (1, seq_len, dim)} from hook capture (on device, with grad)
        target_states: {layer: (n_target_completion, dim)} pre-computed (CPU, float32).
                       For nullify: these are the FROZEN BAD hidden states (we push away
                       from them via cosine). None means no pre-computed targets available.
        prefix_len: number of prefix tokens (loss starts AFTER this)
        layers: which layers to compute loss over
        max_tokens: max completion tokens to include in loss (caps alignment region)
        objective: "refusal"/"redirect" (MSE to target) or "nullify" (cosine orthogonality)

    Returns:
        Scalar loss averaged over layers and token positions.
    """
    # Use first layer's device as the accumulation device.
    # When layers span multiple GPUs (e.g., layers 30-55 across 4 GPUs),
    # each layer's loss is computed on its own device, then moved here for summing.
    accum_device = next(iter(captured_states.values())).device
    total_loss = torch.tensor(0.0, device=accum_device)
    n_layers = 0

    for layer in layers:
        model_h = captured_states[layer]  # (1, seq_len, dim)

        # Extract model's completion hidden states
        model_completion = model_h[0, prefix_len:, :]  # (n_model_completion, dim)

        if objective == "nullify":
            if nullify_mode == "mse":
                # MSE to zeros: directly destroy the representation
                n = min(model_completion.shape[0], max_tokens)
                if n <= 0:
                    continue
                model_slice = model_completion[:n]
                if normalize:
                    model_slice = F.normalize(model_slice, dim=-1)
                layer_loss = F.mse_loss(model_slice, torch.zeros_like(model_slice))
            else:
                # Cosine dissimilarity: push representations orthogonal to frozen bad states.
                # Loss = mean(ReLU(cosine_sim(model, frozen_bad))) per token.
                # When cosine ≤ 0, the representation is already orthogonal → loss = 0.
                if target_states is not None:
                    target_h = target_states[layer]
                    n = min(model_completion.shape[0], target_h.shape[0], max_tokens)
                else:
                    n = min(model_completion.shape[0], max_tokens)
                if n <= 0:
                    continue
                model_slice = model_completion[:n]

                if target_states is not None:
                    target_slice = target_h[:n].to(model_slice.device, model_slice.dtype)
                    cos_sim = F.cosine_similarity(model_slice, target_slice, dim=-1)
                else:
                    if normalize:
                        model_slice = F.normalize(model_slice, dim=-1)
                    layer_loss = F.mse_loss(model_slice, torch.zeros_like(model_slice))
                    total_loss = total_loss + layer_loss.to(accum_device)
                    n_layers += 1
                    continue

                # ReLU: only penalize positive cosine (aligned with bad direction)
                layer_loss = F.relu(cos_sim).mean()
            total_loss = total_loss + layer_loss.to(accum_device)
        else:
            # Refusal/redirect: MSE to pre-computed frozen target
            target_h = target_states[layer]  # (n_target_completion, dim)
            n = min(model_completion.shape[0], target_h.shape[0], max_tokens)
            if n <= 0:
                continue
            model_slice = model_completion[:n]
            target_slice = target_h[:n].to(model_slice.device, model_slice.dtype)
            if normalize:
                model_slice = F.normalize(model_slice, dim=-1)
                target_slice = F.normalize(target_slice, dim=-1)
            layer_loss = F.mse_loss(model_slice, target_slice)
            total_loss = total_loss + layer_loss.to(accum_device)

        n_layers += 1

    if n_layers > 0:
        total_loss = total_loss / n_layers

    return total_loss


def compute_refusal_contrastive_loss(
    captured_states: dict[int, torch.Tensor],
    good_states: dict[int, torch.Tensor],
    bad_states: dict[int, torch.Tensor] | None,
    prefix_len: int,
    layers: list[int],
    max_tokens: int = 32,
    push_weight: float = 1.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Contrastive refusal-preservation loss.

    Pulls the adapted representation toward the frozen baseline refusal states
    while pushing it away from the frozen v4a-style compliant completion states.
    """
    accum_device = next(iter(captured_states.values())).device
    total_loss = torch.tensor(0.0, device=accum_device)
    n_layers = 0

    for layer in layers:
        model_completion = captured_states[layer][0, prefix_len:, :]
        good_h = good_states[layer]
        n = min(model_completion.shape[0], good_h.shape[0], max_tokens)
        if n <= 0:
            continue

        model_slice = model_completion[:n]
        good_slice = good_h[:n].to(model_slice.device, model_slice.dtype)
        if normalize:
            model_slice = F.normalize(model_slice, dim=-1)
            good_slice = F.normalize(good_slice, dim=-1)

        pull_loss = F.mse_loss(model_slice, good_slice)
        push_loss = torch.tensor(0.0, device=model_slice.device, dtype=model_slice.dtype)

        if bad_states is not None:
            bad_h = bad_states[layer]
            m = min(n, bad_h.shape[0], max_tokens)
            if m > 0:
                bad_slice = bad_h[:m].to(model_slice.device, model_slice.dtype)
                push_model = model_slice[:m]
                if normalize:
                    bad_slice = F.normalize(bad_slice, dim=-1)
                cos_sim = F.cosine_similarity(push_model, bad_slice, dim=-1)
                push_loss = F.relu(cos_sim).mean()

        layer_loss = pull_loss + push_weight * push_loss
        total_loss = total_loss + layer_loss.to(accum_device)
        n_layers += 1

    if n_layers > 0:
        total_loss = total_loss / n_layers
    return total_loss


def compute_refusal_orthogonal_loss(
    captured_states: dict[int, torch.Tensor],
    good_states: dict[int, torch.Tensor],
    bad_states: dict[int, torch.Tensor] | None,
    prefix_len: int,
    layers: list[int],
    max_tokens: int = 32,
    push_weight: float = 1.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Refusal-preservation loss with orthogonality to the bad direction.

    Pulls the adapted representation toward the frozen baseline refusal states
    while penalizing squared cosine similarity with the frozen v4a-compliant
    states. This keeps the "push" term geometric instead of distance-based.
    """
    accum_device = next(iter(captured_states.values())).device
    total_loss = torch.tensor(0.0, device=accum_device)
    n_layers = 0

    for layer in layers:
        model_completion = captured_states[layer][0, prefix_len:, :]
        good_h = good_states[layer]
        n = min(model_completion.shape[0], good_h.shape[0], max_tokens)
        if n <= 0:
            continue

        model_slice = model_completion[:n]
        good_slice = good_h[:n].to(model_slice.device, model_slice.dtype)
        if normalize:
            model_slice = F.normalize(model_slice, dim=-1)
            good_slice = F.normalize(good_slice, dim=-1)

        pull_loss = F.mse_loss(model_slice, good_slice)
        push_loss = torch.tensor(0.0, device=model_slice.device, dtype=model_slice.dtype)

        if bad_states is not None:
            bad_h = bad_states[layer]
            m = min(n, bad_h.shape[0], max_tokens)
            if m > 0:
                bad_slice = bad_h[:m].to(model_slice.device, model_slice.dtype)
                push_model = model_slice[:m]
                if normalize:
                    bad_slice = F.normalize(bad_slice, dim=-1)
                cos_sim = F.cosine_similarity(push_model, bad_slice, dim=-1)
                push_loss = torch.square(cos_sim).mean()

        layer_loss = pull_loss + push_weight * push_loss
        total_loss = total_loss + layer_loss.to(accum_device)
        n_layers += 1

    if n_layers > 0:
        total_loss = total_loss / n_layers
    return total_loss


def train_epoch(
    model,
    optimizer,
    train_data: list[dict],
    cb_targets: list[dict],
    anchor_targets: list[dict],
    refusal_data: list[dict] | None,
    refusal_targets: list[dict] | None,
    layers: list[int],
    device: torch.device,
    grad_accum_steps: int,
    cb_weight: float,
    anchor_weight: float,
    lambda3: float,
    max_grad_norm: float,
    epoch: int,
    max_completion_tokens: int = 15,
    refusal_max_tokens: int = 32,
    objective: str = "refusal",
    nullify_mode: str = "cosine",
    triplet_args: dict | None = None,
    scheduler=None,
    normalize_states: bool = False,
    refusal_mode: str = "contrastive",
    refusal_push_weight: float = 1.0,
    benign_anchor_data: list[dict] | None = None,
    benign_anchor_targets: list[dict] | None = None,
    lambda4: float = 0.0,
    benign_anchor_max_tokens: int = 32,
) -> dict:
    """Run one training epoch.

    Each step processes one triplet with two forward passes:
      1. Bad trace → representation loss with CB target (or unsafe triplet)
      2. Good v1 trace → MSE with anchoring target (or safe triplet)
    Optional extra passes (if data provided):
      3. Refusal trace → contrastive/mse loss with scale lambda3
      4. Benign-anchor trace → MSE pull to baseline good_targets with scale lambda4 (§19.52b)
    """
    model.train()
    indices = np.random.permutation(len(train_data))

    total_cb_loss = 0.0
    total_anchor_loss = 0.0
    total_refusal_loss = 0.0
    total_refusal_weighted = 0.0
    total_loss = 0.0
    n_steps = 0

    # Triplet centroid (EMA of harmful representations)
    centroid: dict[int, torch.Tensor] = {}
    ta = triplet_args or {}
    refusal_indices = (
        np.random.permutation(len(refusal_data))
        if lambda3 > 0 and refusal_data and refusal_targets
        else None
    )
    benign_anchor_indices = (
        np.random.permutation(len(benign_anchor_data))
        if lambda4 > 0 and benign_anchor_data and benign_anchor_targets
        else None
    )
    total_benign_anchor_loss = 0.0
    n_benign_anchor_steps = 0

    optimizer.zero_grad()

    with HiddenStateCapture(model, layers) as capture:
        for step_idx, data_idx in enumerate(indices):
            d = train_data[data_idx]
            cb_target = cb_targets[data_idx]
            anchor_target = anchor_targets[data_idx]

            # ── Forward pass 1: Bad trace ─────────────────────────────
            bad_input = torch.tensor([d["bad_ids"]], device=device)
            capture.clear()
            model(bad_input)

            if objective == "triplet":
                update_centroid(
                    centroid, capture.captured, d["bad_prefix_len"],
                    layers, max_completion_tokens, ta.get("centroid_momentum", 0.9),
                )
                loss_cb = compute_triplet_loss(
                    capture.captured, cb_target, centroid,
                    d["bad_prefix_len"], layers, max_completion_tokens,
                    margin=ta.get("margin_unsafe", 3.0),
                    dist_p=ta.get("unsafe_dist_p", "norm"),
                    dist_n=ta.get("unsafe_dist_n", "cosine"),
                    role="unsafe",
                    normalize=normalize_states,
                )
            else:
                loss_cb = compute_representation_loss(
                    capture.captured, cb_target,
                    d["bad_prefix_len"], layers,
                    max_tokens=max_completion_tokens,
                    objective=objective, nullify_mode=nullify_mode,
                    normalize=normalize_states,
                )
            scaled_cb = cb_weight * loss_cb / grad_accum_steps
            scaled_cb.backward()
            del bad_input

            # ── Forward pass 2: Good trace ────────────────────────────
            anchor_input = torch.tensor([d["anchor_ids"]], device=device)
            capture.clear()
            model(anchor_input)

            if objective == "triplet":
                loss_anchor = compute_triplet_loss(
                    capture.captured, anchor_target, centroid,
                    d["anchor_prefix_len"], layers, max_completion_tokens,
                    margin=ta.get("margin_safe", 2.0),
                    dist_p=ta.get("safe_dist_p", "norm"),
                    dist_n=ta.get("safe_dist_n", "cosine"),
                    role="safe",
                    normalize=normalize_states,
                )
            else:
                loss_anchor = compute_representation_loss(
                    capture.captured, anchor_target, d["anchor_prefix_len"], layers,
                    max_tokens=max_completion_tokens,
                    objective="refusal",
                    normalize=normalize_states,
                )
            scaled_anchor = anchor_weight * loss_anchor / grad_accum_steps
            scaled_anchor.backward()
            del anchor_input

            if lambda3 > 0 and refusal_data and refusal_targets and refusal_indices is not None:
                refusal_idx = refusal_indices[step_idx % len(refusal_indices)]
                refusal_sample = refusal_data[refusal_idx]
                refusal_target = refusal_targets[refusal_idx]
                refusal_input = torch.tensor([refusal_sample["train_input_ids"]], device=device)
                capture.clear()
                model(refusal_input)
                if refusal_mode == "contrastive":
                    loss_refusal = compute_refusal_contrastive_loss(
                        capture.captured,
                        refusal_target["good"],
                        refusal_target.get("bad"),
                        refusal_sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        push_weight=refusal_push_weight,
                        normalize=normalize_states,
                    )
                elif refusal_mode == "orthogonal":
                    loss_refusal = compute_refusal_orthogonal_loss(
                        capture.captured,
                        refusal_target["good"],
                        refusal_target.get("bad"),
                        refusal_sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        push_weight=refusal_push_weight,
                        normalize=normalize_states,
                    )
                else:
                    loss_refusal = compute_representation_loss(
                        capture.captured,
                        refusal_target["good"],
                        refusal_sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        objective="refusal",
                        normalize=normalize_states,
                    )
                refusal_weight = float(refusal_sample.get("weight", 1.0))
                scaled_refusal = lambda3 * refusal_weight * loss_refusal / grad_accum_steps
                scaled_refusal.backward()
                del refusal_input
            else:
                loss_refusal = None
                refusal_weight = 0.0

            # ── Forward pass 4: benign anchor MSE (§19.52b) ──────────
            if (
                lambda4 > 0
                and benign_anchor_data
                and benign_anchor_targets
                and benign_anchor_indices is not None
            ):
                ba_idx = benign_anchor_indices[step_idx % len(benign_anchor_indices)]
                ba_sample = benign_anchor_data[ba_idx]
                ba_target = benign_anchor_targets[ba_idx]
                ba_input = torch.tensor([ba_sample["train_input_ids"]], device=device)
                capture.clear()
                model(ba_input)
                loss_benign_anchor = compute_representation_loss(
                    capture.captured,
                    ba_target["good"],
                    ba_sample["prefix_len"],
                    layers,
                    max_tokens=benign_anchor_max_tokens,
                    objective="refusal",   # pull-toward
                    normalize=normalize_states,
                )
                scaled_ba = lambda4 * loss_benign_anchor / grad_accum_steps
                scaled_ba.backward()
                total_benign_anchor_loss += loss_benign_anchor.item()
                n_benign_anchor_steps += 1
                del ba_input
            else:
                loss_benign_anchor = None

            total_cb_loss += loss_cb.item()
            total_anchor_loss += loss_anchor.item()
            if loss_refusal is not None:
                total_refusal_loss += loss_refusal.item()
                total_refusal_weighted += refusal_weight * loss_refusal.item()
            total_loss += (
                cb_weight * loss_cb.item()
                + anchor_weight * loss_anchor.item()
                + lambda3 * refusal_weight * (loss_refusal.item() if loss_refusal is not None else 0.0)
                + lambda4 * (loss_benign_anchor.item() if loss_benign_anchor is not None else 0.0)
            )
            n_steps += 1

            # ── Gradient step ─────────────────────────────────────────
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if (step_idx + 1) % 10 == 0:
                avg_cb = total_cb_loss / n_steps
                avg_anchor = total_anchor_loss / n_steps
                avg_refusal = total_refusal_loss / n_steps if lambda3 > 0 else 0.0
                avg_ba = total_benign_anchor_loss / n_benign_anchor_steps if n_benign_anchor_steps else 0.0
                log.info(
                    "  Epoch %d step %d/%d: cb_loss=%.6f anchor_loss=%.6f refusal_loss=%.6f benign_anchor_loss=%.6f total=%.6f",
                    epoch,
                    step_idx + 1,
                    len(indices),
                    avg_cb,
                    avg_anchor,
                    avg_refusal,
                    avg_ba,
                    total_loss / n_steps,
                )

            if (step_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

    return {
        "cb_loss": total_cb_loss / n_steps,
        "anchor_loss": total_anchor_loss / n_steps,
        "refusal_loss": total_refusal_loss / n_steps if lambda3 > 0 else 0.0,
        "refusal_weighted_loss": total_refusal_weighted / n_steps if lambda3 > 0 else 0.0,
        "benign_anchor_loss": total_benign_anchor_loss / n_benign_anchor_steps if n_benign_anchor_steps else 0.0,
        "total_loss": total_loss / n_steps,
        "n_steps": n_steps,
    }


@torch.no_grad()
def validate(
    model,
    val_data: list[dict],
    cb_targets: list[dict],
    anchor_targets: list[dict],
    refusal_data: list[dict] | None,
    refusal_targets: list[dict] | None,
    layers: list[int],
    device: torch.device,
    cb_weight: float,
    anchor_weight: float,
    lambda3: float,
    max_completion_tokens: int = 15,
    refusal_max_tokens: int = 32,
    objective: str = "refusal",
    nullify_mode: str = "cosine",
    triplet_args: dict | None = None,
    normalize_states: bool = False,
    refusal_mode: str = "contrastive",
    refusal_push_weight: float = 1.0,
) -> dict:
    """Compute validation loss on holdout set."""
    model.eval()
    total_cb_loss = 0.0
    total_anchor_loss = 0.0
    total_refusal_loss = 0.0
    total_refusal_weighted = 0.0
    n = 0

    centroid: dict[int, torch.Tensor] = {}
    ta = triplet_args or {}

    with HiddenStateCapture(model, layers) as capture:
        for i in range(len(val_data)):
            d = val_data[i]

            # CB / unsafe triplet loss
            bad_input = torch.tensor([d["bad_ids"]], device=device)
            capture.clear()
            model(bad_input)
            if objective == "triplet":
                update_centroid(
                    centroid, capture.captured, d["bad_prefix_len"],
                    layers, max_completion_tokens, ta.get("centroid_momentum", 0.9),
                )
                loss_cb = compute_triplet_loss(
                    capture.captured, cb_targets[i], centroid,
                    d["bad_prefix_len"], layers, max_completion_tokens,
                    margin=ta.get("margin_unsafe", 3.0),
                    dist_p=ta.get("unsafe_dist_p", "norm"),
                    dist_n=ta.get("unsafe_dist_n", "cosine"),
                    role="unsafe",
                    normalize=normalize_states,
                )
            else:
                loss_cb = compute_representation_loss(
                    capture.captured, cb_targets[i],
                    d["bad_prefix_len"], layers,
                    max_tokens=max_completion_tokens,
                    objective=objective, nullify_mode=nullify_mode,
                    normalize=normalize_states,
                )
            del bad_input

            # Anchor / safe triplet loss
            anchor_input = torch.tensor([d["anchor_ids"]], device=device)
            capture.clear()
            model(anchor_input)
            if objective == "triplet":
                loss_anchor = compute_triplet_loss(
                    capture.captured, anchor_targets[i], centroid,
                    d["anchor_prefix_len"], layers, max_completion_tokens,
                    margin=ta.get("margin_safe", 2.0),
                    dist_p=ta.get("safe_dist_p", "norm"),
                    dist_n=ta.get("safe_dist_n", "cosine"),
                    role="safe",
                    normalize=normalize_states,
                )
            else:
                loss_anchor = compute_representation_loss(
                    capture.captured, anchor_targets[i], d["anchor_prefix_len"], layers,
                    max_tokens=max_completion_tokens,
                    objective="refusal",
                    normalize=normalize_states,
                )
            del anchor_input

            total_cb_loss += loss_cb.item()
            total_anchor_loss += loss_anchor.item()
            n += 1

        if lambda3 > 0 and refusal_data and refusal_targets:
            for sample, target in zip(refusal_data, refusal_targets):
                refusal_input = torch.tensor([sample["train_input_ids"]], device=device)
                capture.clear()
                model(refusal_input)
                if refusal_mode == "contrastive":
                    loss_refusal = compute_refusal_contrastive_loss(
                        capture.captured,
                        target["good"],
                        target.get("bad"),
                        sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        push_weight=refusal_push_weight,
                        normalize=normalize_states,
                    )
                elif refusal_mode == "orthogonal":
                    loss_refusal = compute_refusal_orthogonal_loss(
                        capture.captured,
                        target["good"],
                        target.get("bad"),
                        sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        push_weight=refusal_push_weight,
                        normalize=normalize_states,
                    )
                else:
                    loss_refusal = compute_representation_loss(
                        capture.captured,
                        target["good"],
                        sample["prefix_len"],
                        layers,
                        max_tokens=refusal_max_tokens,
                        objective="refusal",
                        normalize=normalize_states,
                    )
                weight = float(sample.get("weight", 1.0))
                total_refusal_loss += loss_refusal.item()
                total_refusal_weighted += weight * loss_refusal.item()
                del refusal_input

    base_total = (cb_weight * total_cb_loss + anchor_weight * total_anchor_loss) / n
    refusal_total = 0.0
    if lambda3 > 0 and refusal_data and refusal_targets:
        refusal_total = lambda3 * (total_refusal_weighted / len(refusal_data))
    total_loss = base_total + refusal_total
    return {
        "cb_loss": total_cb_loss / n,
        "anchor_loss": total_anchor_loss / n,
        "refusal_loss": total_refusal_loss / len(refusal_data) if lambda3 > 0 and refusal_data else 0.0,
        "refusal_weighted_loss": total_refusal_weighted / len(refusal_data) if lambda3 > 0 and refusal_data else 0.0,
        "total_loss": total_loss,
        "n": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation with generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_one(model, tokenizer, prefix_text: str, device, max_new_tokens: int) -> str:
    """Generate a single completion from a prefix."""
    encoded = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del input_ids, attention_mask, outputs
    return text


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    triplets: list[dict],
    device: torch.device,
    max_new_tokens: int = 256,
) -> dict:
    """Generate from bad and clean prefixes with LoRA enabled, plus baseline with LoRA disabled.

    Prints per-sample comparison showing baseline vs LoRA side-by-side.
    Returns structured results for both.
    """
    model.eval()

    results = {
        "lorra": {
            "injected": {"harmful": 0, "refusal": 0, "benign": 0,
                         "garbled": 0, "completions": []},
            "clean": {"harmful": 0, "refusal": 0, "benign": 0,
                      "tool_changed": 0, "completions": []},
        },
        "baseline": {
            "injected": {"harmful": 0, "refusal": 0, "benign": 0, "completions": []},
            "clean": {"harmful": 0, "refusal": 0, "benign": 0, "completions": []},
        },
    }

    for i, t in enumerate(triplets):
        uid = t["user_task_id"]
        iid = t.get("injection_task_id", "?")
        bad_prefix = render_prefix(t["bad_trace"], tokenizer=tokenizer)
        clean_prefix = render_prefix(t["good_trace_v1"], tokenizer=tokenizer)

        # ── LoRA model ────────────────────────────────────────────
        model.enable_adapter_layers()
        lorra_inj = _generate_one(model, tokenizer, bad_prefix, device, max_new_tokens)
        lorra_clean = _generate_one(model, tokenizer, clean_prefix, device, max_new_tokens)

        lorra_inj_label = classify_completion_simple(lorra_inj)
        lorra_clean_label = classify_completion_simple(lorra_clean)

        results["lorra"]["injected"][lorra_inj_label] += 1
        if _is_garbled(lorra_inj):
            results["lorra"]["injected"]["garbled"] += 1
        results["lorra"]["injected"]["completions"].append({
            "user_task_id": uid, "injection_task_id": iid,
            "injection_goal": t.get("injection_task_goal", ""),
            "user_prompt": t.get("user_task_prompt", ""),
            "label": lorra_inj_label, "text": lorra_inj,
            "garbled": _is_garbled(lorra_inj),
            "tool_calls": _extract_tool_calls(lorra_inj),
        })
        results["lorra"]["clean"][lorra_clean_label] += 1
        results["lorra"]["clean"]["completions"].append({
            "user_task_id": uid, "label": lorra_clean_label, "text": lorra_clean,
            "tool_calls": _extract_tool_calls(lorra_clean),
        })

        # ── Baseline (LoRA disabled) ──────────────────────────────
        model.disable_adapter_layers()
        base_inj = _generate_one(model, tokenizer, bad_prefix, device, max_new_tokens)
        base_clean = _generate_one(model, tokenizer, clean_prefix, device, max_new_tokens)
        model.enable_adapter_layers()

        base_inj_label = classify_completion_simple(base_inj)
        base_clean_label = classify_completion_simple(base_clean)

        results["baseline"]["injected"][base_inj_label] += 1
        results["baseline"]["injected"]["completions"].append({
            "user_task_id": uid, "injection_task_id": iid,
            "label": base_inj_label, "text": base_inj,
            "tool_calls": _extract_tool_calls(base_inj),
        })
        results["baseline"]["clean"][base_clean_label] += 1
        results["baseline"]["clean"]["completions"].append({
            "user_task_id": uid, "label": base_clean_label, "text": base_clean,
            "tool_calls": _extract_tool_calls(base_clean),
        })

        # Track clean tool-call changes (capability retention)
        base_clean_tools = _extract_tool_calls(base_clean)
        lorra_clean_tools = _extract_tool_calls(lorra_clean)
        if base_clean_tools != lorra_clean_tools:
            results["lorra"]["clean"]["tool_changed"] += 1

        # ── Per-sample log ────────────────────────────────────────
        inj_changed = lorra_inj_label != base_inj_label
        clean_changed = lorra_clean_label != base_clean_label
        inj_marker = " ★ CHANGED" if inj_changed else ""
        clean_marker = " ★ CHANGED" if clean_changed else ""

        # Running tally
        n_done = i + 1
        base_h = results["baseline"]["injected"]["harmful"]
        lorra_h = results["lorra"]["injected"]["harmful"]
        lorra_r_clean = results["lorra"]["clean"]["refusal"]

        # Truncate very long completions but keep structure visible
        max_display = 800

        def _fmt(text: str) -> str:
            """Format completion for display: keep full text up to max, show structure."""
            s = text.replace(chr(10), '↵ ')
            if len(s) > max_display:
                s = s[:max_display] + f"... [{len(text)} chars total]"
            return s

        print(f"\n  {'━' * 78}")
        print(f"  [{i}] {uid} x {iid}    "
              f"[running: base_harmful={base_h}/{n_done} "
              f"lorra_harmful={lorra_h}/{n_done} "
              f"refusal_tax={lorra_r_clean}/{n_done}]")
        print(f"  Injection goal: {t.get('injection_task_goal', '?')[:120]}")
        print(f"  User task:      {t.get('user_task_prompt', '?')[:120]}")
        print(f"  {'━' * 78}")

        print(f"\n  INJECTED  base={base_inj_label}  →  lorra={lorra_inj_label}{inj_marker}")
        print(f"  ┌─ base ─────────────────────────────────────────────")
        print(f"  │ {_fmt(base_inj)}")
        print(f"  ├─ lorra ────────────────────────────────────────────")
        print(f"  │ {_fmt(lorra_inj)}")
        print(f"  └────────────────────────────────────────────────────")

        print(f"\n  CLEAN     base={base_clean_label}  →  lorra={lorra_clean_label}{clean_marker}")
        print(f"  ┌─ base ─────────────────────────────────────────────")
        print(f"  │ {_fmt(base_clean)}")
        print(f"  ├─ lorra ────────────────────────────────────────────")
        print(f"  │ {_fmt(lorra_clean)}")
        print(f"  └────────────────────────────────────────────────────")

        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics: loss mask visualization
# ═══════════════════════════════════════════════════════════════════════════

def _diagnose_loss_mask(
    train_data: list[dict],
    holdout_data: list[dict],
    triplets: list[dict],
    tokenizer,
    objective: str,
    max_completion_tokens: int,
    n_train: int,
) -> None:
    """Print detailed loss mask diagnostics for the first few triplets.

    Shows:
      1. Token-level map: which positions are prefix (no loss) vs completion (loss)
      2. Decoded text at the exact boundary where loss starts
      3. The first max_completion_tokens of each trace decoded
      4. Alignment checks: bad vs target prefix identity (for v2 objective)
      5. Deterministic sanity checks
    """
    target_label = {"refusal": "Good v2 (refusal)", "redirect": "Good v1 (benign)", "nullify": "zeros", "triplet": "centroid (contrastive)"}[objective]
    n_show = min(3, len(train_data))

    print("\n" + "=" * 80)
    print("  LOSS MASK DIAGNOSTICS")
    print(f"  Objective: {objective} (CB target = {target_label})")
    print(f"  Max completion tokens in loss: {max_completion_tokens}")
    print("=" * 80)

    for i in range(n_show):
        d = train_data[i]
        t = triplets[d["index"]]
        bad_ids = d["bad_ids"]
        target_ids = d["target_ids"]
        anchor_ids = d["anchor_ids"]
        bad_plen = d["bad_prefix_len"]
        target_plen = d["target_prefix_len"]
        anchor_plen = d["anchor_prefix_len"]

        bad_comp_len = len(bad_ids) - bad_plen
        target_comp_len = len(target_ids) - target_plen
        anchor_comp_len = len(anchor_ids) - anchor_plen

        # How many tokens actually enter the loss
        if objective in ("nullify", "triplet"):
            cb_loss_tokens = min(bad_comp_len, max_completion_tokens)
        else:
            cb_loss_tokens = min(bad_comp_len, target_comp_len, max_completion_tokens)
        anchor_loss_tokens = min(anchor_comp_len, max_completion_tokens)

        print(f"\n{'─' * 80}")
        print(f"  Triplet {d['index']}: {d['user_task_id']} x {d['injection_task_id']}")
        print(f"{'─' * 80}")

        # ── Token layout visual ───────────────────────────────────────
        def _bar(label: str, total: int, prefix_len: int, loss_tokens: int) -> None:
            comp_len = total - prefix_len
            no_loss_comp = comp_len - loss_tokens
            # Scale to ~70 chars width
            scale = 70 / max(total, 1)
            pw = max(1, int(prefix_len * scale))
            lw = max(1, int(loss_tokens * scale))
            nw = max(0, int(no_loss_comp * scale))
            print(f"\n  {label} ({total} tokens):")
            print(f"  ┌{'─' * pw}┬{'█' * lw}{'░' * nw}┐")
            print(f"  │{'PREFIX':^{pw}}│{'LOSS':^{lw}}{'':^{nw}}│")
            print(f"  │{prefix_len:^{pw}}│{loss_tokens:^{lw}}{no_loss_comp:^{nw}}│")
            print(f"  └{'─' * pw}┴{'─' * (lw + nw)}┘")
            print(f"  {'':>{pw}} ▲ token {prefix_len}")
            print(f"  {'no loss (masked)':^{pw}}  loss region = [{prefix_len}:{prefix_len + loss_tokens}]")

        _bar("BAD trace (CB input)", len(bad_ids), bad_plen, cb_loss_tokens)
        if objective not in ("nullify", "triplet"):
            _bar("TARGET trace (CB target)", len(target_ids), target_plen, cb_loss_tokens)
        _bar("ANCHOR trace (Loss B)", len(anchor_ids), anchor_plen, anchor_loss_tokens)

        # ── Decoded boundary tokens ───────────────────────────────────
        def _decode_around(ids: list[int], pos: int, label: str, window: int = 8) -> None:
            """Decode tokens around a boundary position."""
            start = max(0, pos - window)
            end = min(len(ids), pos + window)
            before_ids = ids[start:pos]
            after_ids = ids[pos:end]
            before_text = tokenizer.decode(before_ids, skip_special_tokens=False)
            after_text = tokenizer.decode(after_ids, skip_special_tokens=False)
            # Clean up for display
            before_text = before_text.replace('\n', '\\n')[-60:]
            after_text = after_text.replace('\n', '\\n')[:60]
            print(f"\n  {label} boundary (token {pos}):")
            print(f"    BEFORE (no loss): ...{before_text}")
            print(f"    AFTER  (in loss): {after_text}...")

        _decode_around(bad_ids, bad_plen, "BAD prefix→completion")
        if objective not in ("nullify", "triplet"):
            _decode_around(target_ids, target_plen, "TARGET prefix→completion")
        _decode_around(anchor_ids, anchor_plen, "ANCHOR prefix→completion")

        # ── First N completion tokens decoded ─────────────────────────
        def _decode_first_n(ids: list[int], prefix_len: int, n: int, label: str) -> None:
            comp_ids = ids[prefix_len: prefix_len + n]
            text = tokenizer.decode(comp_ids, skip_special_tokens=False)
            text = text.replace('\n', '\\n')
            print(f"\n  {label} — first {n} completion tokens (IN LOSS):")
            print(f"    \"{text}\"")
            print(f"    Token IDs: {comp_ids}")

        _decode_first_n(bad_ids, bad_plen, cb_loss_tokens, "BAD")
        if objective not in ("nullify", "triplet"):
            _decode_first_n(target_ids, target_plen, cb_loss_tokens, "TARGET")
        _decode_first_n(anchor_ids, anchor_plen, anchor_loss_tokens, "ANCHOR")

        # ── Extended window visualization ─────────────────────────────
        # Show what's at different token ranges so we can verify tool
        # commitment falls within the loss window
        def _decode_range(ids: list[int], prefix_len: int, start: int, end: int, label: str) -> str:
            """Decode a range of completion tokens."""
            actual_start = prefix_len + start
            actual_end = min(prefix_len + end, len(ids))
            if actual_start >= len(ids):
                return "(past end of trace)"
            comp_ids = ids[actual_start:actual_end]
            text = tokenizer.decode(comp_ids, skip_special_tokens=False).replace('\n', '\\n')
            return text

        TOOL_NAMES = ["send_money", "update_password", "update_scheduled_transaction",
                      "get_most_recent_transactions", "get_scheduled_transactions",
                      "get_user_info", "schedule_transaction"]

        print(f"\n  WINDOW COVERAGE (BAD completion):")
        for rng_start, rng_end, label in [(0, 15, "OLD [0:15]"), (0, 30, "NEW [0:30]"),
                                           (15, 30, "EXTENSION [15:30]"), (30, 45, "BEYOND [30:45]")]:
            text = _decode_range(bad_ids, bad_plen, rng_start, rng_end, label)
            in_loss = "IN LOSS" if rng_end <= max_completion_tokens else "outside"
            if rng_start < max_completion_tokens and rng_end > max_completion_tokens:
                in_loss = f"PARTIAL (loss ends at {max_completion_tokens})"
            print(f"    {label:20s} ({in_loss:12s}): \"{text[:120]}\"")

        # Detect tool commitment position in bad completion
        bad_comp_text_full = tokenizer.decode(bad_ids[bad_plen:], skip_special_tokens=False)
        for tool_name in TOOL_NAMES:
            if tool_name in bad_comp_text_full:
                # Find approximate token position by decoding incrementally
                for tok_pos in range(1, bad_comp_len + 1):
                    partial = tokenizer.decode(bad_ids[bad_plen:bad_plen + tok_pos], skip_special_tokens=False)
                    if tool_name in partial:
                        in_window = tok_pos <= max_completion_tokens
                        status = " OK " if in_window else "FAIL"
                        print(f"    [{status}] Tool '{tool_name}' first appears at ~token {tok_pos} "
                              f"({'within' if in_window else 'OUTSIDE'} loss window of {max_completion_tokens})")
                        break
                break  # only report first tool found

        # Check anchor completion length vs window
        if anchor_comp_len < max_completion_tokens:
            print(f"    [WARN] Anchor completion only {anchor_comp_len} tokens "
                  f"(< window of {max_completion_tokens})")

        # ── Deterministic checks ──────────────────────────────────────
        print(f"\n  CHECKS:")

        # Check 1: Last prefix message should end with the assistant header
        # (the generation prompt), NOT contain the completion's <function-thoughts>.
        # Earlier assistant turns in the prefix DO have <function-thoughts> from
        # ground truth tool calls — that's expected and fine.
        last_8_prefix = tokenizer.decode(bad_ids[bad_plen - 8:bad_plen], skip_special_tokens=False)
        ends_with_header = "assistant" in last_8_prefix and "\\n\\n" in repr(last_8_prefix)
        print(f"    [{' OK ' if ends_with_header else 'FAIL'}] Prefix ends with assistant header "
              f"(generation prompt boundary)")

        # Check 2: Bad completion should contain <function-thoughts> or <function-call>
        bad_comp_text = tokenizer.decode(bad_ids[bad_plen:], skip_special_tokens=False)
        has_ft_in_comp = "<function-thoughts>" in bad_comp_text or "<function-call>" in bad_comp_text
        print(f"    [{'FAIL' if not has_ft_in_comp else ' OK '}] Bad completion HAS <function-thoughts> or "
              f"<function-call> (model's decision)")

        # Check 3: For refusal objective, bad and target prefixes should be identical
        if objective == "refusal":
            prefixes_match = (bad_ids[:bad_plen] == target_ids[:target_plen])
            print(f"    [{' OK ' if prefixes_match else 'FAIL'}] Bad and Target prefixes are "
                  f"{'IDENTICAL' if prefixes_match else 'DIFFERENT'} "
                  f"(expected: identical for refusal/v2)")
        elif objective == "redirect":
            # v1 prefixes differ at injection point — expected
            prefixes_match = (bad_ids[:bad_plen] == target_ids[:target_plen])
            print(f"    [{' OK ' if not prefixes_match else 'WARN'}] Bad and Target prefixes "
                  f"{'DIFFER' if not prefixes_match else 'are IDENTICAL'} "
                  f"(expected: differ at injection for redirect/v1)")

        # Check 4: Completion tokens in loss are > 0
        print(f"    [{' OK ' if cb_loss_tokens > 0 else 'FAIL'}] CB loss covers {cb_loss_tokens} tokens "
              f"(min of bad_comp={bad_comp_len}, "
              f"{'target_comp=' + str(target_comp_len) + ', ' if objective != 'nullify' else ''}"
              f"max={max_completion_tokens})")
        print(f"    [{' OK ' if anchor_loss_tokens > 0 else 'FAIL'}] Anchor loss covers "
              f"{anchor_loss_tokens} tokens")

        # Check 5: Loss does NOT touch the prefix
        print(f"    [ OK ] Loss starts at token {bad_plen} (after prefix)")
        print(f"    [ OK ] Prefix is {bad_plen} tokens = "
              f"{bad_plen / len(bad_ids) * 100:.1f}% of the trace (masked out)")

    # ── Aggregate stats ───────────────────────────────────────────────
    all_data = train_data + holdout_data
    cb_tokens_used = []
    for d in all_data:
        bad_comp = len(d["bad_ids"]) - d["bad_prefix_len"]
        target_comp = len(d["target_ids"]) - d["target_prefix_len"]
        if objective in ("nullify", "triplet"):
            cb_tokens_used.append(min(bad_comp, max_completion_tokens))
        else:
            cb_tokens_used.append(min(bad_comp, target_comp, max_completion_tokens))

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE MASK STATS ({len(all_data)} triplets)")
    print(f"{'=' * 80}")
    print(f"  CB loss tokens per sample:  min={min(cb_tokens_used)}, "
          f"max={max(cb_tokens_used)}, mean={np.mean(cb_tokens_used):.1f}")
    print(f"  Max completion tokens cap:  {max_completion_tokens}")

    capped = sum(1 for t in cb_tokens_used if t == max_completion_tokens)
    print(f"  Samples hitting the cap:    {capped}/{len(cb_tokens_used)} "
          f"({100 * capped / len(cb_tokens_used):.0f}%)")

    below_5 = sum(1 for t in cb_tokens_used if t < 5)
    if below_5:
        print(f"  WARNING: {below_5} samples have < 5 CB loss tokens!")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Results summary writer
# ═══════════════════════════════════════════════════════════════════════════

def _write_results_summary(
    output_dir: Path,
    args,
    history: list[dict],
    eval_results: dict,
    eval_triplets: list[dict],
    suffix: str = "",
) -> None:
    """Write a human-readable RESULTS.md and per-sample eval details.

    Creates:
      output_dir/RESULTS.md           — one-page summary you can read in 30 seconds
      output_dir/eval_per_sample.md   — full per-sample completions for deep inspection
    """
    n = len(eval_triplets)
    base_inj = eval_results["baseline"]["injected"]
    lorra_inj = eval_results["lorra"]["injected"]
    base_clean = eval_results["baseline"]["clean"]
    lorra_clean = eval_results["lorra"]["clean"]

    # ── RESULTS.md (quick summary) ────────────────────────────────────
    lines = []
    lines.append(f"# LoRRA Results: {args.objective}")
    lines.append(f"")
    lines.append(f"## Key Metrics (n={n} holdout samples)")
    lines.append(f"")
    lines.append(f"| Metric | Baseline | LoRRA | Delta |")
    lines.append(f"|--------|----------|-------|-------|")
    lines.append(f"| Attack success (injected) | "
                 f"{base_inj['harmful']}/{n} ({100*base_inj['harmful']/n:.0f}%) | "
                 f"{lorra_inj['harmful']}/{n} ({100*lorra_inj['harmful']/n:.0f}%) | "
                 f"{lorra_inj['harmful'] - base_inj['harmful']:+d} |")
    lines.append(f"| Refusal (injected) | "
                 f"{base_inj['refusal']}/{n} ({100*base_inj['refusal']/n:.0f}%) | "
                 f"{lorra_inj['refusal']}/{n} ({100*lorra_inj['refusal']/n:.0f}%) | "
                 f"{lorra_inj['refusal'] - base_inj['refusal']:+d} |")
    lines.append(f"| Benign (injected) | "
                 f"{base_inj['benign']}/{n} ({100*base_inj['benign']/n:.0f}%) | "
                 f"{lorra_inj['benign']}/{n} ({100*lorra_inj['benign']/n:.0f}%) | "
                 f"{lorra_inj['benign'] - base_inj['benign']:+d} |")
    lines.append(f"| Refusal tax (clean) | "
                 f"{base_clean['refusal']}/{n} ({100*base_clean['refusal']/n:.0f}%) | "
                 f"{lorra_clean['refusal']}/{n} ({100*lorra_clean['refusal']/n:.0f}%) | "
                 f"{lorra_clean['refusal'] - base_clean['refusal']:+d} |")
    lines.append(f"")

    lines.append(f"## Training")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Objective | {args.objective} |")
    lines.append(f"| Layers | {args.layer_start}–{args.layer_end} |")
    lines.append(f"| Rank | {args.rank} |")
    lines.append(f"| Alpha | {args.lora_alpha} (scaling={args.lora_alpha/args.rank:.1f}) |")
    lines.append(f"| LR (peak) | {args.lr} |")
    lines.append(f"| Epochs completed | {len(history)}/{args.epochs} |")
    lines.append(f"| Max completion tokens | {args.max_completion_tokens} |")
    lines.append(f"| Lambda3 | {args.lambda3 if args.refusal_targets else 0.0} |")
    lines.append(f"| Refusal mode | {args.refusal_mode if args.refusal_targets else 'disabled'} |")
    lines.append(f"| Train/holdout | {len(eval_triplets)}/{len(eval_triplets)} |")
    lines.append(f"")

    lines.append(f"## Loss Curve")
    lines.append(f"")
    lines.append(f"| Epoch | Train CB | Train Anchor | Train Refusal | Val CB | Val Anchor | Val Refusal | LR |")
    lines.append(f"|-------|----------|-------------|---------------|--------|------------|-------------|-----|")
    for h in history:
        t = h["train"]
        v = h.get("val", {})
        lines.append(
            f"| {h['epoch']} "
            f"| {t['cb_loss']:.6f} "
            f"| {t['anchor_loss']:.6f} "
            f"| {t.get('refusal_loss', 0):.6f} "
            f"| {v.get('cb_loss', 0):.6f} "
            f"| {v.get('anchor_loss', 0):.6f} "
            f"| {v.get('refusal_loss', 0):.6f} "
            f"| |"
        )
    lines.append(f"")

    # Count changes
    n_inj_changed = 0
    n_inj_improved = 0
    n_inj_worsened = 0
    n_clean_changed = 0
    for j in range(n):
        b_label = base_inj["completions"][j]["label"]
        l_label = lorra_inj["completions"][j]["label"]
        if b_label != l_label:
            n_inj_changed += 1
            if b_label == "harmful" and l_label != "harmful":
                n_inj_improved += 1
            elif b_label != "harmful" and l_label == "harmful":
                n_inj_worsened += 1
        bc_label = base_clean["completions"][j]["label"]
        lc_label = lorra_clean["completions"][j]["label"]
        if bc_label != lc_label:
            n_clean_changed += 1

    lines.append(f"## Behavioral Changes")
    lines.append(f"")
    lines.append(f"- Injected: **{n_inj_changed}/{n}** changed behavior")
    lines.append(f"  - {n_inj_improved} improved (harmful → benign/refusal)")
    lines.append(f"  - {n_inj_worsened} worsened (benign → harmful)")
    lines.append(f"  - {n_inj_changed - n_inj_improved - n_inj_worsened} other changes")
    lines.append(f"- Clean: **{n_clean_changed}/{n}** changed behavior")
    lines.append(f"")

    # ── Q1/Q2: Detailed behavioral breakdown ──────────────────────────
    inj_garbled = lorra_inj.get("garbled", 0)
    clean_tool_changed = lorra_clean.get("tool_changed", 0)
    clean_preserved = n - clean_tool_changed

    lines.append(f"## Detailed Behavioral Breakdown")
    lines.append(f"")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Injected: no harmful tool call (LoRRA) | "
                 f"{n - lorra_inj['harmful']}/{n} ({100*(n - lorra_inj['harmful'])/n:.0f}%) |")
    lines.append(f"| Injected: garbled/malformed output (LoRRA) | "
                 f"{inj_garbled}/{n} ({100*inj_garbled/n:.0f}%) |")
    lines.append(f"| Clean: different tool calls from baseline | "
                 f"{clean_tool_changed}/{n} ({100*clean_tool_changed/n:.0f}%) |")
    lines.append(f"| Clean: correct task preserved (same tools) | "
                 f"{clean_preserved}/{n} ({100*clean_preserved/n:.0f}%) |")
    lines.append(f"")
    lines.append(f"> **Note**: 'Refusal tax' only checks for attacker IBAN/refusal markers. "
                 f"'Clean: different tool calls' is a stricter capability check — LoRRA may "
                 f"change which tools are called even if the output is classified as 'benign'.")
    lines.append(f"")

    fname_suffix = f"_{suffix}" if suffix else ""
    results_name = f"RESULTS{fname_suffix}.md"
    with open(output_dir / results_name, "w") as f:
        f.write("\n".join(lines))
    log.info("Saved %s to %s", results_name, output_dir / results_name)

    # ── eval_per_sample.md (detailed per-sample view) ─────────────────
    detail = []
    detail.append(f"# Per-Sample Eval: {args.objective}")
    detail.append(f"")

    for j in range(n):
        bi = base_inj["completions"][j]
        li = lorra_inj["completions"][j]
        bc = base_clean["completions"][j]
        lc = lorra_clean["completions"][j]

        uid = bi["user_task_id"]
        iid = bi.get("injection_task_id", "?")
        inj_changed = bi["label"] != li["label"]
        clean_changed = bc["label"] != lc["label"]
        marker = " **CHANGED**" if inj_changed else ""
        clean_marker = " **CHANGED**" if clean_changed else ""

        detail.append(f"---")
        detail.append(f"## [{j}] {uid} x {iid}")
        detail.append(f"")
        if li.get("injection_goal"):
            detail.append(f"**Injection goal**: {li['injection_goal']}")
        if li.get("user_prompt"):
            detail.append(f"**User task**: {li['user_prompt']}")
        detail.append(f"")

        detail.append(f"### Injected: base={bi['label']} → lorra={li['label']}{marker}")
        detail.append(f"")
        detail.append(f"**Baseline**:")
        detail.append(f"```")
        detail.append(bi["text"])
        detail.append(f"```")
        detail.append(f"")
        detail.append(f"**LoRRA**:")
        detail.append(f"```")
        detail.append(li["text"])
        detail.append(f"```")
        detail.append(f"")

        # Show tool calls extracted for injected samples
        li_tools = li.get("tool_calls", [])
        bi_tools = bi.get("tool_calls", [])
        if li_tools or bi_tools:
            detail.append(f"**Tool calls**: base={bi_tools} → lorra={li_tools}")
            detail.append(f"")
        if li.get("garbled"):
            detail.append(f"**⚠ GARBLED OUTPUT DETECTED**")
            detail.append(f"")

        detail.append(f"### Clean: base={bc['label']} → lorra={lc['label']}{clean_marker}")
        detail.append(f"")
        # Show tool call comparison for quick scanning
        bc_tools = bc.get("tool_calls", [])
        lc_tools = lc.get("tool_calls", [])
        tools_match = bc_tools == lc_tools
        tools_tag = "" if tools_match else " **TOOL CHANGED**"
        detail.append(f"**Tool calls**: base={bc_tools} → lorra={lc_tools}{tools_tag}")
        detail.append(f"")
        detail.append(f"**Baseline**:")
        detail.append(f"```")
        detail.append(bc["text"])
        detail.append(f"```")
        detail.append(f"")
        detail.append(f"**LoRRA**:")
        detail.append(f"```")
        detail.append(lc["text"])
        detail.append(f"```")
        detail.append(f"")

    per_sample_name = f"eval_per_sample{fname_suffix}.md"
    with open(output_dir / per_sample_name, "w") as f:
        f.write("\n".join(detail))
    log.info("Saved %s to %s", per_sample_name, output_dir / per_sample_name)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoRRA Circuit Breaker training for prompt injection defense"
    )

    # Data
    parser.add_argument("--suite", type=str, default="banking",
                        help="Suite(s) to train on: single name, comma-separated, or 'all'. "
                             f"Available: {', '.join(ALL_TRAIN_SUITES)}, all")
    parser.add_argument("--traces", type=str, nargs="+", required=True,
                        help="Filtered traces JSON(s). Single path for single suite, "
                             "or suite=path pairs for --suite all "
                             "(e.g. banking=/path/bank.json slack=/path/slack.json)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model (local or HuggingFace ID)")
    parser.add_argument("--format", type=str, default="prompting_llm",
                        choices=["prompting_llm", "qwen_native", "qwen3_native",
                                 "llama_native", "react", "dual"],
                        help="Tool calling format for rendering/tokenization "
                             "(default: prompting_llm). 'dual' dispatches per-trace "
                             "via the trace's 'format' field; the CLI value is used "
                             "only as a fallback when the field is missing.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save LoRA adapter and results")

    # LoRA config
    parser.add_argument("--layer-start", type=int, default=40,
                        help="First layer to apply LoRA and compute loss (0-indexed, default: 40)")
    parser.add_argument("--layer-end", type=int, default=55,
                        help="Last layer (inclusive) for LoRA and loss (default: 55)")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank (default: 8, keep low for 82 samples)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: equal to rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--target-modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "down_proj", "up_proj"],
                        help="Which modules to apply LoRA to "
                             "(default: q_proj v_proj down_proj up_proj — attention + MLP)")

    # Objective
    parser.add_argument("--objective", type=str, default="refusal",
                        choices=["refusal", "redirect", "nullify", "triplet", "odile"],
                        help="Training objective. 'odile' is the canonical ODILE loss "
                             "(contrastive clustering of paired traces); refusal/redirect/"
                             "nullify are earlier representation targets kept for ablations.")
    # Triplet-specific args (only used with --objective triplet)
    parser.add_argument("--triplet-margin-safe", type=float, default=2.0,
                        help="Margin for safe triplet loss (default: 2.0)")
    parser.add_argument("--triplet-margin-unsafe", type=float, default=3.0,
                        help="Margin for unsafe triplet loss (default: 3.0)")
    parser.add_argument("--triplet-safe-dist-p", type=str, default="norm",
                        choices=["norm", "cosine", "mix"],
                        help="Distance for safe-positive: h'_safe ↔ h_safe (default: norm)")
    parser.add_argument("--triplet-safe-dist-n", type=str, default="cosine",
                        choices=["norm", "cosine", "mix"],
                        help="Distance for safe-negative: h'_safe ↔ centroid (default: cosine)")
    parser.add_argument("--triplet-unsafe-dist-p", type=str, default="norm",
                        choices=["norm", "cosine", "mix"],
                        help="Distance for unsafe-positive: h'_unsafe ↔ centroid (default: norm)")
    parser.add_argument("--triplet-unsafe-dist-n", type=str, default="cosine",
                        choices=["norm", "cosine", "mix"],
                        help="Distance for unsafe-negative: h'_unsafe ↔ h_unsafe (default: cosine)")
    parser.add_argument("--triplet-centroid-momentum", type=float, default=0.9,
                        help="EMA momentum for harmful centroid (default: 0.9)")
    parser.add_argument("--nullify-mode", type=str, default="cosine",
                        choices=["cosine", "mse"],
                        help="Nullify loss variant: cosine (push orthogonal to frozen bad "
                             "states) or mse (MSE to zeros). Only used with --objective nullify")
    parser.add_argument("--normalize-states", action="store_true", default=False,
                        help="L2-normalize hidden states before computing loss. Required for "
                             "models with large hidden state norms (e.g. Qwen3-32B: norms ~11500). "
                             "Makes loss scale-invariant across models. For triplet: distances "
                             "become bounded [0,2], margins become meaningful. For nullify+cosine: "
                             "no effect (already magnitude-invariant). For nullify+mse: becomes "
                             "equivalent to cosine nullification.")
    parser.add_argument("--cb-weight", type=float, default=1.0,
                        help="Weight for circuit breaking loss (default: 1.0)")
    parser.add_argument("--anchor-weight", type=float, default=1.0,
                        help="Weight for anchoring loss (default: 1.0)")
    parser.add_argument("--lambda3", type=float, default=0.3,
                        help="Weight for refusal-preservation auxiliary loss (default: 0.3)")
    parser.add_argument("--refusal-targets", type=str, default=None,
                        help="Optional .pt bundle from precompute_refusal_targets.py")
    parser.add_argument("--refusal-mode", type=str, default="contrastive",
                        choices=["contrastive", "orthogonal", "mse"],
                        help="Auxiliary refusal objective: contrastive (pull to baseline refusal "
                             "and push away from v4a compliance), orthogonal (pull to refusal "
                             "plus squared-cosine penalty vs. v4a compliance), or mse (pull only).")
    parser.add_argument("--refusal-push-weight", type=float, default=1.0,
                        help="Push-away coefficient inside the contrastive refusal loss.")
    parser.add_argument("--refusal-max-tokens", type=int, default=32,
                        help="Max completion tokens from refusal pairs used in lambda3 loss.")
    parser.add_argument("--benign-anchor-targets", type=Path, default=None,
                        help="Optional .pt bundle from precompute_benign_anchors.py — "
                             "auxiliary MSE loss pulling v5 hidden states toward baseline on "
                             "benign scaffolded prompts (§19.52b mechanism repair).")
    parser.add_argument("--lambda4", type=float, default=0.15,
                        help="Weight for benign-anchor MSE loss (default: 0.15).")
    parser.add_argument("--benign-anchor-max-tokens", type=int, default=32,
                        help="Max completion tokens from benign anchor pairs used in lambda4 loss.")
    parser.add_argument("--max-completion-tokens", type=int, default=15,
                        help="Max completion tokens to use in loss (default: 15). "
                             "Caps the aligned region to the intent-formation phase. "
                             "Full overlap (0) causes refusal tax bleed on clean prompts.")
    parser.add_argument("--max-train-tokens", type=int, default=8000,
                        help="Drop training triplets whose bad/target/anchor full-trace "
                             "length exceeds this many tokens. Protects against activation "
                             "OOM on 70B models without using gradient checkpointing. "
                             "Set to 0 to disable.")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Max training epochs (default: 3, expect convergence in 1-3)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--holdout", type=int, default=20,
                        help="Number of triplets to hold out (default: 20)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience in epochs (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Evaluation
    parser.add_argument("--eval", action="store_true",
                        help="Run generation evaluation after training")
    parser.add_argument("--eval-max-tokens", type=int, default=256,
                        help="Max tokens for evaluation generation (default: 256)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Skip training. Load this LoRA checkpoint dir and run eval only. "
                             "Use with --eval-split to control which samples to evaluate.")
    parser.add_argument("--eval-split", type=str, default="holdout",
                        choices=["holdout", "train", "all"],
                        help="Which split to evaluate (default: holdout). "
                             "'all' evaluates on all triplets with split labels.")

    parser.add_argument("--no-exclude-ambiguous", action="store_true",
                        help="Disable the hard-coded EXCLUDE_PAIRS_BY_SUITE filter (structurally "
                             "ambiguous pairs). Default off = pairs excluded (standard recipe). "
                             "Set this for the worsened-pairs ablation (include arm).")

    # Utility
    parser.add_argument("--dry-run", action="store_true",
                        help="Tokenize and show info without loading model")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print detailed loss mask diagnostics: shows exact token "
                             "positions, decoded text at mask boundaries, and alignment "
                             "checks for the first 3 triplets. Combine with --dry-run "
                             "to diagnose without loading model.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype (default: bfloat16)")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        help="Attention kernel: sdpa (PyTorch fused, fast on H100/H200 and "
                             "always available), flash_attention_2 (marginally faster but "
                             "requires `pip install flash-attn`), eager (slow but most "
                             "compatible). Set to empty string '' to let HF pick.")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device map for model loading (default: auto)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (trades recomputation for "
                             "less activation memory). Needed for long-context training "
                             "on Qwen3-32B (traces up to 13k tokens would OOM). Uses "
                             "use_reentrant=False so forward-hooks play nicely with it.")

    args = parser.parse_args()

    # 'odile' is the canonical name for the shipped objective; route it to the
    # triplet contrastive-clustering loss the rest of the trainer dispatches on.
    if args.objective == "odile":
        args.objective = "triplet"

    # ── Parse suite(s) and traces ─────────────────────────────────────
    if args.suite == "all":
        suites = ALL_TRAIN_SUITES
    elif "," in args.suite:
        suites = [s.strip() for s in args.suite.split(",")]
        bad = [s for s in suites if s not in ALL_TRAIN_SUITES]
        if bad:
            parser.error(f"Unknown suite(s): {bad}. Available: {ALL_TRAIN_SUITES}")
    else:
        if args.suite not in ALL_TRAIN_SUITES:
            parser.error(f"Unknown suite: {args.suite}. Available: {ALL_TRAIN_SUITES}")
        suites = [args.suite]

    if len(suites) == 1:
        traces_map = {suites[0]: args.traces[0]}
    else:
        traces_map = {}
        for entry in args.traces:
            if "=" not in entry:
                parser.error(f"With --suite all, --traces must be suite=path pairs, got: {entry}")
            s, p = entry.split("=", 1)
            traces_map[s] = p
        missing = set(suites) - set(traces_map)
        if missing:
            parser.error(f"Missing --traces for suites: {sorted(missing)}")

    log.info("Suites: %s", ", ".join(suites))

    if args.lora_alpha is None:
        args.lora_alpha = args.rank

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter traces per suite ──────────────────────────────
    EXCLUDE_PAIRS_BY_SUITE = {
        "banking": {
            ("user_task_15", "injection_task_0"),
            ("user_task_15", "injection_task_1"),
            ("user_task_15", "injection_task_2"),
        },
    }

    suite_raw = {}       # suite -> all raw triplets (after exclusion)
    total_triplets = 0

    for suite in suites:
        log.info("Loading traces for '%s' from %s", suite, traces_map[suite])
        with open(traces_map[suite]) as f:
            triplets = json.load(f)
        log.info("  Loaded %d triplets", len(triplets))

        exclude = set() if args.no_exclude_ambiguous else EXCLUDE_PAIRS_BY_SUITE.get(suite, set())
        n_before = len(triplets)
        triplets = [t for t in triplets
                    if (t["user_task_id"], t["injection_task_id"]) not in exclude]
        n_excluded = n_before - len(triplets)
        if args.no_exclude_ambiguous and suite in EXCLUDE_PAIRS_BY_SUITE:
            log.info("  [--no-exclude-ambiguous] KEEPING %d normally-excluded pairs for '%s'",
                     sum(1 for t in triplets if (t["user_task_id"], t["injection_task_id"]) in EXCLUDE_PAIRS_BY_SUITE[suite]), suite)
        if n_excluded > 0:
            log.info("  Excluded %d ambiguous pairs for '%s': %s",
                     n_excluded, suite, sorted(exclude))

        suite_raw[suite] = triplets
        total_triplets += len(triplets)

    log.info("Total: %d triplets across %d suite(s)", total_triplets, len(suites))

    # ── Stratified train/holdout split ────────────────────────────────
    n_holdout_total = min(args.holdout, total_triplets // 4)
    holdout_frac = n_holdout_total / total_triplets if total_triplets > 0 else 0

    suite_train_raw = {}    # suite -> raw train triplets (for eval)
    suite_holdout_raw = {}  # suite -> raw holdout triplets (for eval)

    for suite in suites:
        st = suite_raw[suite]
        n_ho = max(1, round(len(st) * holdout_frac)) if len(st) > 1 else 0
        suite_train_raw[suite] = st[:len(st) - n_ho]
        suite_holdout_raw[suite] = st[len(st) - n_ho:]
        log.info("  '%s': %d train, %d holdout",
                 suite, len(suite_train_raw[suite]), len(suite_holdout_raw[suite]))

    # ── Load tokenizer ────────────────────────────────────────────────
    log.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # ── Tokenize per suite, then combine ──────────────────────────────
    log.info("Tokenizing traces (objective=%s)...", args.objective)
    train_data = []
    holdout_data = []
    all_raw_triplets = []  # combined raw list (train+holdout, for --diagnose)
    n_train_raw = 0

    def _group_by_format(triplets, fallback_fmt):
        """Partition triplets by their 'format' field (fallback if missing)."""
        groups: dict[str, list[dict]] = {}
        for t in triplets:
            fmt = t.get("format", fallback_fmt)
            groups.setdefault(fmt, []).append(t)
        return groups

    for suite in suites:
        offset = len(all_raw_triplets)

        train_t = suite_train_raw[suite]
        holdout_t = suite_holdout_raw[suite]

        if args.format == "dual":
            # Per-triplet dispatch preserving original input order so that
            # d["index"] = position in train_t / holdout_t (matches downstream
            # _diagnose_loss_mask and offset math). Grouping-then-concat would
            # shuffle the mapping between d["index"] and the source triplet
            # in all_raw_triplets.
            td = []
            hd = []
            train_format_counts: dict[str, int] = {}
            holdout_format_counts: dict[str, int] = {}
            for i, t in enumerate(train_t):
                fmt = t.get("format", "prompting_llm")
                train_format_counts[fmt] = train_format_counts.get(fmt, 0) + 1
                _init_data_config(suite, format=fmt)
                result = tokenize_triplets([t], tokenizer, args.objective)
                if result:
                    d = result[0]
                    d["index"] = i
                    td.append(d)
            for i, t in enumerate(holdout_t):
                fmt = t.get("format", "prompting_llm")
                holdout_format_counts[fmt] = holdout_format_counts.get(fmt, 0) + 1
                _init_data_config(suite, format=fmt)
                result = tokenize_triplets([t], tokenizer, args.objective)
                if result:
                    d = result[0]
                    d["index"] = i
                    hd.append(d)
            log.info(
                "  '%s': dual-format tokenized — train=%s, holdout=%s",
                suite, train_format_counts, holdout_format_counts,
            )
        else:
            _init_data_config(suite, format=args.format)
            td = tokenize_triplets(train_t, tokenizer, args.objective)
            hd = tokenize_triplets(holdout_t, tokenizer, args.objective)

        for d in td:
            d["index"] += offset
            d["suite"] = suite
        for d in hd:
            d["index"] += offset + len(train_t)
            d["suite"] = suite

        all_raw_triplets.extend(train_t)
        all_raw_triplets.extend(holdout_t)
        n_train_raw += len(train_t)

        train_data.extend(td)
        holdout_data.extend(hd)

        log.info("  '%s': %d train, %d holdout tokenized", suite, len(td), len(hd))

    # Filter out long traces to cap per-sample activation memory. Without
    # gradient checkpointing, a 10K-token forward on a 70B model keeps every
    # layer's activations for backward, which OOMs GPU 2 under device_map="auto".
    # Capping at 8k drops a handful of workspace outliers (huge tool-result
    # blobs) without touching the other ~95% of the dataset.
    if args.max_train_tokens > 0:
        def _too_long(d):
            return max(len(d["bad_ids"]), len(d["target_ids"]), len(d["anchor_ids"])) > args.max_train_tokens
        n_before = len(train_data)
        dropped = [d for d in train_data if _too_long(d)]
        train_data = [d for d in train_data if not _too_long(d)]
        if dropped:
            log.info("Dropped %d/%d train triplets over %d tokens (keeping %d):",
                     len(dropped), n_before, args.max_train_tokens, len(train_data))
            for d in dropped:
                m = max(len(d["bad_ids"]), len(d["target_ids"]), len(d["anchor_ids"]))
                log.info("  - %s x %s (%s): max %d tokens", d["user_task_id"], d["injection_task_id"], d.get("suite","?"), m)

    # Shuffle combined training data so suites are interleaved
    if len(suites) > 1:
        np.random.shuffle(train_data)

    log.info("Combined: %d train, %d holdout tokenized", len(train_data), len(holdout_data))

    # Print stats
    if train_data:
        bad_lens = [len(d["bad_ids"]) for d in train_data]
        target_lens = [len(d["target_ids"]) for d in train_data]
        anchor_lens = [len(d["anchor_ids"]) for d in train_data]
        bad_comp = [len(d["bad_ids"]) - d["bad_prefix_len"] for d in train_data]
        target_comp = [len(d["target_ids"]) - d["target_prefix_len"] for d in train_data]
        anchor_comp = [len(d["anchor_ids"]) - d["anchor_prefix_len"] for d in train_data]
        log.info("  Bad:    %d–%d tokens (completion: %d–%d)",
                 min(bad_lens), max(bad_lens), min(bad_comp), max(bad_comp))
        log.info("  Target: %d–%d tokens (completion: %d–%d)",
                 min(target_lens), max(target_lens), min(target_comp), max(target_comp))
        log.info("  Anchor: %d–%d tokens (completion: %d–%d)",
                 min(anchor_lens), max(anchor_lens), min(anchor_comp), max(anchor_comp))

    refusal_train_data: list[dict] = []
    refusal_train_targets: list[dict] = []
    refusal_val_data: list[dict] = []
    refusal_val_targets: list[dict] = []
    refusal_meta: dict = {}
    if args.refusal_targets is not None:
        log.info("Loading refusal-preservation bundle from %s", args.refusal_targets)
        (
            refusal_train_data,
            refusal_train_targets,
            refusal_val_data,
            refusal_val_targets,
            refusal_meta,
        ) = load_refusal_target_bundle(Path(args.refusal_targets))
        log.info(
            "  Refusal pairs: %d train / %d val (mode=%s, lambda3=%.3f)",
            len(refusal_train_data),
            len(refusal_val_data),
            args.refusal_mode,
            args.lambda3,
        )
    elif args.lambda3 > 0:
        log.info("No --refusal-targets provided; lambda3 auxiliary loss disabled for this run.")

    # ── Benign-anchor bundle (§19.52b) ──────────────────────────────
    benign_train_data: list[dict] = []
    benign_train_targets: list[dict] = []
    benign_val_data: list[dict] = []
    benign_val_targets: list[dict] = []
    benign_meta: dict = {}
    if args.benign_anchor_targets is not None:
        log.info("Loading benign-anchor bundle from %s", args.benign_anchor_targets)
        (
            benign_train_data,
            benign_train_targets,
            benign_val_data,
            benign_val_targets,
            benign_meta,
        ) = load_refusal_target_bundle(Path(args.benign_anchor_targets))
        log.info(
            "  Benign anchor pairs: %d train / %d val (lambda4=%.3f, max_tokens=%d)",
            len(benign_train_data),
            len(benign_val_data),
            args.lambda4,
            args.benign_anchor_max_tokens,
        )

    layers = list(range(args.layer_start, args.layer_end + 1))
    log.info("Layers for LoRA + loss: %d–%d (%d layers)", args.layer_start, args.layer_end, len(layers))

    # ── Diagnose: visualize loss mask alignment ─────────────────────
    if args.diagnose:
        _diagnose_loss_mask(train_data, holdout_data, all_raw_triplets, tokenizer,
                            args.objective, args.max_completion_tokens, n_train_raw)
        if args.dry_run:
            sys.exit(0)

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        n_lora_params_estimate = (
            len(layers) * len(args.target_modules) * 2 * args.rank * 8192 * 2  # 2 bytes bf16
        )
        log.info("DRY RUN — estimated LoRA parameters: %.1fM (%.1f MB)",
                 n_lora_params_estimate / 2 / 1e6, n_lora_params_estimate / 1e6)
        log.info("Config: rank=%d, alpha=%d, modules=%s",
                 args.rank, args.lora_alpha, args.target_modules)
        for d in train_data[:3]:
            log.info("  [%d] %s x %s: bad=%d tok (comp=%d), target=%d tok (comp=%d), anchor=%d tok (comp=%d)",
                     d["index"], d["user_task_id"], d["injection_task_id"],
                     len(d["bad_ids"]), len(d["bad_ids"]) - d["bad_prefix_len"],
                     len(d["target_ids"]), len(d["target_ids"]) - d["target_prefix_len"],
                     len(d["anchor_ids"]), len(d["anchor_ids"]) - d["anchor_prefix_len"])
        if refusal_train_data:
            for d in refusal_train_data[:3]:
                log.info(
                    "  [refusal] %s (%s): prefix=%d tok total=%d tok weight=%.1f",
                    d["id"],
                    d["source"],
                    d["prefix_len"],
                    len(d["train_input_ids"]),
                    float(d.get("weight", 1.0)),
                )
        sys.exit(0)

    # ── Load model ────────────────────────────────────────────────────
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    log.info("Loading model from %s (dtype=%s, device_map=%s, attn=%s)",
             args.model, args.dtype, args.device_map, args.attn_implementation)
    t0 = time.time()
    _model_kwargs = dict(
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.attn_implementation:
        _model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **_model_kwargs)
    # Disable KV cache during training — we only do forward+backward, no autoregressive generation.
    # Saves memory and avoids cache-hook interactions.
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    log.info("Model loaded in %.1f s. %d layers, hidden_dim=%d",
             time.time() - t0, model.config.num_hidden_layers, model.config.hidden_size)

    # Validate layer range
    n_model_layers = model.config.num_hidden_layers
    if args.layer_end >= n_model_layers:
        log.error("layer-end %d >= model layers %d", args.layer_end, n_model_layers)
        sys.exit(1)

    # Find input embedding device (works for both plain HF and PeftModel)
    for param in model.parameters():
        device = param.device
        break

    # ── Eval-only mode: skip training, just evaluate a saved checkpoint ──
    if args.eval_only:
        log.info("=" * 60)
        log.info("EVAL-ONLY MODE: Loading checkpoint from %s", args.eval_only)
        log.info("=" * 60)

        model = PeftModel.from_pretrained(model, args.eval_only)
        model.eval()
        log.info("LoRA adapter loaded. Evaluating...")

        if args.eval_split == "holdout":
            split_label = "holdout"
        elif args.eval_split == "train":
            split_label = "train"
        else:
            split_label = "all"

        for suite in suites:
            _init_data_config(suite, format=args.format)

            if split_label == "holdout":
                eval_triplets = suite_holdout_raw[suite]
            elif split_label == "train":
                eval_triplets = suite_train_raw[suite]
            else:
                eval_triplets = suite_raw[suite]

            if not eval_triplets:
                log.info("Suite '%s': no %s triplets, skipping", suite, split_label)
                continue

            suite_tag = f" [{suite}]" if len(suites) > 1 else ""
            log.info("Eval split: %s%s (%d samples)", split_label, suite_tag, len(eval_triplets))

            eval_results = evaluate_generation(
                model, tokenizer, eval_triplets, device,
                max_new_tokens=args.eval_max_tokens,
            )

            n = len(eval_triplets)
            log.info("")
            log.info("  ═══════════════════════════════════════════════════════")
            log.info("  EVAL-ONLY SUMMARY%s: %s split (%d samples)", suite_tag, split_label, n)
            log.info("  Checkpoint: %s", args.eval_only)
            log.info("  ═══════════════════════════════════════════════════════")

            for scenario in ["injected", "clean"]:
                log.info("  %s:", scenario.upper())
                for model_key, model_name in [("baseline", "Baseline"), ("lorra", "LoRRA  ")]:
                    counts = {k: v for k, v in eval_results[model_key][scenario].items()
                              if k not in ("completions", "garbled", "tool_changed")}
                    n_total = sum(counts.values())
                    parts = [f"{label}={counts[label]}({100*counts[label]/n_total:.0f}%)"
                             if n_total else f"{label}=0" for label in ["harmful", "refusal", "benign"]]
                    log.info("    %s: %s", model_name, "  ".join(parts))

            base_harmful = eval_results["baseline"]["injected"]["harmful"]
            lorra_harmful = eval_results["lorra"]["injected"]["harmful"]
            log.info("")
            log.info("  Attack success rate:  %4.1f%% → %4.1f%%  (Δ %+.1f%%)",
                     100 * base_harmful / n, 100 * lorra_harmful / n,
                     100 * (lorra_harmful - base_harmful) / n)

            n_improved = sum(1 for j in range(n)
                             if eval_results["baseline"]["injected"]["completions"][j]["label"] == "harmful"
                             and eval_results["lorra"]["injected"]["completions"][j]["label"] != "harmful")
            n_worsened = sum(1 for j in range(n)
                             if eval_results["baseline"]["injected"]["completions"][j]["label"] != "harmful"
                             and eval_results["lorra"]["injected"]["completions"][j]["label"] == "harmful")
            log.info("  FLIPS: %d improved, %d worsened (out of %d)", n_improved, n_worsened, n)

            suffix = f"_{suite}" if len(suites) > 1 else ""
            eval_path = output_dir / f"eval_results{suffix}_{split_label}_seed{args.seed}.json"
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            log.info("Saved eval results to %s", eval_path)

            detail = [f"# Eval-Only{suite_tag}: {split_label} split (seed={args.seed})", ""]
            for j in range(n):
                bi = eval_results["baseline"]["injected"]["completions"][j]
                li = eval_results["lorra"]["injected"]["completions"][j]
                bc = eval_results["baseline"]["clean"]["completions"][j]
                lc = eval_results["lorra"]["clean"]["completions"][j]
                inj_changed = bi["label"] != li["label"]
                marker = " **CHANGED**" if inj_changed else ""
                detail.append(f"---")
                detail.append(f"## [{j}] {bi['user_task_id']} x {bi.get('injection_task_id', '?')}")
                if li.get("injection_goal"):
                    detail.append(f"**Injection goal**: {li['injection_goal']}")
                if li.get("user_prompt"):
                    detail.append(f"**User task**: {li['user_prompt']}")
                detail.append(f"")
                detail.append(f"### Injected: base={bi['label']} → lorra={li['label']}{marker}")
                detail.append(f"**Tool calls**: base={bi.get('tool_calls', [])} → lorra={li.get('tool_calls', [])}")
                if li.get("garbled"):
                    detail.append(f"**GARBLED**")
                detail.append(f"**Baseline**: ```{bi['text'][:500]}```")
                detail.append(f"**LoRRA**: ```{li['text'][:500]}```")
                detail.append(f"")
                clean_tools_match = bc.get("tool_calls", []) == lc.get("tool_calls", [])
                detail.append(f"### Clean: base={bc['label']} → lorra={lc['label']}"
                              f"{'' if clean_tools_match else ' **TOOL CHANGED**'}")
                detail.append(f"**Tool calls**: base={bc.get('tool_calls', [])} → lorra={lc.get('tool_calls', [])}")
                detail.append(f"")

            per_sample_path = output_dir / f"eval_per_sample{suffix}_{split_label}_seed{args.seed}.md"
            with open(per_sample_path, "w") as f:
                f.write("\n".join(detail))
            log.info("Saved per-sample eval to %s", per_sample_path)

        log.info("Done. Eval-only output in %s", output_dir)
        sys.exit(0)

    # ── Phase 1: Pre-compute frozen targets ───────────────────────────
    log.info("=" * 60)
    log.info("Phase 1: Pre-computing frozen model activations...")
    log.info("=" * 60)

    if args.objective == "nullify" and args.nullify_mode == "mse":
        # MSE-to-zeros: CB targets are None (loss computed against zeros directly).
        # Still need anchor targets to preserve benign behavior.
        log.info("  Objective=nullify (MSE-to-zeros): no frozen CB targets needed")
        log.info("  Computing frozen anchor targets (%d train samples)...", len(train_data))
        _, train_anchor_targets = _precompute_nullify_targets(
            model, train_data, layers, device)
        train_cb_targets = [None] * len(train_data)
        torch.cuda.empty_cache()
        if holdout_data:
            log.info("  Computing frozen anchor targets (%d holdout samples)...", len(holdout_data))
            _, holdout_anchor_targets = _precompute_nullify_targets(
                model, holdout_data, layers, device)
            holdout_cb_targets = [None] * len(holdout_data)
            torch.cuda.empty_cache()
        else:
            holdout_cb_targets, holdout_anchor_targets = [], []
    elif args.objective in ("nullify", "triplet"):
        mode_label = "triplet (contrastive clustering)" if args.objective == "triplet" else "nullify (cosine)"
        log.info("  Objective=%s: pre-computing frozen BAD + GOOD hidden states", mode_label)
        log.info("  (cosine loss pushes LoRA states orthogonal to frozen bad direction)")
        log.info("  Computing frozen bad + anchor targets (%d train samples)...", len(train_data))
        train_cb_targets, train_anchor_targets = _precompute_nullify_targets(
            model, train_data, layers, device)
        torch.cuda.empty_cache()
        if holdout_data:
            log.info("  Computing frozen bad + anchor targets (%d holdout samples)...", len(holdout_data))
            holdout_cb_targets, holdout_anchor_targets = _precompute_nullify_targets(
                model, holdout_data, layers, device)
            torch.cuda.empty_cache()
        else:
            holdout_cb_targets, holdout_anchor_targets = [], []
    else:
        log.info("  Computing train targets (%d samples)...", len(train_data))
        train_cb_targets, train_anchor_targets = precompute_targets(model, train_data, layers, device)
        torch.cuda.empty_cache()

        if holdout_data:
            log.info("  Computing holdout targets (%d samples)...", len(holdout_data))
            holdout_cb_targets, holdout_anchor_targets = precompute_targets(model, holdout_data, layers, device)
            torch.cuda.empty_cache()
        else:
            holdout_cb_targets, holdout_anchor_targets = [], []

        # Report target sizes
        if train_cb_targets:
            example_shapes = {l: train_cb_targets[0][l].shape for l in layers[:3]}
            log.info("  Target shapes (first sample, first 3 layers): %s", example_shapes)

    # ── Phase 2: Apply LoRA ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 2: Applying LoRA (rank=%d, alpha=%d, layers %d–%d, modules=%s)",
             args.rank, args.lora_alpha, args.layer_start, args.layer_end, args.target_modules)
    log.info("=" * 60)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        layers_to_transform=layers,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        # use_reentrant=False is required for our HiddenStateCapture forward-hooks:
        # reentrant checkpointing runs the first forward under torch.no_grad(), so
        # hooks capture detached tensors and the loss has no autograd path back to
        # the LoRA weights (seen empirically: cb_loss flat, anchor_loss exactly 0).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled (use_reentrant=False)")

    # ── Phase 3: Training ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 3: Training (epochs=%d, lr=%s, grad_accum=%d, patience=%d)",
             args.epochs, args.lr, args.grad_accum, args.patience)
    log.info("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # LR scheduler: linear warmup (10% of steps) + cosine decay to 10% of peak LR.
    # Stabilizes early training and prevents overfitting in later epochs.
    steps_per_epoch = math.ceil(len(train_data) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    log.info("  LR scheduler: warmup %d steps, cosine decay over %d total steps",
             warmup_steps, total_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    triplet_args = {
        "margin_safe": args.triplet_margin_safe,
        "margin_unsafe": args.triplet_margin_unsafe,
        "safe_dist_p": args.triplet_safe_dist_p,
        "safe_dist_n": args.triplet_safe_dist_n,
        "unsafe_dist_p": args.triplet_unsafe_dist_p,
        "unsafe_dist_n": args.triplet_unsafe_dist_n,
        "centroid_momentum": args.triplet_centroid_momentum,
    } if args.objective == "triplet" else None

    for epoch in range(1, args.epochs + 1):
        log.info("─" * 60)
        log.info("Epoch %d/%d", epoch, args.epochs)
        log.info("─" * 60)

        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            cb_targets=train_cb_targets,
            anchor_targets=train_anchor_targets,
            refusal_data=refusal_train_data,
            refusal_targets=refusal_train_targets,
            layers=layers,
            device=device,
            grad_accum_steps=args.grad_accum,
            cb_weight=args.cb_weight,
            anchor_weight=args.anchor_weight,
            lambda3=args.lambda3 if refusal_train_data else 0.0,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            max_completion_tokens=args.max_completion_tokens,
            refusal_max_tokens=args.refusal_max_tokens,
            objective=args.objective,
            nullify_mode=args.nullify_mode,
            triplet_args=triplet_args,
            scheduler=scheduler,
            normalize_states=args.normalize_states,
            refusal_mode=args.refusal_mode,
            refusal_push_weight=args.refusal_push_weight,
            benign_anchor_data=benign_train_data,
            benign_anchor_targets=benign_train_targets,
            lambda4=args.lambda4 if benign_train_data else 0.0,
            benign_anchor_max_tokens=args.benign_anchor_max_tokens,
        )
        current_lr = scheduler.get_last_lr()[0]
        log.info(
            "  Train: cb=%.6f  anchor=%.6f  refusal=%.6f  total=%.6f  lr=%.2e",
            train_metrics["cb_loss"],
            train_metrics["anchor_loss"],
            train_metrics["refusal_loss"],
            train_metrics["total_loss"],
            current_lr,
        )

        # Validation
        if holdout_data:
            val_metrics = validate(
                model=model,
                val_data=holdout_data,
                cb_targets=holdout_cb_targets,
                anchor_targets=holdout_anchor_targets,
                refusal_data=refusal_val_data,
                refusal_targets=refusal_val_targets,
                layers=layers,
                device=device,
                cb_weight=args.cb_weight,
                anchor_weight=args.anchor_weight,
                lambda3=args.lambda3 if refusal_train_data else 0.0,
                max_completion_tokens=args.max_completion_tokens,
                refusal_max_tokens=args.refusal_max_tokens,
                objective=args.objective,
                nullify_mode=args.nullify_mode,
                triplet_args=triplet_args,
                normalize_states=args.normalize_states,
                refusal_mode=args.refusal_mode,
                refusal_push_weight=args.refusal_push_weight,
            )
            log.info(
                "  Val:   cb=%.6f  anchor=%.6f  refusal=%.6f  total=%.6f",
                val_metrics["cb_loss"],
                val_metrics["anchor_loss"],
                val_metrics["refusal_loss"],
                val_metrics["total_loss"],
            )

            # Early stopping
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                patience_counter = 0
                # Save best checkpoint
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                log.info("  New best! Saved to %s", best_dir)
            else:
                patience_counter += 1
                log.info("  No improvement (%d/%d patience)", patience_counter, args.patience)
                if patience_counter >= args.patience:
                    log.info("  Early stopping triggered!")
                    break
        else:
            val_metrics = {}

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save per-epoch checkpoint — enables mid-training evals ("halfway weights")
        epoch_dir = output_dir / f"epoch{epoch}"
        model.save_pretrained(epoch_dir)
        log.info("  Saved epoch checkpoint to %s", epoch_dir)

        torch.cuda.empty_cache()

    # Save final checkpoint
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    log.info("Saved final adapter to %s", final_dir)

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        "suite": args.suite,
        "suites": suites,
        "model": args.model,
        "traces": args.traces,
        "traces_map": traces_map,
        "objective": args.objective,
        "nullify_mode": args.nullify_mode,
        "triplet_args": triplet_args,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "layers": layers,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "lr": args.lr,
        "epochs_completed": len(history),
        "epochs_max": args.epochs,
        "grad_accum": args.grad_accum,
        "cb_weight": args.cb_weight,
        "anchor_weight": args.anchor_weight,
        "lambda3": args.lambda3 if refusal_train_data else 0.0,
        "refusal_targets": args.refusal_targets,
        "refusal_mode": args.refusal_mode,
        "refusal_push_weight": args.refusal_push_weight,
        "refusal_max_tokens": args.refusal_max_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "n_train": len(train_data),
        "n_holdout": len(holdout_data),
        "n_refusal_train": len(refusal_train_data),
        "n_refusal_val": len(refusal_val_data),
        "refusal_metadata": refusal_meta,
        "best_val_loss": best_val_loss if holdout_data else None,
        "seed": args.seed,
        "normalize_states": args.normalize_states,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Phase 4: Evaluation ───────────────────────────────────────────
    if args.eval:
        log.info("=" * 60)
        log.info("Phase 4: Generation evaluation")
        log.info("=" * 60)

        # Load best checkpoint weights into existing model (no second model load)
        best_dir = output_dir / "best"
        if best_dir.exists():
            log.info("Loading best LoRA weights from %s (in-place)", best_dir)
            best_weights = load_peft_weights(best_dir)
            set_peft_model_state_dict(model, best_weights)
            model.eval()

        for suite in suites:
            ho = suite_holdout_raw[suite]
            tr = suite_train_raw[suite]
            eval_triplets = ho if ho else tr[:10]
            if not eval_triplets:
                continue

            suite_tag = f" [{suite}]" if len(suites) > 1 else ""
            log.info("Evaluating%s on %d triplets...", suite_tag, len(eval_triplets))

            if args.format == "dual":
                # Group by per-triplet format, eval each group with its config,
                # then merge results. evaluate_generation relies on module-level
                # globals (render_prefix, classify_completion_simple, …) set via
                # _init_data_config, so they must match each triplet's format.
                fmt_groups: dict[str, list[dict]] = {}
                orig_positions: dict[int, int] = {}  # eval_triplets idx → position in merged output
                for i, t in enumerate(eval_triplets):
                    fmt = t.get("format", "prompting_llm")
                    fmt_groups.setdefault(fmt, []).append(t)
                log.info("  dual-format eval groups: %s",
                         {k: len(v) for k, v in fmt_groups.items()})
                eval_results = {
                    "lorra": {
                        "injected": {"harmful": 0, "refusal": 0, "benign": 0,
                                     "garbled": 0, "completions": []},
                        "clean": {"harmful": 0, "refusal": 0, "benign": 0,
                                  "tool_changed": 0, "completions": []},
                    },
                    "baseline": {
                        "injected": {"harmful": 0, "refusal": 0, "benign": 0, "completions": []},
                        "clean": {"harmful": 0, "refusal": 0, "benign": 0, "completions": []},
                    },
                }
                for fmt, group in fmt_groups.items():
                    _init_data_config(suite, format=fmt)
                    fmt_results = evaluate_generation(
                        model, tokenizer, group, device,
                        max_new_tokens=args.eval_max_tokens,
                    )
                    for role in ("lorra", "baseline"):
                        for scenario in ("injected", "clean"):
                            for key, val in fmt_results[role][scenario].items():
                                if isinstance(val, int):
                                    eval_results[role][scenario][key] += val
                                elif isinstance(val, list):
                                    eval_results[role][scenario][key].extend(val)
            else:
                _init_data_config(suite, format=args.format)
                eval_results = evaluate_generation(
                    model, tokenizer, eval_triplets, device,
                    max_new_tokens=args.eval_max_tokens,
                )

            log.info("")
            log.info("  ═══════════════════════════════════════════════════════")
            log.info("  EVAL SUMMARY%s: Baseline vs LoRRA (%s)", suite_tag, args.objective)
            log.info("  ═══════════════════════════════════════════════════════")

            for scenario in ["injected", "clean"]:
                log.info("  %s:", scenario.upper())
                for model_key, model_name in [("baseline", "Baseline"), ("lorra", "LoRRA  ")]:
                    counts = {k: v for k, v in eval_results[model_key][scenario].items() if k != "completions"}
                    n = sum(counts.values())
                    parts = []
                    for label in ["harmful", "refusal", "benign"]:
                        c = counts[label]
                        parts.append(f"{label}={c}({100 * c / n:.0f}%)" if n else f"{label}=0")
                    log.info("    %s: %s", model_name, "  ".join(parts))

            n = len(eval_triplets)
            base_harmful = eval_results["baseline"]["injected"]["harmful"]
            lorra_harmful = eval_results["lorra"]["injected"]["harmful"]
            base_refusal_clean = eval_results["baseline"]["clean"]["refusal"]
            lorra_refusal_clean = eval_results["lorra"]["clean"]["refusal"]

            log.info("")
            log.info("  ─── KEY METRICS%s (n=%d) ───────────────────────────", suite_tag, n)
            log.info("  Attack success rate:  %4.1f%% → %4.1f%%  (Δ %+.1f%%)",
                     100 * base_harmful / n, 100 * lorra_harmful / n,
                     100 * (lorra_harmful - base_harmful) / n)
            log.info("  Refusal tax (clean):  %4.1f%% → %4.1f%%  (Δ %+.1f%%)",
                     100 * base_refusal_clean / n, 100 * lorra_refusal_clean / n,
                     100 * (lorra_refusal_clean - base_refusal_clean) / n)
            log.info("  ─────────────────────────────────────────────────────")

            n_improved = 0
            n_worsened = 0
            n_other_flip = 0
            n_clean_flip = 0
            for j in range(n):
                bl = eval_results["baseline"]["injected"]["completions"][j]["label"]
                ll = eval_results["lorra"]["injected"]["completions"][j]["label"]
                if bl != ll:
                    if bl == "harmful" and ll != "harmful":
                        n_improved += 1
                    elif bl != "harmful" and ll == "harmful":
                        n_worsened += 1
                    else:
                        n_other_flip += 1
                bcl = eval_results["baseline"]["clean"]["completions"][j]["label"]
                lcl = eval_results["lorra"]["clean"]["completions"][j]["label"]
                if bcl != lcl:
                    n_clean_flip += 1

            log.info("  FLIPS (injected):  %d improved, %d worsened, %d other  (out of %d)",
                     n_improved, n_worsened, n_other_flip, n)
            log.info("  FLIPS (clean):     %d changed  (out of %d)", n_clean_flip, n)
            log.info("  ═════════════════════════════════════════════════════")

            suffix = f"_{suite}" if len(suites) > 1 else ""
            eval_path = output_dir / f"eval_results{suffix}.json"
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            log.info("Saved eval results to %s", eval_path)

            _write_results_summary(output_dir, args, history, eval_results, eval_triplets,
                                   suffix=suite if len(suites) > 1 else "")

    log.info("Done. Output in %s", output_dir)


if __name__ == "__main__":
    main()
