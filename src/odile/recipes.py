"""Canonical ODILE training and serving recipes.

Each training recipe reproduces one of the published ODILE adapters
(https://huggingface.co/memo-ozdincer/ODILE), keyed by backbone. ODILE is trained
with a single objective (the ODILE loss); run a recipe with::

    odile train llama-70b          # headline Llama-3.3-70B
    odile train qwen3-8b

Hyperparameters follow the paper (rank 16, alpha 32, lr 5e-5, 5 epochs, seed 42;
depth-scaled per-backbone LoRA layer bands). Paths are repo-relative by default and
can be overridden with environment variables:

    ODILE_DATA_ROOT      training-trace root             (default: <repo>/data)
    ODILE_ADAPTERS_ROOT  where adapters are written/read (default: <repo>/adapters)

The base model defaults to the Hugging Face id; point it at a local mirror with a
trailing override, e.g. ``odile train llama-70b -- --model /path/to/Llama-3.3-70B``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# <repo>/src/odile/recipes.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ODILE_DATA_ROOT", str(REPO_ROOT / "data")))
TRACES_ROOT = DATA_ROOT / "traces"
ADAPTERS_ROOT = Path(os.environ.get("ODILE_ADAPTERS_ROOT", str(REPO_ROOT / "adapters")))

# Per-backbone adapter directory names. These match the subfolders published at
# https://huggingface.co/memo-ozdincer/ODILE and the default `odile train` output
# location, so `train -> serve -> eval` round-trips without extra path wiring.
ADAPTER_DIRS: dict[str, str] = {
    "llama-70b": "ODILE_Llama-3.3-70B",
    "llama-8b": "ODILE_Llama-3.1-8B",
    "qwen2.5-7b": "ODILE_Qwen2.5-7B",
    "qwen2.5-14b": "ODILE_Qwen2.5-14B",
    "qwen3-8b": "ODILE_Qwen3-8B",
    "qwen3-32b": "ODILE_Qwen3-32B",
    "qwen3-next": "ODILE_Qwen3-Next-80B",
}

_SUITES = ("banking", "slack", "travel", "workspace")


def _mixed_traces(family: str) -> tuple[str, ...]:
    """The four AgentDojo paired-trace files (Odette benign / Odile harmful twins)."""
    return tuple(f"{s}={TRACES_ROOT}/{family}/traces_{s}_mixed.json" for s in _SUITES)


@dataclass(frozen=True)
class TrainRecipe:
    name: str
    description: str
    objective_choices: tuple[str, ...]
    args: tuple[str, ...]


@dataclass(frozen=True)
class ServeRecipe:
    name: str
    description: str
    base_model: str
    served_model_name: str
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    max_lora_rank: int
    max_loras: int
    lora_modules: tuple[tuple[str, str], ...]


def _train_args(
    *,
    traces: tuple[str, ...],
    model: str,
    fmt: str,
    layer_start: int,
    layer_end: int,
    max_completion_tokens: int,
    eval_max_tokens: int,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Shared LoRA-training argument template (paper §app:hparams).

    ``max_completion_tokens`` caps the loss to the early intent-formation window
    (W=15 by default in the trainer; raised here per backbone so the window clears
    the leading reasoning block on thinking models).
    """
    return (
        "--suite", "all",
        "--traces", *traces,
        "--model", model,
        "--format", fmt,
        "--rank", "16",
        "--lora-alpha", "32",
        "--layer-start", str(layer_start),
        "--layer-end", str(layer_end),
        "--lr", "5e-5",
        "--grad-accum", "1",
        "--max-completion-tokens", str(max_completion_tokens),
        "--epochs", "5",
        "--seed", "42",
        "--attn-implementation", "sdpa",
        *extra,
        "--eval",
        "--eval-max-tokens", str(eval_max_tokens),
    )


# Qwen3-Next-80B is a Gated-DeltaNet hybrid MoE: q/v_proj exist only in the
# full-attention layers and bare up/down_proj would match all 512 routed experts,
# so the target modules are set explicitly to the DeltaNet path, the full-attention
# path, and the always-active shared expert.
_QWEN3_NEXT_TARGET_MODULES = (
    "--target-modules",
    "in_proj_qkvz",
    "out_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
)

_OBJECTIVE = ("odile",)


TRAIN_RECIPES: dict[str, TrainRecipe] = {
    "llama-70b": TrainRecipe(
        name="llama-70b",
        description="ODILE on Llama-3.3-70B-Instruct (paper headline backbone). LoRA layers 30-55, dual tool-call format.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("llama"),
            model="meta-llama/Llama-3.3-70B-Instruct",
            fmt="dual",
            layer_start=30,
            layer_end=55,
            max_completion_tokens=50,
            eval_max_tokens=256,
        ),
    ),
    "llama-8b": TrainRecipe(
        name="llama-8b",
        description="ODILE on Llama-3.1-8B-Instruct. LoRA layers 12-22, dual tool-call format.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("llama"),
            model="meta-llama/Llama-3.1-8B-Instruct",
            fmt="dual",
            layer_start=12,
            layer_end=22,
            max_completion_tokens=50,
            eval_max_tokens=256,
        ),
    ),
    "qwen2.5-7b": TrainRecipe(
        name="qwen2.5-7b",
        description="ODILE on Qwen2.5-7B-Instruct. LoRA layers 10-19, dual tool-call format.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("qwen2.5"),
            model="Qwen/Qwen2.5-7B-Instruct",
            fmt="dual",
            layer_start=10,
            layer_end=19,
            max_completion_tokens=50,
            eval_max_tokens=256,
        ),
    ),
    "qwen2.5-14b": TrainRecipe(
        name="qwen2.5-14b",
        description="ODILE on Qwen2.5-14B-Instruct. LoRA layers 18-33, Qwen-native tool-call format.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("qwen2.5"),
            model="Qwen/Qwen2.5-14B-Instruct",
            fmt="qwen_native",
            layer_start=18,
            layer_end=33,
            max_completion_tokens=50,
            eval_max_tokens=256,
        ),
    ),
    "qwen3-8b": TrainRecipe(
        name="qwen3-8b",
        description="ODILE on Qwen3-8B. LoRA layers 13-25, Qwen3-native (thinking-aware) format; loss window raised to clear the auto-think block.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("qwen3"),
            model="Qwen/Qwen3-8B",
            fmt="qwen3_native",
            layer_start=13,
            layer_end=25,
            max_completion_tokens=512,
            eval_max_tokens=512,
        ),
    ),
    "qwen3-32b": TrainRecipe(
        name="qwen3-32b",
        description="ODILE on Qwen3-32B. LoRA layers 24-44; loss window raised to clear the auto-think block. The trainer l2-normalizes hidden states for this high-norm backbone.",
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=_mixed_traces("qwen3"),
            model="Qwen/Qwen3-32B",
            fmt="dual",
            layer_start=24,
            layer_end=44,
            max_completion_tokens=512,
            eval_max_tokens=512,
        ),
    ),
    "qwen3-next": TrainRecipe(
        name="qwen3-next",
        description=(
            "ODILE on Qwen3-Next-80B-A3B-Thinking, a nonstandard-attention "
            "(Gated-DeltaNet hybrid MoE) backbone. LoRA layers 18-33 across the "
            "DeltaNet, full-attention, and shared-expert paths. Gradient "
            "checkpointing is intentionally off (incompatible with the DeltaNet "
            "recompute); fits at batch 1 on 4x80GB."
        ),
        objective_choices=_OBJECTIVE,
        args=_train_args(
            traces=tuple(
                f"{s}={TRACES_ROOT}/qwen3-next/traces_{s}_filtered.json" for s in _SUITES
            ),
            model="Qwen/Qwen3-Next-80B-A3B-Thinking",
            fmt="qwen3_native",
            layer_start=18,
            layer_end=33,
            max_completion_tokens=512,
            eval_max_tokens=512,
            extra=_QWEN3_NEXT_TARGET_MODULES,
        ),
    ),
}


def _serve(name: str, base_model: str, tensor_parallel_size: int) -> ServeRecipe:
    """Serve a base model + its ODILE adapter on vLLM.

    The base is exposed as ``base`` and the adapter as ``odile`` so the eval grid
    can run ``--models base odile`` against a single server.
    """
    return ServeRecipe(
        name=name,
        description=f"Serve {base_model} + ODILE adapter on vLLM.",
        base_model=base_model,
        served_model_name="base",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=16384,
        max_num_seqs=128,
        max_lora_rank=16,
        max_loras=1,
        lora_modules=(("odile", str(ADAPTERS_ROOT / ADAPTER_DIRS[name])),),
    )


SERVE_RECIPES: dict[str, ServeRecipe] = {
    "llama-70b": _serve("llama-70b", "meta-llama/Llama-3.3-70B-Instruct", 4),
    "llama-8b": _serve("llama-8b", "meta-llama/Llama-3.1-8B-Instruct", 1),
    "qwen2.5-7b": _serve("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", 1),
    "qwen2.5-14b": _serve("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", 1),
    "qwen3-8b": _serve("qwen3-8b", "Qwen/Qwen3-8B", 1),
    "qwen3-32b": _serve("qwen3-32b", "Qwen/Qwen3-32B", 2),
    "qwen3-next": _serve("qwen3-next", "Qwen/Qwen3-Next-80B-A3B-Thinking", 4),
}
