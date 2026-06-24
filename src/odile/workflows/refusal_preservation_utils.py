from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "we cannot",
    "we can't",
    "i will not",
    "i won't",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i'm not able to",
    "i am unable",
    "i must decline",
    "i cannot assist with",
    "i can't assist with",
    "i cannot help with",
    "i can't help with",
    "i cannot provide",
    "i can't provide",
    "would not be appropriate",
    "against the law",
    "illegal",
    "unethical",
    "harmful",
)

WEAK_BASELINE_CATEGORY_KEYWORDS = (
    "hate",
    "harassment",
    "bullying",
    "discrimination",
    "stereotype",
    "injustice",
    "physical harm",
    "violence",
    "violent extremism",
    "terrorism",
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_prompt(text: str) -> str:
    return normalize_whitespace(text).strip()


def prompt_key(text: str) -> str:
    return normalize_prompt(text).lower()


def detect_refusal_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def is_weak_baseline_category(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in WEAK_BASELINE_CATEGORY_KEYWORDS)


def build_user_chat_prefix_ids(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(map(int, token_ids))
    encoded = tokenizer(prompt, add_special_tokens=True)
    return list(map(int, encoded["input_ids"]))


def weighted_sample_without_replacement(
    items: list[dict],
    k: int,
    rng: random.Random,
    weight_key: str = "sample_weight",
) -> list[dict]:
    if k >= len(items):
        copied = list(items)
        rng.shuffle(copied)
        return copied

    weighted_scores: list[tuple[float, dict]] = []
    for item in items:
        weight = max(float(item.get(weight_key, 1.0)), 1e-6)
        score = rng.random() ** (1.0 / weight)
        weighted_scores.append((score, item))
    weighted_scores.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in weighted_scores[:k]]


def load_toolsword_candidates(root: Path) -> list[dict]:
    candidates: list[dict] = []
    for path in sorted((root / "data").glob("*.json")):
        with path.open() as fh:
            payload = json.load(fh)
        for idx, item in enumerate(payload):
            prompt = normalize_prompt(item.get("query", ""))
            if not prompt:
                continue
            stage = item.get("stage", "unknown")
            scenario = item.get("scenario", "unknown")
            weak = is_weak_baseline_category(f"{stage} {scenario} {prompt}")
            weight = 2.0 if weak else 1.0
            candidates.append(
                {
                    "id": f"toolsword:{path.stem}:{idx}",
                    "source": "toolsword",
                    "source_file": path.name,
                    "category": f"{stage}/{scenario}",
                    "stage": stage,
                    "scenario": scenario,
                    "prompt": prompt,
                    "reference_refusal": None,
                    "sample_weight": weight,
                    "loss_weight": weight,
                    "weak_baseline_category": weak,
                }
            )
    return candidates


def load_circuit_breakers_candidates(root: Path) -> list[dict]:
    with (root / "data" / "circuit_breakers_train.json").open() as fh:
        payload = json.load(fh)

    candidates: list[dict] = []
    for idx, item in enumerate(payload):
        prompt = normalize_prompt(item.get("prompt", ""))
        if not prompt:
            continue
        category = normalize_whitespace(item.get("category", "unknown"))
        weak = is_weak_baseline_category(category)
        weight = 2.0 if weak else 1.0
        candidates.append(
            {
                "id": f"circuit_breakers:{idx}",
                "source": "circuit_breakers",
                "source_file": "circuit_breakers_train.json",
                "category": category,
                "prompt": prompt,
                "reference_refusal": normalize_whitespace(item.get("llama3_output", "")) or None,
                "sample_weight": weight,
                "loss_weight": weight,
                "weak_baseline_category": weak,
            }
        )
    return candidates


def load_agentharm_harmful_prompts(paths: list[Path]) -> list[dict]:
    prompts: list[dict] = []
    for path in paths:
        with path.open() as fh:
            payload = json.load(fh)
        behaviors = payload.get("behaviors", payload)
        for item in behaviors:
            prompts.append(
                {
                    "id": item.get("id", f"{path.name}:{len(prompts)}"),
                    "prompt": normalize_prompt(item["prompt"]),
                }
            )
    return prompts


@torch.no_grad()
def compute_text_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    local_files_only: bool = False,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = functional.normalize(pooled, dim=-1)
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


def assign_splits_by_source(
    items: list[dict],
    val_fraction: float,
    seed: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["source"], []).append(dict(item))

    rng = random.Random(seed)
    out: list[dict] = []
    for source, group in grouped.items():
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        if len(group) > 1:
            n_val = max(1, min(n_val, len(group) - 1))
        else:
            n_val = 0
        for idx, item in enumerate(group):
            item["split"] = "val" if idx < n_val else "train"
            out.append(item)
        log.info("Assigned splits for %s: %d train / %d val", source, len(group) - n_val, n_val)
    return sorted(out, key=lambda item: (item["source"], item["split"], item["id"]))


def load_refusal_target_bundle(
    path: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    good_targets = payload["good_targets"]
    bad_targets = payload.get("bad_targets")
    metadata = payload.get("metadata", {})

    train_samples: list[dict] = []
    val_samples: list[dict] = []
    train_targets: list[dict] = []
    val_targets: list[dict] = []

    for idx, sample in enumerate(samples):
        sample = dict(sample)
        sample["weight"] = float(sample.get("weight", sample.get("loss_weight", sample.get("sample_weight", 1.0))))
        sample["train_input_ids"] = (
            sample.get("train_input_ids")
            or sample.get("bad_full_ids")
            or sample.get("v4a_full_ids")
            or sample.get("good_full_ids")
            or sample.get("full_ids")
        )
        if sample["train_input_ids"] is None:
            raise ValueError(f"Refusal sample {sample.get('id', idx)} is missing train_input_ids")
        target_entry = {
            "good": good_targets[idx],
            "bad": None if bad_targets is None else bad_targets[idx],
        }
        if sample.get("split") == "val":
            val_samples.append(sample)
            val_targets.append(target_entry)
        else:
            train_samples.append(sample)
            train_targets.append(target_entry)

    return train_samples, train_targets, val_samples, val_targets, metadata
