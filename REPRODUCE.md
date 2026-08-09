# Reproducing ODILE

End-to-end: install → train an adapter → serve it → evaluate on AgentDojo. The headline
result is Llama-3.3-70B; smaller backbones run on a single GPU.

## 1. Install

```bash
uv venv --python 3.11
uv pip install -e ".[training,serve]"
```

`[training]` pulls `torch`/`transformers`/`peft`; `[serve]` pulls `vllm`. The core install
(CLI + AgentDojo evaluation client) needs neither.

## 2. Training data

The paired training traces ship in `data/traces/` — one set per backbone family. Each
suite file holds **(Odette, Odile) twins**: a benign and a harmful completion that share the
same user task, tools, and preceding trajectory and differ at the designated tool result.
Both completions are produced by the *base* model under greedy decoding; ODILE's loss
contrasts them. No download or regeneration is needed.

```
data/traces/llama/        traces_{banking,slack,travel,workspace}_mixed.json     # Llama-3.1-8B, Llama-3.3-70B
data/traces/qwen2.5/      traces_{...}_mixed.json                                # Qwen2.5-7B, Qwen2.5-14B
data/traces/qwen3/        traces_{...}_mixed.json                                # Qwen3-8B, Qwen3-32B
data/traces/qwen3-next/   traces_{...}_filtered.json                            # Qwen3-Next-80B
```

## 3. Train

```bash
odile recipes            # list recipes
odile train llama-70b    # writes to adapters/ODILE_Llama-3.3-70B
```

Every recipe uses the ODILE loss with the paper hyperparameters: LoRA **rank 16, α 32**,
targets `q/v/up/down_proj`, **AdamW lr 5e-5**, cosine schedule, **5 epochs**, **seed 42**.
The loss is applied only on the early intent-formation window of each completion. Per-backbone
settings:

| recipe        | base model                              | LoRA layers | eval `--format` |
|---------------|-----------------------------------------|-------------|-----------------|
| `llama-70b`   | meta-llama/Llama-3.3-70B-Instruct       | 30–55       | `llama_native`  |
| `llama-8b`    | meta-llama/Llama-3.1-8B-Instruct        | 12–22       | `llama_native`  |
| `qwen2.5-7b`  | Qwen/Qwen2.5-7B-Instruct                | 10–19       | `qwen_native`   |
| `qwen2.5-14b` | Qwen/Qwen2.5-14B-Instruct               | 18–33       | `qwen_native`   |
| `qwen3-8b`    | Qwen/Qwen3-8B                           | 13–25       | `qwen3_native`  |
| `qwen3-32b`   | Qwen/Qwen3-32B                          | 24–44       | `qwen3_native`  |
| `qwen3-next`  | Qwen/Qwen3-Next-80B-A3B-Thinking        | 18–33       | `qwen3_native`  |

Notes:
- **Hardware / time:** ~1 h on a single H100 for most backbones; Llama-70B and Qwen3-Next
  use 4×H100.
- **Base model:** defaults to the Hugging Face id above (downloaded on first use). Point at a
  local mirror with a trailing override: `odile train llama-70b -- --model /path/to/model`.
- **Output:** `adapters/ODILE_<Backbone>/` by default (override with `--output-dir`, or set
  `ODILE_ADAPTERS_ROOT`). This matches both `odile serve` and the published HF subfolders.
- **GPUs:** select with `--gpus 0,1,2,3`.
- **Qwen3 (thinking) backbones** raise the completion window past the leading reasoning block;
  **Qwen3-32B** additionally l2-normalizes hidden states (high hidden-state norm), and
  **Qwen3-Next** trains with gradient checkpointing off (incompatible with its DeltaNet path).
  These are handled by the recipes — no manual flags.

You can skip training and use the published adapters instead:

```bash
python scripts/download_adapters.py llama-70b      # -> adapters/ODILE_Llama-3.3-70B
```

## 4. Serve

```bash
odile serve llama-70b      # vLLM serves base (name "base") + adapter (name "odile") on :27547
```

This requires the adapter to exist locally (trained or downloaded). Adjust `--port`,
`--max-model-len`, `--max-num-seqs` as needed; tensor-parallel size is set per recipe.

## 5. Evaluate on AgentDojo

With the server up, run the grid against it. The default grid is the **four AgentDojo task
suites** (banking, slack, travel, workspace) × the **five standard attacks** (`direct`,
`ignore_previous`, `system_message`, `injecagent`, `important_instructions`), for both the
undefended base and ODILE:

```bash
odile eval grid --models base odile --format llama_native --output-dir results/
```

For Qwen backbones set `--format` per the table above (`qwen_native` / `qwen3_native`). Each
suite × attack × model cell writes one JSON to `results/`. Existing cells are skipped, so the
grid is resumable.

**Metrics.** `ASR` is AgentDojo's hand-coded security predicate (a harmful tool call was
executed). `Benign utility` is task success on the no-attack split — measure it with
`--attacks benign`:

```bash
odile eval grid --models base odile --attacks benign --format llama_native --output-dir results/
```

### Expected headline (Llama-3.3-70B)

ASR drops from **14.04% (base) to 0.01% (ODILE)** while benign utility stays at **59.8%**
(vs. 59.9% base), at 1× inference cost. In these runs, attacked continuations contain no
parseable attacker-directed tool call.

## 6. Inspect a single trace

```bash
odile traces list results/odile_important_instructions_banking.json
odile traces sbs results/baseline_important_instructions_banking.json results/odile_important_instructions_banking.json
```

Or run the standalone jam demo (no server needed):

```bash
python examples/jam_demo.py --base meta-llama/Llama-3.1-8B-Instruct --adapter adapters/ODILE_Llama-3.1-8B
```
