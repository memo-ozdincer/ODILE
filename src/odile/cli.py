from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from odile.catalog import build_catalogs
from odile.recipes import ADAPTER_DIRS, ADAPTERS_ROOT, REPO_ROOT, SERVE_RECIPES, TRAIN_RECIPES

# The five standard AgentDojo attacks evaluated in the paper.
STANDARD_ATTACKS = [
    "direct",
    "ignore_previous",
    "system_message",
    "injecagent",
    "important_instructions",
]
# The four AgentDojo task suites.
STANDARD_SUITES = ["banking", "slack", "travel", "workspace"]


def _run_command(command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    subprocess.run(command, check=True, cwd=cwd or REPO_ROOT, env=merged_env)


def _print_recipes() -> None:
    print("Train recipes (objective is the ODILE loss):")
    for recipe in TRAIN_RECIPES.values():
        print(f"  {recipe.name:12}  {recipe.description}")
    print()
    print("Serve recipes:")
    for recipe in SERVE_RECIPES.values():
        print(f"  {recipe.name:12}  {recipe.description}")


def _train(args: argparse.Namespace) -> None:
    recipe = TRAIN_RECIPES[args.recipe]
    if args.objective not in recipe.objective_choices:
        raise SystemExit(f"Recipe {args.recipe} only supports objectives: {', '.join(recipe.objective_choices)}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(ADAPTERS_ROOT / ADAPTER_DIRS.get(args.recipe, f"ODILE_{args.recipe}"))

    command = [
        sys.executable,
        "-m",
        "odile.workflows.lorra_train",
        *recipe.args,
        "--output-dir",
        output_dir,
    ]
    if "--objective" not in recipe.args:
        command.extend(["--objective", args.objective])

    # Allow trailing override args (last-value-wins on argparse).
    extra = list(getattr(args, "extra_args", []) or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)

    env = {}
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTORCH_ALLOC_CONF"] = os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _run_command(command, env=env)


def _serve(args: argparse.Namespace) -> None:
    recipe = SERVE_RECIPES[args.recipe]
    lora_modules: list[str] = []
    for name, path in recipe.lora_modules:
        if Path(path).exists():
            lora_modules.append(f"{name}={path}")
    if not lora_modules:
        raise SystemExit(
            f"No adapter found for serve recipe {args.recipe}. "
            f"Run `python scripts/download_adapters.py {args.recipe}` first, or train it with `odile train {args.recipe}`."
        )

    command = [
        "vllm",
        "serve",
        recipe.base_model,
        "--served-model-name",
        recipe.served_model_name,
        "--tensor-parallel-size",
        str(recipe.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len or recipe.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs or recipe.max_num_seqs),
        "--enable-lora",
        "--max-lora-rank",
        str(recipe.max_lora_rank),
        "--max-loras",
        str(recipe.max_loras),
        "--lora-modules",
        *lora_modules,
        "--port",
        str(args.port),
        "--disable-uvicorn-access-log",
    ]
    _run_command(command)


def _eval_grid(args: argparse.Namespace) -> None:
    for suite in args.suites:
        for attack in args.attacks:
            for model in args.models:
                label_prefix = "baseline" if model == "base" else model
                label = f"{label_prefix}_{attack}_{suite}"
                out_path = Path(args.output_dir) / f"{label}.json"
                if out_path.exists():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "odile.workflows.eval_lorra_multiturn",
                    "--format",
                    args.format,
                    "--attack",
                    attack,
                    "--suite",
                    suite,
                    "--model",
                    model,
                    "--port",
                    str(args.port),
                    "--base-url",
                    args.base_url,
                    "--workers",
                    str(args.workers),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--label",
                    label,
                    "--adapter",
                    model,
                    "--output-dir",
                    args.output_dir,
                ]
                _run_command(command, env={"LLAMA_NATIVE_COT": "0"})


def _trace_tool(args: argparse.Namespace) -> None:
    # argparse REMAINDER chokes on --flags; pull the slice past the trace
    # subcommand from sys.argv directly so all flags survive verbatim.
    # exec the trace_tool script directly rather than spawning a subprocess
    # so stdout flows back to whoever called us without buffering surprises.
    try:
        traces_idx = sys.argv.index("traces")
    except ValueError:
        forward: list[str] = []
    else:
        forward = sys.argv[traces_idx + 2 :]  # skip 'traces' and the subcommand name
    argv = ["odile-trace_tool", args.trace_command, *forward]
    sys.argv = argv
    from odile.workflows import trace_tool as _trace_tool_mod
    sys.exit(_trace_tool_mod.main())


def _build_react_bundle(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "odile.workflows.build_react_bundle",
        "--in-dir",
        args.in_dir,
        "--out-dir",
        args.out_dir,
        "--source-suffix",
        args.source_suffix,
        "--suites",
        *args.suites,
    ]
    _run_command(command)


def _catalog_build() -> None:
    paths = build_catalogs()
    for path in paths:
        print(path)


def main() -> None:
    # Short-circuit `odile traces ...` so the trace_tool keeps full control of
    # its own flag parsing (argparse's REMAINDER mishandles --flags).
    if len(sys.argv) >= 3 and sys.argv[1] == "traces":
        from odile.workflows import trace_tool as _trace_tool_mod
        sys.argv = ["odile-trace_tool"] + sys.argv[2:]
        sys.exit(_trace_tool_mod.main())

    parser = argparse.ArgumentParser(prog="odile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipes_parser = subparsers.add_parser("recipes", help="List ODILE train and serve recipes.")
    recipes_parser.set_defaults(func=lambda args: _print_recipes())

    train_parser = subparsers.add_parser("train", help="Train an ODILE adapter for a backbone.")
    train_parser.add_argument("recipe", choices=sorted(TRAIN_RECIPES))
    train_parser.add_argument("--objective", default="odile", help="Training objective (default: odile).")
    train_parser.add_argument("--gpus")
    train_parser.add_argument("--output-dir")
    # Trailing override args after a literal `--` are forwarded verbatim to
    # lorra_train (last-value-wins on argparse). Captured via parse_known_args
    # below in main(); not declared as a positional here to avoid REMAINDER
    # eating sibling --options.
    train_parser.set_defaults(func=_train, extra_args=[])

    serve_parser = subparsers.add_parser("serve", help="Serve a backbone + its ODILE adapter on vLLM.")
    serve_parser.add_argument("recipe", choices=sorted(SERVE_RECIPES))
    serve_parser.add_argument("--port", type=int, default=27547)
    serve_parser.add_argument("--max-model-len", type=int)
    serve_parser.add_argument("--max-num-seqs", type=int)
    serve_parser.set_defaults(func=_serve)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation workflows.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    grid_parser = eval_subparsers.add_parser(
        "grid",
        help="Run the AgentDojo grid (4 suites x 5 standard attacks) against a served endpoint.",
    )
    grid_parser.add_argument("--models", nargs="+", default=["base", "odile"])
    grid_parser.add_argument("--suites", nargs="+", default=list(STANDARD_SUITES))
    grid_parser.add_argument("--attacks", nargs="+", default=list(STANDARD_ATTACKS))
    grid_parser.add_argument(
        "--format",
        default="llama_native",
        help="Tool-call format. llama_native for Llama; qwen_native / qwen3_native for Qwen backbones.",
    )
    grid_parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    grid_parser.add_argument("--port", type=int, default=27547)
    grid_parser.add_argument("--base-url", default="http://localhost:27547/v1")
    grid_parser.add_argument("--workers", type=int, default=64)
    grid_parser.add_argument("--max-new-tokens", type=int, default=2048)
    grid_parser.set_defaults(func=_eval_grid)

    traces_parser = subparsers.add_parser("traces", help="Trace query and bundle helpers.")
    traces_subparsers = traces_parser.add_subparsers(dest="trace_command", required=True)

    for command_name in ("list", "query", "sbs", "group"):
        command_parser = traces_subparsers.add_parser(
            command_name,
            help=f"Forward to trace_tool {command_name}.",
            add_help=False,  # let trace_tool handle its own --help
        )
        command_parser.set_defaults(func=_trace_tool)

    bundle_parser = traces_subparsers.add_parser("build-react-bundle", help="Build the dual-format react bundle.")
    bundle_parser.add_argument("--in-dir", required=True)
    bundle_parser.add_argument("--out-dir", required=True)
    bundle_parser.add_argument("--source-suffix", default="filtered")
    bundle_parser.add_argument("--suites", nargs="+", default=list(STANDARD_SUITES))
    bundle_parser.set_defaults(func=_build_react_bundle)

    catalog_parser = subparsers.add_parser("catalog", help="Generate the machine-readable recipe catalog.")
    catalog_subparsers = catalog_parser.add_subparsers(dest="catalog_command", required=True)
    catalog_build_parser = catalog_subparsers.add_parser("build")
    catalog_build_parser.set_defaults(func=lambda args: _catalog_build())

    args, unknown = parser.parse_known_args()
    # For `train`, treat anything after `--` (or any unrecognized flags) as
    # trailing override args forwarded verbatim to lorra_train.
    if getattr(args, "command", None) == "train":
        if unknown and unknown[0] == "--":
            unknown = unknown[1:]
        args.extra_args = unknown
    elif unknown:
        # Unknown args are not allowed for other subcommands.
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.func(args)


if __name__ == "__main__":
    main()
