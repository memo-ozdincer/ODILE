#!/usr/bin/env python3
"""Download published ODILE adapters from the Hugging Face Hub.

    python scripts/download_adapters.py                  # all backbones
    python scripts/download_adapters.py llama-70b        # one backbone
    python scripts/download_adapters.py llama-70b qwen3-8b --dest adapters

Adapters land in ``<dest>/ODILE_<Backbone>/``, which is exactly where
``odile serve`` and ``odile train`` expect them.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

REPO_ID = "memo-ozdincer/ODILE"

# Mirrors odile.recipes.ADAPTER_DIRS, kept standalone so this script runs with
# only huggingface-hub installed (before the training extras).
ADAPTER_DIRS = {
    "llama-70b": "ODILE_Llama-3.3-70B",
    "llama-8b": "ODILE_Llama-3.1-8B",
    "qwen2.5-7b": "ODILE_Qwen2.5-7B",
    "qwen2.5-14b": "ODILE_Qwen2.5-14B",
    "qwen3-8b": "ODILE_Qwen3-8B",
    "qwen3-32b": "ODILE_Qwen3-32B",
    "qwen3-next": "ODILE_Qwen3-Next-80B",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("backbones", nargs="*", help=f"recipe names (default: all). Options: {', '.join(ADAPTER_DIRS)}")
    parser.add_argument("--dest", default="adapters", help="destination directory (default: ./adapters)")
    args = parser.parse_args()

    names = args.backbones or list(ADAPTER_DIRS)
    unknown = [n for n in names if n not in ADAPTER_DIRS]
    if unknown:
        sys.exit(f"unknown backbone(s): {', '.join(unknown)}. Choose from: {', '.join(ADAPTER_DIRS)}")

    for name in names:
        subfolder = ADAPTER_DIRS[name]
        print(f"downloading {subfolder} from {REPO_ID} ...")
        snapshot_download(repo_id=REPO_ID, allow_patterns=f"{subfolder}/*", local_dir=args.dest)
        print(f"  -> {args.dest}/{subfolder}")


if __name__ == "__main__":
    main()
