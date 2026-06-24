#!/usr/bin/env python3
"""Build v7-dual trace bundle: llama_native + react format copies of each triplet.

For each existing triplet in traces_{suite}_mixed.json (or _filtered.json), emit
two copies into the v7-dual bundle — one with format="llama_native" (unchanged),
one with format="react" (same message data, different format field).

The actual format-specific rendering happens at training time via the per-suite
config modules. We do NOT pre-render here; the triplet's structured messages
are format-agnostic.

Usage:
    python -m odile.workflows.build_react_bundle \\
        --in-dir data/traces/llama \\
        --out-dir data/traces/llama \\
        --suites banking slack travel workspace
"""

import argparse
import copy
import json
import logging
import pathlib
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_react_bundle")


def load_triplets(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def emit_pair(triplet: dict) -> list[dict]:
    """Emit two copies of the triplet: one llama_native, one react."""
    ln = copy.deepcopy(triplet)
    ln["format"] = "llama_native"

    react = copy.deepcopy(triplet)
    react["format"] = "react"
    react["source_format"] = triplet.get("format", "llama_native")

    return [ln, react]


def build_for_suite(
    in_dir: pathlib.Path,
    out_dir: pathlib.Path,
    suite: str,
    source_suffix: str,
) -> tuple[int, int]:
    src = in_dir / f"traces_{suite}_{source_suffix}.json"
    if not src.exists():
        log.warning("  no source file: %s", src)
        return (0, 0)

    triplets = load_triplets(src)
    combined = []
    for t in triplets:
        combined.extend(emit_pair(t))

    out_file = out_dir / f"traces_{suite}_mixed.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(combined, f)

    n_in = len(triplets)
    n_out = len(combined)
    log.info("  %s: %d in → %d out (ln=%d + react=%d)", suite, n_in, n_out, n_in, n_in)
    return (n_in, n_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--suites", nargs="+", default=["banking", "slack", "travel", "workspace"])
    ap.add_argument(
        "--source-suffix",
        default="filtered",
        help="Source file stem after 'traces_{suite}_'. Typically 'filtered' or 'mixed'.",
    )
    args = ap.parse_args()

    log.info("v7-dual bundle build: %s → %s", args.in_dir, args.out_dir)
    log.info("Suites: %s, source suffix: %s", args.suites, args.source_suffix)

    if not args.in_dir.exists():
        log.error("Input dir does not exist: %s", args.in_dir)
        sys.exit(1)

    total_in = 0
    total_out = 0
    for suite in args.suites:
        n_in, n_out = build_for_suite(args.in_dir, args.out_dir, suite, args.source_suffix)
        total_in += n_in
        total_out += n_out

    log.info("Total: %d input triplets → %d output (×2 formats)", total_in, total_out)

    manifest = {
        "source_dir": str(args.in_dir),
        "source_suffix": args.source_suffix,
        "suites": args.suites,
        "total_input_triplets": total_in,
        "total_output_triplets": total_out,
        "format_split": {"llama_native": total_in, "react": total_in},
    }
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s/MANIFEST.json", args.out_dir)


if __name__ == "__main__":
    main()
