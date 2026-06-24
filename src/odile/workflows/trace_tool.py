#!/usr/bin/env python3
"""
trace_tool.py — query, group, and side-by-side view JSON traces / eval outputs.

Four subcommands. All stdlib only.

  list FILE [--depth N]
      Print JSON structure (keys + list lengths) up to --depth.

  query FILE --path DOTTED [--where K=V] [--limit N] [--key-only]
                           [--max-bytes B]
      Walk a dotted path; print matches. Path syntax:
        messages                      → the messages array
        messages.0.content            → indexed
        messages.*.role               → wildcard across list/dict
        results.*.security            → mixed
        .                             → root

  sbs FILE_A FILE_B --path P [--path-b P_B] [--width W] [--limit N]
                              [--max-bytes B]
      Side-by-side render of two JSONs at the same (or two different)
      paths. Terminal-width columns.

  group FILES... --by KEY [--inner-path P] [--bins B] [--max-print N]
      Bucket many JSONs by the value at KEY. Prints counts + filenames
      per bucket. KEY can be a dotted path; --inner-path makes group
      iterate through rows within each file before bucketing.

Examples:
  trace_tool list bundle.json --depth 2
  trace_tool query eval.json --path results.* --where security=true --limit 3
  trace_tool sbs run_a.json run_b.json --path results.0.messages --max-bytes 4000
  trace_tool group results/v7a_dual_*.json --by aggregate.security \\
                   --bins '<0.05,<0.2,<0.5,>=0.5'
  trace_tool group bundle.json --by format --inner-path '*'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import shutil
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def _load(path: str) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _walk(node: Any, parts: list[str]) -> Iterator[tuple[str, Any]]:
    if not parts or (len(parts) == 1 and parts[0] == ""):
        yield "", node
        return
    head, *rest = parts
    if head == "*":
        if isinstance(node, list):
            for i, v in enumerate(node):
                for crumb, val in _walk(v, rest):
                    yield f"[{i}]" + (f".{crumb}" if crumb else ""), val
        elif isinstance(node, dict):
            for k, v in node.items():
                for crumb, val in _walk(v, rest):
                    yield f".{k}" + (f".{crumb}" if crumb else ""), val
        return
    if head.isdigit() and isinstance(node, list):
        i = int(head)
        if 0 <= i < len(node):
            for crumb, val in _walk(node[i], rest):
                yield f"[{i}]" + (f".{crumb}" if crumb else ""), val
        return
    if isinstance(node, dict) and head in node:
        for crumb, val in _walk(node[head], rest):
            yield f".{head}" + (f".{crumb}" if crumb else ""), val


def _split_path(p: str) -> list[str]:
    if p in (".", ""):
        return []
    return p.split(".")


def _matches_where(node: Any, where: list[tuple[str, str]]) -> bool:
    for k, v in where:
        if not isinstance(node, dict) or k not in node:
            return False
        if str(node[k]).lower() != v.lower():
            return False
    return True


def _render(val: Any, max_bytes: int) -> str:
    if isinstance(val, str):
        s = val
    else:
        s = json.dumps(val, indent=2, ensure_ascii=False, default=str)
    if len(s) > max_bytes:
        s = s[:max_bytes] + f"\n… [truncated, {len(s) - max_bytes} more bytes]"
    return s


# -----------------------------------------------------------------------------
# list


def cmd_list(args: argparse.Namespace) -> int:
    data = _load(args.file)

    def walk(node: Any, depth: int, prefix: str) -> None:
        if depth < 0:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                kind = type(v).__name__
                if isinstance(v, (dict, list)):
                    sz = len(v)
                    print(f"{prefix}{k}: {kind}[{sz}]")
                    if depth > 0:
                        walk(v, depth - 1, prefix + "  ")
                else:
                    s = repr(v)
                    if len(s) > 80:
                        s = s[:80] + "…"
                    print(f"{prefix}{k}: {kind} = {s}")
        elif isinstance(node, list):
            if not node:
                print(f"{prefix}(empty list)")
                return
            print(f"{prefix}[0] (representative of {len(node)}):")
            if depth > 0:
                walk(node[0], depth - 1, prefix + "  ")

    walk(data, args.depth, "")
    return 0


# -----------------------------------------------------------------------------
# query


def cmd_query(args: argparse.Namespace) -> int:
    data = _load(args.file)
    parts = _split_path(args.path)
    where = [tuple(w.split("=", 1)) for w in args.where] if args.where else []
    n = 0
    for crumb, val in _walk(data, parts):
        if where and not _matches_where(val, where):
            continue
        print(f"─── {crumb or '.'} " + "─" * max(0, 60 - len(crumb)))
        if args.key_only:
            if isinstance(val, dict):
                print(", ".join(val.keys()))
            elif isinstance(val, list):
                print(f"list[{len(val)}]")
            else:
                print(type(val).__name__)
        else:
            print(_render(val, args.max_bytes))
        n += 1
        if args.limit and n >= args.limit:
            break
    if n == 0:
        print(f"(no matches for path {args.path!r})", file=sys.stderr)
        return 1
    return 0


# -----------------------------------------------------------------------------
# sbs


def _columnize(left: str, right: str, width: int) -> str:
    half = (width - 3) // 2
    lwrap = [textwrap.fill(ln, half, replace_whitespace=False, drop_whitespace=False) or "" for ln in left.splitlines() or [""]]
    rwrap = [textwrap.fill(ln, half, replace_whitespace=False, drop_whitespace=False) or "" for ln in right.splitlines() or [""]]
    ll = "\n".join(lwrap).splitlines()
    rl = "\n".join(rwrap).splitlines()
    out = []
    for i in range(max(len(ll), len(rl))):
        l = ll[i] if i < len(ll) else ""
        r = rl[i] if i < len(rl) else ""
        out.append(f"{l:<{half}} │ {r:<{half}}")
    return "\n".join(out)


def cmd_sbs(args: argparse.Namespace) -> int:
    a = _load(args.file_a)
    b = _load(args.file_b)
    path_a = _split_path(args.path_a)
    path_b = _split_path(args.path_b or args.path_a)

    matches_a = list(_walk(a, path_a))
    matches_b = list(_walk(b, path_b))

    if not matches_a or not matches_b:
        print(f"no matches (a={len(matches_a)}, b={len(matches_b)})", file=sys.stderr)
        return 1

    width = args.width if args.width > 0 else shutil.get_terminal_size((160, 40)).columns
    pairs = min(len(matches_a), len(matches_b), args.limit or 1_000_000)
    header_a = Path(args.file_a).name
    header_b = Path(args.file_b).name
    for i in range(pairs):
        ca, va = matches_a[i]
        cb, vb = matches_b[i]
        print("═" * width)
        print(f"{header_a}  at  {ca or '.'}    │    {header_b}  at  {cb or '.'}")
        print("═" * width)
        print(_columnize(_render(va, args.max_bytes), _render(vb, args.max_bytes), width))
    return 0


# -----------------------------------------------------------------------------
# group — the new mode


def _parse_bins(spec: str) -> list[tuple[str, float | None]]:
    """Parse '<0.05,<0.2,<0.5,>=0.5' → ordered bins.

    Each element is (label, threshold) where threshold is the upper bound
    for `<X` bins, or None for the catch-all `>=X` bucket.
    """
    bins: list[tuple[str, float | None]] = []
    for raw in spec.split(","):
        s = raw.strip()
        if s.startswith("<"):
            bins.append((s, float(s[1:])))
        elif s.startswith(">="):
            bins.append((s, None))
        else:
            raise ValueError(f"bad bin spec {s!r}; use '<X' or '>=X'")
    return bins


def _bucket_value(value: Any, bins: list[tuple[str, float | None]]) -> str:
    """Return the bucket label for `value` against ordered bins."""
    if isinstance(value, bool):  # bool is int in python — handle first
        return str(value)
    if isinstance(value, (int, float)):
        for label, upper in bins:
            if upper is None:
                return label
            if value < upper:
                return label
        return bins[-1][0]
    return str(value)


def cmd_group(args: argparse.Namespace) -> int:
    files = []
    for pattern in args.files:
        files.extend(_glob.glob(pattern) if any(c in pattern for c in "*?[") else [pattern])
    files = sorted(set(files))
    if not files:
        print("no files matched", file=sys.stderr)
        return 1

    by_parts = _split_path(args.by)
    inner_parts = _split_path(args.inner_path) if args.inner_path else None
    bins = _parse_bins(args.bins) if args.bins else None

    buckets: dict[str, list[str]] = defaultdict(list)
    for fp in files:
        try:
            data = _load(fp)
        except Exception as e:
            buckets[f"<error: {type(e).__name__}>"].append(fp)
            continue

        if inner_parts is None:
            # Pull the value directly from each file
            vals = [v for _, v in _walk(data, by_parts)]
            if not vals:
                buckets["<missing>"].append(fp)
                continue
            v = vals[0]
            label = _bucket_value(v, bins) if bins else str(v)
            buckets[label].append(fp)
        else:
            # Iterate inner rows; bucket each row, but record the file (with row index)
            for crumb, row in _walk(data, inner_parts):
                vals = [v for _, v in _walk(row, by_parts)]
                if not vals:
                    label = "<missing>"
                else:
                    v = vals[0]
                    label = _bucket_value(v, bins) if bins else str(v)
                buckets[label].append(f"{fp}::{crumb}")

    # Stable display order: bin order if bins given, else by descending count
    if bins:
        order = [b[0] for b in bins] + ["<missing>", "<error>"]
        sorted_keys = [k for k in order if k in buckets] + [k for k in buckets if k not in order]
    else:
        sorted_keys = sorted(buckets, key=lambda k: -len(buckets[k]))

    for k in sorted_keys:
        items = buckets[k]
        print(f"\n{k:<20} {len(items)} items")
        for it in items[: args.max_print]:
            print(f"  {it}")
        if len(items) > args.max_print:
            print(f"  … and {len(items) - args.max_print} more")
    return 0


# -----------------------------------------------------------------------------
# main


def main() -> int:
    ap = argparse.ArgumentParser(prog="trace_tool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="print JSON structure")
    l.add_argument("file")
    l.add_argument("--depth", type=int, default=2)
    l.set_defaults(func=cmd_list)

    q = sub.add_parser("query", help="walk a dotted path and print matches")
    q.add_argument("file")
    q.add_argument("--path", default=".", help="dotted path (default: root)")
    q.add_argument("--where", action="append", default=[], help="k=v filter on matched dicts (repeatable)")
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--key-only", action="store_true")
    q.add_argument("--max-bytes", type=int, default=50_000)
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("sbs", help="side-by-side two files at the same path")
    s.add_argument("file_a")
    s.add_argument("file_b")
    s.add_argument("--path", dest="path_a", default=".")
    s.add_argument("--path-b", dest="path_b", default=None)
    s.add_argument("--width", type=int, default=0)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--max-bytes", type=int, default=8_000)
    s.set_defaults(func=cmd_sbs)

    g = sub.add_parser("group", help="bucket many JSONs by a path value")
    g.add_argument("files", nargs="+", help="file paths or globs")
    g.add_argument("--by", required=True, help="dotted path to the value to bucket on")
    g.add_argument("--inner-path", default=None,
                   help="iterate this path within each file (e.g. 'results.*' or '*') before bucketing")
    g.add_argument("--bins", default=None,
                   help="comma-separated numeric bins like '<0.05,<0.2,<0.5,>=0.5'")
    g.add_argument("--max-print", type=int, default=10,
                   help="max items to list per bucket (default 10)")
    g.set_defaults(func=cmd_group)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
