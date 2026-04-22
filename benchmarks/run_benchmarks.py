#!/usr/bin/env python3
"""
pdfminer.six Rust-vs-Python benchmark runner.

Usage
-----
    uv run python benchmarks/run_benchmarks.py [OPTIONS]

Options
-------
    --rounds N      Number of outer timeit repetitions (default: 5)
    --number N      Inner-loop iterations per round (default: 200)
    --filter GLOB   Only run benchmark groups matching GLOB (fnmatch, default: *)
    --modules LIST  Comma-separated list of modules to run.
                    Choices: predictor,lzw,matrix,xref,e2e  (default: all except e2e)
    --all           Include end-to-end benchmarks (slow)
    --json FILE     Write JSON results to FILE in addition to stdout
    --no-color      Disable ANSI colour codes in output

Exit codes
----------
    0  — all benchmarks ran, at least one result collected
    1  — no benchmarks ran (wrong --filter, missing Rust extension, etc.)
    2  — import error

Examples
--------
    # Quick run of fast micro-benchmarks only
    uv run python benchmarks/run_benchmarks.py

    # Include end-to-end benchmarks on real PDFs
    uv run python benchmarks/run_benchmarks.py --all

    # Only matrix micro-benchmarks, 10 rounds
    uv run python benchmarks/run_benchmarks.py --modules matrix --rounds 10

    # Save JSON for later comparison
    uv run python benchmarks/run_benchmarks.py --json results.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

# Ensure the repo root is on the Python path so that both `benchmarks.*` and
# `pdfminer.*` are importable when running from an arbitrary working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pdfminer.six Rust vs Python benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rounds", type=int, default=3,
                   help="Number of timeit repetitions (default: 3)")
    p.add_argument("--number", type=int, default=0,
                   help="Inner-loop iterations per round; 0=auto-calibrate (default: 0)")
    p.add_argument("--filter", dest="filter_glob", default="*",
                   help="Only run groups matching this fnmatch pattern (default: *)")
    p.add_argument("--modules", default="predictor,lzw,matrix,xref",
                   help="Comma-separated modules to run (default: predictor,lzw,matrix,xref)")
    p.add_argument("--all", dest="run_all", action="store_true",
                   help="Include end-to-end benchmarks (adds 'e2e' to --modules)")
    p.add_argument("--json", dest="json_output", default=None,
                   help="Write JSON results to this file")
    p.add_argument("--no-color", dest="no_color", action="store_true",
                   help="Disable ANSI colour codes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

_MODULE_NAMES = {
    "predictor": "benchmarks.bench_predictor",
    "lzw":       "benchmarks.bench_lzw",
    "matrix":    "benchmarks.bench_matrix",
    "xref":      "benchmarks.bench_xref",
    "e2e":       "benchmarks.bench_e2e",
}


def _load_modules(names: list[str]) -> list:  # type: ignore[type-arg]
    """Import and return the benchmark modules for the given short names."""
    import importlib
    modules = []
    for name in names:
        full = _MODULE_NAMES.get(name)
        if full is None:
            print(f"Unknown benchmark module: {name!r}. Choices: {list(_MODULE_NAMES)}", file=sys.stderr)
            sys.exit(2)
        try:
            modules.append(importlib.import_module(full))
        except ImportError as exc:
            print(f"Could not import {full}: {exc}", file=sys.stderr)
            sys.exit(2)
    return modules


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _colorise(text: str, color: str, *, no_color: bool) -> str:
    if no_color:
        return text
    return f"{color}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Enhanced report with colour
# ---------------------------------------------------------------------------

def _print_report(suite, *, filter_glob: str, no_color: bool) -> None:  # type: ignore[no-untyped-def]
    from collections import defaultdict
    from benchmarks.bench_utils import BenchmarkResult

    # Group results
    by_group: dict[str, dict[str, BenchmarkResult]] = defaultdict(dict)
    for r in suite.results:
        by_group[r.group][r.variant] = r

    col_w = [50, 10, 18, 12]
    header = (
        f"{'Benchmark':<{col_w[0]}}"
        f"{'Variant':<{col_w[1]}}"
        f"{'Best/call (µs)':>{col_w[2]}}"
        f"{'Speedup':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    bold_header = _colorise(header, _BOLD, no_color=no_color)

    print(sep)
    print(bold_header)
    print(sep)

    printed_any = False
    for group in sorted(by_group):
        if not fnmatch.fnmatch(group, filter_glob):
            continue
        variants = by_group[group]
        py_res = variants.get("python")
        rs_res = variants.get("rust")

        for variant_key in ("python", "rust"):
            res = variants.get(variant_key)
            if res is None:
                continue
            us = res.best_sec * 1e6
            speedup_str = ""
            color = _RESET
            if variant_key == "rust" and py_res is not None:
                ratio = py_res.best_sec / res.best_sec
                speedup_str = f"{ratio:.2f}x"
                if ratio >= 3.0:
                    color = _GREEN
                elif ratio >= 1.0:
                    color = _YELLOW
                else:
                    color = _RED
            elif variant_key == "python":
                speedup_str = "(baseline)"

            line = (
                f"{res.name:<{col_w[0]}}"
                f"{variant_key:<{col_w[1]}}"
                f"{us:>{col_w[2]}.3f}"
                f"{speedup_str:>{col_w[3]}}"
            )
            if variant_key == "rust":
                line = _colorise(line, color, no_color=no_color)
            print(line)
            printed_any = True

    print(sep)

    if not printed_any:
        print(f"  (no results matched filter {filter_glob!r})")

    # Summary: geometric mean speedup
    speedups = [v for v in suite.speedup_map().values() if v is not None]
    if speedups:
        import math
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        label = _colorise(f"  Geometric mean speedup: {geomean:.2f}x", _BOLD, no_color=no_color)
        print(label)
        print(f"  (over {len(speedups)} paired Python/Rust benchmarks)")
    print()


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _write_json(suite, path: str) -> None:  # type: ignore[no-untyped-def]
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "rounds": suite.rounds,
            "number": suite.number,
        },
        "results": [
            {
                "name":     r.name,
                "group":    r.group,
                "variant":  r.variant,
                "best_sec": r.best_sec,
                "rounds":   r.rounds,
            }
            for r in suite.results
        ],
        "speedups": {
            k: v for k, v in suite.speedup_map().items() if v is not None
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"JSON results written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    module_names = [m.strip() for m in args.modules.split(",") if m.strip()]
    if args.run_all and "e2e" not in module_names:
        module_names.append("e2e")

    print(f"\npdfminer.six benchmark suite")
    print(f"  modules : {', '.join(module_names)}")
    print(f"  rounds  : {args.rounds}")
    print(f"  number  : {args.number}")
    print(f"  filter  : {args.filter_glob!r}")

    # Check Rust availability
    try:
        import pdfminer_core  # noqa: F401
        print("  rust    : available\n")
    except ImportError:
        print("  rust    : NOT AVAILABLE (Python-only results)\n")

    bench_modules = _load_modules(module_names)

    from benchmarks.bench_utils import BenchmarkSuite
    suite = BenchmarkSuite(rounds=args.rounds, number=args.number)

    # Register all cases
    for mod in bench_modules:
        mod.register(suite)

    # Filter to matching groups only (avoids timing unused cases)
    if args.filter_glob != "*":
        suite._cases = [
            c for c in suite._cases
            if fnmatch.fnmatch(c[0], args.filter_glob)
        ]

    if not suite._cases:
        print("No benchmark cases matched the filter. Exiting.", file=sys.stderr)
        return 1

    print(f"Running {len(suite._cases)} benchmark cases...")
    t0 = time.perf_counter()
    suite.run()
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s.\n")

    _print_report(suite, filter_glob=args.filter_glob, no_color=args.no_color)

    if args.json_output:
        _write_json(suite, args.json_output)

    return 0 if suite.results else 1


if __name__ == "__main__":
    sys.exit(main())
