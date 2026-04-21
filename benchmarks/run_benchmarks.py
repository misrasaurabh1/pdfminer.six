#!/usr/bin/env python3
"""Run all benchmarks and print a comparison table.

Usage
-----
    # Run all benchmarks (Python only)
    python benchmarks/run_benchmarks.py

    # Run only benchmarks whose name matches a substring
    python benchmarks/run_benchmarks.py --filter codec

    # Show Rust column (when a Rust extension is installed the numbers will
    # differ; without one the column simply mirrors Python)
    python benchmarks/run_benchmarks.py --rust

    # Output a GitHub-flavoured Markdown table
    python benchmarks/run_benchmarks.py --markdown
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

_BENCHMARKS_DIR = Path(__file__).parent


def _discover_modules(filter_str: str | None = None) -> list[Any]:
    """Import and return all benchmark_*.py modules (optionally filtered)."""
    repo_root = str(_BENCHMARKS_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    modules = []
    for path in sorted(_BENCHMARKS_DIR.glob("benchmark_*.py")):
        mod_name = f"benchmarks.{path.stem}"
        if filter_str and filter_str.lower() not in path.stem.lower():
            continue
        try:
            mod = importlib.import_module(mod_name)
            modules.append(mod)
        except Exception as exc:
            print(f"  [WARN] Could not import {mod_name}: {exc}", file=sys.stderr)
    return modules


def _collect_bench_functions(module: Any) -> list[tuple[str, Any]]:
    """Return (display_name, callable) pairs for all bench_* functions."""
    pairs = []
    for name in dir(module):
        if not name.startswith("bench_"):
            continue
        fn = getattr(module, name)
        if callable(fn):
            # Build a human-readable label: module_stem + function name
            stem = module.__name__.split(".")[-1].replace("benchmark_", "")
            label = f"{stem} / {name.removeprefix('bench_')}"
            pairs.append((label, fn))
    return pairs


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_NAME = 40
_COL_MS = 12
_COL_SPD = 9


def _fmt_ms(ms: float) -> str:
    if math.isnan(ms):
        return "N/A"
    if ms < 0.001:
        return f"{ms*1000:.3f} µs"
    if ms < 1:
        return f"{ms:.3f} ms"
    return f"{ms:.1f} ms"


def _speedup(py_ms: float, rust_ms: float) -> str:
    if math.isnan(py_ms) or math.isnan(rust_ms) or rust_ms == 0:
        return "N/A"
    return f"{py_ms / rust_ms:.1f}x"


def _print_table(
    rows: list[tuple[str, float, float]],
    markdown: bool = False,
    show_rust: bool = False,
) -> None:
    if markdown:
        _print_table_md(rows, show_rust=show_rust)
    else:
        _print_table_plain(rows, show_rust=show_rust)


def _print_table_plain(
    rows: list[tuple[str, float, float]],
    show_rust: bool = False,
) -> None:
    sep = "-" * (_COL_NAME + 2 + _COL_MS + 3 + (_COL_MS + 3 + _COL_SPD + 2 if show_rust else 0))
    if show_rust:
        hdr = (
            f"{'Component':<{_COL_NAME}} | {'Python':>{_COL_MS}} | "
            f"{'Rust':>{_COL_MS}} | {'Speedup':>{_COL_SPD}}"
        )
    else:
        hdr = f"{'Component':<{_COL_NAME}} | {'Python':>{_COL_MS}}"
    print(hdr)
    print(sep)
    for label, py_ms, rust_ms in rows:
        py_str = _fmt_ms(py_ms)
        if show_rust:
            rust_str = _fmt_ms(rust_ms)
            spd_str = _speedup(py_ms, rust_ms)
            print(
                f"{label:<{_COL_NAME}} | {py_str:>{_COL_MS}} | "
                f"{rust_str:>{_COL_MS}} | {spd_str:>{_COL_SPD}}"
            )
        else:
            print(f"{label:<{_COL_NAME}} | {py_str:>{_COL_MS}}")
    print(sep)


def _print_table_md(
    rows: list[tuple[str, float, float]],
    show_rust: bool = False,
) -> None:
    if show_rust:
        print(f"| {'Component':<{_COL_NAME}} | {'Python':>{_COL_MS}} | {'Rust':>{_COL_MS}} | {'Speedup':>{_COL_SPD}} |")
        print(f"|{'-'*(_COL_NAME+2)}|{'-'*(_COL_MS+2)}|{'-'*(_COL_MS+2)}|{'-'*(_COL_SPD+2)}|")
        for label, py_ms, rust_ms in rows:
            py_str = _fmt_ms(py_ms)
            rust_str = _fmt_ms(rust_ms)
            spd_str = _speedup(py_ms, rust_ms)
            print(f"| {label:<{_COL_NAME}} | {py_str:>{_COL_MS}} | {rust_str:>{_COL_MS}} | {spd_str:>{_COL_SPD}} |")
    else:
        print(f"| {'Component':<{_COL_NAME}} | {'Python':>{_COL_MS}} |")
        print(f"|{'-'*(_COL_NAME+2)}|{'-'*(_COL_MS+2)}|")
        for label, py_ms, _rust_ms in rows:
            py_str = _fmt_ms(py_ms)
            print(f"| {label:<{_COL_NAME}} | {py_str:>{_COL_MS}} |")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_fn(fn: Any) -> float:
    """Call a bench_* function and return the reported ms value."""
    try:
        result = fn()
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)
    except Exception as exc:
        print(f"    [ERROR] {fn.__name__}: {exc}", file=sys.stderr)
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pdfminer.six benchmark suite.",
    )
    parser.add_argument(
        "--filter",
        metavar="STR",
        help="Only run benchmarks whose module name contains STR.",
    )
    parser.add_argument(
        "--rust",
        action="store_true",
        help="Show a Rust column by running each benchmark twice (baseline for when a Rust extension replaces the Python impl).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output a GitHub-flavoured Markdown table.",
    )
    args = parser.parse_args()

    print("Discovering benchmarks…", file=sys.stderr)
    modules = _discover_modules(filter_str=args.filter)
    if not modules:
        print("No benchmark modules found.", file=sys.stderr)
        return 1

    rows: list[tuple[str, float, float]] = []

    for mod in modules:
        bench_fns = _collect_bench_functions(mod)
        for label, fn in bench_fns:
            print(f"  Running {label}…", file=sys.stderr, end=" ", flush=True)
            t0 = time.perf_counter()
            py_ms = _run_fn(fn)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)", file=sys.stderr)

            # --rust: run again (Rust extension may have been swapped in
            # by an environment variable or import hook, but in practice
            # the same Python code runs – this is a placeholder for the
            # workflow where the Rust extension changes the underlying impl)
            rust_ms = float("nan")
            if args.rust:
                rust_ms = _run_fn(fn)

            rows.append((label, py_ms, rust_ms))

    print()
    _print_table(rows, markdown=args.markdown, show_rust=args.rust)
    return 0


if __name__ == "__main__":
    sys.exit(main())
