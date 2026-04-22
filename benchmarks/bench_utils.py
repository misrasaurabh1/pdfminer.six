"""
Shared infrastructure for pdfminer.six Rust-vs-Python benchmarks.

Usage
-----
Import BenchmarkSuite, instantiate it, call .add() to register cases, then
call .run() to execute them and .report() to print a formatted table.

Each benchmark case is a function that accepts no arguments and returns
whatever value it produces (for correctness cross-checking in tests).

Timing is done with ``timeit.Timer`` repeated ``rounds`` times;  the *best*
of all rounds is kept (standard practice — it most closely reflects the
underlying algorithm speed, with minimal scheduling noise).
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str          # human-readable label
    group: str         # e.g. "predictor_png", "lzw", "matrix"
    variant: str       # "python" or "rust"
    best_sec: float    # best elapsed seconds over all rounds
    rounds: int        # how many rounds were timed


@dataclass
class BenchmarkSuite:
    """Collect and run a set of benchmark cases."""

    rounds: int = 3           # number of timeit repetitions per case
    number: int = 0           # inner-loop reps; 0 = auto-calibrate per case
    results: list[BenchmarkResult] = field(default_factory=list)
    _cases: list[tuple[str, str, str, Callable[[], object]]] = field(
        default_factory=list, repr=False
    )

    def add(
        self,
        group: str,
        variant: str,
        name: str,
        fn: Callable[[], object],
    ) -> None:
        """Register a benchmark case.

        Parameters
        ----------
        group:   category name (used to pair Python vs Rust results)
        variant: "python" or "rust"
        name:    display name
        fn:      zero-argument callable to time
        """
        self._cases.append((group, variant, name, fn))

    def run(self) -> None:
        """Execute all registered cases and store results."""
        self.results.clear()
        for group, variant, name, fn in self._cases:
            timer = timeit.Timer(fn)
            # Auto-calibrate number so each round takes ~200ms max.
            # This keeps fast micro-benchmarks accurate while preventing
            # slow e2e cases (100ms+/call) from blowing up the total runtime.
            if self.number > 0:
                number = self.number
            else:
                # Auto-calibrate: min 5 iterations, target ~1s per round.
                # min() needs enough samples to find the scheduling-noise floor —
                # n=1 lets a single slow OS-preemption run dominate; n>=5 gives the
                # scheduler enough chances to produce at least one clean run.
                one_shot = timeit.timeit(fn, number=1)
                number = max(5, int(1.0 / one_shot))
            times = timer.repeat(repeat=self.rounds, number=number)
            best_per_call = min(times) / number
            self.results.append(
                BenchmarkResult(
                    name=name,
                    group=group,
                    variant=variant,
                    best_sec=best_per_call,
                    rounds=self.rounds,
                )
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, *, show_speedup: bool = True) -> str:
        """Return a formatted text table of results.

        When show_speedup is True, each Rust entry includes a "speedup"
        column that shows Rust vs its paired Python implementation.
        """
        if not self.results:
            return "(no results — call .run() first)"

        # Group results by group name.
        from collections import defaultdict
        by_group: dict[str, dict[str, BenchmarkResult]] = defaultdict(dict)
        for r in self.results:
            by_group[r.group][r.variant] = r

        lines: list[str] = []
        header = f"{'Benchmark':<45} {'Variant':<8} {'Best/call (µs)':>16} {'Speedup':>10}"
        sep = "-" * len(header)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for group in sorted(by_group):
            variants = by_group[group]
            py_res = variants.get("python")
            rs_res = variants.get("rust")

            for variant_key in ("python", "rust"):
                res = variants.get(variant_key)
                if res is None:
                    continue
                us = res.best_sec * 1e6
                speedup_str = ""
                if show_speedup and variant_key == "rust" and py_res is not None:
                    ratio = py_res.best_sec / res.best_sec
                    speedup_str = f"{ratio:.2f}x"
                elif show_speedup and variant_key == "python":
                    speedup_str = "(baseline)"
                lines.append(
                    f"{res.name:<45} {variant_key:<8} {us:>16.3f} {speedup_str:>10}"
                )

        lines.append(sep)
        return "\n".join(lines)

    def speedup_map(self) -> dict[str, float | None]:
        """Return {group: speedup_ratio} where ratio = python_time / rust_time.

        A ratio > 1.0 means Rust is faster.  None means no paired data.
        """
        from collections import defaultdict
        by_group: dict[str, dict[str, BenchmarkResult]] = defaultdict(dict)
        for r in self.results:
            by_group[r.group][r.variant] = r

        out: dict[str, float | None] = {}
        for group, variants in by_group.items():
            py = variants.get("python")
            rs = variants.get("rust")
            if py is not None and rs is not None:
                out[group] = py.best_sec / rs.best_sec
            else:
                out[group] = None
        return out
