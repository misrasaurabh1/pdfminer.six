"""
Benchmarks: PDF transformation matrix operations.

Covers mult_matrix, translate_matrix, apply_matrix_pt, apply_matrix_rect,
and apply_matrix_norm — each has both a pure-Python and a Rust implementation.
These are called very frequently during PDF rendering (once per graphics state
operation), so even small per-call savings multiply up on large documents.
"""

from __future__ import annotations

import random

from benchmarks.bench_utils import BenchmarkSuite
import pdfminer.utils as _utils_module
from pdfminer.utils import (
    Matrix,
    Point,
    Rect,
    apply_matrix_norm,
    apply_matrix_pt,
    apply_matrix_rect,
    mult_matrix,
    translate_matrix,
)

try:
    import pdfminer_core as _core

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Force-Python wrappers (bypass the Rust fast-path for baseline measurement)
# ---------------------------------------------------------------------------

def _py(fn_name: str, *args):  # type: ignore[no-untyped-def]
    old = _utils_module._HAS_RUST  # noqa: SLF001
    try:
        _utils_module._HAS_RUST = False
        return getattr(_utils_module, fn_name)(*args)
    finally:
        _utils_module._HAS_RUST = old


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_RNG = random.Random(7)

def _rand_mat() -> Matrix:
    return tuple(_RNG.uniform(-100, 100) for _ in range(6))  # type: ignore[return-value]

def _rand_pt() -> Point:
    return (_RNG.uniform(0, 600), _RNG.uniform(0, 800))

def _rand_rect() -> Rect:
    x0 = _RNG.uniform(0, 500)
    y0 = _RNG.uniform(0, 700)
    return (x0, y0, x0 + _RNG.uniform(1, 100), y0 + _RNG.uniform(1, 100))


# Pre-generate a list of operand tuples so the benchmark body is pure arithmetic.
_N = 500

_MATRIX_PAIRS:  list[tuple[Matrix, Matrix]] = [(_rand_mat(), _rand_mat()) for _ in range(_N)]
_MATRIX_POINTS: list[tuple[Matrix, Point]]  = [(_rand_mat(), _rand_pt())  for _ in range(_N)]
_MATRIX_RECTS:  list[tuple[Matrix, Rect]]   = [(_rand_mat(), _rand_rect()) for _ in range(_N)]


# ---------------------------------------------------------------------------
# Register benchmarks
# ---------------------------------------------------------------------------

def register(suite: BenchmarkSuite) -> None:
    """Add all matrix benchmarks to *suite*."""

    # ---- mult_matrix ----
    suite.add(
        group="matrix_mult",
        variant="python",
        name="mult_matrix (500 pairs) [py]",
        fn=lambda: [_py("mult_matrix", m1, m0) for m1, m0 in _MATRIX_PAIRS],
    )
    if _RUST_AVAILABLE:
        suite.add(
            group="matrix_mult",
            variant="rust",
            name="mult_matrix (500 pairs) [rs]",
            fn=lambda: [_core.mult_matrix(m1, m0) for m1, m0 in _MATRIX_PAIRS],
        )

    # ---- translate_matrix ----
    suite.add(
        group="matrix_translate",
        variant="python",
        name="translate_matrix (500 calls) [py]",
        fn=lambda: [_py("translate_matrix", m, pt) for m, pt in _MATRIX_POINTS],
    )
    if _RUST_AVAILABLE:
        suite.add(
            group="matrix_translate",
            variant="rust",
            name="translate_matrix (500 calls) [rs]",
            fn=lambda: [_core.translate_matrix(m, pt) for m, pt in _MATRIX_POINTS],
        )

    # ---- apply_matrix_pt ----
    suite.add(
        group="matrix_apply_pt",
        variant="python",
        name="apply_matrix_pt (500 calls) [py]",
        fn=lambda: [_py("apply_matrix_pt", m, pt) for m, pt in _MATRIX_POINTS],
    )
    if _RUST_AVAILABLE:
        suite.add(
            group="matrix_apply_pt",
            variant="rust",
            name="apply_matrix_pt (500 calls) [rs]",
            fn=lambda: [_core.apply_matrix_pt(m, pt) for m, pt in _MATRIX_POINTS],
        )

    # ---- apply_matrix_norm ----
    suite.add(
        group="matrix_apply_norm",
        variant="python",
        name="apply_matrix_norm (500 calls) [py]",
        fn=lambda: [_py("apply_matrix_norm", m, pt) for m, pt in _MATRIX_POINTS],
    )
    if _RUST_AVAILABLE:
        suite.add(
            group="matrix_apply_norm",
            variant="rust",
            name="apply_matrix_norm (500 calls) [rs]",
            fn=lambda: [_core.apply_matrix_norm(m, pt) for m, pt in _MATRIX_POINTS],
        )

    # ---- apply_matrix_rect ----
    suite.add(
        group="matrix_apply_rect",
        variant="python",
        name="apply_matrix_rect (500 calls) [py]",
        fn=lambda: [_py("apply_matrix_rect", m, r) for m, r in _MATRIX_RECTS],
    )
    if _RUST_AVAILABLE:
        suite.add(
            group="matrix_apply_rect",
            variant="rust",
            name="apply_matrix_rect (500 calls) [rs]",
            fn=lambda: [_core.apply_matrix_rect(m, r) for m, r in _MATRIX_RECTS],
        )
