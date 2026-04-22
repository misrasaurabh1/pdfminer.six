"""
Benchmarks: PNG and TIFF predictor decompression.

Covers apply_png_predictor and apply_tiff_predictor with a range of
image sizes and filter types (0-None, 1-Sub, 2-Up, 3-Average, 4-Paeth).

Both the pure-Python fallback (from pdfminer.utils) and the Rust
implementation (from pdfminer_core) are timed.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from benchmarks.bench_utils import BenchmarkSuite

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Import both implementations
# ---------------------------------------------------------------------------

import pdfminer.utils as _utils_module
from pdfminer.utils import (
    apply_png_predictor as _dispatch_png,
    apply_tiff_predictor as _dispatch_tiff,
)

try:
    import pdfminer_core as _core

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _py_png(pred, colors, columns, bpc, data):
    """Call the pure-Python PNG predictor."""
    old = _utils_module._HAS_RUST  # noqa: SLF001
    try:
        _utils_module._HAS_RUST = False
        return _utils_module.apply_png_predictor(pred, colors, columns, bpc, data)
    finally:
        _utils_module._HAS_RUST = old


def _py_tiff(colors, columns, bpc, data):
    """Call the pure-Python TIFF predictor."""
    old = _utils_module._HAS_RUST  # noqa: SLF001
    try:
        _utils_module._HAS_RUST = False
        return _utils_module.apply_tiff_predictor(colors, columns, bpc, data)
    finally:
        _utils_module._HAS_RUST = old


def _rs_png(pred, colors, columns, bpc, data):
    return bytes(_core.apply_png_predictor(pred, colors, columns, bpc, data))


def _rs_tiff(colors, columns, bpc, data):
    return bytes(_core.apply_tiff_predictor(colors, columns, bpc, data))


# ---------------------------------------------------------------------------
# Synthetic data builders  (mirrors those in test_rust_predictor.py)
# ---------------------------------------------------------------------------

_RNG = random.Random(42)

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVERAGE = 3
FILTER_PAETH = 4


def _paeth(left: int, above: int, upper_left: int) -> int:
    p = left + above - upper_left
    pa = abs(p - left)
    pb = abs(p - above)
    pc = abs(p - upper_left)
    if pa <= pb and pa <= pc:
        return left
    elif pb <= pc:
        return above
    return upper_left


def _make_raw_rows(colors: int, columns: int, bpc: int, nrows: int) -> list[bytes]:
    nbytes = colors * columns * bpc // 8
    return [bytes(_RNG.randint(0, 255) for _ in range(nbytes)) for _ in range(nrows)]


def _encode_png(filter_type: int, raw_rows: list[bytes], bpp: int) -> bytes:
    out = bytearray()
    prev = bytes(len(raw_rows[0]))
    for row in raw_rows:
        n = len(row)
        if filter_type == FILTER_NONE:
            enc = row
        elif filter_type == FILTER_SUB:
            enc = bytes([(row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(n)])
        elif filter_type == FILTER_UP:
            enc = bytes([(row[i] - prev[i]) & 0xFF for i in range(n)])
        elif filter_type == FILTER_AVERAGE:
            enc = bytes(
                [(row[i] - (((row[i - bpp] if i >= bpp else 0) + prev[i]) // 2)) & 0xFF for i in range(n)]
            )
        elif filter_type == FILTER_PAETH:
            enc = bytes(
                [
                    (row[i] - _paeth(row[i - bpp] if i >= bpp else 0, prev[i], prev[i - bpp] if i >= bpp else 0))
                    & 0xFF
                    for i in range(n)
                ]
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        out += bytes([filter_type]) + enc
        prev = row
    return bytes(out)


def _encode_tiff(raw_rows: list[bytes], bpp: int) -> bytes:
    """Encode using TIFF predictor 2 (delta encoding)."""
    out = bytearray()
    for row in raw_rows:
        n = len(row)
        enc = bytearray(n)
        for i in range(n):
            enc[i] = (row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF
        out += enc
    return bytes(out)


# ---------------------------------------------------------------------------
# Pre-build test payloads at import time (shared across all runs)
# ---------------------------------------------------------------------------

# (label_suffix, colors, columns, bpc, nrows)
_PNG_SCENARIOS: list[tuple[str, int, int, int, int]] = [
    ("small_1c_8bpc",    1,   64,  8,  10),
    ("medium_3c_8bpc",   3,  256,  8,  50),
    ("large_3c_8bpc",    3,  800,  8, 100),
    ("large_4c_8bpc",    4,  800,  8, 100),
]

_TIFF_SCENARIOS: list[tuple[str, int, int, int, int]] = [
    ("small_1c_8bpc",    1,   64,  8,  10),
    ("medium_3c_8bpc",   3,  256,  8,  50),
    ("large_3c_8bpc",    3,  800,  8, 100),
    ("large_4c_8bpc",    4,  800,  8, 100),
]

_PNG_FILTER_TYPES = [
    ("none",    FILTER_NONE),
    ("sub",     FILTER_SUB),
    ("up",      FILTER_UP),
    ("average", FILTER_AVERAGE),
    ("paeth",   FILTER_PAETH),
]


def _build_png_payloads() -> dict[str, tuple[int, int, int, int, bytes]]:
    """Return {key: (pred, colors, columns, bpc, encoded_data)} for PNG."""
    payloads: dict[str, tuple[int, int, int, int, bytes]] = {}
    for label, colors, columns, bpc, nrows in _PNG_SCENARIOS:
        bpp = max(1, colors * bpc // 8)
        raw_rows = _make_raw_rows(colors, columns, bpc, nrows)
        for ft_name, ft in _PNG_FILTER_TYPES:
            encoded = _encode_png(ft, raw_rows, bpp)
            key = f"{label}_filter_{ft_name}"
            payloads[key] = (10 + ft, colors, columns, bpc, encoded)
    return payloads


def _build_tiff_payloads() -> dict[str, tuple[int, int, int, bytes]]:
    """Return {key: (colors, columns, bpc, encoded_data)} for TIFF."""
    payloads: dict[str, tuple[int, int, int, bytes]] = {}
    for label, colors, columns, bpc, nrows in _TIFF_SCENARIOS:
        bpp = colors * (bpc // 8)
        raw_rows = _make_raw_rows(colors, columns, bpc, nrows)
        encoded = _encode_tiff(raw_rows, bpp)
        payloads[label] = (colors, columns, bpc, encoded)
    return payloads


_PNG_PAYLOADS = _build_png_payloads()
_TIFF_PAYLOADS = _build_tiff_payloads()


# ---------------------------------------------------------------------------
# Register benchmarks
# ---------------------------------------------------------------------------

def register(suite: BenchmarkSuite) -> None:
    """Add all predictor benchmarks to *suite*."""

    # --- PNG predictor ---
    for key, (pred, colors, columns, bpc, data) in sorted(_PNG_PAYLOADS.items()):
        group = f"png_{key}"
        name = f"PNG predictor {key}"

        # Python baseline
        suite.add(
            group=group,
            variant="python",
            name=f"{name} [py]",
            fn=lambda p=pred, c=colors, col=columns, b=bpc, d=data: _py_png(p, c, col, b, d),
        )

        # Rust implementation
        if _RUST_AVAILABLE:
            suite.add(
                group=group,
                variant="rust",
                name=f"{name} [rs]",
                fn=lambda p=pred, c=colors, col=columns, b=bpc, d=data: _rs_png(p, c, col, b, d),
            )

    # --- TIFF predictor ---
    for key, (colors, columns, bpc, data) in sorted(_TIFF_PAYLOADS.items()):
        group = f"tiff_{key}"
        name = f"TIFF predictor {key}"

        suite.add(
            group=group,
            variant="python",
            name=f"{name} [py]",
            fn=lambda c=colors, col=columns, b=bpc, d=data: _py_tiff(c, col, b, d),
        )

        if _RUST_AVAILABLE:
            suite.add(
                group=group,
                variant="rust",
                name=f"{name} [rs]",
                fn=lambda c=colors, col=columns, b=bpc, d=data: _rs_tiff(c, col, b, d),
            )
