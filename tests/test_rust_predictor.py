"""Tests for Rust-accelerated PNG and TIFF predictor functions.

Verifies that the Rust implementation produces byte-identical output to the
Python fallback for all filter types, image dimensions, bit depths, and color
counts covered by the pdfminer predictor code paths.
"""

import random

import pytest

import pdfminer.utils as _utils_module
from pdfminer.utils import (
    apply_png_predictor as py_apply_png_predictor,
    apply_tiff_predictor as py_apply_tiff_predictor,
    paeth_predictor as _paeth,
)

try:
    import pdfminer_core as _core

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built"
)


def _python_apply_png_predictor(
    pred: int,
    colors: int,
    columns: int,
    bitspercomponent: int,
    data: bytes,
) -> bytes:
    """Call the pure-Python PNG predictor, bypassing any Rust fast-path."""
    old = _utils_module._HAS_RUST  # noqa: SLF001
    try:
        _utils_module._HAS_RUST = False
        return _utils_module.apply_png_predictor(pred, colors, columns, bitspercomponent, data)
    finally:
        _utils_module._HAS_RUST = old


def _python_apply_tiff_predictor(
    colors: int,
    columns: int,
    bitspercomponent: int,
    data: bytes,
) -> bytes:
    """Call the pure-Python TIFF predictor, bypassing any Rust fast-path."""
    old = _utils_module._HAS_RUST  # noqa: SLF001
    try:
        _utils_module._HAS_RUST = False
        return _utils_module.apply_tiff_predictor(colors, columns, bitspercomponent, data)
    finally:
        _utils_module._HAS_RUST = old


def _rust_apply_png_predictor(
    pred: int,
    colors: int,
    columns: int,
    bitspercomponent: int,
    data: bytes,
) -> bytes:
    return bytes(_core.apply_png_predictor(pred, colors, columns, bitspercomponent, data))


def _rust_apply_tiff_predictor(
    colors: int,
    columns: int,
    bitspercomponent: int,
    data: bytes,
) -> bytes:
    return bytes(_core.apply_tiff_predictor(colors, columns, bitspercomponent, data))


# ---------------------------------------------------------------------------
# Helpers to build synthetic encoded data for each PNG filter type.
# ---------------------------------------------------------------------------

def _encode_png_none(raw_rows: list[bytes]) -> bytes:
    """Prepend filter byte 0 (None) to each row."""
    out = bytearray()
    for row in raw_rows:
        out += bytes([0]) + row
    return bytes(out)


def _encode_png_sub(raw_rows: list[bytes], bpp: int) -> bytes:
    """Apply Sub filter (type 1) to each row."""
    out = bytearray()
    for row in raw_rows:
        encoded = bytearray(len(row))
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            encoded[i] = (row[i] - left) & 0xFF
        out += bytes([1]) + bytes(encoded)
    return bytes(out)


def _encode_png_up(raw_rows: list[bytes]) -> bytes:
    """Apply Up filter (type 2) to each row."""
    out = bytearray()
    prev = bytes(len(raw_rows[0]))
    for row in raw_rows:
        encoded = bytearray((b - p) & 0xFF for b, p in zip(row, prev))
        out += bytes([2]) + bytes(encoded)
        prev = row
    return bytes(out)


def _encode_png_average(raw_rows: list[bytes], bpp: int) -> bytes:
    """Apply Average filter (type 3) to each row."""
    out = bytearray()
    prev = bytes(len(raw_rows[0]))
    for row in raw_rows:
        encoded = bytearray(len(row))
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            above = prev[i]
            encoded[i] = (row[i] - ((left + above) // 2)) & 0xFF
        out += bytes([3]) + bytes(encoded)
        prev = row
    return bytes(out)


def _encode_png_paeth(raw_rows: list[bytes], bpp: int) -> bytes:
    """Apply Paeth filter (type 4) to each row."""
    out = bytearray()
    prev = bytes(len(raw_rows[0]))
    for row in raw_rows:
        encoded = bytearray(len(row))
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            above = prev[i]
            ul = prev[i - bpp] if i >= bpp else 0
            encoded[i] = (row[i] - _paeth(left, above, ul)) & 0xFF
        out += bytes([4]) + bytes(encoded)
        prev = row
    return bytes(out)


# ---------------------------------------------------------------------------
# Fixtures & parametrize helpers
# ---------------------------------------------------------------------------

# (colors, columns, bitspercomponent)
PNG_PARAMS = [
    (1, 1, 8),
    (1, 10, 8),
    (3, 10, 8),
    (4, 10, 8),
    (1, 100, 8),
    (3, 100, 8),
    (4, 100, 8),
    (1, 8, 1),   # 1-bit depth, 8 columns => 1 byte per row
]

TIFF_PARAMS = [
    (1, 1, 8),
    (1, 10, 8),
    (3, 10, 8),
    (4, 10, 8),
    (1, 100, 8),
    (3, 100, 8),
    (4, 100, 8),
]

ROWS = 5  # number of scanlines per synthetic image
SEED = 42


def _make_raw_rows(colors: int, columns: int, bitspercomponent: int, nrows: int) -> list[bytes]:
    rng = random.Random(SEED)
    nbytes = colors * columns * bitspercomponent // 8
    return [bytes(rng.randint(0, 255) for _ in range(nbytes)) for _ in range(nrows)]


# ---------------------------------------------------------------------------
# PNG predictor tests
# ---------------------------------------------------------------------------

class TestPngPredictorFilterNone:
    @pytest.mark.parametrize("colors,columns,bpc", PNG_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        raw_rows = _make_raw_rows(colors, columns, bpc, ROWS)
        encoded = _encode_png_none(raw_rows)
        py_out = _python_apply_png_predictor(10, colors, columns, bpc, encoded)
        rs_out = _rust_apply_png_predictor(10, colors, columns, bpc, encoded)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


class TestPngPredictorFilterSub:
    @pytest.mark.parametrize("colors,columns,bpc", PNG_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        raw_rows = _make_raw_rows(colors, columns, bpc, ROWS)
        bpp = max(1, colors * bpc // 8)
        encoded = _encode_png_sub(raw_rows, bpp)
        py_out = _python_apply_png_predictor(11, colors, columns, bpc, encoded)
        rs_out = _rust_apply_png_predictor(11, colors, columns, bpc, encoded)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


class TestPngPredictorFilterUp:
    @pytest.mark.parametrize("colors,columns,bpc", PNG_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        raw_rows = _make_raw_rows(colors, columns, bpc, ROWS)
        encoded = _encode_png_up(raw_rows)
        py_out = _python_apply_png_predictor(12, colors, columns, bpc, encoded)
        rs_out = _rust_apply_png_predictor(12, colors, columns, bpc, encoded)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


class TestPngPredictorFilterAverage:
    @pytest.mark.parametrize("colors,columns,bpc", PNG_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        raw_rows = _make_raw_rows(colors, columns, bpc, ROWS)
        bpp = max(1, colors * bpc // 8)
        encoded = _encode_png_average(raw_rows, bpp)
        py_out = _python_apply_png_predictor(13, colors, columns, bpc, encoded)
        rs_out = _rust_apply_png_predictor(13, colors, columns, bpc, encoded)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


class TestPngPredictorFilterPaeth:
    @pytest.mark.parametrize("colors,columns,bpc", PNG_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        raw_rows = _make_raw_rows(colors, columns, bpc, ROWS)
        bpp = max(1, colors * bpc // 8)
        encoded = _encode_png_paeth(raw_rows, bpp)
        py_out = _python_apply_png_predictor(14, colors, columns, bpc, encoded)
        rs_out = _rust_apply_png_predictor(14, colors, columns, bpc, encoded)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


class TestPngPredictorKnownValues:
    """Hand-crafted known encoded/decoded pairs for each filter type."""

    def test_filter_none_single_pixel(self) -> None:
        encoded = bytes([0, 0xAB])
        assert _rust_apply_png_predictor(10, 1, 1, 8, encoded) == bytes([0xAB])

    def test_filter_sub_basic(self) -> None:
        # raw = [10, 20, 30] => sub = [10, 10, 10]
        encoded = bytes([1, 10, 10, 10])
        assert _rust_apply_png_predictor(10, 1, 3, 8, encoded) == bytes([10, 20, 30])

    def test_filter_up_basic(self) -> None:
        # row1 raw=[5,10], up=[5,10] (no prior); row2 raw=[7,15], up=[2,5]
        encoded = bytes([2, 5, 10, 2, 2, 5])
        assert _rust_apply_png_predictor(10, 1, 2, 8, encoded) == bytes([5, 10, 7, 15])

    def test_filter_average_basic(self) -> None:
        # left=0, above=0 => average=0, decoded = encoded
        encoded = bytes([3, 50])
        assert _rust_apply_png_predictor(10, 1, 1, 8, encoded) == bytes([50])

    def test_filter_paeth_basic(self) -> None:
        # all context=0 => paeth=0, decoded = encoded
        encoded = bytes([4, 77])
        assert _rust_apply_png_predictor(10, 1, 1, 8, encoded) == bytes([77])

    def test_multi_row_all_filter_types(self) -> None:
        """5 rows, one per filter type, 3 columns, 1 color, 8 bpc."""
        raw_rows = [bytes([10 * (i + 1), 20 * (i + 1), 30 * (i + 1)]) for i in range(5)]
        filter_types = [0, 1, 2, 3, 4]
        bpp = 1
        out = bytearray()
        prev = bytes(3)
        for ft, row in zip(filter_types, raw_rows):
            if ft == 0:
                enc = row
            elif ft == 1:
                enc = bytes([(row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(3)])
            elif ft == 2:
                enc = bytes([(row[i] - prev[i]) & 0xFF for i in range(3)])
            elif ft == 3:
                enc = bytes(
                    [(row[i] - (((row[i - bpp] if i >= bpp else 0) + prev[i]) // 2)) & 0xFF for i in range(3)]
                )
            else:  # 4
                enc = bytes(
                    [
                        (
                            row[i]
                            - _paeth(
                                row[i - bpp] if i >= bpp else 0,
                                prev[i],
                                prev[i - bpp] if i >= bpp else 0,
                            )
                        )
                        & 0xFF
                        for i in range(3)
                    ]
                )
            out += bytes([ft]) + enc
            prev = row

        encoded = bytes(out)
        py_out = _python_apply_png_predictor(10, 1, 3, 8, encoded)
        rs_out = _rust_apply_png_predictor(10, 1, 3, 8, encoded)
        assert rs_out == py_out
        assert rs_out == b"".join(raw_rows)


# ---------------------------------------------------------------------------
# TIFF predictor tests
# ---------------------------------------------------------------------------

class TestTiffPredictorKnownValues:
    def test_single_row_identity(self) -> None:
        """First bpp bytes are returned as-is; subsequent bytes accumulate."""
        data = bytes([10, 20, 30])
        assert _rust_apply_tiff_predictor(1, 3, 8, data) == bytes([10, 30, 60])

    def test_single_pixel(self) -> None:
        data = bytes([42])
        assert _rust_apply_tiff_predictor(1, 1, 8, data) == bytes([42])

    def test_rgb_row(self) -> None:
        """3-component, 2 pixels: [R0,G0,B0, dR,dG,dB]."""
        encoded = bytes([100, 150, 200, 10, 10, 10])
        expected = bytes([100, 150, 200, 110, 160, 210])
        assert _rust_apply_tiff_predictor(3, 2, 8, encoded) == expected


class TestTiffPredictorMatchesPython:
    @pytest.mark.parametrize("colors,columns,bpc", TIFF_PARAMS)
    def test_rust_matches_python(self, colors: int, columns: int, bpc: int) -> None:
        rng = random.Random(SEED + 1)
        nbytes = colors * columns * (bpc // 8)
        nrows = 5
        data = bytes(rng.randint(0, 255) for _ in range(nbytes * nrows))
        py_out = _python_apply_tiff_predictor(colors, columns, bpc, data)
        rs_out = _rust_apply_tiff_predictor(colors, columns, bpc, data)
        assert rs_out == py_out, f"Mismatch for colors={colors} columns={columns} bpc={bpc}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_png_unsupported_bitspercomponent(self) -> None:
        with pytest.raises(ValueError):
            _rust_apply_png_predictor(10, 1, 1, 16, bytes([0, 1, 2]))

    def test_png_unknown_filter_type(self) -> None:
        with pytest.raises(ValueError):
            _rust_apply_png_predictor(10, 1, 1, 8, bytes([5, 42]))

    def test_tiff_unsupported_bitspercomponent(self) -> None:
        with pytest.raises(ValueError):
            _rust_apply_tiff_predictor(1, 1, 1, bytes([42]))


# ---------------------------------------------------------------------------
# Integration: public API uses Rust when available
# ---------------------------------------------------------------------------

class TestPublicApiUsesRust:
    def test_png_predictor_returns_bytes(self) -> None:
        """The public apply_png_predictor returns bytes (not list/bytearray)."""
        encoded = bytes([0, 0xAB])
        result = py_apply_png_predictor(10, 1, 1, 8, encoded)
        assert isinstance(result, bytes)
        assert result == bytes([0xAB])

    def test_tiff_predictor_returns_bytes(self) -> None:
        encoded = bytes([5, 5, 5])
        result = py_apply_tiff_predictor(1, 3, 8, encoded)
        assert isinstance(result, bytes)
        assert result == bytes([5, 10, 15])
