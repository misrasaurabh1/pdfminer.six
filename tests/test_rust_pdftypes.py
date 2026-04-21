"""Tests for Rust PDF stream filter-chain decode pipeline."""

import zlib

import pytest

from pdfminer.pdftypes import (
    LITERALS_ASCII85_DECODE,
    LITERALS_ASCIIHEX_DECODE,
    LITERALS_FLATE_DECODE,
    PDFObjRef,
    PDFStream,
    resolve1,
)


# ---------------------------------------------------------------------------
# Smoke-test: all symbols that `unstructured` imports must be available
# ---------------------------------------------------------------------------


def test_pdftypes_imports() -> None:
    """Critical: all symbols accessed by downstream consumers must import."""
    assert LITERALS_FLATE_DECODE is not None
    assert LITERALS_ASCII85_DECODE is not None
    assert LITERALS_ASCIIHEX_DECODE is not None
    assert PDFObjRef is not None
    assert PDFStream is not None
    assert resolve1 is not None


# ---------------------------------------------------------------------------
# PDFStream attribute access (unstructured compatibility)
# ---------------------------------------------------------------------------


def test_pdfstream_attributes() -> None:
    """PDFStream must expose .data, .rawdata, .attrs, .objid, .genno, .decipher."""
    raw = b"hello"
    stream = PDFStream({"Length": len(raw)}, raw)
    assert stream.rawdata == raw
    assert stream.data is None
    assert stream.attrs == {"Length": len(raw)}
    assert stream.objid is None
    assert stream.genno is None
    assert stream.decipher is None


def test_pdfstream_set_objid() -> None:
    stream = PDFStream({}, b"")
    stream.set_objid(42, 0)
    assert stream.objid == 42
    assert stream.genno == 0


def test_pdfstream_get_rawdata() -> None:
    raw = b"raw bytes"
    stream = PDFStream({}, raw)
    assert stream.get_rawdata() == raw


# ---------------------------------------------------------------------------
# FlateDecode
# ---------------------------------------------------------------------------


def test_pdfstream_decode_flate_basic() -> None:
    original = b"Hello, world! " * 100
    compressed = zlib.compress(original)
    stream = PDFStream(
        {"Filter": b"FlateDecode", "Length": len(compressed)},
        compressed,
    )
    assert stream.get_data() == original


def test_pdfstream_decode_flate_empty() -> None:
    compressed = zlib.compress(b"")
    stream = PDFStream({"Filter": b"FlateDecode"}, compressed)
    assert stream.get_data() == b""


# ---------------------------------------------------------------------------
# Multi-filter chain (FlateDecode + FlateDecode)
# ---------------------------------------------------------------------------


def test_pdfstream_decode_chain() -> None:
    """Two FlateDecode filters applied in sequence."""
    original = b"chained compression test " * 50
    layer1 = zlib.compress(original)
    layer2 = zlib.compress(layer1)

    from pdfminer.psparser import LIT

    stream = PDFStream(
        {
            "Filter": [LIT("FlateDecode"), LIT("FlateDecode")],
            "Length": len(layer2),
        },
        layer2,
    )
    assert stream.get_data() == original


# ---------------------------------------------------------------------------
# PDFObjRef.resolve
# ---------------------------------------------------------------------------


def test_pdfobjref_repr() -> None:
    ref = PDFObjRef(None, 7)
    assert "7" in repr(ref)


def test_pdfobjref_resolve_default() -> None:
    """Resolving with no document returns default."""
    ref = PDFObjRef(None, 1)
    with pytest.raises(AssertionError):
        # doc is None → assert self.doc is not None fires
        ref.resolve()


def test_resolve1_non_ref() -> None:
    """resolve1 on a plain value returns the value unchanged."""
    assert resolve1(42) == 42
    assert resolve1(b"bytes") == b"bytes"
    assert resolve1(None) is None


# ---------------------------------------------------------------------------
# get_filters()
# ---------------------------------------------------------------------------


def test_get_filters_empty() -> None:
    stream = PDFStream({}, b"")
    assert stream.get_filters() == []


def test_get_filters_single() -> None:
    from pdfminer.psparser import LIT

    stream = PDFStream({"Filter": LIT("FlateDecode")}, b"")
    filters = stream.get_filters()
    assert len(filters) == 1
    assert filters[0][0] == LIT("FlateDecode")


# ---------------------------------------------------------------------------
# Rust fast path is exercised when available
# ---------------------------------------------------------------------------


def test_rust_decode_stream_filters_available() -> None:
    """If the Rust extension is built, decode_stream_filters must be importable."""
    try:
        from pdfminer_core import decode_stream_filters  # type: ignore[import-not-found]

        assert callable(decode_stream_filters)
    except ImportError:
        pytest.skip("pdfminer_core not built — skipping Rust-specific test")


def test_rust_flate_roundtrip() -> None:
    """Direct call to decode_stream_filters for FlateDecode."""
    try:
        from pdfminer_core import decode_stream_filters  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("pdfminer_core not built")

    original = b"Rust flate roundtrip test " * 200
    compressed = zlib.compress(original)
    result = decode_stream_filters(compressed, [("FlateDecode", {})])
    assert result == original


def test_rust_passthrough_dct() -> None:
    """DCTDecode should return data unchanged (pass-through)."""
    try:
        from pdfminer_core import decode_stream_filters  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("pdfminer_core not built")

    data = b"\xff\xd8\xff" + b"\x00" * 100  # fake JPEG header
    result = decode_stream_filters(data, [("DCTDecode", {})])
    assert result == data
