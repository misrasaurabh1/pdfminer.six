"""
Benchmarks: PDF cross-reference table parsing.

Covers two Rust functions that have direct Python equivalents:
  - parse_xref_table  (Rust) vs the Python loop in PDFXRef.load()
  - parse_xref_stream (Rust) vs the Python loop in PDFXRefStream.load()
  - scan_pdf_objects  (Rust) vs PDFXRefFallback (regex scan)

Synthetic xref table/stream data is generated so the benchmarks are
self-contained and do not require real PDF files on disk.
"""

from __future__ import annotations

import re
import struct

from benchmarks.bench_utils import BenchmarkSuite

try:
    from pdfminer_core import parse_xref_stream, parse_xref_table, scan_pdf_objects

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Python equivalents (extracted from pdfdocument.py / pdfparser.py)
# ---------------------------------------------------------------------------

def _py_parse_xref_table(data: bytes) -> list[tuple[int, int, int]]:
    """
    Mimics the core of PDFXRef.load() — parses a standard xref table
    and returns [(objid, genno, offset), ...] for in-use entries.
    """
    result = []
    current_objid = 0
    for raw_line in data.split(b"\n"):
        line = raw_line.strip()
        if not line or line == b"xref" or line.startswith(b"trailer"):
            continue
        parts = line.split()
        if len(parts) == 2:
            current_objid = int(parts[0])
        elif len(parts) == 3:
            offset = int(parts[0])
            genno = int(parts[1])
            use = parts[2]
            if use == b"n":
                result.append((current_objid, genno, offset))
            current_objid += 1
    return result


_PDFOBJ_CUE = re.compile(rb"^(\d+)\s+(\d+)\s+obj\b")

def _py_scan_pdf_objects(data: bytes) -> list[tuple[int, int, int]]:
    """
    Mimics PDFXRefFallback: scans byte data for lines matching '<n> <n> obj'.
    Returns [(objid, genno, byte_offset), ...].
    """
    results = []
    pos = 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        if end == -1:
            end = len(data)
        line = data[pos:end]
        m = _PDFOBJ_CUE.match(line)
        if m:
            objid = int(m.group(1))
            genno = int(m.group(2))
            results.append((objid, genno, pos))
        pos = end + 1
    return results


def _read_be(data: bytes, width: int) -> int:
    v = 0
    for b in data[:width]:
        v = (v << 8) | b
    return v


def _py_parse_xref_stream(
    data: bytes, w: list[int], index: list[int]
) -> list[tuple[int, int, int, int]]:
    """
    Mimics the Python xref-stream parser in PDFDocument._read_xref_stream().
    Returns [(entry_type, objid, field2, field3), ...].
    """
    w1, w2, w3 = w
    entry_size = w1 + w2 + w3
    results = []
    data_pos = 0
    for chunk_start in range(0, len(index), 2):
        start_objid = index[chunk_start]
        count = index[chunk_start + 1]
        for i in range(count):
            if data_pos + entry_size > len(data):
                break
            entry_type = _read_be(data[data_pos:], w1) if w1 > 0 else 1
            f2 = _read_be(data[data_pos + w1:], w2)
            f3 = _read_be(data[data_pos + w1 + w2:], w3)
            results.append((entry_type, start_objid + i, f2, f3))
            data_pos += entry_size
    return results


# ---------------------------------------------------------------------------
# Build synthetic payloads
# ---------------------------------------------------------------------------

def _build_xref_table(n_objects: int) -> bytes:
    """Generate a syntactically-valid xref table for *n_objects* objects."""
    lines = [b"xref", f"0 {n_objects}".encode()]
    for i in range(n_objects):
        offset = 1000 + i * 512
        genno = 0
        use = b"n"
        lines.append(f"{offset:010d} {genno:05d} ".encode() + use)
    lines.append(b"trailer")
    return b"\n".join(lines) + b"\n"


def _build_xref_stream(n_objects: int) -> tuple[bytes, list[int], list[int]]:
    """
    Generate a compressed xref stream payload (already decoded — just raw entry bytes).
    Entry format: type(1), offset(4), genno(2)  => w = [1, 4, 2], entry_size = 7.
    """
    w = [1, 4, 2]
    index = [0, n_objects]
    rows = []
    for i in range(n_objects):
        entry_type = 1
        offset = 1000 + i * 512
        genno = 0
        rows.append(struct.pack(">BIBH", entry_type, offset, 0, genno)[:7])
        # Pack: B=type(1 byte), I=offset(4 bytes), H=genno(2 bytes) = 7 bytes
    data = b"".join(rows)
    return data, w, index


def _build_scan_data(n_objects: int) -> bytes:
    """Generate a fake PDF body with n_objects object-header lines."""
    chunks = []
    pos = 0
    for i in range(1, n_objects + 1):
        # some padding bytes before each object marker
        padding = b"% comment line\n" * 3
        header = f"{i} 0 obj\n".encode()
        body = b"<< /Type /Dummy >>\nendobj\n"
        chunks.append(padding + header + body)
    return b"".join(chunks)


# Payload sizes
_SMALL = 100
_MEDIUM = 1_000
_LARGE = 5_000

_TABLE_SMALL,  = (_build_xref_table(_SMALL),)
_TABLE_MEDIUM, = (_build_xref_table(_MEDIUM),)
_TABLE_LARGE,  = (_build_xref_table(_LARGE),)

_STREAM_SMALL,  _W, _IDX_S  = _build_xref_stream(_SMALL)
_STREAM_MEDIUM, _W, _IDX_M  = _build_xref_stream(_MEDIUM)
_STREAM_LARGE,  _W, _IDX_L  = _build_xref_stream(_LARGE)

_SCAN_SMALL,  = (_build_scan_data(_SMALL),)
_SCAN_MEDIUM, = (_build_scan_data(_MEDIUM),)
_SCAN_LARGE,  = (_build_scan_data(_LARGE),)

_STREAM_W = [1, 4, 2]


# ---------------------------------------------------------------------------
# Register benchmarks
# ---------------------------------------------------------------------------

def register(suite: BenchmarkSuite) -> None:
    """Add all xref benchmarks to *suite*."""

    # ---- parse_xref_table ----
    for label, data in [
        ("xref_table_small",  _TABLE_SMALL),
        ("xref_table_medium", _TABLE_MEDIUM),
        ("xref_table_large",  _TABLE_LARGE),
    ]:
        suite.add(
            group=label,
            variant="python",
            name=f"parse_xref_table {label} [py]",
            fn=lambda d=data: _py_parse_xref_table(d),
        )
        if _RUST_AVAILABLE:
            suite.add(
                group=label,
                variant="rust",
                name=f"parse_xref_table {label} [rs]",
                fn=lambda d=data: parse_xref_table(d),
            )

    # ---- parse_xref_stream ----
    for label, data, idx in [
        ("xref_stream_small",  _STREAM_SMALL,  _IDX_S),
        ("xref_stream_medium", _STREAM_MEDIUM, _IDX_M),
        ("xref_stream_large",  _STREAM_LARGE,  _IDX_L),
    ]:
        suite.add(
            group=label,
            variant="python",
            name=f"parse_xref_stream {label} [py]",
            fn=lambda d=data, i=idx: _py_parse_xref_stream(d, _STREAM_W, i),
        )
        if _RUST_AVAILABLE:
            suite.add(
                group=label,
                variant="rust",
                name=f"parse_xref_stream {label} [rs]",
                fn=lambda d=data, i=idx: parse_xref_stream(d, _STREAM_W, i),
            )

    # ---- scan_pdf_objects ----
    for label, data in [
        ("scan_objects_small",  _SCAN_SMALL),
        ("scan_objects_medium", _SCAN_MEDIUM),
        ("scan_objects_large",  _SCAN_LARGE),
    ]:
        suite.add(
            group=label,
            variant="python",
            name=f"scan_pdf_objects {label} [py]",
            fn=lambda d=data: _py_scan_pdf_objects(d),
        )
        if _RUST_AVAILABLE:
            suite.add(
                group=label,
                variant="rust",
                name=f"scan_pdf_objects {label} [rs]",
                fn=lambda d=data: scan_pdf_objects(d),
            )
