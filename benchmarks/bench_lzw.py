"""
Benchmarks: LZW decompression.

Compares the pure-Python ``lzwdecode`` (LZWDecoder class in pdfminer/lzw.py)
against the Rust ``lzw_decode`` (pdfminer_core.lzw_decode).

Test payloads are synthetically constructed with a minimal literal encoder
(same approach as the Rust unit tests in pdfminer_core/src/codecs/lzw.rs).
We also include a highly-compressible run-length payload to exercise the
table-growth (bit-width expansion) code paths.
"""

from __future__ import annotations

from benchmarks.bench_utils import BenchmarkSuite
from pdfminer.lzw import lzwdecode as _py_lzwdecode

try:
    from pdfminer_core import lzw_decode as _rs_lzwdecode

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal LZW encoder (no compression, just literal packing)
# Used to produce valid LZW bitstreams from arbitrary raw bytes.
# ---------------------------------------------------------------------------

def _lzw_encode_literals(data: bytes) -> bytes:
    """Pack raw bytes into a 9-bit literal LZW bitstream (CLEAR + literals + EOD)."""
    out = bytearray()
    bit_buf: int = 0
    bit_count: int = 0

    def push_code(code: int, nbits: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf = (bit_buf << nbits) | code
        bit_count += nbits
        while bit_count >= 8:
            bit_count -= 8
            out.append((bit_buf >> bit_count) & 0xFF)

    push_code(256, 9)  # CLEAR
    for b in data:
        push_code(b, 9)
    push_code(257, 9)  # EOD
    if bit_count > 0:
        out.append((bit_buf << (8 - bit_count)) & 0xFF)
    return bytes(out)


def _lzw_encode_compressible(data: bytes) -> bytes:
    """
    Encode using the real LZW algorithm (with compression) so that the
    decoder table grows and bit-widths expand.  This exercises the 9->10->11->12
    bit code-width transitions in both the Python and Rust decoders.

    This is a simplified but correct LZW encoder (PDF earlychange=1 semantics):
    code width bumps when table reaches 511 / 1023 / 2047 entries.
    """
    CLEAR = 256
    EOD = 257

    out = bytearray()
    bit_buf: int = 0
    bit_count: int = 0

    def flush_bits() -> None:
        nonlocal bit_buf, bit_count
        while bit_count >= 8:
            bit_count -= 8
            out.append((bit_buf >> bit_count) & 0xFF)

    def push_code(code: int, nbits: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf = (bit_buf << nbits) | code
        bit_count += nbits
        flush_bits()

    # Initialise table
    table: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    table[b"CLEAR"] = CLEAR  # sentinel – not a real entry
    next_code = 258  # 256 = CLEAR, 257 = EOD
    nbits = 9

    push_code(CLEAR, nbits)

    w = b""
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
        else:
            push_code(table[w], nbits)
            if next_code <= 4095:
                table[wc] = next_code
                next_code += 1
                # earlychange = 1: bump nbits when table length hits 511/1023/2047
                if next_code == 512:
                    nbits = 10
                elif next_code == 1024:
                    nbits = 11
                elif next_code == 2048:
                    nbits = 12
            else:
                # Table full — emit CLEAR and reset
                push_code(CLEAR, nbits)
                table = {bytes([i]): i for i in range(256)}
                next_code = 258
                nbits = 9
            w = bytes([byte])

    if w:
        push_code(table[w], nbits)
    push_code(EOD, nbits)
    if bit_count > 0:
        out.append((bit_buf << (8 - bit_count)) & 0xFF)
    return bytes(out)


# ---------------------------------------------------------------------------
# Build test payloads
# ---------------------------------------------------------------------------

import random as _random

_RNG = _random.Random(99)

def _make_random_bytes(n: int) -> bytes:
    return bytes(_RNG.randint(0, 255) for _ in range(n))

def _make_repetitive_bytes(n: int) -> bytes:
    """Highly compressible: 8-byte pattern repeated."""
    pattern = b"\x41\x42\x43\x44\x10\x20\x30\x40"
    return (pattern * (n // len(pattern) + 1))[:n]


# Payloads: (label, encoded_bytes)
_PAYLOADS: list[tuple[str, bytes]] = []

# 1 kB random — minimal compression
_PAYLOADS.append(("1kb_random_literal",      _lzw_encode_literals(_make_random_bytes(1024))))
# 8 kB random
_PAYLOADS.append(("8kb_random_literal",      _lzw_encode_literals(_make_random_bytes(8192))))
# 32 kB random
_PAYLOADS.append(("32kb_random_literal",     _lzw_encode_literals(_make_random_bytes(32768))))
# 1 kB repetitive — triggers many table hits and bit-width growth
_PAYLOADS.append(("1kb_repetitive_compressed", _lzw_encode_compressible(_make_repetitive_bytes(1024))))
# 8 kB repetitive
_PAYLOADS.append(("8kb_repetitive_compressed", _lzw_encode_compressible(_make_repetitive_bytes(8192))))
# 32 kB repetitive — exercises table-full and CLEAR resets
_PAYLOADS.append(("32kb_repetitive_compressed", _lzw_encode_compressible(_make_repetitive_bytes(32768))))


# ---------------------------------------------------------------------------
# Register benchmarks
# ---------------------------------------------------------------------------

def register(suite: BenchmarkSuite) -> None:
    """Add all LZW benchmarks to *suite*."""
    for label, encoded in _PAYLOADS:
        group = f"lzw_{label}"
        name = f"LZW {label}"

        suite.add(
            group=group,
            variant="python",
            name=f"{name} [py]",
            fn=lambda d=encoded: _py_lzwdecode(d),
        )

        if _RUST_AVAILABLE:
            suite.add(
                group=group,
                variant="rust",
                name=f"{name} [rs]",
                fn=lambda d=encoded: _rs_lzwdecode(d),
            )
