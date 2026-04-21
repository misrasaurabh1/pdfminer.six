"""Tests for the Rust LZW decoder implementation.

These tests verify that the Rust implementation produces byte-identical output
to the pure-Python LZWDecoder, and that the Python wrapper correctly delegates
to Rust when available.
"""

import pytest

from pdfminer.lzw import LZWDecoder, lzwdecode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pack_codes(codes: list[int], nbits_list: list[int]) -> bytes:
    """Pack a list of variable-width codes into a byte string (MSB-first)."""
    bits = 0
    n = 0
    out: list[int] = []
    for code, nb in zip(codes, nbits_list):
        bits = (bits << nb) | code
        n += nb
        while n >= 8:
            n -= 8
            out.append((bits >> n) & 0xFF)
    if n > 0:
        out.append((bits << (8 - n)) & 0xFF)
    return bytes(out)


def encode_literals(payload: bytes) -> bytes:
    """Produce a valid PDF-compatible LZW stream encoding `payload` as literals.

    No compression: each byte is emitted as a single 9-bit code.  The encoder
    correctly tracks table growth and upgrades code width at the same thresholds
    as the Python decoder.
    """
    codes: list[int] = [256]  # CLEAR
    nbits_list: list[int] = [9]

    nbits = 9
    table_len = 258  # 0-255 + CLEAR + EOD

    for i, b in enumerate(payload):
        codes.append(b)
        nbits_list.append(nbits)
        # First literal after CLEAR does not add a table entry.
        if i > 0:
            table_len += 1
            if table_len == 511:
                nbits = 10
            elif table_len == 1023:
                nbits = 11
            elif table_len == 2047:
                nbits = 12

    codes.append(257)  # EOD
    nbits_list.append(nbits)
    return pack_codes(codes, nbits_list)


def python_decode(data: bytes) -> bytes:
    """Always use the pure-Python decoder regardless of _HAS_RUST."""
    from io import BytesIO

    return b"".join(LZWDecoder(BytesIO(data)).run())


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_HAS_RUST: bool = False
try:
    from pdfminer_core import lzw_decode as _lzw_decode_rust  # noqa: F401

    _HAS_RUST = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestLzwDecode:
    def test_empty_input(self) -> None:
        """Empty byte string decodes to empty bytes."""
        assert lzwdecode(b"") == b""

    def test_clear_then_eod(self) -> None:
        """Stream with just CLEAR + EOD decodes to empty bytes."""
        data = encode_literals(b"")
        assert lzwdecode(data) == b""

    def test_single_byte(self) -> None:
        """Single literal byte round-trips correctly."""
        data = encode_literals(b"A")
        assert lzwdecode(data) == b"A"

    def test_hello(self) -> None:
        """ASCII string round-trips."""
        payload = b"Hello, world!"
        assert lzwdecode(encode_literals(payload)) == payload

    def test_all_byte_values(self) -> None:
        """All 256 byte values exercise the 9→10 bit-width transition."""
        payload = bytes(range(256))
        assert lzwdecode(encode_literals(payload)) == payload

    def test_null_bytes(self) -> None:
        payload = b"\x00" * 32
        assert lzwdecode(encode_literals(payload)) == payload

    def test_high_bytes(self) -> None:
        payload = bytes(range(128, 256))
        assert lzwdecode(encode_literals(payload)) == payload

    @pytest.mark.parametrize(
        "known_bytes, expected",
        [
            # CLEAR(9) + 'A'(9) + EOD(9) → b"A"
            (bytes([0x80, 0x10, 0x60, 0x20]), b"A"),
        ],
    )
    def test_known_vectors(self, known_bytes: bytes, expected: bytes) -> None:
        assert lzwdecode(known_bytes) == expected


# ---------------------------------------------------------------------------
# Rust-vs-Python parity
# ---------------------------------------------------------------------------


class TestRustPythonParity:
    """When Rust is available, the Rust and Python paths must agree byte-for-byte."""

    @pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")
    def test_parity_empty(self) -> None:
        assert python_decode(b"") == lzwdecode(b"")

    @pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")
    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"A",
            b"Hello, world!",
            bytes(range(256)),
            b"\x00" * 100,
            b"\xff" * 100,
        ],
    )
    def test_parity_round_trip(self, payload: bytes) -> None:
        encoded = encode_literals(payload)
        assert python_decode(encoded) == lzwdecode(encoded)

    @pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")
    def test_wrapper_uses_rust(self) -> None:
        """lzwdecode() delegates to the Rust path when _HAS_RUST is True."""
        from pdfminer import lzw as _lzw_mod

        assert _lzw_mod._HAS_RUST is True


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_truncated_input(self) -> None:
        """Truncated streams should not raise – just return partial output."""
        data = encode_literals(b"Hello")
        # Drop the last two bytes
        lzwdecode(data[:-2])  # should not raise

    def test_corrupt_data_silently_truncated(self) -> None:
        """Corrupt data (out-of-range code) should be silently truncated."""
        # Craft a stream with CLEAR + a valid literal + an invalid code.
        # Use 9-bit code 259 (> initial table size of 258) after a single literal.
        codes = [256, 65, 259, 257]  # CLEAR + 'A' + invalid(259) + EOD
        nbits_list = [9, 9, 9, 9]
        data = pack_codes(codes, nbits_list)
        # Should return b"A" and stop, not raise.
        result = lzwdecode(data)
        assert result == b"A"
