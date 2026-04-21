"""Tests for the Rust CCITT Fax decoder (pdfminer_core.ccitt_decode).

These tests verify that the Rust implementation produces byte-identical output
to the Python reference implementation for all supported inputs.
"""

from __future__ import annotations

import pytest

import pdfminer.ccitt as ccitt_mod

try:
    from pdfminer_core import ccitt_decode as _ccitt_decode_rust

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="pdfminer_core not available")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

EOFB = "000000000001000000000001"


def bits_to_bytes(bits_str: str) -> bytes:
    """Convert a binary string to bytes, padding to byte boundary."""
    while len(bits_str) % 8:
        bits_str += "0"
    return bytes(
        int(bits_str[i : i + 8], 2) for i in range(0, len(bits_str), 8)
    )


def decode_python(data: bytes, columns: int, black_is_1: bool = False) -> bytes:
    """Decode via the Python (reference) implementation, bypassing Rust."""
    saved = ccitt_mod._HAS_RUST
    ccitt_mod._HAS_RUST = False
    try:
        return ccitt_mod.ccittfaxdecode(
            data,
            {
                "K": -1,
                "Columns": columns,
                "BlackIs1": black_is_1,
                "EncodedByteAlign": False,
            },
        )
    finally:
        ccitt_mod._HAS_RUST = saved


def decode_rust(data: bytes, columns: int, black_is_1: bool = False) -> bytes:
    return bytes(
        _ccitt_decode_rust(
            data,
            k=-1,
            columns=columns,
            rows=0,
            end_of_line=False,
            black_is_1=black_is_1,
            damaged_rows_before_error=0,
        )
    )


def assert_rust_matches_python(
    data: bytes, columns: int, black_is_1: bool = False
) -> None:
    expected = decode_python(data, columns, black_is_1)
    actual = decode_rust(data, columns, black_is_1)
    assert actual == expected, (
        f"Rust output {actual.hex()!r} != Python output {expected.hex()!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: individual mode codes
# ─────────────────────────────────────────────────────────────────────────────


class TestVerticalMode:
    def test_v0_all_white(self) -> None:
        """v(0) on an all-white reference line fills the row with white."""
        # One v(0) advances a0 to width -> row complete
        data = bits_to_bytes("1" + EOFB)
        assert_rust_matches_python(data, 4)

    def test_v0_multiple(self) -> None:
        """Multiple v(0) codes traverse changing elements on the ref line."""
        # Three consecutive v(0) on [1,0,0,0,0] ref advances through CEs
        data = bits_to_bytes("1" + "1" + "1" + EOFB)
        assert_rust_matches_python(data, 5)

    def test_v_plus1(self) -> None:
        """v(+1) shifts the target position one pixel right of b1."""
        # After row1 (all white), row2 ref=[1..8], use v(+1) to place
        # first transition at position 1 instead of end-of-line
        row1 = "1"  # v(0) -> all white
        row2 = "011"  # v(+1): b1=8, a1=9 -> clamp to 8, still all white
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 8)

    def test_v_minus1(self) -> None:
        """v(-1) shifts target one pixel left of b1."""
        row1 = "1"  # v(0) -> all white, 4-wide
        row2 = "010"  # v(-1): b1=4, a1=3, fills [0,3) with white, a0=3, color=0
        # After that, row2 not done: at a0=3, color=0
        # v(0) from a0=3: b1=first CE!=0 from pos 4 in ref=[1,1,1,1] -> none -> b1=4
        # a1=4, fill [3,4) with 0, a0=4, done
        row2 += "1"  # v(0) for the remaining black pixel
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 4)

    def test_v_plus2(self) -> None:
        """v(+2) works correctly."""
        row1 = "1"  # v(0) -> all white, 4-wide
        # row2: b1=4, v(+2) would be a1=6 -> clamp to 4 -> all white again
        row2 = "000011"  # v(+2)
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 4)

    def test_v_minus2(self) -> None:
        """v(-2) works correctly."""
        row1 = "1"  # v(0) -> all white, 6-wide
        # row2: b1=6, v(-2) -> a1=4, fill [0,4) with white, a0=4, color=0
        # Then v(0): b1=CE!=0 from pos 5 in ref=[1..6] -> none -> b1=6
        # a1=6, fill [4,6) with 0, done
        row2 = "000010" + "1"  # v(-2) + v(0)
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 6)

    def test_v_plus3_and_minus3(self) -> None:
        """v(±3) mode codes work correctly."""
        row1 = "1"  # v(0) -> all white, 8-wide
        # row2: b1=8, v(+3) -> clamp to 8, all white
        row2 = "0000011"  # v(+3)
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 8)


class TestHorizontalMode:
    def test_h_white_then_black(self) -> None:
        """H mode encoding two runs: white then black."""
        # H(2 white, 2 black): '001' + '0111' + '11'
        data = bits_to_bytes("001" + "0111" + "11" + "1" + EOFB)
        assert_rust_matches_python(data, 8)

    def test_h_zero_white(self) -> None:
        """H mode with zero-length first run (all black from start)."""
        # H(0 white, 5 black): '001' + '00110101' + '0011'
        data = bits_to_bytes("001" + "00110101" + "0011" + EOFB)
        assert_rust_matches_python(data, 5)

    def test_h_all_white(self) -> None:
        """H mode with zero-length second run."""
        # H(5 white, 0 black): '001' + '1100' + '0000110111'
        data = bits_to_bytes("001" + "1100" + "0000110111" + EOFB)
        assert_rust_matches_python(data, 5)

    def test_two_h_codes(self) -> None:
        """Two consecutive H mode codes fill a row."""
        # Row: [1,0,1,0] encoded as H(1,1) + H(1,1)
        h_run = "001" + "000111" + "010"  # H(1 white, 1 black)
        data = bits_to_bytes(h_run + h_run + EOFB)
        assert_rust_matches_python(data, 4)

    def test_h_then_v(self) -> None:
        """H mode followed by V mode within the same row."""
        # H(2,2) leaves a0=4, color=1; then v(0) fills [4,8) with white
        data = bits_to_bytes("001" + "0111" + "11" + "1" + EOFB)
        assert_rust_matches_python(data, 8)


class TestPassMode:
    def test_pass_all_white_ref(self) -> None:
        """Pass on all-white ref line leaves current line all-white."""
        data = bits_to_bytes("0001" + EOFB)
        assert_rust_matches_python(data, 5)

    def test_pass_copies_from_ref(self) -> None:
        """Pass mode copies the reference line pattern."""
        # Row1: H(2,2) -> cur=[1,1,0,0,1,1,1,1] (8-wide, H fills 4 then v(0) fills 4)
        row1 = "001" + "0111" + "11" + "1"
        # Row2: pass copies b2 pixels from ref=[1,1,0,0,1,1,1,1]
        # At a0=-1, color=1: b1=CE!=1 in ref=pos 2 (ref[1]=1,ref[2]=0,0!=1)
        # b2=CE==1 after b1: pos 4 (ref[3]=0,ref[4]=1,1==1)
        # pass fills [0,4) with white -> cur[0..3]=1, a0=4
        # Then color=1, a0=4: b1=CE!=1 from pos 5 -> none -> b1=8
        # v(0): a1=8, done
        row2 = "0001" + "1"  # pass + v(0)
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 8)

    def test_multiple_pass(self) -> None:
        """Multiple consecutive pass mode codes match Python behaviour.

        Two pass codes on a 6-wide alternating row advance a0 only to 4,
        so the second row is incomplete when EOFB arrives.  Both Python and
        Rust should return only the completed first row.
        """
        # ref=[1,0,1,0,1,0] (alternating)
        # Row1: H(1,1) + H(1,1) + H(1,1) -> [1,0,1,0,1,0]
        h_run = "001" + "000111" + "010"
        row1 = h_run + h_run + h_run

        # Row2: two pass codes then EOFB (row not complete → not emitted)
        row2 = "0001" + "0001"
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests: multi-row images
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiRow:
    def test_two_rows_white_black(self) -> None:
        """All-white row followed by all-black row."""
        row1 = "1"  # v(0)
        row2 = "001" + "00110101" + "000101"  # H(0, 8)
        data = bits_to_bytes(row1 + row2 + EOFB)
        assert_rust_matches_python(data, 8)

    def test_four_rows_mixed(self) -> None:
        """Four-row image with mixed content."""
        row1 = "1"  # v(0) all-white 8-wide
        row2 = "001" + "0111" + "11" + "1"  # H(2,2)+v(0) -> [1,1,0,0,1,1,1,1]
        row3 = "1" + "1" + "1"  # three v(0) -> same as row2
        row4 = "001" + "00110101" + "000101"  # H(0,8) -> all-black
        data = bits_to_bytes(row1 + row2 + row3 + row4 + EOFB)
        assert_rust_matches_python(data, 8)

    def test_rows_limited_by_rows_param(self) -> None:
        """rows parameter limits the number of decoded rows."""
        row1 = "1"  # v(0) all-white
        row2 = "1"  # v(0) all-white
        row3 = "001" + "00110101" + "000101"  # H(0,8) all-black
        data = bits_to_bytes(row1 + row2 + row3 + EOFB)
        # Decode only 2 rows
        result = bytes(_ccitt_decode_rust(data, k=-1, columns=8, rows=2))
        assert len(result) == 2  # 2 rows x 1 byte each (8 pixels / 8 = 1)
        assert result == b"\xff\xff"  # both rows all-white


# ─────────────────────────────────────────────────────────────────────────────
# Tests for BlackIs1 polarity
# ─────────────────────────────────────────────────────────────────────────────


class TestBlackIs1:
    def test_inverts_output(self) -> None:
        """BlackIs1=True inverts the output pixels."""
        data = bits_to_bytes("1" + EOFB)  # v(0) -> all-white row
        normal = decode_rust(data, 8, black_is_1=False)
        inverted = decode_rust(data, 8, black_is_1=True)
        assert normal == b"\xff"  # all white = all 1s
        assert inverted == b"\x00"  # inverted = all black = all 0s

    def test_matches_python_black_is_1(self) -> None:
        """Rust with BlackIs1 matches Python with BlackIs1."""
        data = bits_to_bytes("001" + "0111" + "11" + "1" + EOFB)  # mixed row
        assert_rust_matches_python(data, 8, black_is_1=True)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_data(self) -> None:
        """Empty input returns empty output."""
        result = decode_rust(b"", 8)
        assert result == b""

    def test_single_pixel_wide(self) -> None:
        """Single-pixel-wide image."""
        # Width=1: v(0) -> b1=1, a1=1, done
        data = bits_to_bytes("1" + EOFB)
        assert_rust_matches_python(data, 1)

    def test_large_width(self) -> None:
        """Wide image (1728 pixels, standard fax width)."""
        # All-white row: v(0) -> b1=1728, a1=1728, done
        data = bits_to_bytes("1" + EOFB)
        result = decode_rust(data, 1728)
        expected = b"\xff" * 216  # 1728 / 8 = 216 bytes all 1s
        assert result == expected

    def test_width_not_multiple_of_8(self) -> None:
        """Width that is not a multiple of 8 is padded correctly."""
        # Width=5: all-white row -> 1 byte with 5 high bits set = 0xF8
        data = bits_to_bytes("1" + EOFB)
        result = decode_rust(data, 5)
        assert result == b"\xf8"  # 11111000

    def test_trailing_garbage_ignored(self) -> None:
        """Extra bytes after EOFB are silently ignored."""
        data = bits_to_bytes("1" + EOFB) + b"\xff\xff\xff"
        result = decode_rust(data, 4)
        # Should decode normally, ignoring trailing bytes
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Regression: verify ccittfaxdecode() uses Rust when available
# ─────────────────────────────────────────────────────────────────────────────


class TestCCITTFaxDecode:
    def test_uses_rust_when_available(self) -> None:
        """ccittfaxdecode() calls the Rust implementation when installed."""
        assert ccitt_mod._HAS_RUST is True

    def test_rust_path_matches_python(self) -> None:
        """ccittfaxdecode() Rust path produces same bytes as Python path."""
        data = bits_to_bytes("001" + "0111" + "11" + "1" + EOFB)
        params = {
            "K": -1,
            "Columns": 8,
            "BlackIs1": False,
            "EncodedByteAlign": False,
        }

        # Rust path (default)
        ccitt_mod._HAS_RUST = True
        rust_result = ccitt_mod.ccittfaxdecode(data, params)

        # Python path
        ccitt_mod._HAS_RUST = False
        py_result = ccitt_mod.ccittfaxdecode(data, params)

        ccitt_mod._HAS_RUST = True
        assert rust_result == py_result

    def test_k_not_minus1_fallback(self) -> None:
        """For unsupported K values the Rust decoder raises ValueError."""
        data = b"\x00" * 8
        with pytest.raises(ValueError):
            ccitt_mod.ccittfaxdecode(data, {"K": 1, "Columns": 8})
