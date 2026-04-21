"""Tests for Rust ASCII85 and ASCIIHex decoders."""

import base64

import pytest

from pdfminer.ascii85 import ascii85decode, asciihexdecode


# ── ASCII85 tests ────────────────────────────────────────────────────────────


def test_ascii85_basic() -> None:
    # "9jqo^" encodes "Man " in ASCII85
    assert ascii85decode(b"9jqo^~>") == b"Man "


def test_ascii85_with_delimiters() -> None:
    assert ascii85decode(b"<~9jqo^~>") == b"Man "


def test_ascii85_z_shorthand() -> None:
    # 'z' encodes 4 zero bytes
    assert ascii85decode(b"z~>") == b"\x00\x00\x00\x00"


def test_ascii85_partial_group_1_byte() -> None:
    # "!!" (2 chars) → 1 byte 0x00
    assert ascii85decode(b"!!~>") == b"\x00"


def test_ascii85_partial_group_2_bytes() -> None:
    # "!!*" (3 chars) → 2 bytes
    assert ascii85decode(b"!!*~>") == b"\x00\x01"


def test_ascii85_partial_group_3_bytes() -> None:
    assert ascii85decode(b"!!*-~>") == b"\x00\x01\x02"


def test_ascii85_hello_world() -> None:
    assert ascii85decode(b"87cURD_*#4DfTZ)+T~>") == b"Hello, World!"


def test_ascii85_no_end_marker() -> None:
    # Works without ~>
    assert ascii85decode(b"9jqo^") == b"Man "


def test_ascii85_returns_bytes() -> None:
    result = ascii85decode(b"9jqo^~>")
    assert isinstance(result, bytes)


# ── ASCIIHex tests ───────────────────────────────────────────────────────────


def test_asciihex_basic() -> None:
    assert asciihexdecode(b"48656c6c6f") == b"Hello"


def test_asciihex_with_spaces() -> None:
    assert asciihexdecode(b"48 65 6c 6c 6f") == b"Hello"


def test_asciihex_with_terminator() -> None:
    assert asciihexdecode(b"48656c6c6f>") == b"Hello"


def test_asciihex_uppercase() -> None:
    assert asciihexdecode(b"48656C6C6F") == b"Hello"


def test_asciihex_stop_at_terminator() -> None:
    # Data after '>' is ignored
    assert asciihexdecode(b"4865>6c6c6f") == b"\x48\x65"


def test_asciihex_empty() -> None:
    assert asciihexdecode(b"") == b""


def test_asciihex_only_terminator() -> None:
    assert asciihexdecode(b">") == b""


def test_asciihex_returns_bytes() -> None:
    result = asciihexdecode(b"48656c6c6f")
    assert isinstance(result, bytes)


# ── Round-trip tests using Python reference encoders ────────────────────────


@pytest.mark.parametrize(
    "plaintext",
    [
        b"",
        b"A",
        b"AB",
        b"ABC",
        b"ABCD",
        b"Hello, World!",
        b"\x00\x01\x02\x03",
        bytes(range(256)),
    ],
)
def test_ascii85_round_trip(plaintext: bytes) -> None:
    """Decode output of Python's a85encode must match the original."""
    encoded = base64.a85encode(plaintext)
    assert ascii85decode(encoded + b"~>") == plaintext


@pytest.mark.parametrize(
    "plaintext",
    [
        b"",
        b"A",
        b"Hello",
        b"\x00\xff",
        bytes(range(256)),
    ],
)
def test_asciihex_round_trip(plaintext: bytes) -> None:
    """Encode as hex then decode must return the original."""
    encoded = plaintext.hex().encode()
    assert asciihexdecode(encoded) == plaintext
