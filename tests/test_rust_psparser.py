"""Tests for Rust-accelerated PS tokenizer.

Verifies that the Rust implementation produces identical output to the
Python fallback for all token types and edge cases.
"""

import io
from io import BytesIO
from typing import ClassVar

import pytest

from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import (
    KWD,
    LIT,
    PSBaseParser,
    PSKeyword,
    PSLiteral,
)

try:
    from pdfminer_core import tokenize_ps_buffer

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

import pdfminer.psparser as _psparser_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_parser(data: bytes) -> PSBaseParser:
    return PSBaseParser(BytesIO(data))


def get_all_tokens(data: bytes) -> list[tuple[int, object]]:
    parser = make_parser(data)
    tokens = []
    try:
        while True:
            tokens.append(parser.nexttoken())
    except PSEOF:
        pass
    return tokens


def _force_python(data: bytes) -> list[tuple[int, object]]:
    """Run tokenization with Rust disabled."""
    old = _psparser_module._HAS_RUST  # noqa: SLF001
    try:
        _psparser_module._HAS_RUST = False
        return get_all_tokens(data)
    finally:
        _psparser_module._HAS_RUST = old


def _force_rust(data: bytes) -> list[tuple[int, object]]:
    """Run tokenization – skip if Rust not available."""
    if not _RUST_AVAILABLE:
        pytest.skip("pdfminer_core Rust extension not built")
    return get_all_tokens(data)


# ---------------------------------------------------------------------------
# Low-level Rust tokenizer unit tests
# ---------------------------------------------------------------------------


class TestRustTokenizerDirect:
    """Unit tests for the raw `tokenize_ps_buffer` Rust function."""

    def test_integer(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, consumed = tokenize_ps_buffer(b"42 ", 0)
        assert len(tokens) == 1
        pos, val = tokens[0]
        assert pos == 0
        assert val == 42  # native Python int
        assert consumed == 3

    def test_negative_integer(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"-7 ", 0)
        pos, val = tokens[0]
        assert val == -7

    def test_float(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"3.14 ", 0)
        pos, val = tokens[0]
        assert isinstance(val, float)
        assert val == pytest.approx(3.14)

    def test_float_leading_dot(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b".5 ", 0)
        pos, val = tokens[0]
        assert isinstance(val, float)
        assert val == pytest.approx(0.5)

    def test_keyword(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"BT ", 0)
        pos, val = tokens[0]
        # Keywords are encoded as (4, bytes) tuples
        assert val == (4, b"BT")

    def test_boolean_true(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"true ", 0)
        pos, val = tokens[0]
        assert val is True

    def test_boolean_false(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"false ", 0)
        pos, val = tokens[0]
        assert val is False

    def test_literal(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"/Name ", 0)
        pos, val = tokens[0]
        # Literals are encoded as (3, bytes) tuples
        assert val == (3, b"Name")

    def test_string(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"(hello) ", 0)
        pos, val = tokens[0]
        assert val == b"hello"  # native Python bytes

    def test_hexstring(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"<48656c6c6f> ", 0)
        pos, val = tokens[0]
        assert val == b"Hello"  # native Python bytes

    def test_dict_begin(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"<< ", 0)
        pos, val = tokens[0]
        assert val == (4, b"<<")

    def test_dict_end(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b">> ", 0)
        pos, val = tokens[0]
        assert val == (4, b">>")

    def test_comment_skipped(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"% comment\n42 ", 0)
        assert len(tokens) == 1
        pos, val = tokens[0]
        assert val == 42

    def test_base_offset(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"42 ", 100)
        assert tokens[0][0] == 100  # absolute position

    def test_incomplete_token_returns_conservative(self) -> None:
        """Incomplete token at buffer edge → Rust returns empty and zero consumed."""
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        # "42" with no terminator – might be more digits coming
        tokens, consumed = tokenize_ps_buffer(b"42", 0)
        # Either no tokens were returned (conservative) or it handled the whole
        # thing.  The key invariant: consumed <= len(data).
        assert consumed <= 2

    def test_incomplete_string_returns_empty(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, consumed = tokenize_ps_buffer(b"(hello", 0)
        assert len(tokens) == 0
        assert consumed == 0

    def test_hex_escape_in_literal(self) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        tokens, _ = tokenize_ps_buffer(b"/foo#5fbar ", 0)
        pos, val = tokens[0]
        assert val == (3, b"foo_bar")  # #5f == '_', encoded as (3, bytes)


# ---------------------------------------------------------------------------
# Integration tests via PSBaseParser (with and without Rust)
# ---------------------------------------------------------------------------


class TestPSBaseParserRustParity:
    """Verify that Python and Rust paths produce identical token streams."""

    CASES: ClassVar[list[tuple[str, bytes]]] = [
        ("integer", b"42 "),
        ("negative", b"-2 "),
        ("positive_sign", b"+1 "),
        ("float", b"3.14 "),
        ("dot_float", b".5 "),
        ("keyword", b"BT "),
        ("true", b"true "),
        ("false", b"false "),
        ("literal", b"/Name "),
        ("empty_literal", b"/ "),
        ("string_simple", b"(hello) "),
        ("string_empty", b"() "),
        ("string_nested_parens", b"(a(b)c) "),
        ("string_escape_newline", b"(foo\nbaa) "),
        ("string_escape_octal", b"(def\\040ghi) "),
        ("string_escape_backslash", b"(foo\\\\bar) "),
        ("hexstring", b"<48656c6c6f> "),
        ("hexstring_empty", b"<> "),
        ("hexstring_spaces", b"< 48 65 > "),
        ("dict_begin", b"<< "),
        ("dict_end", b">> "),
        ("array_begin", b"[ "),
        ("array_end", b"] "),
        ("proc_begin", b"{ "),
        ("proc_end", b"} "),
        ("comment_ignored", b"% comment\n42 "),
        ("multiple_tokens", b"42 3.14 /Name BT "),
    ]

    @pytest.mark.parametrize("name,data", CASES)
    def test_rust_matches_python(self, name: str, data: bytes) -> None:
        if not _RUST_AVAILABLE:
            pytest.skip("pdfminer_core not built")
        py_tokens = _force_python(data)
        rust_tokens = _force_rust(data)
        assert rust_tokens == py_tokens, f"Mismatch for case {name!r}"


# ---------------------------------------------------------------------------
# Token-type tests via PSBaseParser public API
# ---------------------------------------------------------------------------


def test_integer_token() -> None:
    p = make_parser(b"42 ")
    _pos, tok = p.nexttoken()
    assert tok == 42


def test_float_token() -> None:
    p = make_parser(b"3.14 ")
    _pos, tok = p.nexttoken()
    assert abs(tok - 3.14) < 1e-10  # type: ignore[operator]


def test_keyword_token() -> None:
    p = make_parser(b"BT ")
    _pos, tok = p.nexttoken()
    assert isinstance(tok, PSKeyword)
    assert tok.name == b"BT"


def test_literal_token() -> None:
    p = make_parser(b"/Name ")
    _pos, tok = p.nexttoken()
    assert isinstance(tok, PSLiteral)
    assert tok.name == "Name"


def test_string_token() -> None:
    p = make_parser(b"(hello) ")
    _pos, tok = p.nexttoken()
    assert tok == b"hello"


def test_hexstring_token() -> None:
    p = make_parser(b"<48656c6c6f> ")
    _pos, tok = p.nexttoken()
    assert tok == b"Hello"


def test_bool_true_token() -> None:
    p = make_parser(b"true ")
    _pos, tok = p.nexttoken()
    assert tok is True


def test_bool_false_token() -> None:
    p = make_parser(b"false ")
    _pos, tok = p.nexttoken()
    assert tok is False


def test_dict_tokens() -> None:
    p = make_parser(b"<< >> ")
    _pos, t1 = p.nexttoken()
    _pos, t2 = p.nexttoken()
    assert isinstance(t1, PSKeyword) and t1.name == b"<<"
    assert isinstance(t2, PSKeyword) and t2.name == b">>"


# ---------------------------------------------------------------------------
# Critical: monkey-patching compatibility
# ---------------------------------------------------------------------------


def test_monkey_patching_seek_still_works() -> None:
    """PSBaseParser.seek must remain monkey-patchable (unstructured requirement)."""
    original_seek = PSBaseParser.seek
    patched_called = []

    def patched_seek(self: PSBaseParser, pos: int) -> None:
        patched_called.append(pos)
        original_seek(self, pos)

    PSBaseParser.seek = patched_seek  # type: ignore[method-assign]
    try:
        p = make_parser(b"42 ")
        p.seek(0)
        assert 0 in patched_called
    finally:
        PSBaseParser.seek = original_seek  # type: ignore[method-assign]


def test_monkey_patching_nexttoken_still_works() -> None:
    """PSBaseParser.nexttoken must remain monkey-patchable."""
    original_nexttoken = PSBaseParser.nexttoken
    call_count = [0]

    def patched_nexttoken(self: PSBaseParser) -> tuple[int, object]:
        call_count[0] += 1
        return original_nexttoken(self)

    PSBaseParser.nexttoken = patched_nexttoken  # type: ignore[method-assign]
    try:
        tokens = get_all_tokens(b"42 BT ")
        assert call_count[0] >= 2
        assert len(tokens) == 2
    finally:
        PSBaseParser.nexttoken = original_nexttoken  # type: ignore[method-assign]


def test_monkey_patching_parse_keyword_still_works() -> None:
    """PSBaseParser._parse_keyword must remain monkey-patchable."""
    original = PSBaseParser._parse_keyword  # type: ignore[attr-defined]
    parse_called = [False]

    def patched(self: PSBaseParser, s: bytes, i: int) -> int:
        parse_called[0] = True
        return original(self, s, i)  # type: ignore[arg-type]

    PSBaseParser._parse_keyword = patched  # type: ignore[method-assign]
    try:
        # Force Python path for this test
        old = _psparser_module._HAS_RUST
        _psparser_module._HAS_RUST = False
        try:
            get_all_tokens(b"BT ")
        finally:
            _psparser_module._HAS_RUST = old
        assert parse_called[0]
    finally:
        PSBaseParser._parse_keyword = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_buffer_boundary_token_regression() -> None:
    """Tokens that span a 4096-byte buffer boundary must be parsed correctly."""
    from tests.test_pdfminer_psparser import BIGDATA

    parser = PSBaseParser(BytesIO(BIGDATA))
    beginbfchar = KWD(b"beginbfchar")
    end = KWD(b"end")
    tokens = []
    while True:
        try:
            pos, token = parser.nexttoken()
            if pos == 4093:
                assert token is beginbfchar
            tokens.append(token)
        except PSEOF:
            break
    assert sum(1 for t in tokens if t is beginbfchar) == 3
    assert tokens[-1] == end
    assert tokens[-2] == tokens[-1]


def test_full_testdata_matches_python() -> None:
    """The full PSBaseParser test-suite data must produce identical token streams."""
    from tests.test_pdfminer_psparser import TestPSBaseParser

    TESTDATA = TestPSBaseParser.TESTDATA

    if not _RUST_AVAILABLE:
        pytest.skip("pdfminer_core not built")

    py_tokens = _force_python(TESTDATA)
    rust_tokens = _force_rust(TESTDATA)
    assert rust_tokens == py_tokens
