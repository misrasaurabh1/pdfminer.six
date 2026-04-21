"""Tests for Rust PDF interpreter acceleration.

Verifies that:
- LITERAL_FONT and the Python classes remain importable / subclassable.
- wrapt-style patching of PDFPageInterpreter.init_resources works.
- Text extraction from sample PDFs produces correct output.
- The Rust helper functions (when available) produce values identical to the
  pure-Python fallback.
"""

import pytest

from pdfminer.pdfinterp import LITERAL_FONT, PDFPageInterpreter, PDFResourceManager

# ---------------------------------------------------------------------------
# Try to import the Rust extension so tests can be skipped gracefully.
# ---------------------------------------------------------------------------
try:
    import pdfminer_core as _core  # type: ignore[import-not-found]

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Compatibility / import tests  (always run — no Rust required)
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """Critical unstructured imports must work."""
    from pdfminer.pdfinterp import LITERAL_FONT, PDFPageInterpreter, PDFResourceManager

    assert LITERAL_FONT is not None
    assert PDFPageInterpreter is not None
    assert PDFResourceManager is not None


def test_literal_font_value() -> None:
    """LITERAL_FONT must be the PSLiteral for 'Font'."""
    from pdfminer.psparser import LIT

    assert LITERAL_FONT == LIT("Font")


def test_subclassing_interpreter() -> None:
    """PDFPageInterpreter must be subclassable and do_TJ overridable."""

    class CustomInterp(PDFPageInterpreter):
        called = False

        def do_TJ(self, seq: object) -> None:  # type: ignore[override]
            CustomInterp.called = True
            super().do_TJ(seq)  # type: ignore[arg-type]

    assert issubclass(CustomInterp, PDFPageInterpreter)


def test_subclassing_resource_manager() -> None:
    """PDFResourceManager must be subclassable with get_font overridable."""
    from collections.abc import Mapping

    class CustomRM(PDFResourceManager):
        def get_font(self, objid: object, spec: Mapping[str, object]) -> object:
            return super().get_font(objid, spec)

    assert issubclass(CustomRM, PDFResourceManager)


def test_patch_function_wrapper() -> None:
    """wrapt-style patching must work on PDFPageInterpreter.init_resources."""
    original = PDFPageInterpreter.init_resources

    def patched(self: PDFPageInterpreter, res: object) -> None:
        original(self, res)  # type: ignore[arg-type]

    PDFPageInterpreter.init_resources = patched  # type: ignore[method-assign]
    try:
        assert PDFPageInterpreter.init_resources is patched
    finally:
        PDFPageInterpreter.init_resources = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Rust function correctness tests  (skipped when extension not built)
# ---------------------------------------------------------------------------


def test_rust_available_flag() -> None:
    """_HAS_RUST flag in pdfinterp matches actual extension availability."""
    import pdfminer.pdfinterp as _pdfinterp

    assert _pdfinterp._HAS_RUST == _RUST_AVAILABLE  # noqa: SLF001


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_apply_text_advance_identity() -> None:
    """apply_text_advance(identity, tx, ty) == (tx, ty)."""
    result = _core.apply_text_advance((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), 10.0, 20.0)
    assert result == (10.0, 20.0)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_apply_text_advance_scaled() -> None:
    """apply_text_advance with non-identity matrix matches manual Python."""
    matrix = (2.0, 0.0, 0.0, 3.0, 5.0, 7.0)
    tx, ty = 4.0, 6.0
    a, b, c, d, e, f = matrix
    expected_e = tx * a + ty * c + e  # 4*2 + 6*0 + 5 = 13
    expected_f = tx * b + ty * d + f  # 4*0 + 6*3 + 7 = 25
    result = _core.apply_text_advance(matrix, tx, ty)
    assert result == pytest.approx((expected_e, expected_f))


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_mult_matrix_rust_identity() -> None:
    """mult_matrix_rust with identity matrices returns identity."""
    I = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    result = _core.mult_matrix_rust(I, I)
    assert result == pytest.approx(I)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_mult_matrix_rust_matches_python() -> None:
    """mult_matrix_rust produces the same result as pdfminer.utils.mult_matrix."""
    from pdfminer.utils import mult_matrix as py_mult

    m1 = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    m2 = (1.5, 0.5, 0.25, 2.0, 1.0, -1.0)
    py_result = py_mult(m1, m2)
    rs_result = _core.mult_matrix_rust(m1, m2)
    assert rs_result == pytest.approx(py_result)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_text_state_to_matrix_identity_ctm() -> None:
    """text_state_to_matrix with identity CTM returns the text matrix unchanged."""
    tm = (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    result = _core.text_state_to_matrix(tm, ctm)
    assert result == pytest.approx(tm)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_calculate_char_advances_empty() -> None:
    """Empty sequence returns empty list."""
    result = _core.calculate_char_advances(12.0, 100.0, [], True)
    assert result == []


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_calculate_char_advances_numeric_only() -> None:
    """Numeric-only TJ elements produce correct horizontal displacements."""
    # tx = -adj * 0.001 * fontsize * (scaling * 0.01)
    # adj=500, fontsize=10, scaling=100 => tx = -500 * 0.001 * 10 * 1.0 = -5.0
    result = _core.calculate_char_advances(10.0, 100.0, [(None, 500.0)], True)
    assert len(result) == 1
    tx, ty = result[0]
    assert tx == pytest.approx(-5.0)
    assert ty == pytest.approx(0.0)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core Rust extension not built")
def test_calculate_char_advances_vertical() -> None:
    """Vertical text adjustments go in the y direction."""
    result = _core.calculate_char_advances(10.0, 100.0, [(None, 200.0)], False)
    assert len(result) == 1
    tx, ty = result[0]
    assert tx == pytest.approx(0.0)
    assert ty == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Integration: text extraction from sample PDFs
# ---------------------------------------------------------------------------


def test_extract_text_simple1() -> None:
    """simple1.pdf must produce non-empty text."""
    from tests.helpers import absolute_sample_path

    from pdfminer.high_level import extract_text

    path = absolute_sample_path("simple1.pdf")
    text = extract_text(str(path))
    assert len(text) > 0, "Failed to extract text from simple1.pdf"


def test_extract_text_simple2() -> None:
    """simple2.pdf must produce non-empty text."""
    from tests.helpers import absolute_sample_path

    from pdfminer.high_level import extract_text

    path = absolute_sample_path("simple2.pdf")
    text = extract_text(str(path))
    assert len(text) > 0, "Failed to extract text from simple2.pdf"
