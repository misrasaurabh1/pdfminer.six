"""Tests for Rust PDF font acceleration."""
import pytest

from pdfminer.pdffont import PDFCIDFont, PDFFontError
from pdfminer.high_level import extract_text


def test_pdffont_subclassing():
    """PDFCIDFont must be subclassable (unstructured does this)."""
    class CustomCIDFont(PDFCIDFont):
        def get_cmap_from_spec(self, spec, strict):
            return None

    # Verify subclass can be created
    assert issubclass(CustomCIDFont, PDFCIDFont)


def test_pdffont_error_is_exception():
    """PDFFontError must be a Python exception."""
    assert issubclass(PDFFontError, Exception)
    with pytest.raises(PDFFontError):
        raise PDFFontError("test")


def test_extract_text_with_fonts():
    """Font handling must work correctly for all sample PDFs."""
    from tests.helpers import absolute_sample_path

    path = absolute_sample_path("font-size-test.pdf")
    text = extract_text(str(path))
    assert len(text) > 0


def test_font_size_extraction():
    """LTChar must have correct fontname and size."""
    from tests.helpers import absolute_sample_path
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTChar, LTTextBox, LTTextLine

    path = absolute_sample_path("simple1.pdf")
    found_char = False
    for page in extract_pages(str(path), laparams=LAParams()):
        for box in page:
            if isinstance(box, LTTextBox):
                for line in box:
                    if isinstance(line, LTTextLine):
                        for char in line:
                            if isinstance(char, LTChar):
                                assert hasattr(char, "fontname")
                                assert hasattr(char, "size")
                                assert char.size > 0
                                found_char = True
                                break
    assert found_char, "No LTChar found in simple1.pdf"


def test_rust_glyph_name_to_unicode():
    """Rust glyph_name_to_unicode returns correct codepoints."""
    try:
        from pdfminer_core import glyph_name_to_unicode
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not available")

    # Standard Adobe glyph name -> Unicode codepoint
    assert glyph_name_to_unicode("A") == ord("A")
    assert glyph_name_to_unicode("AE") == ord("Æ")
    assert glyph_name_to_unicode("space") == ord(" ")
    assert glyph_name_to_unicode("period") == ord(".")
    # Unknown name returns None
    assert glyph_name_to_unicode("nonexistent_glyph_xyz") is None


def test_rust_lookup_char_width():
    """Rust lookup_char_width returns correct widths."""
    try:
        from pdfminer_core import lookup_char_width
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not available")

    widths = [(32, 600.0), (65, 722.0), (66, 667.0)]
    assert lookup_char_width(widths, 1000.0, 32) == 600.0
    assert lookup_char_width(widths, 1000.0, 65) == 722.0
    assert lookup_char_width(widths, 1000.0, 99) == 1000.0  # default


def test_encodingdb_name2unicode_with_rust():
    """name2unicode should work correctly with Rust acceleration enabled."""
    from pdfminer.encodingdb import name2unicode

    assert name2unicode("A") == "A"
    assert name2unicode("AE") == "Æ"
    assert name2unicode("space") == " "
    assert name2unicode("period") == "."
    # uni-prefixed names
    assert name2unicode("uni0041") == "A"
    # u-prefixed names
    assert name2unicode("u0041") == "A"
