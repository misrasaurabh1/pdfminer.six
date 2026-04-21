"""Tests that verify the package installs and works correctly."""
import pytest


def test_import_pdfminer():
    """pdfminer must be importable."""
    import pdfminer

    assert hasattr(pdfminer, "__version__")


def test_import_all_public_modules():
    """All public pdfminer modules must be importable."""
    from pdfminer.ascii85 import ascii85decode, asciihexdecode
    from pdfminer.cmapdb import CMap, CMapDB
    from pdfminer.converter import (
        HTMLConverter,
        PDFPageAggregator,
        TextConverter,
        XMLConverter,
    )
    from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
    from pdfminer.layout import (
        LAParams,
        LTAnno,
        LTChar,
        LTContainer,
        LTCurve,
        LTFigure,
        LTImage,
        LTItem,
        LTLayoutContainer,
        LTLine,
        LTPage,
        LTRect,
        LTTextBox,
        LTTextBoxHorizontal,
        LTTextBoxVertical,
        LTTextLine,
    )
    from pdfminer.pdffont import PDFCIDFont, PDFFontError
    from pdfminer.pdfinterp import LITERAL_FONT, PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdftypes import (
        LITERALS_ASCII85_DECODE,
        LITERALS_ASCIIHEX_DECODE,
        LITERALS_FLATE_DECODE,
        PDFObjRef,
        PDFStream,
        resolve1,
    )
    from pdfminer.psexceptions import PSSyntaxError
    from pdfminer.psparser import KWD, LIT, PSEOF, PSBaseParser, PSKeyword, PSLiteral
    from pdfminer.utils import open_filename
    from pdfminer import settings as pdfminer_settings

    assert hasattr(pdfminer_settings, "STRICT")


def test_version_string_format():
    """pdfminer.__version__ must be comparable (unstructured version-checks it)."""
    import pdfminer

    version = pdfminer.__version__
    assert isinstance(version, str)
    # Must be comparable with >= "20251230"
    assert version >= "20251230", f"Version {version!r} too old"


def test_unstructured_monkey_patching_compat():
    """All monkey-patching that unstructured does must still work."""
    from pdfminer.psparser import PSBaseParser

    # Test seek patching
    original_seek = PSBaseParser.seek

    def patched_seek(self, pos):
        self.eof = False
        return original_seek(self, pos)

    PSBaseParser.seek = patched_seek
    PSBaseParser.seek = original_seek  # restore

    # Test nexttoken patching
    original_nexttoken = PSBaseParser.nexttoken
    PSBaseParser.nexttoken = lambda self: original_nexttoken(self)
    PSBaseParser.nexttoken = original_nexttoken
    print("PASS: PSBaseParser monkey-patching works")


def test_unstructured_subclassing_compat():
    """All subclassing that unstructured does must still work."""
    from pdfminer.pdffont import PDFCIDFont
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

    class CustomInterp(PDFPageInterpreter):
        def do_TJ(self, seq):
            super().do_TJ(seq)

    class CustomRM(PDFResourceManager):
        def get_font(self, objid, spec):
            return super().get_font(objid, spec)

    class CustomCIDFont(PDFCIDFont):
        def get_cmap_from_spec(self, spec, strict):
            try:
                return super().get_cmap_from_spec(spec, strict)
            except Exception:
                return None

    assert issubclass(CustomInterp, PDFPageInterpreter)
    assert issubclass(CustomRM, PDFResourceManager)
    assert issubclass(CustomCIDFont, PDFCIDFont)
    print("PASS: All subclassing works")


def test_wrapt_patching_compat():
    """wrapt.patch_function_wrapper must work (unstructured uses this)."""
    try:
        import wrapt
        from pdfminer.pdfinterp import PDFPageInterpreter

        @wrapt.patch_function_wrapper("pdfminer.pdfinterp", "PDFPageInterpreter.init_resources")
        def patched(wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        print("PASS: wrapt patching works")
    except ImportError:
        pytest.skip("wrapt not installed")


def test_extract_text_simple():
    """extract_text must work on simple PDFs."""
    from pdfminer.high_level import extract_text

    text = extract_text("samples/simple1.pdf")
    assert len(text) > 0


def test_extract_pages_ltpage():
    """extract_pages must return LTPage objects."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTPage

    pages = list(extract_pages("samples/simple1.pdf", laparams=LAParams()))
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, LTPage)


def test_ltchar_attributes():
    """LTChar must have all attributes unstructured accesses."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTChar, LTTextBox, LTTextLine

    for page in extract_pages("samples/simple1.pdf", laparams=LAParams()):
        for box in page:
            if isinstance(box, LTTextBox):
                for line in box:
                    if isinstance(line, LTTextLine):
                        for char in line:
                            if isinstance(char, LTChar):
                                assert hasattr(char, "x0")
                                assert hasattr(char, "y0")
                                assert hasattr(char, "x1")
                                assert hasattr(char, "y1")
                                assert hasattr(char, "matrix")
                                assert callable(char.get_text)
                                char.rendermode = 0  # Must be settable
                                assert char.rendermode == 0
                                return
    pytest.fail("No LTChar found in simple1.pdf")
