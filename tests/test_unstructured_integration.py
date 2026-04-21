"""Integration tests simulating unstructured library's usage of pdfminer."""
import io
import pytest


def test_full_pdf_processing_pipeline():
    """Simulate the full pipeline unstructured uses."""
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import (
        LAParams,
        LTChar,
        LTContainer,
        LTImage,
        LTPage,
        LTTextBox,
    )
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    # Simulate CustomPDFPageInterpreter from unstructured
    class CustomInterp(PDFPageInterpreter):
        def render_char(
            self,
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        ):
            result = super().render_char(
                matrix, font, fontsize, scaling, rise, text, textwidth, textdisp, ncs, graphicstate
            )
            # unstructured patches render mode into new characters
            # find chars added to device since last call
            return result

    rsrcmgr = PDFResourceManager()
    laparams = LAParams(line_margin=0.5, word_margin=0.1)
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with open("samples/simple1.pdf", "rb") as fp:
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            layout = device.get_result()
            assert isinstance(layout, LTPage)

            # Walk layout like unstructured does
            for element in layout:
                if isinstance(element, LTContainer):
                    for sub in element:
                        pass  # just verify iteration works
            break
    print("PASS: Full pipeline works")


def test_psparser_imports_match_unstructured():
    """All psparser symbols imported by unstructured must exist."""
    from pdfminer.psparser import (
        END_KEYWORD,
        KWD,
        PSEOF,
        PSBaseParser,
        PSBaseParserToken,
        PSKeyword,
        literal_name,
        log,
    )

    # Verify PSBaseParser has the methods unstructured monkey-patches
    assert hasattr(PSBaseParser, "seek")
    assert hasattr(PSBaseParser, "_parse_keyword")
    assert hasattr(PSBaseParser, "nexttoken")
    assert hasattr(PSBaseParser, "fillbuf")
    assert hasattr(PSBaseParser, "_parse_main")
    assert hasattr(PSBaseParser, "_add_token")
    print("PASS: All psparser imports present")


def test_layout_imports_match_unstructured():
    """All layout symbols imported by unstructured must exist."""
    from pdfminer.layout import (
        LAParams,
        LTChar,
        LTContainer,
        LTFigure,
        LTImage,
        LTItem,
        LTLayoutContainer,
        LTTextBox,
        LTTextLine,
    )

    # Verify LTChar has the attributes unstructured accesses
    assert hasattr(LTChar, "get_text")
    print("PASS: All layout imports present")


def test_pdftypes_imports_match_unstructured():
    """All pdftypes symbols imported by unstructured must exist."""
    from pdfminer.pdftypes import (
        LITERALS_ASCII85_DECODE,
        LITERALS_ASCIIHEX_DECODE,
        LITERALS_FLATE_DECODE,
        PDFObjRef,
        PDFStream,
        resolve1,
    )

    assert isinstance(LITERALS_FLATE_DECODE, (list, tuple, set, frozenset))
    assert isinstance(LITERALS_ASCII85_DECODE, (list, tuple, set, frozenset))
    assert isinstance(LITERALS_ASCIIHEX_DECODE, (list, tuple, set, frozenset))
    print("PASS: All pdftypes imports present")


def test_pdfstream_attributes():
    """PDFStream must have the attributes unstructured accesses."""
    from pdfminer.pdftypes import PDFStream

    # Check the attributes unstructured uses
    assert hasattr(PDFStream, "get_rawdata")
    assert hasattr(PDFStream, "get_data")
    assert hasattr(PDFStream, "get_filters")
    # These are instance attributes set during construction:
    # stream.objid, stream.genno, stream.decipher, stream.data, stream.rawdata, stream.attrs
    print("PASS: PDFStream has required attributes/methods")


def test_version_check_as_unstructured_does():
    """Replicate the version check in unstructured patches/pdfminer.py."""
    import pdfminer

    # unstructured does: if pdfminer.__version__ <= "20240706":
    # Our version must be newer than that
    assert pdfminer.__version__ > "20240706"
    print(f"PASS: pdfminer version {pdfminer.__version__} passes unstructured check")


def test_custom_resource_manager_subclassing():
    """Simulate CustomPDFResourceManager from pdfminer_utils.py."""
    from pdfminer import settings as pdfminer_settings
    from pdfminer.pdffont import PDFFontError
    from pdfminer.pdfinterp import LITERAL_FONT, PDFResourceManager
    from pdfminer.psparser import literal_name

    class CustomPDFResourceManager(PDFResourceManager):
        def get_font(self, objid, spec):
            subtype = literal_name(spec["Subtype"]) if "Subtype" in spec else "Type1"

            if subtype in ("CIDFontType0", "CIDFontType2"):
                if objid and objid in self._cached_fonts:
                    return self._cached_fonts[objid]
                if pdfminer_settings.STRICT and spec.get("Type") is not LITERAL_FONT:
                    raise PDFFontError("Type is not /Font")
                # In production this would create a CustomPDFCIDFont
                # Here we just verify the method can be overridden
                return super().get_font(objid, spec)

            return super().get_font(objid, spec)

    rsrcmgr = CustomPDFResourceManager()
    assert isinstance(rsrcmgr, PDFResourceManager)
    assert isinstance(rsrcmgr, CustomPDFResourceManager)
    print("PASS: CustomPDFResourceManager subclassing works")


def test_custom_interpreter_do_tj():
    """Simulate CustomPDFPageInterpreter.do_TJ patching render mode."""
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTChar
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    class CustomPDFPageInterpreter(PDFPageInterpreter):
        def _patch_current_chars_with_render_mode(self, start: int):
            cur_item = getattr(self.device, "cur_item", None)
            if not cur_item:
                return
            render_mode = self.textstate.render
            for obj in getattr(cur_item, "_objs", ())[start:]:
                if isinstance(obj, LTChar):
                    obj.rendermode = render_mode

        def do_TJ(self, seq):
            start = len(getattr(getattr(self.device, "cur_item", None), "_objs", ()))
            super().do_TJ(seq)
            self._patch_current_chars_with_render_mode(start)

    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = CustomPDFPageInterpreter(rsrcmgr, device)

    with open("samples/simple1.pdf", "rb") as fp:
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            break  # Just test that it runs without errors

    print("PASS: CustomPDFPageInterpreter.do_TJ works")
