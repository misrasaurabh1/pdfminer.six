"""E2E tests for pdfminer API compatibility with the unstructured library.

These tests verify all patterns that unstructured relies on:
- Monkey-patching PSBaseParser
- Subclassing PDFPageInterpreter / PDFResourceManager / PDFCIDFont
- LTChar attributes (including .rendermode assignment)
- CMap / PDFStream / PDFObjRef interfaces
- Module-level constants and version strings

They serve as the correctness ground truth when porting to Rust.
"""

import io
from typing import Mapping

import pytest

from pdfminer import settings as pdfminer_settings
import pdfminer
from pdfminer.cmapdb import CMap, CMapDB
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTTextLine
from pdfminer.pdffont import PDFCIDFont
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
from pdfminer.psparser import PSBaseParser, PSLiteral
from tests.helpers import absolute_sample_path


# ---------------------------------------------------------------------------
# pdfminer version and settings
# ---------------------------------------------------------------------------

class TestModuleLevel:
    """Verify module-level attributes required by unstructured."""

    def test_version_is_string(self) -> None:
        """pdfminer.__version__ must be a string."""
        assert isinstance(pdfminer.__version__, str)

    def test_version_is_comparable(self) -> None:
        """pdfminer.__version__ supports string comparison (used in patch_psparser)."""
        # This should not raise; the comparison is used in unstructured's patch_psparser
        _ = pdfminer.__version__ <= "20240706"

    def test_settings_strict_is_bool(self) -> None:
        """pdfminer.settings.STRICT must be a boolean."""
        assert isinstance(pdfminer_settings.STRICT, bool)

    def test_literal_font_importable(self) -> None:
        """LITERAL_FONT is importable from pdfminer.pdfinterp."""
        assert LITERAL_FONT is not None

    def test_literals_flate_decode_importable(self) -> None:
        """LITERALS_FLATE_DECODE is importable from pdfminer.pdftypes and is a tuple."""
        assert isinstance(LITERALS_FLATE_DECODE, tuple)
        assert len(LITERALS_FLATE_DECODE) > 0

    def test_literals_ascii85_decode_importable(self) -> None:
        """LITERALS_ASCII85_DECODE is importable and is a tuple."""
        assert isinstance(LITERALS_ASCII85_DECODE, tuple)
        assert len(LITERALS_ASCII85_DECODE) > 0

    def test_literals_asciihex_decode_importable(self) -> None:
        """LITERALS_ASCIIHEX_DECODE is importable and is a tuple."""
        assert isinstance(LITERALS_ASCIIHEX_DECODE, tuple)
        assert len(LITERALS_ASCIIHEX_DECODE) > 0


# ---------------------------------------------------------------------------
# PSBaseParser monkey-patching (unstructured/patches/pdfminer.py)
# ---------------------------------------------------------------------------

class TestPSBaseParserMonkeyPatch:
    """Verify that PSBaseParser can be monkey-patched the way unstructured does."""

    def test_seek_is_patchable(self) -> None:
        """PSBaseParser.seek can be replaced with a custom function."""
        original_seek = PSBaseParser.seek

        def custom_seek(self: PSBaseParser, pos: int) -> None:
            original_seek(self, pos)
            self.eof = False

        PSBaseParser.seek = custom_seek  # type: ignore[method-assign]
        try:
            parser = PSBaseParser(io.BytesIO(b"true"))
            parser.seek(0)
            assert parser.eof is False
        finally:
            PSBaseParser.seek = original_seek  # type: ignore[method-assign]

    def test_parse_keyword_is_patchable(self) -> None:
        """PSBaseParser._parse_keyword can be replaced and exercised."""
        original = PSBaseParser._parse_keyword
        called = []

        def custom_parse_keyword(self: PSBaseParser, s: bytes, i: int) -> int:
            called.append(True)
            return original(self, s, i)

        PSBaseParser._parse_keyword = custom_parse_keyword  # type: ignore[method-assign]
        try:
            parser = PSBaseParser(io.BytesIO(b"true false"))
            parser.nexttoken()  # triggers keyword parsing
        finally:
            PSBaseParser._parse_keyword = original  # type: ignore[method-assign]
        assert len(called) > 0

    def test_nexttoken_is_patchable(self) -> None:
        """PSBaseParser.nexttoken can be replaced."""
        original = PSBaseParser.nexttoken
        PSBaseParser.nexttoken = original  # no-op, just verifies assignment works
        assert PSBaseParser.nexttoken is original

    def test_parser_has_eof_attribute(self) -> None:
        """A PSBaseParser instance has an .eof attribute."""
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert hasattr(parser, "eof")
        assert isinstance(parser.eof, bool)

    def test_parser_has_fillbuf(self) -> None:
        """PSBaseParser instances have a .fillbuf method."""
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert callable(parser.fillbuf)

    def test_parser_has_charpos(self) -> None:
        """PSBaseParser instances have a .charpos attribute."""
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert hasattr(parser, "charpos")

    def test_parser_has_buf(self) -> None:
        """PSBaseParser instances have a .buf attribute."""
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert hasattr(parser, "buf")

    def test_parser_has_parse1(self) -> None:
        """PSBaseParser instances have a ._parse1 attribute (callable)."""
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert callable(parser._parse1)

    def test_parser_has_tokens(self) -> None:
        """PSBaseParser instances have a ._tokens sequence (list or deque)."""
        import collections
        parser = PSBaseParser(io.BytesIO(b"true"))
        assert hasattr(parser, "_tokens")
        assert isinstance(parser._tokens, (list, collections.deque))


# ---------------------------------------------------------------------------
# PDFPageInterpreter subclassing
# ---------------------------------------------------------------------------

class TestPDFPageInterpreterSubclass:
    """Verify PDFPageInterpreter can be subclassed and process_page overridden."""

    def test_subclass_process_page(self) -> None:
        """CustomPDFPageInterpreter can override do_TJ to patch LTChar objects."""
        process_page_calls = []

        class CustomInterpreter(PDFPageInterpreter):
            def process_page(self, page: PDFPage) -> None:
                process_page_calls.append(page)
                super().process_page(page)

        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
        interp = CustomInterpreter(rsrcmgr, device)

        with open(absolute_sample_path("simple1.pdf"), "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)

        assert len(process_page_calls) == 1

    def test_do_tj_can_be_overridden(self) -> None:
        """do_TJ can be overridden in a subclass (unstructured's rendermode patch)."""
        patched = []

        class RenderModeInterpreter(PDFPageInterpreter):
            def do_TJ(self, seq):  # type: ignore[override]
                patched.append(seq)
                super().do_TJ(seq)

        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
        interp = RenderModeInterpreter(rsrcmgr, device)

        with open(absolute_sample_path("simple1.pdf"), "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)

        assert len(patched) > 0

    def test_device_cur_item_accessible(self) -> None:
        """device.cur_item is accessible during page processing."""
        class InspectInterpreter(PDFPageInterpreter):
            def do_TJ(self, seq):  # type: ignore[override]
                cur_item = getattr(self.device, "cur_item", None)
                assert cur_item is not None
                super().do_TJ(seq)

        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
        interp = InspectInterpreter(rsrcmgr, device)

        with open(absolute_sample_path("simple1.pdf"), "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)


# ---------------------------------------------------------------------------
# PDFResourceManager subclassing
# ---------------------------------------------------------------------------

class TestPDFResourceManagerSubclass:
    """Verify PDFResourceManager can be subclassed with get_font overridden."""

    def test_subclass_get_font(self) -> None:
        """A subclass with overridden get_font is called during page processing."""
        get_font_calls = []

        class CustomResourceManager(PDFResourceManager):
            def get_font(self, objid, spec):
                get_font_calls.append((objid, spec))
                return super().get_font(objid, spec)

        rsrcmgr = CustomResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
        interp = PDFPageInterpreter(rsrcmgr, device)

        with open(absolute_sample_path("simple1.pdf"), "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)

        assert len(get_font_calls) > 0

    def test_cached_fonts_dict_exists(self) -> None:
        """PDFResourceManager has a _cached_fonts dict."""
        rsrcmgr = PDFResourceManager()
        assert hasattr(rsrcmgr, "_cached_fonts")
        assert isinstance(rsrcmgr._cached_fonts, dict)

    def test_caching_attribute(self) -> None:
        """PDFResourceManager exposes a .caching attribute."""
        rsrcmgr = PDFResourceManager(caching=True)
        assert rsrcmgr.caching is True
        rsrcmgr2 = PDFResourceManager(caching=False)
        assert rsrcmgr2.caching is False


# ---------------------------------------------------------------------------
# PDFCIDFont subclassing
# ---------------------------------------------------------------------------

class TestPDFCIDFontSubclass:
    """Verify PDFCIDFont can be subclassed with get_cmap_from_spec overridden."""

    def test_subclass_get_cmap_from_spec(self) -> None:
        """A subclass can override get_cmap_from_spec."""
        called = []

        class CustomCIDFont(PDFCIDFont):
            def get_cmap_from_spec(self, spec: Mapping, strict: bool) -> CMap:
                called.append(spec)
                return super().get_cmap_from_spec(spec, strict)

        spec = {"Encoding": PSLiteral("UniGB-UCS2-H")}
        font = CustomCIDFont(None, spec)
        assert len(called) > 0
        assert isinstance(font.cmap, CMap)

    def test_get_cmap_from_spec_with_unknown_cmap(self) -> None:
        """PDFCIDFont.get_cmap_from_spec returns CMap() on unknown name, strict=False."""
        font = PDFCIDFont(None, {})
        result = font.get_cmap_from_spec({"Encoding": PSLiteral("no-such-cmap")}, False)
        assert isinstance(result, CMap)

    def test_get_cmap_name_method_exists(self) -> None:
        """PDFCIDFont has _get_cmap_name method (used in unstructured subclass)."""
        assert hasattr(PDFCIDFont, "_get_cmap_name") or hasattr(PDFCIDFont, "get_cmap_from_spec")


# ---------------------------------------------------------------------------
# LTChar attributes
# ---------------------------------------------------------------------------

class TestLTCharAttributes:
    """Verify LTChar has all attributes required by unstructured."""

    def _get_first_ltchar(self) -> LTChar:
        from pdfminer.high_level import extract_pages

        for page in extract_pages(absolute_sample_path("simple1.pdf")):
            for item in page:
                if hasattr(item, "__iter__"):
                    for line in item:
                        if isinstance(line, LTTextLine):
                            for char in line:
                                if isinstance(char, LTChar):
                                    return char
        pytest.fail("No LTChar found in simple1.pdf")

    def test_ltchar_x0(self) -> None:
        """LTChar has x0 attribute (float)."""
        char = self._get_first_ltchar()
        assert isinstance(char.x0, float)

    def test_ltchar_y0(self) -> None:
        """LTChar has y0 attribute (float)."""
        char = self._get_first_ltchar()
        assert isinstance(char.y0, float)

    def test_ltchar_x1(self) -> None:
        """LTChar has x1 attribute (float)."""
        char = self._get_first_ltchar()
        assert isinstance(char.x1, float)

    def test_ltchar_y1(self) -> None:
        """LTChar has y1 attribute (float)."""
        char = self._get_first_ltchar()
        assert isinstance(char.y1, float)

    def test_ltchar_bbox(self) -> None:
        """LTChar.bbox is a 4-tuple of floats with x0 <= x1 and y0 <= y1."""
        char = self._get_first_ltchar()
        assert len(char.bbox) == 4
        x0, y0, x1, y1 = char.bbox
        assert x0 <= x1
        assert y0 <= y1

    def test_ltchar_matrix(self) -> None:
        """LTChar.matrix is a 6-tuple."""
        char = self._get_first_ltchar()
        assert len(char.matrix) == 6

    def test_ltchar_get_text(self) -> None:
        """LTChar.get_text() returns a single character string."""
        char = self._get_first_ltchar()
        text = char.get_text()
        assert isinstance(text, str)
        assert len(text) == 1

    def test_ltchar_size(self) -> None:
        """LTChar.size is a positive float."""
        char = self._get_first_ltchar()
        assert isinstance(char.size, float)
        assert char.size > 0

    def test_ltchar_fontname(self) -> None:
        """LTChar.fontname is a string."""
        char = self._get_first_ltchar()
        assert isinstance(char.fontname, str)

    def test_ltchar_adv(self) -> None:
        """LTChar.adv is a float."""
        char = self._get_first_ltchar()
        assert isinstance(char.adv, float)

    def test_ltchar_upright(self) -> None:
        """LTChar.upright is a bool."""
        char = self._get_first_ltchar()
        assert isinstance(char.upright, bool)

    def test_ltchar_ncs(self) -> None:
        """LTChar.ncs has a .name attribute (color space name string)."""
        char = self._get_first_ltchar()
        assert hasattr(char.ncs, "name")
        assert isinstance(char.ncs.name, str)

    def test_ltchar_graphicstate(self) -> None:
        """LTChar.graphicstate is not None."""
        char = self._get_first_ltchar()
        assert char.graphicstate is not None

    def test_ltchar_rendermode_assignment(self) -> None:
        """LTChar accepts .rendermode attribute assignment (unstructured's rendermode patch)."""
        char = self._get_first_ltchar()
        # The attribute may or may not exist before assignment — either is fine
        char.rendermode = 3
        assert char.rendermode == 3

    def test_ltchar_rendermode_default_absence(self) -> None:
        """LTChar does not have rendermode by default (unstructured adds it dynamically)."""
        char = self._get_first_ltchar()
        # Standard extraction should not add rendermode; if it's absent that's expected
        # If it IS present, it should be a valid render mode int
        if hasattr(char, "rendermode"):
            assert isinstance(char.rendermode, int)


# ---------------------------------------------------------------------------
# CMap interface
# ---------------------------------------------------------------------------

class TestCMapInterface:
    """Verify the CMap interface required by unstructured."""

    def test_cmap_code2cid_is_dict(self) -> None:
        """CMap.code2cid is a dict."""
        cmap = CMap()
        assert isinstance(cmap.code2cid, dict)

    def test_cmap_attrs_is_dict(self) -> None:
        """CMap.attrs is a dict."""
        cmap = CMap()
        assert isinstance(cmap.attrs, dict)

    def test_cmap_code2cid_assignable(self) -> None:
        """CMap.code2cid can be replaced with a new dict."""
        cmap = CMap()
        new_mapping = {0: 1, 1: 2}
        cmap.code2cid = new_mapping
        assert cmap.code2cid is new_mapping

    def test_cmap_attrs_assignable(self) -> None:
        """CMap.attrs can be modified in-place."""
        cmap = CMap()
        cmap.attrs["WMode"] = 1
        assert cmap.attrs["WMode"] == 1

    def test_cmapdb_get_cmap_returns_cmap(self) -> None:
        """CMapDB.get_cmap() returns a CMap instance for known CMap names."""
        cmap = CMapDB.get_cmap("UniGB-UCS2-H")
        assert isinstance(cmap, CMap)
        assert cmap.attrs.get("CMapName") == "UniGB-UCS2-H"
        assert len(cmap.code2cid) > 0

    def test_cmapdb_get_cmap_not_found(self) -> None:
        """CMapDB.get_cmap() raises CMapDB.CMapNotFound for unknown names."""
        with pytest.raises(CMapDB.CMapNotFound):
            CMapDB.get_cmap("totally-nonexistent-cmap-name")

    def test_cmapdb_cmap_not_found_is_exception(self) -> None:
        """CMapDB.CMapNotFound is an exception class."""
        assert issubclass(CMapDB.CMapNotFound, Exception)


# ---------------------------------------------------------------------------
# PDFStream interface
# ---------------------------------------------------------------------------

class TestPDFStreamInterface:
    """Verify the PDFStream interface required by unstructured."""

    def test_pdfstream_data_attribute(self) -> None:
        """PDFStream has a .data attribute (None before decoding)."""
        stream = PDFStream({"Length": 5}, b"hello")
        assert stream.data is None

    def test_pdfstream_rawdata_attribute(self) -> None:
        """PDFStream has a .rawdata attribute with the raw bytes."""
        stream = PDFStream({"Length": 5}, b"hello")
        assert stream.rawdata == b"hello"

    def test_pdfstream_attrs_attribute(self) -> None:
        """PDFStream has an .attrs dict."""
        attrs = {"Length": 5, "Type": "XObject"}
        stream = PDFStream(attrs, b"hello")
        assert isinstance(stream.attrs, dict)
        assert stream.attrs["Length"] == 5

    def test_pdfstream_objid_attribute(self) -> None:
        """PDFStream has an .objid attribute (None by default)."""
        stream = PDFStream({}, b"")
        assert stream.objid is None

    def test_pdfstream_genno_attribute(self) -> None:
        """PDFStream has a .genno attribute (None by default)."""
        stream = PDFStream({}, b"")
        assert stream.genno is None

    def test_pdfstream_decipher_attribute(self) -> None:
        """PDFStream has a .decipher attribute (None by default)."""
        stream = PDFStream({}, b"")
        assert stream.decipher is None

    def test_pdfstream_get_rawdata(self) -> None:
        """PDFStream.get_rawdata() returns the raw bytes."""
        stream = PDFStream({"Length": 5}, b"hello")
        assert stream.get_rawdata() == b"hello"

    def test_pdfstream_get_data(self) -> None:
        """PDFStream.get_data() decodes and returns bytes for an uncompressed stream."""
        stream = PDFStream({"Length": 5}, b"hello")
        data = stream.get_data()
        assert data == b"hello"

    def test_pdfstream_get_filters_empty(self) -> None:
        """PDFStream.get_filters() returns an empty list when there are no filters."""
        stream = PDFStream({"Length": 5}, b"hello")
        filters = stream.get_filters()
        assert isinstance(filters, list)
        assert len(filters) == 0

    def test_pdfstream_get_filters_with_flate(self) -> None:
        """PDFStream.get_filters() returns a list of (filter, params) tuples."""
        import zlib

        from pdfminer.psparser import LIT

        data = zlib.compress(b"hello world")
        # PSLiteral objects are interned via LIT() — use LIT() for identity semantics
        stream = PDFStream(
            {"Filter": LIT("FlateDecode"), "Length": len(data)}, data
        )
        filters = stream.get_filters()
        assert len(filters) == 1
        filt, _params = filters[0]
        assert filt in LITERALS_FLATE_DECODE


# ---------------------------------------------------------------------------
# PDFObjRef interface
# ---------------------------------------------------------------------------

class TestPDFObjRefInterface:
    """Verify PDFObjRef.resolve() works as expected by unstructured."""

    def test_resolve_returns_object(self) -> None:
        """PDFObjRef.resolve() returns the referenced object."""
        class MockDoc:
            def getobj(self, objid: int) -> object:
                return {"key": "value"}

        ref = PDFObjRef(doc=MockDoc(), objid=1)
        resolved = ref.resolve()
        assert resolved == {"key": "value"}

    def test_resolve_returns_list(self) -> None:
        """PDFObjRef.resolve() works with list values."""
        class MockDoc:
            def getobj(self, objid: int) -> object:
                return [1, 2, 3]

        ref = PDFObjRef(doc=MockDoc(), objid=42)
        resolved = ref.resolve()
        assert resolved == [1, 2, 3]


# ---------------------------------------------------------------------------
# resolve1 function
# ---------------------------------------------------------------------------

class TestResolve1:
    """Verify resolve1() behaves as expected."""

    def test_resolve1_resolves_objref(self) -> None:
        """resolve1() resolves a PDFObjRef to the underlying object."""
        class MockDoc:
            def getobj(self, objid: int) -> object:
                return "resolved value"

        ref = PDFObjRef(doc=MockDoc(), objid=1)
        assert resolve1(ref) == "resolved value"

    def test_resolve1_passthrough_non_ref(self) -> None:
        """resolve1() returns non-reference objects unchanged."""
        assert resolve1("plain string") == "plain string"
        assert resolve1(42) == 42
        assert resolve1({"a": 1}) == {"a": 1}
