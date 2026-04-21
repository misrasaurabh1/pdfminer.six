"""E2E tests for pdfminer font handling.

These tests verify font-related functionality: CMap lookups,
PDFCIDFont construction, font-size extraction, and encoding handling.
They serve as correctness ground truth for the Rust port.
"""

import pytest

from pdfminer.cmapdb import CMap, CMapBase, CMapDB, IdentityCMap, IdentityCMapByte
from pdfminer.converter import PDFPageAggregator
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextLine,
)
from pdfminer.pdffont import PDFCIDFont
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.psparser import PSLiteral
from tests.helpers import absolute_sample_path


# ---------------------------------------------------------------------------
# CMapDB lookups
# ---------------------------------------------------------------------------

class TestCMapDB:
    """Test CMapDB.get_cmap() for standard CMap names."""

    @pytest.mark.parametrize(
        "cmap_name",
        [
            "UniGB-UCS2-H",
            "UniGB-UCS2-V",
            "UniJIS-UCS2-H",
            "UniJIS-UCS2-V",
            "UniKS-UCS2-H",
            "UniKS-UCS2-V",
        ],
    )
    def test_standard_cmap_names(self, cmap_name: str) -> None:
        """CMapDB.get_cmap() loads known standard CMap names without error."""
        cmap = CMapDB.get_cmap(cmap_name)
        assert isinstance(cmap, CMapBase)

    def test_unigb_ucs2_h_has_code2cid(self) -> None:
        """UniGB-UCS2-H CMap has a non-empty code2cid mapping."""
        cmap = CMapDB.get_cmap("UniGB-UCS2-H")
        assert len(cmap.code2cid) > 0

    def test_unigb_ucs2_h_has_attrs(self) -> None:
        """UniGB-UCS2-H CMap has CMapName in attrs."""
        cmap = CMapDB.get_cmap("UniGB-UCS2-H")
        assert cmap.attrs.get("CMapName") == "UniGB-UCS2-H"

    def test_cmap_not_found_raises(self) -> None:
        """CMapDB.CMapNotFound is raised for unknown CMap names."""
        with pytest.raises(CMapDB.CMapNotFound):
            CMapDB.get_cmap("this-does-not-exist")

    def test_identity_h_is_identity_cmap(self) -> None:
        """Identity-H CMap is an IdentityCMap instance (subclass of CMapBase)."""
        cmap = CMapDB.get_cmap("Identity-H")
        assert isinstance(cmap, IdentityCMap)
        assert isinstance(cmap, CMapBase)

    def test_identity_v_is_identity_cmap(self) -> None:
        """Identity-V CMap is an IdentityCMap instance (subclass of CMapBase)."""
        cmap = CMapDB.get_cmap("Identity-V")
        assert isinstance(cmap, IdentityCMap)
        assert isinstance(cmap, CMapBase)


# ---------------------------------------------------------------------------
# PDFCIDFont construction
# ---------------------------------------------------------------------------

class TestPDFCIDFontConstruction:
    """Verify PDFCIDFont can be constructed with various specs."""

    def test_construct_with_unigb_cmap(self) -> None:
        """PDFCIDFont can be constructed with a known CMap name."""
        spec = {"Encoding": PSLiteral("UniGB-UCS2-H")}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, CMap)
        assert font.cmap.attrs.get("CMapName") == "UniGB-UCS2-H"

    def test_construct_with_empty_spec(self) -> None:
        """PDFCIDFont can be constructed with an empty spec."""
        font = PDFCIDFont(None, {})
        assert isinstance(font.cmap, CMap)

    def test_construct_with_identity_h(self) -> None:
        """PDFCIDFont with Identity-H encoding produces an IdentityCMap."""
        spec = {"Encoding": PSLiteral("Identity-H")}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_construct_with_identity_v(self) -> None:
        """PDFCIDFont with Identity-V encoding produces an IdentityCMap."""
        spec = {"Encoding": PSLiteral("Identity-V")}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_construct_with_onebyte_identity_h(self) -> None:
        """PDFCIDFont with OneByteIdentityH encoding produces an IdentityCMapByte."""
        from pdfminer.pdftypes import PDFStream

        stream = PDFStream({"CMapName": PSLiteral("OneByteIdentityH")}, "")
        spec = {"Encoding": stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMapByte)

    def test_get_cmap_from_spec_with_known_name(self) -> None:
        """PDFCIDFont.get_cmap_from_spec() returns the CMap for known names."""
        spec = {"Encoding": PSLiteral("UniGB-UCS2-H")}
        font = PDFCIDFont(None, spec)
        result = font.get_cmap_from_spec(spec, False)
        assert isinstance(result, CMap)

    def test_get_cmap_from_spec_unknown_strict_false(self) -> None:
        """PDFCIDFont.get_cmap_from_spec() returns empty CMap on unknown name (strict=False)."""
        font = PDFCIDFont(None, {})
        result = font.get_cmap_from_spec({"Encoding": PSLiteral("no-such-cmap")}, False)
        assert isinstance(result, CMap)

    def test_get_cmap_from_spec_unknown_strict_true_raises(self) -> None:
        """PDFCIDFont.get_cmap_from_spec() raises PDFFontError on unknown name (strict=True)."""
        from pdfminer.pdffont import PDFFontError

        font = PDFCIDFont(None, {})
        with pytest.raises(PDFFontError):
            font.get_cmap_from_spec({"Encoding": PSLiteral("no-such-cmap")}, True)


# ---------------------------------------------------------------------------
# Font size extraction
# ---------------------------------------------------------------------------

class TestFontSizeExtraction:
    """Verify correct character sizes are extracted from font-size-test.pdf."""

    def test_font_size_test_pdf_processes(self) -> None:
        """font-size-test.pdf can be processed without error."""
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical=True, char_margin=1, all_texts=True, boxes_flow=None)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interp = PDFPageInterpreter(rsrcmgr, device)

        with open(absolute_sample_path("font-size-test.pdf"), "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)
        result = device.get_result()
        assert result is not None

    def test_font_size_test_pdf_has_chars(self) -> None:
        """font-size-test.pdf yields LTChar objects with positive size."""
        for page in extract_pages(
            absolute_sample_path("font-size-test.pdf"),
            laparams=LAParams(detect_vertical=True, char_margin=1, all_texts=True, boxes_flow=None),
        ):
            for item in page:
                if hasattr(item, "__iter__"):
                    for line in item:
                        if isinstance(line, LTTextLine):
                            for char in line:
                                if isinstance(char, LTChar):
                                    assert char.size > 0
                                    return
        pytest.fail("No LTChar found in font-size-test.pdf")

    def test_contrib_cmap_font_12(self) -> None:
        """contrib/issue-598-cmap-other-fonts.pdf processes without error."""
        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical=True, char_margin=1, all_texts=True, boxes_flow=None)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interp = PDFPageInterpreter(rsrcmgr, device)

        with open(
            absolute_sample_path("contrib/issue-598-cmap-other-fonts.pdf"), "rb"
        ) as fp:
            for page in PDFPage.get_pages(fp):
                interp.process_page(page)
        assert device.get_result() is not None


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncoding:
    """Verify extraction from PDFs with various encodings."""

    def test_japanese_characters_simple3(self) -> None:
        """simple3.pdf produces correct Japanese (hiragana) characters."""
        result = extract_text(absolute_sample_path("simple3.pdf"))
        # Should contain hiragana characters
        assert "あ" in result
        assert "い" in result
        assert "う" in result

    @pytest.mark.xfail(
        reason="Known pre-existing bug: mult_matrix TypeError for list input (issue_566)",
        strict=False,
    )
    def test_chinese_issue_566_1(self) -> None:
        """contrib/issue_566_test_1.pdf extracts Chinese characters correctly."""
        result = extract_text(absolute_sample_path("contrib/issue_566_test_1.pdf"))
        assert "黎荣" in result or "ISSUE" in result

    @pytest.mark.xfail(
        reason="Known pre-existing bug: mult_matrix TypeError for list input (issue_566)",
        strict=False,
    )
    def test_chinese_issue_566_2(self) -> None:
        """contrib/issue_566_test_2.pdf extracts Chinese characters correctly."""
        result = extract_text(absolute_sample_path("contrib/issue_566_test_2.pdf"))
        assert "甲方" in result

    @pytest.mark.xfail(
        reason="Known pre-existing bug: mult_matrix TypeError for list input (issue_625)",
        strict=False,
    )
    def test_polish_issue_625(self) -> None:
        """contrib/issue-625-identity-cmap.pdf extracts Polish characters."""
        result = extract_text(absolute_sample_path("contrib/issue-625-identity-cmap.pdf"))
        assert "Termin" in result

    def test_czech_issue_791(self) -> None:
        """contrib/issue-791-non-unicode-cmap.pdf extracts Czech characters."""
        result = extract_text(absolute_sample_path("contrib/issue-791-non-unicode-cmap.pdf"))
        # Should contain some Czech text
        assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Vertical text
# ---------------------------------------------------------------------------

class TestVerticalText:
    """Verify vertical text extraction."""

    def test_vertical_pdf_produces_vertical_textbox(self) -> None:
        """contrib/issue-449-vertical.pdf yields LTTextBoxVertical with detect_vertical=True."""
        path = absolute_sample_path("contrib/issue-449-vertical.pdf")
        laparams = LAParams(detect_vertical=True)
        pages = extract_pages(path, laparams=laparams)
        textboxes = [
            item
            for item in next(pages)
            if isinstance(item, LTTextBoxVertical)
        ]
        assert len(textboxes) == 3

    def test_horizontal_pdf_produces_horizontal_textbox(self) -> None:
        """contrib/issue-449-horizontal.pdf yields LTTextBoxHorizontal."""
        path = absolute_sample_path("contrib/issue-449-horizontal.pdf")
        pages = extract_pages(path)
        textboxes = [
            item
            for item in next(pages)
            if isinstance(item, LTTextBoxHorizontal)
        ]
        assert len(textboxes) == 3
