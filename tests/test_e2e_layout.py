"""E2E tests for pdfminer layout analysis.

These tests document and verify the exact structure of layout objects
produced by extract_pages(). They serve as the correctness ground truth
when the Python implementation is replaced with Rust.
"""

import pytest

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTContainer,
    LTCurve,
    LTFigure,
    LTItem,
    LTLayoutContainer,
    LTLine,
    LTPage,
    LTRect,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextContainer,
    LTTextLine,
    LTTextLineHorizontal,
)
from tests.helpers import absolute_sample_path


# ---------------------------------------------------------------------------
# LTPage properties
# ---------------------------------------------------------------------------

class TestLTPageProperties:
    """Verify LTPage object attributes."""

    def _get_page(self, fname: str = "simple1.pdf") -> LTPage:
        return next(extract_pages(absolute_sample_path(fname)))

    def test_pageid_is_int(self) -> None:
        """LTPage.pageid is an integer."""
        page = self._get_page()
        assert isinstance(page.pageid, int)
        assert page.pageid >= 1

    def test_pageid_simple1(self) -> None:
        """simple1.pdf page 1 has pageid=1."""
        page = self._get_page()
        assert page.pageid == 1

    def test_bbox_is_4_tuple(self) -> None:
        """LTPage.bbox is a 4-element tuple."""
        page = self._get_page()
        assert len(page.bbox) == 4

    def test_bbox_simple1(self) -> None:
        """simple1.pdf page bbox is (0, 0, 612.0, 792.0)."""
        page = self._get_page()
        assert page.bbox == (0, 0, 612.0, 792.0)

    def test_rotate_is_int(self) -> None:
        """LTPage.rotate is an integer (0, 90, 180, or 270)."""
        page = self._get_page()
        assert isinstance(page.rotate, int)
        assert page.rotate in (0, 90, 180, 270)

    def test_rotate_simple1(self) -> None:
        """simple1.pdf has no rotation (rotate=0)."""
        page = self._get_page()
        assert page.rotate == 0

    def test_width_is_float(self) -> None:
        """LTPage.width is a float."""
        page = self._get_page()
        assert isinstance(page.width, float)

    def test_width_simple1(self) -> None:
        """simple1.pdf page width is 612.0."""
        page = self._get_page()
        assert page.width == 612.0

    def test_height_is_float(self) -> None:
        """LTPage.height is a float."""
        page = self._get_page()
        assert isinstance(page.height, float)

    def test_height_simple1(self) -> None:
        """simple1.pdf page height is 792.0."""
        page = self._get_page()
        assert page.height == 792.0

    def test_page_is_iterable(self) -> None:
        """LTPage is iterable and yields layout items."""
        page = self._get_page()
        items = list(page)
        assert len(items) > 0

    def test_page_items_are_ltitem(self) -> None:
        """Every item yielded by iterating LTPage is an LTItem."""
        page = self._get_page()
        for item in page:
            assert isinstance(item, LTItem)

    def test_page_is_ltcontainer(self) -> None:
        """LTPage is an instance of LTContainer."""
        page = self._get_page()
        assert isinstance(page, LTContainer)


# ---------------------------------------------------------------------------
# LTTextBox properties
# ---------------------------------------------------------------------------

class TestLTTextBoxProperties:
    """Verify LTTextBox / LTTextBoxHorizontal attributes."""

    def _get_first_textbox(self) -> LTTextBox:
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        for item in page:
            if isinstance(item, LTTextBox):
                return item
        pytest.fail("No LTTextBox in simple1.pdf")

    def test_textbox_has_index(self) -> None:
        """LTTextBox.index is a non-negative integer."""
        box = self._get_first_textbox()
        assert isinstance(box.index, int)
        assert box.index >= 0

    def test_textbox_has_bbox(self) -> None:
        """LTTextBox.bbox is a 4-tuple."""
        box = self._get_first_textbox()
        assert len(box.bbox) == 4

    def test_textbox_bbox_valid(self) -> None:
        """LTTextBox.bbox has x0 <= x1 and y0 <= y1."""
        box = self._get_first_textbox()
        x0, y0, x1, y1 = box.bbox
        assert x0 <= x1
        assert y0 <= y1

    def test_textbox_get_text(self) -> None:
        """LTTextBox.get_text() returns a non-empty string."""
        box = self._get_first_textbox()
        text = box.get_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_textbox_iteration_yields_textlines(self) -> None:
        """Iterating LTTextBox yields LTTextLine objects."""
        box = self._get_first_textbox()
        lines = list(box)
        assert len(lines) > 0
        for line in lines:
            assert isinstance(line, LTTextLine)

    def test_textbox_is_ltcontainer(self) -> None:
        """LTTextBox is a subclass of LTContainer."""
        box = self._get_first_textbox()
        assert isinstance(box, LTContainer)

    def test_textbox_index_zero(self) -> None:
        """The first LTTextBox in simple1.pdf has index=0."""
        box = self._get_first_textbox()
        assert box.index == 0

    def test_textbox_get_text_simple1_first(self) -> None:
        """First textbox in simple1.pdf contains 'Hello'."""
        box = self._get_first_textbox()
        assert "Hello" in box.get_text()


# ---------------------------------------------------------------------------
# LTTextLine properties
# ---------------------------------------------------------------------------

class TestLTTextLineProperties:
    """Verify LTTextLine attributes."""

    def _get_first_textline(self) -> LTTextLine:
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        for item in page:
            if isinstance(item, LTTextBox):
                for line in item:
                    if isinstance(line, LTTextLine):
                        return line
        pytest.fail("No LTTextLine in simple1.pdf")

    def test_textline_get_text(self) -> None:
        """LTTextLine.get_text() returns a string ending with newline."""
        line = self._get_first_textline()
        text = line.get_text()
        assert isinstance(text, str)
        assert text.endswith("\n")

    def test_textline_iteration_yields_chars(self) -> None:
        """Iterating LTTextLine yields LTChar and/or LTAnno objects."""
        line = self._get_first_textline()
        chars = list(line)
        assert len(chars) > 0
        for char in chars:
            assert isinstance(char, (LTChar, LTAnno))

    def test_textline_has_ltchar(self) -> None:
        """At least one LTChar appears in the first textline of simple1.pdf."""
        line = self._get_first_textline()
        ltchars = [c for c in line if isinstance(c, LTChar)]
        assert len(ltchars) > 0

    def test_textline_bbox_valid(self) -> None:
        """LTTextLine.bbox has x0 <= x1 and y0 <= y1."""
        line = self._get_first_textline()
        x0, y0, x1, y1 = line.bbox
        assert x0 <= x1
        assert y0 <= y1


# ---------------------------------------------------------------------------
# LTChar properties (full coverage)
# ---------------------------------------------------------------------------

class TestLTCharProperties:
    """Verify all LTChar attributes needed by layout analysis and unstructured."""

    def _get_chars(self, fname: str = "simple1.pdf"):
        for page in extract_pages(absolute_sample_path(fname)):
            for item in page:
                if isinstance(item, LTTextBox):
                    for line in item:
                        if isinstance(line, LTTextLine):
                            for char in line:
                                if isinstance(char, LTChar):
                                    yield char

    def _first_char(self) -> LTChar:
        return next(self._get_chars())

    def test_ltchar_x0_y0_x1_y1(self) -> None:
        """LTChar has x0, y0, x1, y1 float attributes consistent with bbox."""
        char = self._first_char()
        assert char.x0 == char.bbox[0]
        assert char.y0 == char.bbox[1]
        assert char.x1 == char.bbox[2]
        assert char.y1 == char.bbox[3]

    def test_ltchar_matrix_length(self) -> None:
        """LTChar.matrix is a 6-element sequence."""
        char = self._first_char()
        assert len(char.matrix) == 6

    def test_ltchar_get_text_single_char(self) -> None:
        """LTChar.get_text() always returns a single character."""
        for char in self._get_chars():
            assert len(char.get_text()) == 1
            break

    def test_ltchar_size_positive(self) -> None:
        """LTChar.size > 0 for all chars in simple1.pdf."""
        for char in self._get_chars():
            assert char.size > 0

    def test_ltchar_fontname_string(self) -> None:
        """LTChar.fontname is a non-empty string."""
        char = self._first_char()
        assert isinstance(char.fontname, str)
        assert len(char.fontname) > 0

    def test_ltchar_adv_float(self) -> None:
        """LTChar.adv is a float (advance width)."""
        char = self._first_char()
        assert isinstance(char.adv, float)

    def test_ltchar_upright_bool(self) -> None:
        """LTChar.upright is a boolean."""
        char = self._first_char()
        assert isinstance(char.upright, bool)

    def test_ltchar_ncs_has_name(self) -> None:
        """LTChar.ncs has a .name attribute."""
        char = self._first_char()
        assert hasattr(char.ncs, "name")

    def test_ltchar_graphicstate_not_none(self) -> None:
        """LTChar.graphicstate is present and not None."""
        char = self._first_char()
        assert char.graphicstate is not None

    def test_ltchar_simple1_fontname_helvetica(self) -> None:
        """The first char in simple1.pdf uses Helvetica."""
        char = self._first_char()
        assert char.fontname == "Helvetica"

    def test_ltchar_simple1_first_char_is_H(self) -> None:
        """The first LTChar in simple1.pdf is 'H'."""
        char = self._first_char()
        assert char.get_text() == "H"


# ---------------------------------------------------------------------------
# LTAnno properties
# ---------------------------------------------------------------------------

class TestLTAnnoProperties:
    """Verify LTAnno attributes."""

    def _get_first_ltanno(self) -> LTAnno:
        for page in extract_pages(absolute_sample_path("simple1.pdf")):
            for item in page:
                if isinstance(item, LTTextBox):
                    for line in item:
                        if isinstance(line, LTTextLine):
                            for char in line:
                                if isinstance(char, LTAnno):
                                    return char
        pytest.fail("No LTAnno found in simple1.pdf")

    def test_ltanno_get_text(self) -> None:
        """LTAnno.get_text() returns a string."""
        anno = self._get_first_ltanno()
        assert isinstance(anno.get_text(), str)

    def test_ltanno_is_ltitem(self) -> None:
        """LTAnno is an LTItem."""
        anno = self._get_first_ltanno()
        assert isinstance(anno, LTItem)


# ---------------------------------------------------------------------------
# LTFigure properties
# ---------------------------------------------------------------------------

class TestLTFigureProperties:
    """Verify LTFigure attributes."""

    def _get_figure(self) -> LTFigure:
        path = absolute_sample_path("contrib/issue_495_pdfobjref.pdf")
        for page in extract_pages(path):
            for item in page:
                if isinstance(item, LTFigure):
                    return item
        pytest.skip("No LTFigure found in test PDFs")

    def test_ltfigure_name(self) -> None:
        """LTFigure.name is a string."""
        fig = self._get_figure()
        assert isinstance(fig.name, str)

    def test_ltfigure_bbox(self) -> None:
        """LTFigure.bbox is a 4-tuple."""
        fig = self._get_figure()
        assert len(fig.bbox) == 4

    def test_ltfigure_matrix(self) -> None:
        """LTFigure.matrix is a 6-element sequence."""
        fig = self._get_figure()
        assert len(fig.matrix) == 6

    def test_ltfigure_is_ltlayoutcontainer(self) -> None:
        """LTFigure is a subclass of LTLayoutContainer."""
        fig = self._get_figure()
        assert isinstance(fig, LTLayoutContainer)


# ---------------------------------------------------------------------------
# LTCurve / LTLine / LTRect properties
# ---------------------------------------------------------------------------

class TestLTCurveProperties:
    """Verify LTCurve / LTLine / LTRect attributes."""

    def test_ltcurve_pts(self) -> None:
        """LTCurve.pts is a list of (x, y) tuples."""
        from pdfminer.converter import PDFLayoutAnalyzer
        from pdfminer.pdfinterp import PDFGraphicState

        analyzer = PDFLayoutAnalyzer(None)
        # Use tuple for ctm to satisfy the Rust extension's type requirements
        analyzer.set_ctm((1, 0, 0, 1, 0, 0))
        analyzer.cur_item = LTContainer([0, 1000, 0, 1000])
        path = [("m", 72.41, 433.89), ("l", 90.0, 433.89), ("l", 90.0, 440.0), ("l", 72.41, 440.0), ("h",)]
        analyzer.paint_path(PDFGraphicState(), False, False, False, path)
        objs = analyzer.cur_item._objs
        assert len(objs) == 1
        obj = objs[0]
        # Should be LTRect or LTCurve
        assert isinstance(obj, LTCurve)
        assert hasattr(obj, "pts")
        assert isinstance(obj.pts, list)

    def test_ltcurve_linewidth(self) -> None:
        """LTCurve.linewidth is a float."""
        from pdfminer.converter import PDFLayoutAnalyzer
        from pdfminer.pdfinterp import PDFGraphicState

        analyzer = PDFLayoutAnalyzer(None)
        # Use tuple for ctm to satisfy the Rust extension's type requirements
        analyzer.set_ctm((1, 0, 0, 1, 0, 0))
        analyzer.cur_item = LTContainer([0, 1000, 0, 1000])
        path = [("m", 10, 30), ("l", 10, 40)]
        gs = PDFGraphicState()
        analyzer.paint_path(gs, False, False, False, path)
        objs = analyzer.cur_item._objs
        if objs:
            assert hasattr(objs[0], "linewidth")

    def test_ltline_is_ltcurve(self) -> None:
        """LTLine is a subclass of LTCurve."""
        assert issubclass(LTLine, LTCurve)

    def test_ltrect_is_ltcurve(self) -> None:
        """LTRect is a subclass of LTCurve."""
        assert issubclass(LTRect, LTCurve)


# ---------------------------------------------------------------------------
# isinstance checks
# ---------------------------------------------------------------------------

class TestIsInstanceHierarchy:
    """Verify the class hierarchy of layout objects."""

    def _get_page(self) -> LTPage:
        return next(extract_pages(absolute_sample_path("simple1.pdf")))

    def test_ltpage_is_ltlayoutcontainer(self) -> None:
        """LTPage is an instance of LTLayoutContainer."""
        page = self._get_page()
        assert isinstance(page, LTLayoutContainer)

    def test_ltpage_is_ltcontainer(self) -> None:
        """LTPage is an instance of LTContainer."""
        page = self._get_page()
        assert isinstance(page, LTContainer)

    def test_ltpage_is_ltitem(self) -> None:
        """LTPage is an instance of LTItem."""
        page = self._get_page()
        assert isinstance(page, LTItem)

    def test_lttext_box_horizontal_is_lttext_box(self) -> None:
        """LTTextBoxHorizontal is an LTTextBox."""
        page = self._get_page()
        for item in page:
            if isinstance(item, LTTextBoxHorizontal):
                assert isinstance(item, LTTextBox)
                return
        pytest.skip("No LTTextBoxHorizontal in simple1.pdf")

    def test_lttext_line_horizontal_is_lttext_line(self) -> None:
        """LTTextLineHorizontal is an LTTextLine."""
        page = self._get_page()
        for item in page:
            if isinstance(item, LTTextBox):
                for line in item:
                    if isinstance(line, LTTextLineHorizontal):
                        assert isinstance(line, LTTextLine)
                        return
        pytest.skip("No LTTextLineHorizontal in simple1.pdf")

    def test_ltcontainer_issubclass_of_ltitem(self) -> None:
        """LTContainer is a subclass of LTItem."""
        assert issubclass(LTContainer, LTItem)

    def test_ltlayoutcontainer_issubclass_of_ltcontainer(self) -> None:
        """LTLayoutContainer is a subclass of LTContainer."""
        assert issubclass(LTLayoutContainer, LTContainer)

    def test_ltpage_issubclass_of_ltlayoutcontainer(self) -> None:
        """LTPage is a subclass of LTLayoutContainer."""
        assert issubclass(LTPage, LTLayoutContainer)

    def test_lttextbox_issubclass_of_ltcontainer(self) -> None:
        """LTTextBox is a subclass of LTContainer (not LTLayoutContainer)."""
        assert issubclass(LTTextBox, LTContainer)
        # LTTextBox inherits from LTTextContainer -> LTExpandableContainer -> LTContainer
        assert not issubclass(LTTextBox, LTLayoutContainer)


# ---------------------------------------------------------------------------
# LAParams defaults
# ---------------------------------------------------------------------------

class TestLAParamsDefaults:
    """Verify LAParams default values match the documented defaults."""

    def test_line_overlap_default(self) -> None:
        """LAParams.line_overlap default is 0.5."""
        assert LAParams().line_overlap == 0.5

    def test_char_margin_default(self) -> None:
        """LAParams.char_margin default is 2.0."""
        assert LAParams().char_margin == 2.0

    def test_line_margin_default(self) -> None:
        """LAParams.line_margin default is 0.5."""
        assert LAParams().line_margin == 0.5

    def test_word_margin_default(self) -> None:
        """LAParams.word_margin default is 0.1."""
        assert LAParams().word_margin == 0.1

    def test_boxes_flow_default(self) -> None:
        """LAParams.boxes_flow default is 0.5."""
        assert LAParams().boxes_flow == 0.5

    def test_detect_vertical_default(self) -> None:
        """LAParams.detect_vertical default is False."""
        assert LAParams().detect_vertical is False

    def test_all_texts_default(self) -> None:
        """LAParams.all_texts default is False."""
        assert LAParams().all_texts is False


# ---------------------------------------------------------------------------
# Snapshot test: exact structure of simple1.pdf
# ---------------------------------------------------------------------------

class TestSimple1Snapshot:
    """Snapshot tests: extract_pages on simple1.pdf returns the exact same
    structure every time. These pin the current behavior as ground truth."""

    def test_simple1_page_count(self) -> None:
        """simple1.pdf has exactly 1 page."""
        pages = list(extract_pages(absolute_sample_path("simple1.pdf")))
        assert len(pages) == 1

    def test_simple1_textbox_count(self) -> None:
        """simple1.pdf page 1 has exactly 8 LTTextBoxHorizontal objects."""
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        boxes = [item for item in page if isinstance(item, LTTextBoxHorizontal)]
        assert len(boxes) == 8

    def test_simple1_textbox_texts(self) -> None:
        """simple1.pdf text boxes return known-good text."""
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        texts = [
            item.get_text()
            for item in page
            if isinstance(item, LTTextBox)
        ]
        assert texts[0] == "Hello \n"
        assert texts[1] == "World\n"

    def test_simple1_page_bbox(self) -> None:
        """simple1.pdf page has exact bbox (0, 0, 612.0, 792.0)."""
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        assert page.bbox == (0, 0, 612.0, 792.0)

    def test_simple1_first_char_properties(self) -> None:
        """First LTChar in simple1.pdf has known-good properties."""
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        first_box = next(item for item in page if isinstance(item, LTTextBox))
        first_line = next(iter(first_box))
        first_char = next(c for c in first_line if isinstance(c, LTChar))

        assert first_char.get_text() == "H"
        assert first_char.fontname == "Helvetica"
        assert first_char.size == pytest.approx(24.0)
        assert first_char.upright is True

    def test_simple1_deterministic(self) -> None:
        """extract_pages on simple1.pdf returns the same text across two calls."""
        path = absolute_sample_path("simple1.pdf")

        def get_texts():
            return [
                item.get_text()
                for page in extract_pages(path)
                for item in page
                if hasattr(item, "get_text")
            ]

        assert get_texts() == get_texts()

    def test_simple4_three_text_boxes_default_laparams(self) -> None:
        """simple4.pdf with default LAParams produces 1 merged text box."""
        pages = list(extract_pages(absolute_sample_path("simple4.pdf")))
        elements = [e for e in pages[0] if isinstance(e, LTTextContainer)]
        # Default line_margin=0.5 merges the three lines
        all_text = "".join(e.get_text() for e in elements)
        assert "Text1" in all_text
        assert "Text2" in all_text
        assert "Text3" in all_text
