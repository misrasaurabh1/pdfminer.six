"""Tests for Rust layout analysis acceleration."""
from pdfminer.layout import LAParams, LTChar, LTContainer, LTTextBox, LTTextLine


def test_all_lt_types_importable() -> None:
    """All LT* types used by unstructured must be importable."""
    from pdfminer.layout import (  # noqa: F401
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


def test_ltchar_rendermode_patchable() -> None:
    """LTChar must support .rendermode attribute assignment (unstructured compat)."""
    from pdfminer.high_level import extract_pages
    from tests.helpers import absolute_sample_path

    path = absolute_sample_path("simple1.pdf")
    found_char = False
    for page in extract_pages(str(path), laparams=LAParams()):
        for box in page:
            if isinstance(box, LTTextBox):
                for line in box:
                    if isinstance(line, LTTextLine):
                        for char in line:
                            if isinstance(char, LTChar):
                                char.rendermode = 0  # Must work!
                                assert char.rendermode == 0
                                found_char = True
    assert found_char, "No LTChar found in test PDF"


def test_isinstance_checks() -> None:
    """isinstance checks must work for all LT types."""
    from pdfminer.high_level import extract_pages
    from tests.helpers import absolute_sample_path

    path = absolute_sample_path("simple1.pdf")
    for page in extract_pages(str(path), laparams=LAParams()):
        assert isinstance(page, LTContainer)
        for element in page:
            if isinstance(element, LTTextBox):
                assert isinstance(element, LTContainer)


def test_laparams_defaults() -> None:
    """LAParams must have correct defaults."""
    lp = LAParams()
    assert lp.line_overlap == 0.5
    assert lp.char_margin == 2.0
    assert lp.line_margin == 0.5
    assert lp.word_margin == 0.1
    assert lp.boxes_flow == 0.5
    assert lp.detect_vertical is False
    assert lp.all_texts is False


def test_layout_analysis_consistent() -> None:
    """Layout analysis must produce consistent results across two runs."""
    from pdfminer.high_level import extract_pages
    from tests.helpers import absolute_sample_path

    path = absolute_sample_path("simple1.pdf")
    pages1 = list(extract_pages(str(path), laparams=LAParams()))
    pages2 = list(extract_pages(str(path), laparams=LAParams()))
    assert len(pages1) == len(pages2)
    for p1, p2 in zip(pages1, pages2, strict=True):
        assert len(list(p1)) == len(list(p2))


def test_rust_acceleration_active() -> None:
    """Verify the Rust acceleration is actually loaded."""
    import pdfminer.layout as lay

    assert lay._HAS_RUST, (
        "pdfminer_core Rust extension not loaded; "
        "run 'maturin develop --release' to build it"
    )


def test_rust_bbox_overlap() -> None:
    """bbox_overlap must correctly identify overlapping/non-overlapping rects."""
    from pdfminer_core import bbox_overlap

    assert bbox_overlap((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)) is True
    assert bbox_overlap((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) is False
    # Touching edges are not overlapping (strict inequality)
    assert bbox_overlap((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0)) is False


def test_rust_group_chars_into_lines() -> None:
    """group_chars_into_lines must group horizontally adjacent chars."""
    from pdfminer_core import group_chars_into_lines

    bboxes = [
        (0.0, 0.0, 5.0, 10.0),
        (5.5, 0.0, 10.0, 10.0),
        (10.5, 0.0, 15.0, 10.0),
        (100.0, 50.0, 110.0, 60.0),  # separate line
    ]
    groups = group_chars_into_lines(bboxes, 0.5, 2.0)
    assert len(groups) == 2
    assert groups[0] == [0, 1, 2]
    assert groups[1] == [3]


def test_rust_plane() -> None:
    """Rust Plane must find overlapping objects and honour removals."""
    from pdfminer_core import Plane

    plane = Plane((0.0, 0.0, 100.0, 100.0), 50.0)
    plane.add_bbox(1, (10.0, 10.0, 30.0, 30.0))
    plane.add_bbox(2, (50.0, 50.0, 80.0, 80.0))
    plane.add_bbox(3, (15.0, 15.0, 25.0, 25.0))

    hits = set(plane.find_overlapping((12.0, 12.0, 20.0, 20.0)))
    assert 1 in hits
    assert 3 in hits
    assert 2 not in hits

    plane.remove_bbox(1)
    hits2 = set(plane.find_overlapping((12.0, 12.0, 20.0, 20.0)))
    assert 1 not in hits2
    assert 3 in hits2
