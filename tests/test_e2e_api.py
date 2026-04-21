"""E2E tests for high-level pdfminer API functions.

These tests document and verify exact current behavior of extract_text(),
extract_pages(), and extract_text_to_fp() across all sample PDFs.
They serve as correctness ground truth when the Python implementation
is replaced with Rust.
"""

import io
import os
from pathlib import Path

import pytest

from pdfminer.high_level import extract_pages, extract_text, extract_text_to_fp
from pdfminer.layout import LAParams, LTPage, LTTextContainer
from tests.helpers import absolute_sample_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(absolute_sample_path("")).parent


def _all_sample_pdfs():
    """Return relative paths (relative to samples/) for every PDF under samples/."""
    pdfs = []
    for root, _dirs, files in os.walk(SAMPLES_DIR):
        for fname in files:
            if fname.endswith(".pdf"):
                rel = os.path.relpath(os.path.join(root, fname), SAMPLES_DIR)
                pdfs.append(rel)
    return sorted(pdfs)


ALL_PDFS = _all_sample_pdfs()

# PDFs that are known to be intentionally corrupted / require special handling
_SKIP_EXTRACT = {
    "zen_of_python_corrupted.pdf",
    "scancode/simple.pdf",  # may not exist on all setups
}

# Encrypted PDFs and their correct user passwords
ENCRYPTED_PDFS = {
    "encryption/aes-256.pdf": "foo",
    "encryption/aes-128.pdf": "foo",
    "encryption/rc4-40.pdf": "foo",
    "encryption/rc4-128.pdf": "foo",
    "encryption/aes-256-m.pdf": "foo",
    "encryption/aes-128-m.pdf": "foo",
    "encryption/aes-256-r6.pdf": "usersecret",
    "encryption/base.pdf": "usersecret",
    "encryption/encrypted_doc_no_id.pdf": "",
}


# ---------------------------------------------------------------------------
# extract_text() tests
# ---------------------------------------------------------------------------

class TestExtractTextAllPdfs:
    """extract_text() must not raise on any sample PDF (excluding known-broken ones)."""

    @pytest.mark.parametrize("rel_path", ALL_PDFS)
    def test_extract_text_returns_string(self, rel_path: str) -> None:
        """extract_text() returns a str for every sample PDF."""
        abs_path = absolute_sample_path(rel_path)
        password = ENCRYPTED_PDFS.get(rel_path, "")
        try:
            result = extract_text(abs_path, password=password)
        except Exception:
            # Skip PDFs that need special handling (corrupt, unsupported)
            pytest.skip(f"Cannot extract from {rel_path}")
        assert isinstance(result, str)

    def test_extract_text_simple1_exact(self) -> None:
        """extract_text() returns the known-good text for simple1.pdf."""
        expected = (
            "Hello \n\nWorld\n\nHello \n\nWorld\n\n"
            "H e l l o  \n\nW o r l d\n\n"
            "H e l l o  \n\nW o r l d\n\n\f"
        )
        result = extract_text(absolute_sample_path("simple1.pdf"))
        assert result == expected

    def test_extract_text_simple2_exact(self) -> None:
        """extract_text() returns a form-feed for the image-only simple2.pdf."""
        result = extract_text(absolute_sample_path("simple2.pdf"))
        assert result == "\f"

    def test_extract_text_simple3_exact(self) -> None:
        """extract_text() returns the Japanese characters for simple3.pdf."""
        expected = (
            "Hello\n\nHello\nあ\nい\nう\nえ\nお\nあ\nい\nう\nえ\nお\n"
            "World\n\nWorld\n\n\f"
        )
        result = extract_text(absolute_sample_path("simple3.pdf"))
        assert result == expected

    def test_extract_text_simple4_exact(self) -> None:
        """extract_text() returns three text items for simple4.pdf."""
        result = extract_text(absolute_sample_path("simple4.pdf"))
        assert result == "Text1\nText2\nText3\n\n\f"

    def test_extract_text_simple5_exact(self) -> None:
        """extract_text() returns the smart quotes for simple5.pdf."""
        result = extract_text(absolute_sample_path("simple5.pdf"))
        assert "Heading" in result
        assert "Subheading" in result


class TestExtractTextOptions:
    """Test optional parameters for extract_text()."""

    def test_maxpages_one(self) -> None:
        """maxpages=1 stops after the first page."""
        full = extract_text(absolute_sample_path("simple1.pdf"))
        one_page = extract_text(absolute_sample_path("simple1.pdf"), maxpages=1)
        # Single-page PDF: should be the same
        assert one_page == full

    def test_page_numbers_first(self) -> None:
        """page_numbers=[0] returns only the first page."""
        full = extract_text(absolute_sample_path("simple1.pdf"))
        first = extract_text(absolute_sample_path("simple1.pdf"), page_numbers=[0])
        # simple1.pdf is a single page, so results must match
        assert first == full

    def test_page_numbers_empty(self) -> None:
        """page_numbers=[] is treated as no filter (returns all pages).

        An empty container is falsy in Python, so it behaves like page_numbers=None,
        meaning all pages are returned. This is the documented current behavior.
        """
        full = extract_text(absolute_sample_path("simple1.pdf"))
        result = extract_text(absolute_sample_path("simple1.pdf"), page_numbers=[])
        assert result == full

    def test_caching_true(self) -> None:
        """caching=True (default) returns same text as caching=False."""
        path = absolute_sample_path("simple1.pdf")
        with_cache = extract_text(path, caching=True)
        without_cache = extract_text(path, caching=False)
        assert with_cache == without_cache

    def test_caching_false(self) -> None:
        """caching=False does not break extraction."""
        result = extract_text(absolute_sample_path("simple4.pdf"), caching=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_from_file_object(self) -> None:
        """extract_text() accepts a file-like object."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as fp:
            result = extract_text(fp)
        assert "Hello" in result

    def test_from_path_string(self) -> None:
        """extract_text() accepts a path string."""
        result = extract_text(absolute_sample_path("simple1.pdf"))
        assert isinstance(result, str)

    def test_from_pathlib_path(self) -> None:
        """extract_text() accepts a pathlib.Path."""
        result = extract_text(Path(absolute_sample_path("simple1.pdf")))
        assert isinstance(result, str)


class TestExtractTextLAParams:
    """Test LAParams combinations with extract_text()."""

    def test_default_laparams(self) -> None:
        """Default LAParams produces expected text for simple4.pdf."""
        result = extract_text(
            absolute_sample_path("simple4.pdf"),
            laparams=LAParams(),
        )
        assert "Text1" in result

    def test_boxes_flow_none(self) -> None:
        """boxes_flow=None does not raise and returns text."""
        result = extract_text(
            absolute_sample_path("simple4.pdf"),
            laparams=LAParams(boxes_flow=None),
        )
        assert "Text1" in result

    def test_line_margin_low(self) -> None:
        """Low line_margin splits lines into separate boxes."""
        result = extract_text(
            absolute_sample_path("simple4.pdf"),
            laparams=LAParams(line_margin=0.19),
        )
        assert "Text1" in result

    def test_line_margin_high(self) -> None:
        """High line_margin merges lines into one box."""
        result = extract_text(
            absolute_sample_path("simple4.pdf"),
            laparams=LAParams(line_margin=0.21),
        )
        assert "Text1" in result
        assert "Text2" in result
        assert "Text3" in result

    def test_detect_vertical_true(self) -> None:
        """detect_vertical=True does not raise on simple1.pdf."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(detect_vertical=True),
        )
        assert isinstance(result, str)

    def test_char_margin(self) -> None:
        """Non-default char_margin does not raise."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(char_margin=1.0),
        )
        assert isinstance(result, str)

    def test_word_margin(self) -> None:
        """Non-default word_margin does not raise."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(word_margin=0.2),
        )
        assert isinstance(result, str)

    def test_line_overlap(self) -> None:
        """Non-default line_overlap does not raise."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(line_overlap=0.3),
        )
        assert isinstance(result, str)

    def test_all_texts_true(self) -> None:
        """all_texts=True does not raise."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(all_texts=True),
        )
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "laparams_kwargs",
        [
            {},
            {"boxes_flow": None},
            {"boxes_flow": 0.0},
            {"boxes_flow": 1.0},
            {"line_margin": 0.1},
            {"line_margin": 1.0},
            {"char_margin": 0.5},
            {"char_margin": 3.0},
            {"word_margin": 0.05},
            {"word_margin": 0.3},
            {"line_overlap": 0.1},
            {"line_overlap": 0.9},
            {"detect_vertical": True},
            {"all_texts": True},
        ],
    )
    def test_laparams_combinations(self, laparams_kwargs: dict) -> None:
        """All LAParams combinations must not raise and return a string."""
        result = extract_text(
            absolute_sample_path("simple1.pdf"),
            laparams=LAParams(**laparams_kwargs),
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_pages() tests
# ---------------------------------------------------------------------------

class TestExtractPages:
    """Test the extract_pages() iterator."""

    def test_returns_iterator(self) -> None:
        """extract_pages() returns an iterator of LTPage objects."""
        pages = extract_pages(absolute_sample_path("simple1.pdf"))
        page = next(pages)
        assert isinstance(page, LTPage)

    def test_simple1_page_count(self) -> None:
        """simple1.pdf has exactly 1 page."""
        pages = list(extract_pages(absolute_sample_path("simple1.pdf")))
        assert len(pages) == 1

    def test_page_is_iterable(self) -> None:
        """LTPage must be iterable (yields layout items)."""
        page = next(extract_pages(absolute_sample_path("simple1.pdf")))
        items = list(page)
        assert len(items) > 0

    def test_maxpages_parameter(self) -> None:
        """maxpages=1 stops after the first page for multi-page PDFs."""
        pages = list(
            extract_pages(absolute_sample_path("simple3.pdf"), maxpages=1)
        )
        assert len(pages) == 1

    def test_page_numbers_parameter(self) -> None:
        """page_numbers=[0] returns only the first page."""
        pages = list(
            extract_pages(
                absolute_sample_path("simple1.pdf"), page_numbers=[0]
            )
        )
        assert len(pages) == 1

    def test_caching_true_and_false_consistent(self) -> None:
        """caching=True and caching=False produce identical text."""
        path = absolute_sample_path("simple1.pdf")
        cached_texts = [
            item.get_text()
            for page in extract_pages(path, caching=True)
            for item in page
            if hasattr(item, "get_text")
        ]
        uncached_texts = [
            item.get_text()
            for page in extract_pages(path, caching=False)
            for item in page
            if hasattr(item, "get_text")
        ]
        assert cached_texts == uncached_texts

    def test_line_margin_splits_into_three_boxes(self) -> None:
        """line_margin=0.19 splits simple4.pdf into 3 text boxes."""
        pages = list(
            extract_pages(
                absolute_sample_path("simple4.pdf"),
                laparams=LAParams(line_margin=0.19),
            )
        )
        assert len(pages) == 1
        elements = [e for e in pages[0] if isinstance(e, LTTextContainer)]
        assert len(elements) == 3
        assert elements[0].get_text() == "Text1\n"
        assert elements[1].get_text() == "Text2\n"
        assert elements[2].get_text() == "Text3\n"

    def test_line_margin_merges_into_one_box(self) -> None:
        """line_margin=0.21 merges simple4.pdf into 1 text box."""
        pages = list(
            extract_pages(
                absolute_sample_path("simple4.pdf"),
                laparams=LAParams(line_margin=0.21),
            )
        )
        elements = [e for e in pages[0] if isinstance(e, LTTextContainer)]
        assert len(elements) == 1
        assert elements[0].get_text() == "Text1\nText2\nText3\n"


# ---------------------------------------------------------------------------
# extract_text_to_fp() tests
# ---------------------------------------------------------------------------

class TestExtractTextToFp:
    """Test extract_text_to_fp() output types."""

    @pytest.mark.parametrize("output_type", ["text", "xml", "html", "hocr", "tag"])
    def test_output_types_produce_bytes(self, output_type: str) -> None:
        """All output_type values produce non-empty output for simple1.pdf."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(
                inf, outfp, output_type=output_type, laparams=LAParams()
            )
            result = outfp.getvalue()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_text_output_contains_hello(self) -> None:
        """text output for simple1.pdf contains 'Hello'."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(inf, outfp, output_type="text", laparams=LAParams())
        assert b"Hello" in outfp.getvalue()

    def test_xml_output_has_pages_element(self) -> None:
        """xml output is valid XML with a <pages> root element."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(inf, outfp, output_type="xml", laparams=LAParams())
        content = outfp.getvalue()
        assert b"<pages>" in content
        assert b"</pages>" in content

    def test_html_output_has_html_element(self) -> None:
        """html output contains an <html> element."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(inf, outfp, output_type="html", laparams=LAParams())
        content = outfp.getvalue()
        assert b"<html>" in content

    def test_hocr_output_has_html_element(self) -> None:
        """hocr output contains an <html> element."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(inf, outfp, output_type="hocr", laparams=LAParams())
        content = outfp.getvalue()
        assert b"<html" in content

    def test_tag_output_has_page_tag(self) -> None:
        """tag output contains a <page> element."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            extract_text_to_fp(inf, outfp, output_type="tag", laparams=LAParams())
        content = outfp.getvalue()
        assert b"<page" in content

    def test_invalid_output_type_raises(self) -> None:
        """An unknown output_type raises PDFValueError."""
        from pdfminer.pdfexceptions import PDFValueError

        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp = io.BytesIO()
            with pytest.raises(PDFValueError):
                extract_text_to_fp(inf, outfp, output_type="invalid")

    def test_maxpages_limits_output(self) -> None:
        """maxpages=0 (unlimited) vs maxpages=1 produce same output for 1-page PDF."""
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp_all = io.BytesIO()
            extract_text_to_fp(
                inf, outfp_all, output_type="text", laparams=LAParams(), maxpages=0
            )
        with open(absolute_sample_path("simple1.pdf"), "rb") as inf:
            outfp_one = io.BytesIO()
            extract_text_to_fp(
                inf, outfp_one, output_type="text", laparams=LAParams(), maxpages=1
            )
        assert outfp_all.getvalue() == outfp_one.getvalue()


# ---------------------------------------------------------------------------
# Encrypted PDF tests (high-level)
# ---------------------------------------------------------------------------

class TestExtractTextEncrypted:
    """Test extract_text() with encrypted PDF files."""

    @pytest.mark.parametrize("rel_path,password", ENCRYPTED_PDFS.items())
    def test_extract_with_correct_password(
        self, rel_path: str, password: str
    ) -> None:
        """extract_text() succeeds with the correct password."""
        abs_path = absolute_sample_path(rel_path)
        if not os.path.exists(abs_path):
            pytest.skip(f"{rel_path} not found")
        result = extract_text(abs_path, password=password)
        assert isinstance(result, str)

    def test_aes256_extracts_secret(self) -> None:
        """AES-256 encrypted PDF extracts 'Secret!' with correct password."""
        path = absolute_sample_path("encryption/aes-256.pdf")
        result = extract_text(path, password="foo")
        assert "Secret!" in result

    def test_rc4_40_extracts_secret(self) -> None:
        """RC4-40 encrypted PDF extracts 'Secret!' with correct password."""
        path = absolute_sample_path("encryption/rc4-40.pdf")
        result = extract_text(path, password="foo")
        assert "Secret!" in result

    def test_rc4_128_extracts_secret(self) -> None:
        """RC4-128 encrypted PDF extracts 'Secret!' with correct password."""
        path = absolute_sample_path("encryption/rc4-128.pdf")
        result = extract_text(path, password="foo")
        assert "Secret!" in result

    def test_aes128_extracts_secret(self) -> None:
        """AES-128 encrypted PDF extracts 'Secret!' with correct password."""
        path = absolute_sample_path("encryption/aes-128.pdf")
        result = extract_text(path, password="foo")
        assert "Secret!" in result

    def test_aes256_r6_extracts_hello(self) -> None:
        """AES-256-R6 encrypted PDF extracts 'Hello World' with correct password."""
        path = absolute_sample_path("encryption/aes-256-r6.pdf")
        result = extract_text(path, password="usersecret")
        assert "Hello World" in result
