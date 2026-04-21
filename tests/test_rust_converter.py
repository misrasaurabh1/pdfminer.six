"""Tests for Rust converter acceleration."""
import io

import pytest

from pdfminer.converter import HTMLConverter, PDFPageAggregator, TextConverter, XMLConverter
from pdfminer.layout import LAParams, LTPage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from tests.helpers import absolute_sample_path


def get_page_aggregator_result(pdf_path: str) -> list[LTPage]:
    """Get LTPage list from PDFPageAggregator for a PDF."""
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    pages = []
    with open(pdf_path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
            layout = device.get_result()
            pages.append(layout)
    return pages


def test_page_aggregator_returns_ltpage() -> None:
    """PDFPageAggregator.get_result() must return LTPage (critical for unstructured)."""
    path = absolute_sample_path("simple1.pdf")
    pages = get_page_aggregator_result(path)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, LTPage)


def test_text_converter() -> None:
    """TextConverter must produce text output."""
    path = absolute_sample_path("simple1.pdf")
    rsrcmgr = PDFResourceManager()
    output = io.StringIO()
    device = TextConverter(rsrcmgr, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
    text = output.getvalue()
    assert len(text) > 0


def test_xml_converter() -> None:
    """XMLConverter must produce valid XML output."""
    path = absolute_sample_path("simple1.pdf")
    rsrcmgr = PDFResourceManager()
    output = io.BytesIO()
    device = XMLConverter(rsrcmgr, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
    device.close()
    xml_text = output.getvalue().decode("utf-8")
    assert "<pages>" in xml_text or "<page" in xml_text


def test_html_converter() -> None:
    """HTMLConverter must produce HTML output."""
    path = absolute_sample_path("simple1.pdf")
    rsrcmgr = PDFResourceManager()
    output = io.BytesIO()
    device = HTMLConverter(rsrcmgr, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
    device.close()
    html_text = output.getvalue().decode("utf-8")
    assert "<html" in html_text
    assert len(html_text) > 0


def test_extract_text_to_fp_formats() -> None:
    """extract_text_to_fp must work for all output types."""
    from pdfminer.high_level import extract_text_to_fp

    path = absolute_sample_path("simple1.pdf")
    for output_type in ["text", "xml", "html"]:
        if output_type == "text":
            output: io.StringIO | io.BytesIO = io.StringIO()
        else:
            output = io.BytesIO()
        with open(path, "rb") as f:
            extract_text_to_fp(f, output, output_type=output_type, laparams=LAParams())
        result = output.getvalue()
        assert len(result) > 0, f"Empty output for {output_type}"


def test_rust_xml_escape() -> None:
    """Rust xml_escape must handle special characters correctly."""
    try:
        from pdfminer_core import xml_escape
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not available")

    assert xml_escape("Hello") == "Hello"
    assert xml_escape("a&b") == "a&amp;b"
    assert xml_escape("<tag>") == "&lt;tag&gt;"
    assert xml_escape('"quoted"') == "&quot;quoted&quot;"
    assert xml_escape("") == ""


def test_rust_format_bbox() -> None:
    """Rust format_bbox must produce correct decimal output."""
    try:
        from pdfminer_core import format_bbox
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not available")

    result = format_bbox((0.0, 1.0, 100.0, 200.0))
    assert result == "0.000,1.000,100.000,200.000"

    result2 = format_bbox((1.23456, 2.34567, 3.45678, 4.56789))
    assert result2 == "1.235,2.346,3.457,4.568"


def test_rust_build_text_output() -> None:
    """Rust build_text_output must join text items."""
    try:
        from pdfminer_core import build_text_output
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not available")

    items = [
        ("Hello ", (0.0, 100.0, 50.0, 110.0)),
        ("World", (50.0, 100.0, 100.0, 110.0)),
    ]
    result = build_text_output(items, 200.0, 300.0)
    assert "Hello" in result
    assert "World" in result
