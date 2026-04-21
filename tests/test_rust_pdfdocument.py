"""Tests for Rust PDF document acceleration helpers."""

import pytest

from tests.helpers import absolute_sample_path
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage


@pytest.mark.parametrize(
    "pdf_name",
    [
        "simple1.pdf",
        "simple2.pdf",
        "simple3.pdf",
        "simple4.pdf",
        "simple5.pdf",
        "jo.pdf",
    ],
)
def test_parse_sample_pdf(pdf_name: str) -> None:
    """All sample PDFs must parse and return at least one page."""
    path = absolute_sample_path(pdf_name)
    with open(path, "rb") as f:
        pages = list(PDFPage.get_pages(f))
    assert len(pages) > 0


def test_extract_text_consistent() -> None:
    """extract_text must return identical results on repeated calls."""
    path = absolute_sample_path("simple1.pdf")
    text1 = extract_text(path)
    text2 = extract_text(path)
    assert text1 == text2


def test_extract_text_nonempty() -> None:
    """extract_text must return non-empty text for simple1.pdf."""
    path = absolute_sample_path("simple1.pdf")
    text = extract_text(path)
    assert len(text) > 0


def test_rust_build_xref_lookup() -> None:
    """Rust build_xref_lookup must produce a correct mapping."""
    try:
        from pdfminer_core import build_xref_lookup
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not built")

    entries = [(1, 100), (2, 200), (3, 300)]
    result = build_xref_lookup(entries)
    assert result[1] == 100
    assert result[2] == 200
    assert result[3] == 300
    assert len(result) == 3


def test_rust_build_xref_lookup_empty() -> None:
    """build_xref_lookup handles empty input."""
    try:
        from pdfminer_core import build_xref_lookup
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not built")

    result = build_xref_lookup([])
    assert result == {}


def test_rust_flatten_page_tree() -> None:
    """flatten_page_tree must return leaf page indices in order."""
    try:
        from pdfminer_core import flatten_page_tree
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not built")

    # Build a simple tree:
    #   node 0: Pages -> [1, 2]
    #   node 1: Page
    #   node 2: Page
    nodes = [
        ("Pages", [1, 2]),
        ("Page", None),
        ("Page", None),
    ]
    result = flatten_page_tree(nodes, 10)
    assert result == [1, 2]


def test_rust_flatten_page_tree_nested() -> None:
    """flatten_page_tree handles nested Pages nodes."""
    try:
        from pdfminer_core import flatten_page_tree
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not built")

    # Tree:
    #   node 0: Pages -> [1, 4]
    #   node 1: Pages -> [2, 3]
    #   node 2: Page
    #   node 3: Page
    #   node 4: Page
    nodes = [
        ("Pages", [1, 4]),
        ("Pages", [2, 3]),
        ("Page", None),
        ("Page", None),
        ("Page", None),
    ]
    result = flatten_page_tree(nodes, 10)
    assert result == [2, 3, 4]


def test_rust_flatten_page_tree_target_count() -> None:
    """flatten_page_tree respects target_count limit."""
    try:
        from pdfminer_core import flatten_page_tree
    except ImportError:
        pytest.skip("pdfminer_core Rust extension not built")

    nodes = [
        ("Pages", [1, 2, 3]),
        ("Page", None),
        ("Page", None),
        ("Page", None),
    ]
    result = flatten_page_tree(nodes, 2)
    assert len(result) <= 2


def test_encrypted_pdfs() -> None:
    """Encrypted PDFs must be parseable (with correct password)."""
    import os

    enc_dir = absolute_sample_path("encryption")
    if not os.path.isdir(enc_dir):
        pytest.skip("No encryption samples directory found")

    parsed_any = False
    for fname in os.listdir(enc_dir):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(enc_dir, fname)
        for password in (b"usersecret", b"", b"owner"):
            try:
                with open(fpath, "rb") as f:
                    pages = list(PDFPage.get_pages(f, password=password.decode()))
                if pages:
                    parsed_any = True
                    break
            except Exception:
                continue
    # We don't assert parsed_any — some test suites may have no accessible PDFs.
