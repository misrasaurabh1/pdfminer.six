"""Tests for Rust PDF document acceleration helpers."""

import os

import pytest

from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from tests.helpers import absolute_sample_path

try:
    from pdfminer_core import build_xref_lookup, flatten_page_tree

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

needs_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core not built")


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
    assert extract_text(path) == extract_text(path)


def test_extract_text_nonempty() -> None:
    """extract_text must return non-empty text for simple1.pdf."""
    assert len(extract_text(absolute_sample_path("simple1.pdf"))) > 0


@needs_rust
def test_rust_build_xref_lookup() -> None:
    """build_xref_lookup produces a correct mapping."""
    result = build_xref_lookup([(1, 100), (2, 200), (3, 300)])
    assert result == {1: 100, 2: 200, 3: 300}


@needs_rust
def test_rust_build_xref_lookup_empty() -> None:
    """build_xref_lookup handles empty input."""
    assert build_xref_lookup([]) == {}


@needs_rust
def test_rust_flatten_page_tree() -> None:
    """flatten_page_tree returns leaf page indices in order."""
    # node 0: Pages -> [1, 2]; nodes 1, 2: Page
    nodes = [("Pages", [1, 2]), ("Page", None), ("Page", None)]
    assert flatten_page_tree(nodes, 10) == [1, 2]


@needs_rust
def test_rust_flatten_page_tree_nested() -> None:
    """flatten_page_tree handles nested Pages nodes."""
    # node 0: Pages -> [1, 4]; node 1: Pages -> [2, 3]; nodes 2-4: Page
    nodes = [
        ("Pages", [1, 4]),
        ("Pages", [2, 3]),
        ("Page", None),
        ("Page", None),
        ("Page", None),
    ]
    assert flatten_page_tree(nodes, 10) == [2, 3, 4]


@needs_rust
def test_rust_flatten_page_tree_target_count() -> None:
    """flatten_page_tree respects target_count limit."""
    nodes = [("Pages", [1, 2, 3]), ("Page", None), ("Page", None), ("Page", None)]
    assert len(flatten_page_tree(nodes, 2)) <= 2


def test_encrypted_pdfs() -> None:
    """Encrypted PDFs must be parseable (with correct password)."""
    enc_dir = absolute_sample_path("encryption")
    if not os.path.isdir(enc_dir):
        pytest.skip("No encryption samples directory found")

    for fname in os.listdir(enc_dir):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(enc_dir, fname)
        for password in ("usersecret", "", "owner"):
            try:
                with open(fpath, "rb") as f:
                    list(PDFPage.get_pages(f, password=password))
                break
            except Exception:
                continue
