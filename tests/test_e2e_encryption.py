"""E2E tests for encrypted PDF handling in pdfminer.

These tests verify decryption of AES and RC4 encrypted PDFs, and that
appropriate exceptions are raised for wrong passwords.

They serve as the correctness ground truth for the Rust port.
"""

import os

import pytest

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTPage
from pdfminer.pdfdocument import PDFDocument, PDFPasswordIncorrect
from pdfminer.pdfparser import PDFParser
from tests.helpers import absolute_sample_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(path: str) -> str:
    return absolute_sample_path(path)


def _exists(path: str) -> bool:
    return os.path.exists(_sample(path))


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class TestEncryptionExceptions:
    """Verify the encryption exception hierarchy."""

    def test_pdfpasswordincorrect_importable(self) -> None:
        """PDFPasswordIncorrect is importable from pdfminer.pdfdocument."""
        assert issubclass(PDFPasswordIncorrect, Exception)

    def test_pdfpasswordincorrect_is_pdfencryption_error(self) -> None:
        """PDFPasswordIncorrect is a subclass of PDFEncryptionError."""
        from pdfminer.pdfdocument import PDFEncryptionError

        assert issubclass(PDFPasswordIncorrect, PDFEncryptionError)

    def test_wrong_password_raises(self) -> None:
        """PDFPasswordIncorrect is raised when decrypting with wrong password."""
        with pytest.raises(PDFPasswordIncorrect):
            with open(_sample("encryption/aes-256.pdf"), "rb") as fp:
                parser = PDFParser(fp)
                PDFDocument(parser, password="definitely-wrong-password")

    def test_wrong_password_aes128_raises(self) -> None:
        """PDFPasswordIncorrect raised for AES-128 with wrong password."""
        with pytest.raises(PDFPasswordIncorrect):
            with open(_sample("encryption/aes-128.pdf"), "rb") as fp:
                parser = PDFParser(fp)
                PDFDocument(parser, password="wrongpassword")

    def test_wrong_password_rc4_40_raises(self) -> None:
        """PDFPasswordIncorrect raised for RC4-40 with wrong password."""
        with pytest.raises(PDFPasswordIncorrect):
            with open(_sample("encryption/rc4-40.pdf"), "rb") as fp:
                parser = PDFParser(fp)
                PDFDocument(parser, password="wrongpassword")

    def test_wrong_password_rc4_128_raises(self) -> None:
        """PDFPasswordIncorrect raised for RC4-128 with wrong password."""
        with pytest.raises(PDFPasswordIncorrect):
            with open(_sample("encryption/rc4-128.pdf"), "rb") as fp:
                parser = PDFParser(fp)
                PDFDocument(parser, password="wrongpassword")


# ---------------------------------------------------------------------------
# AES-256 encrypted PDFs
# ---------------------------------------------------------------------------

class TestAES256:
    """Verify AES-256 encrypted PDFs decrypt correctly."""

    def test_aes256_extract_text(self) -> None:
        """AES-256 encrypted PDF extracts 'Secret!' with correct password."""
        result = extract_text(_sample("encryption/aes-256.pdf"), password="foo")
        assert "Secret!" in result

    def test_aes256_extract_pages_returns_ltpage(self) -> None:
        """extract_pages() on AES-256 PDF returns LTPage objects."""
        pages = list(
            extract_pages(_sample("encryption/aes-256.pdf"), password="foo")
        )
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)

    def test_aes256_document_parses(self) -> None:
        """AES-256 PDF can be parsed into a PDFDocument."""
        with open(_sample("encryption/aes-256.pdf"), "rb") as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser, password="foo")
            assert doc.xrefs

    def test_aes256_r6_extract_text(self) -> None:
        """AES-256-R6 encrypted PDF extracts 'Hello World' with correct password."""
        result = extract_text(
            _sample("encryption/aes-256-r6.pdf"), password="usersecret"
        )
        assert "Hello World" in result

    def test_aes256_r6_wrong_password_raises(self) -> None:
        """PDFPasswordIncorrect is raised for AES-256-R6 with wrong password."""
        with pytest.raises(PDFPasswordIncorrect):
            with open(_sample("encryption/aes-256-r6.pdf"), "rb") as fp:
                parser = PDFParser(fp)
                PDFDocument(parser, password="badpassword")

    def test_aes256_m_extract_text(self) -> None:
        """AES-256-m PDF (metadata-only encryption) extracts 'Secret!'."""
        result = extract_text(_sample("encryption/aes-256-m.pdf"), password="foo")
        assert "Secret!" in result


# ---------------------------------------------------------------------------
# AES-128 encrypted PDFs
# ---------------------------------------------------------------------------

class TestAES128:
    """Verify AES-128 encrypted PDFs decrypt correctly."""

    def test_aes128_extract_text(self) -> None:
        """AES-128 encrypted PDF extracts 'Secret!' with correct password."""
        result = extract_text(_sample("encryption/aes-128.pdf"), password="foo")
        assert "Secret!" in result

    def test_aes128_extract_pages_returns_ltpage(self) -> None:
        """extract_pages() on AES-128 PDF returns LTPage objects."""
        pages = list(
            extract_pages(_sample("encryption/aes-128.pdf"), password="foo")
        )
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)

    def test_aes128_m_extract_text(self) -> None:
        """AES-128-m PDF (metadata-only encryption) extracts 'Secret!'."""
        result = extract_text(_sample("encryption/aes-128-m.pdf"), password="foo")
        assert "Secret!" in result


# ---------------------------------------------------------------------------
# RC4 encrypted PDFs
# ---------------------------------------------------------------------------

class TestRC4:
    """Verify RC4 encrypted PDFs decrypt correctly."""

    def test_rc4_40_extract_text(self) -> None:
        """RC4-40 encrypted PDF extracts 'Secret!' with correct password."""
        result = extract_text(_sample("encryption/rc4-40.pdf"), password="foo")
        assert "Secret!" in result

    def test_rc4_128_extract_text(self) -> None:
        """RC4-128 encrypted PDF extracts 'Secret!' with correct password."""
        result = extract_text(_sample("encryption/rc4-128.pdf"), password="foo")
        assert "Secret!" in result

    def test_rc4_40_extract_pages(self) -> None:
        """extract_pages() on RC4-40 PDF returns LTPage objects."""
        pages = list(
            extract_pages(_sample("encryption/rc4-40.pdf"), password="foo")
        )
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)

    def test_rc4_128_extract_pages(self) -> None:
        """extract_pages() on RC4-128 PDF returns LTPage objects."""
        pages = list(
            extract_pages(_sample("encryption/rc4-128.pdf"), password="foo")
        )
        assert len(pages) == 1
        assert isinstance(pages[0], LTPage)


# ---------------------------------------------------------------------------
# Encrypted PDF without ID (edge case)
# ---------------------------------------------------------------------------

class TestEncryptedDocNoId:
    """Verify handling of encrypted PDF without /ID key in trailer."""

    def test_no_id_doc_opens(self) -> None:
        """encrypted_doc_no_id.pdf can be opened with empty password."""
        with open(_sample("encryption/encrypted_doc_no_id.pdf"), "rb") as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
        # Should parse without exception
        assert doc.info is not None

    def test_no_id_doc_info(self) -> None:
        """encrypted_doc_no_id.pdf has 'European Patent Office' as Producer."""
        with open(_sample("encryption/encrypted_doc_no_id.pdf"), "rb") as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
        assert doc.info == [{"Producer": b"European Patent Office"}]


# ---------------------------------------------------------------------------
# Unencrypted base PDF
# ---------------------------------------------------------------------------

class TestBasePdf:
    """Verify the unencrypted base.pdf (or its encrypted version)."""

    def test_base_pdf_extract_text(self) -> None:
        """encryption/base.pdf extracts 'Secret!' with password 'usersecret'."""
        path = _sample("encryption/base.pdf")
        if not os.path.exists(path):
            pytest.skip("base.pdf not present")
        result = extract_text(path, password="usersecret")
        assert "Secret!" in result


# ---------------------------------------------------------------------------
# Parametrized: all encrypted PDFs
# ---------------------------------------------------------------------------

ENCRYPTED_FILES = [
    ("encryption/aes-256.pdf", "foo", "Secret!"),
    ("encryption/aes-256-m.pdf", "foo", "Secret!"),
    ("encryption/aes-256-r6.pdf", "usersecret", "Hello World"),
    ("encryption/aes-128.pdf", "foo", "Secret!"),
    ("encryption/aes-128-m.pdf", "foo", "Secret!"),
    ("encryption/rc4-40.pdf", "foo", "Secret!"),
    ("encryption/rc4-128.pdf", "foo", "Secret!"),
]


@pytest.mark.parametrize("rel_path,password,expected_text", ENCRYPTED_FILES)
def test_encrypted_pdf_roundtrip(
    rel_path: str, password: str, expected_text: str
) -> None:
    """Each encrypted PDF decrypts and yields the expected text."""
    abs_path = _sample(rel_path)
    if not os.path.exists(abs_path):
        pytest.skip(f"{rel_path} not found")
    result = extract_text(abs_path, password=password)
    assert expected_text in result


@pytest.mark.parametrize(
    "rel_path,correct_password",
    [
        ("encryption/aes-256.pdf", "foo"),
        ("encryption/aes-128.pdf", "foo"),
        ("encryption/rc4-40.pdf", "foo"),
        ("encryption/rc4-128.pdf", "foo"),
        ("encryption/aes-256-r6.pdf", "usersecret"),
    ],
)
def test_wrong_password_raises(rel_path: str, correct_password: str) -> None:
    """A wrong password always raises PDFPasswordIncorrect."""
    abs_path = _sample(rel_path)
    if not os.path.exists(abs_path):
        pytest.skip(f"{rel_path} not found")
    with pytest.raises(PDFPasswordIncorrect):
        with open(abs_path, "rb") as fp:
            parser = PDFParser(fp)
            PDFDocument(parser, password="not-the-right-password-xyz")
