"""Tests for Rust-accelerated XRef parsing helpers and end-to-end PDF loading."""

import pytest

from tests.helpers import absolute_sample_path


class TestRustXrefImport:
    def test_rust_extension_importable(self) -> None:
        """pdfminer_core must be importable and expose the xref helpers."""
        import pdfminer_core  # type: ignore[import]

        assert hasattr(pdfminer_core, "parse_xref_table")
        assert hasattr(pdfminer_core, "scan_pdf_objects")
        assert hasattr(pdfminer_core, "parse_xref_stream")

    def test_has_rust_flag_is_true(self) -> None:
        from pdfminer.pdfdocument import _HAS_RUST  # type: ignore[attr-defined]

        assert _HAS_RUST is True, "Rust extension should be available in this test run"


class TestParseXrefTable:
    def test_basic_table(self) -> None:
        from pdfminer_core import parse_xref_table  # type: ignore[import]

        xref_bytes = (
            b"0 3\n"
            b"0000000000 65535 f\n"
            b"0000000015 00000 n\n"
            b"0000000068 00000 n\n"
        )
        entries = parse_xref_table(xref_bytes)
        # All three entries returned, including free
        assert len(entries) == 3
        # Free entry (objid=0)
        assert entries[0] == (0, 65535, 0, False)
        # In-use entries
        assert entries[1] == (1, 0, 15, True)
        assert entries[2] == (2, 0, 68, True)

    def test_multiple_sections(self) -> None:
        from pdfminer_core import parse_xref_table  # type: ignore[import]

        xref_bytes = (
            b"0 2\n"
            b"0000000000 65535 f\n"
            b"0000000100 00000 n\n"
            b"5 2\n"
            b"0000000200 00000 n\n"
            b"0000000300 00000 n\n"
        )
        entries = parse_xref_table(xref_bytes)
        assert len(entries) == 4
        objids = [e[0] for e in entries]
        assert objids == [0, 1, 5, 6]

    def test_stops_at_trailer(self) -> None:
        from pdfminer_core import parse_xref_table  # type: ignore[import]

        xref_bytes = (
            b"0 1\n"
            b"0000000000 65535 f\n"
            b"trailer\n"
            b"<</Size 1>>\n"
        )
        entries = parse_xref_table(xref_bytes)
        assert len(entries) == 1

    def test_empty_input(self) -> None:
        from pdfminer_core import parse_xref_table  # type: ignore[import]

        assert parse_xref_table(b"") == []

    def test_xref_keyword_skipped(self) -> None:
        from pdfminer_core import parse_xref_table  # type: ignore[import]

        xref_bytes = b"xref\n0 1\n0000000015 00000 n\n"
        entries = parse_xref_table(xref_bytes)
        assert len(entries) == 1
        assert entries[0] == (0, 0, 15, True)


class TestScanPdfObjects:
    def test_basic_scan(self) -> None:
        from pdfminer_core import scan_pdf_objects  # type: ignore[import]

        data = b"some garbage\n1 0 obj\n<</Type /Catalog>>\nendobj\n2 0 obj\nstream\nendstream\nendobj\n"
        results = scan_pdf_objects(data)
        assert len(results) == 2
        objids = [r[0] for r in results]
        assert 1 in objids
        assert 2 in objids

    def test_word_boundary(self) -> None:
        """'objfoo' should NOT match as an object header."""
        from pdfminer_core import scan_pdf_objects  # type: ignore[import]

        data = b"1 0 objfoo\n2 0 obj\n"
        results = scan_pdf_objects(data)
        assert len(results) == 1
        assert results[0][0] == 2

    def test_correct_byte_offset(self) -> None:
        from pdfminer_core import scan_pdf_objects  # type: ignore[import]

        data = b"garbage\n3 0 obj\n"
        results = scan_pdf_objects(data)
        assert len(results) == 1
        offset = results[0][2]
        # Offset should point to the start of the line containing "3 0 obj"
        assert data[offset:].startswith(b"3 0 obj")

    def test_empty_file(self) -> None:
        from pdfminer_core import scan_pdf_objects  # type: ignore[import]

        assert scan_pdf_objects(b"") == []

    def test_no_matches(self) -> None:
        from pdfminer_core import scan_pdf_objects  # type: ignore[import]

        assert scan_pdf_objects(b"%%PDF-1.4\nsome data\n") == []


class TestParseXrefStream:
    def _make_entry(self, entry_type: int, f2: int, f3: int, widths: list[int]) -> bytes:
        """Encode a single xref stream entry."""
        import struct

        parts = []
        for val, width in zip([entry_type, f2, f3], widths):
            if width == 0:
                continue
            parts.append(val.to_bytes(width, "big"))
        return b"".join(parts)

    def test_type1_entry(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        # Type 1: in-use object at file offset
        w = [1, 4, 2]
        data = self._make_entry(1, 1234, 0, w)
        results = parse_xref_stream(data, w, [0, 1])
        assert len(results) == 1
        entry_type, objid, f2, f3 = results[0]
        assert entry_type == 1
        assert objid == 0
        assert f2 == 1234
        assert f3 == 0

    def test_type2_entry(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        # Type 2: compressed object
        w = [1, 2, 2]
        data = self._make_entry(2, 5, 3, w)
        results = parse_xref_stream(data, w, [10, 1])
        assert len(results) == 1
        entry_type, objid, f2, f3 = results[0]
        assert entry_type == 2
        assert objid == 10
        assert f2 == 5
        assert f3 == 3

    def test_type0_free_entry(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        # Type 0: free object
        w = [1, 4, 2]
        data = self._make_entry(0, 0, 65535, w)
        results = parse_xref_stream(data, w, [0, 1])
        assert results[0][0] == 0  # type 0

    def test_default_type_when_w1_zero(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        # If w[0] == 0, the type defaults to 1 per PDF spec
        w = [0, 4, 2]
        # Only f2 and f3 fields
        data = (5678).to_bytes(4, "big") + (0).to_bytes(2, "big")
        results = parse_xref_stream(data, w, [0, 1])
        assert results[0][0] == 1  # default type

    def test_multiple_index_ranges(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        w = [1, 4, 2]
        entry_a = self._make_entry(1, 100, 0, w)
        entry_b = self._make_entry(1, 200, 0, w)
        data = entry_a + entry_b
        results = parse_xref_stream(data, w, [0, 1, 5, 1])
        assert len(results) == 2
        assert results[0][1] == 0  # objid from first range
        assert results[1][1] == 5  # objid from second range

    def test_wrong_w_length_raises(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        with pytest.raises(ValueError):
            parse_xref_stream(b"\x01\x00\x00\x00\x01\x00\x00", [1, 4], [0, 1])

    def test_data_not_divisible_raises(self) -> None:
        from pdfminer_core import parse_xref_stream  # type: ignore[import]

        with pytest.raises(ValueError):
            # entry_size = 7, data length = 5 → not divisible
            parse_xref_stream(b"\x01\x00\x00\x00\x01", [1, 4, 2], [0, 1])


class TestEndToEnd:
    """Verify that real PDFs parse correctly with Rust acceleration active."""

    @pytest.mark.parametrize(
        "pdf_name",
        ["simple1.pdf", "simple2.pdf", "jo.pdf"],
    )
    def test_parse_sample_pdfs(self, pdf_name: str) -> None:
        from pdfminer.pdfpage import PDFPage

        path = absolute_sample_path(pdf_name)
        with open(path, "rb") as f:
            pages = list(PDFPage.get_pages(f))
        assert len(pages) > 0, f"Failed to parse {pdf_name}"

    def test_encrypted_pdf(self) -> None:
        from pdfminer.pdfpage import PDFPage

        path = absolute_sample_path("encryption/aes-256.pdf")
        with open(path, "rb") as f:
            pages = list(PDFPage.get_pages(f, password="foo"))
        assert len(pages) > 0

    def test_extract_text(self) -> None:
        from pdfminer.high_level import extract_text

        text = extract_text(absolute_sample_path("simple1.pdf"))
        assert len(text) > 0, "Expected non-empty text from simple1.pdf"
