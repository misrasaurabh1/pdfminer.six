"""Benchmark PS tokenizer and PDF object parsing."""

import os
import timeit
from io import BytesIO
from pathlib import Path

from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import PSBaseParser

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_SAMPLES = _REPO / "samples"

_DEFAULT_PDF = str(_SAMPLES / "simple1.pdf")


def _make_ps_stream(size_kb: int = 64) -> bytes:
    """Build a synthetic PS token stream of approximately `size_kb` KB."""
    chunk = (
        b"BT\n"
        b"/F1 12 Tf\n"
        b"100 700 Td\n"
        b"(Hello World) Tj\n"
        b"0.5 0.5 0.5 rg\n"
        b"200 600 100 50 re\n"
        b"S\n"
        b"[1 2 3 4 5] array\n"
        b"<<\n"
        b"  /Type /Font\n"
        b"  /Subtype /Type1\n"
        b"  /BaseFont /Helvetica\n"
        b">>\n"
        b"ET\n"
    )
    repeats = max(1, (size_kb * 1024) // len(chunk))
    return chunk * repeats


_PS_STREAM = _make_ps_stream(64)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_tokenize_stream(data: bytes = _PS_STREAM, iterations: int = 10) -> float:
    """Benchmark tokenizing a PS/PDF content stream. Returns ms per call."""

    def run() -> None:
        fp = BytesIO(data)
        parser = PSBaseParser(fp)
        while True:
            try:
                parser.nexttoken()
            except PSEOF:
                break

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


def bench_parse_pdf_objects(pdf_path: str = _DEFAULT_PDF, iterations: int = 10) -> float:
    """Benchmark full PDF object parsing (open file, read xref + objects). Returns ms."""
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    def run() -> None:
        with open(pdf_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            for page in PDFPage.create_pages(doc):
                _ = page.pageid

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Parsing Benchmarks ===")
    print(f"PS stream size: {len(_PS_STREAM):>8} bytes")
    print()

    results = [
        ("Tokenize stream (64KB)", bench_tokenize_stream()),
        (f"Parse PDF objects ({os.path.basename(_DEFAULT_PDF)})", bench_parse_pdf_objects()),
    ]

    print(f"{'Benchmark':<40} | {'ms/call':>10}")
    print("-" * 55)
    for name, ms in results:
        print(f"{name:<40} | {ms:>10.3f}")
