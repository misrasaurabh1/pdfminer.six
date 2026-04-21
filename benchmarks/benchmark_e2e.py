"""End-to-end benchmarks: full PDF processing pipeline."""

import timeit
from io import BytesIO
from pathlib import Path

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LAParams

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_SAMPLES = _REPO / "samples"

SAMPLE_PDFS = [
    str(_SAMPLES / "simple1.pdf"),
    str(_SAMPLES / "simple2.pdf"),
    str(_SAMPLES / "jo.pdf"),
    str(_SAMPLES / "font-size-test.pdf"),
]


def bench_extract_text(pdf_path: str = SAMPLE_PDFS[0], iterations: int = 5) -> tuple[float, int]:
    """Benchmark extract_text. Returns (ms per call, chars extracted)."""
    # Warm-up pass to measure char count
    text = extract_text(pdf_path)
    char_count = len(text)

    total = timeit.timeit(lambda: extract_text(pdf_path), number=iterations)
    return total / iterations * 1000, char_count


def bench_extract_pages(pdf_path: str = SAMPLE_PDFS[0], iterations: int = 5) -> tuple[float, int]:
    """Benchmark extract_pages + full iteration. Returns (ms per call, elements)."""

    def run() -> int:
        count = 0
        for page in extract_pages(pdf_path):
            for element in page:
                count += 1
        return count

    # Warm-up to get element count
    element_count = run()

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000, element_count


def bench_extract_text_to_xml(pdf_path: str = SAMPLE_PDFS[0], iterations: int = 5) -> float:
    """Benchmark XML output via extract_text_to_fp. Returns ms per call."""
    from pdfminer.high_level import extract_text_to_fp

    def run() -> None:
        out = BytesIO()
        with open(pdf_path, "rb") as f:
            extract_text_to_fp(f, out, output_type="xml", codec="utf-8", laparams=LAParams())

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=== E2E Benchmarks ===")
    print()

    existing = [p for p in SAMPLE_PDFS if Path(p).exists()]
    if not existing:
        print("No sample PDFs found. Run from the repo root or adjust SAMPLE_PDFS.")
    else:
        print(f"{'PDF':<30} | {'extract_text ms':>15} | {'chars':>8} | {'extract_pages ms':>16} | {'elements':>8}")
        print("-" * 85)
        for pdf in existing:
            name = os.path.basename(pdf)
            try:
                ms_text, chars = bench_extract_text(pdf)
                ms_pages, elems = bench_extract_pages(pdf)
                print(f"{name:<30} | {ms_text:>15.1f} | {chars:>8} | {ms_pages:>16.1f} | {elems:>8}")
            except Exception as exc:
                print(f"{name:<30} | ERROR: {exc}")
