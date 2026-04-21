"""Benchmark layout analysis: matrix ops, spatial plane, text grouping."""

import random
import timeit
from pathlib import Path

from pdfminer.layout import LAParams
from pdfminer.utils import Plane, apply_matrix_pt, mult_matrix

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_SAMPLES = _REPO / "samples"

_DEFAULT_PDF = str(_SAMPLES / "simple1.pdf")

_M1 = (0.9659, 0.2588, -0.2588, 0.9659, 100.0, 200.0)
_M2 = (1.0, 0.0, 0.0, 1.0, 50.0, 75.0)
_PT = (3.0, 4.0)


class _Box:
    """Minimal bounding-box object compatible with Plane's interface."""

    __hash__ = object.__hash__

    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def _make_boxes(n: int = 1_000) -> list[_Box]:
    rng = random.Random(42)
    boxes = []
    for _ in range(n):
        x0 = rng.uniform(0, 570)
        y0 = rng.uniform(0, 780)
        x1 = x0 + rng.uniform(5, 42)   # absolute right edge, always > x0
        y1 = y0 + rng.uniform(5, 12)   # absolute top edge, always > y0
        boxes.append(_Box(x0, y0, x1, y1))
    return boxes


_BOXES = _make_boxes()
_BBOX_PAGE = (0.0, 0.0, 612.0, 792.0)
_QUERY_BBOX = (100.0, 100.0, 400.0, 600.0)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_matrix_ops(iterations: int = 10_000) -> float:
    """Benchmark 10k matrix multiplications + point transforms. Returns ms total."""

    def run() -> None:
        m = _M1
        for _ in range(iterations):
            m = mult_matrix(_M2, m)
            apply_matrix_pt(m, _PT)

    return timeit.timeit(run, number=1) * 1000


def bench_plane_add_lookup(iterations: int = 100) -> float:
    """Benchmark spatial Plane with 1000 objects: add + find. Returns ms per iteration."""
    boxes = _BOXES

    def run() -> None:
        plane: Plane[_Box] = Plane(_BBOX_PAGE)  # type: ignore[type-var]
        for box in boxes:
            plane.add(box)
        list(plane.find(_QUERY_BBOX))

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


def bench_layout_analysis(pdf_path: str = _DEFAULT_PDF, iterations: int = 10) -> float:
    """Benchmark full layout analysis of all pages. Returns ms per call."""
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    laparams = LAParams()

    def run() -> None:
        with open(pdf_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                device.get_result()

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=== Layout Benchmarks ===")
    print()

    results = [
        ("Matrix ops (10k mult+apply)", bench_matrix_ops()),
        ("Plane add+lookup (1000 objs)", bench_plane_add_lookup()),
        (f"Layout analysis ({os.path.basename(_DEFAULT_PDF)})", bench_layout_analysis()),
    ]

    print(f"{'Benchmark':<40} | {'ms':>10}")
    print("-" * 55)
    for name, ms in results:
        print(f"{name:<40} | {ms:>10.3f}")
