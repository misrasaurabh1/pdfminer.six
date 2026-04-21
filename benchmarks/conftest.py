"""pytest-benchmark fixtures for pdfminer benchmark suite."""

from pathlib import Path

import pytest

_SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture
def sample_pdf_path() -> str:
    return str(_SAMPLES / "simple1.pdf")


@pytest.fixture
def sample_pdf_paths() -> list[str]:
    pdfs = ["simple1.pdf", "simple2.pdf", "jo.pdf", "font-size-test.pdf"]
    return [str(_SAMPLES / name) for name in pdfs if (_SAMPLES / name).exists()]
