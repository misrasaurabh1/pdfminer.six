"""
Benchmarks: end-to-end PDF text extraction.

Uses real PDF files from the samples/ directory to measure full-pipeline
performance.  Because the pipeline calls many Rust-accelerated functions
in combination, the end-to-end benchmark captures total throughput rather
than the per-function micro-benchmark in other modules.

Two measurements are taken for each PDF:
  - "rust"   — normal import, which routes to Rust fast-paths automatically
  - "python" — Rust fast-paths disabled by monkey-patching _HAS_RUST=False
               across every module that checks it

This gives a realistic comparison of the impact of the Rust optimisations
on actual workloads.
"""

from __future__ import annotations

import importlib
import io
import pathlib

from benchmarks.bench_utils import BenchmarkSuite

# ---------------------------------------------------------------------------
# Modules that gate on _HAS_RUST; we need to flip them all to disable Rust.
# ---------------------------------------------------------------------------
_RUST_GATED_MODULES = [
    "pdfminer.utils",
    "pdfminer.lzw",
    "pdfminer.ascii85",
    "pdfminer.runlength",
    "pdfminer.ccitt",
    "pdfminer.pdfdocument",
    "pdfminer.layout",
    "pdfminer.pdfinterp",
    "pdfminer.converter",
]


def _set_has_rust(value: bool) -> None:
    """Flip _HAS_RUST in every gated module."""
    for mod_name in _RUST_GATED_MODULES:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "_HAS_RUST"):
            mod._HAS_RUST = value  # noqa: SLF001


def _extract_text(path: pathlib.Path) -> str:
    """Extract all text from *path* using the high-level pdfminer API."""
    from pdfminer.high_level import extract_text

    return extract_text(str(path))


# ---------------------------------------------------------------------------
# Sample PDFs — relative to the repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_SAMPLE_DIR = _REPO_ROOT / "samples"

_PDFS: list[tuple[str, pathlib.Path]] = [
    ("simple1",       _SAMPLE_DIR / "simple1.pdf"),
    ("simple2",       _SAMPLE_DIR / "simple2.pdf"),
    ("simple3",       _SAMPLE_DIR / "simple3.pdf"),
    ("simple4",       _SAMPLE_DIR / "simple4.pdf"),
    ("simple5",       _SAMPLE_DIR / "simple5.pdf"),
    ("jo",            _SAMPLE_DIR / "jo.pdf"),
    ("font_size",     _SAMPLE_DIR / "font-size-test.pdf"),
    # Larger contrib files — only included if they exist.
    ("ascii85_large", _SAMPLE_DIR / "contrib" / "issue-1008-inline-ascii85.pdf"),
    ("colour_space",  _SAMPLE_DIR / "contrib" / "issue-1061-colour-space-stack.pdf"),
    ("cmap_other",    _SAMPLE_DIR / "contrib" / "issue-598-cmap-other-fonts.pdf"),
]


# ---------------------------------------------------------------------------
# Register benchmarks
# ---------------------------------------------------------------------------

def register(suite: BenchmarkSuite) -> None:
    """Add end-to-end extraction benchmarks to *suite*."""
    for label, path in _PDFS:
        if not path.exists():
            continue

        group = f"e2e_{label}"
        name = f"e2e extract {label}"

        # Python-only baseline (Rust disabled)
        def _py_fn(p: pathlib.Path = path) -> str:
            _set_has_rust(False)
            try:
                return _extract_text(p)
            finally:
                _set_has_rust(True)

        suite.add(
            group=group,
            variant="python",
            name=f"{name} [py]",
            fn=_py_fn,
        )

        # Rust-accelerated (normal)
        def _rs_fn(p: pathlib.Path = path) -> str:
            _set_has_rust(True)
            return _extract_text(p)

        suite.add(
            group=group,
            variant="rust",
            name=f"{name} [rs]",
            fn=_rs_fn,
        )
