"""
pytest-compatible benchmark tests.

These tests verify two things:
  1. Correctness: Rust and Python implementations produce byte-identical output.
  2. Runnability: All benchmark cases can be executed without error.

They do NOT assert specific timing ratios — that is left to the standalone
``run_benchmarks.py`` runner.  This makes the tests safe to run in CI on
machines with variable load.

Run with:
    uv run pytest benchmarks/test_benchmarks.py -v
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pdfminer_core as _core

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _RUST_AVAILABLE, reason="pdfminer_core not built")


# ---------------------------------------------------------------------------
# Correctness: predictor
# ---------------------------------------------------------------------------

class TestPredictorCorrectness:
    """Rust and Python predictors must produce identical output."""

    def test_png_payloads_match(self) -> None:
        from benchmarks.bench_predictor import _PNG_PAYLOADS, _py_png, _rs_png

        for key, (pred, colors, columns, bpc, data) in _PNG_PAYLOADS.items():
            py = _py_png(pred, colors, columns, bpc, data)
            rs = _rs_png(pred, colors, columns, bpc, data)
            assert py == rs, f"Mismatch for PNG payload {key!r}"

    def test_tiff_payloads_match(self) -> None:
        from benchmarks.bench_predictor import _TIFF_PAYLOADS, _py_tiff, _rs_tiff

        for key, (colors, columns, bpc, data) in _TIFF_PAYLOADS.items():
            py = _py_tiff(colors, columns, bpc, data)
            rs = _rs_tiff(colors, columns, bpc, data)
            assert py == rs, f"Mismatch for TIFF payload {key!r}"


# ---------------------------------------------------------------------------
# Correctness: LZW
# ---------------------------------------------------------------------------

class TestLzwCorrectness:
    """Rust and Python LZW decoders must produce identical output."""

    def test_all_payloads_match(self) -> None:
        from benchmarks.bench_lzw import (
            _PAYLOADS,
            _py_lzwdecode,
            _rs_lzwdecode,
        )

        for label, encoded in _PAYLOADS:
            py = _py_lzwdecode(encoded)
            rs = bytes(_rs_lzwdecode(encoded))
            assert py == rs, f"Mismatch for LZW payload {label!r}"


# ---------------------------------------------------------------------------
# Correctness: matrix operations
# ---------------------------------------------------------------------------

class TestMatrixCorrectness:
    """Rust matrix ops must return values within floating-point epsilon of Python."""

    _EPS = 1e-10

    def _close(self, a: tuple, b: tuple) -> bool:
        return all(abs(x - y) < self._EPS for x, y in zip(a, b))

    def test_mult_matrix(self) -> None:
        from benchmarks.bench_matrix import _MATRIX_PAIRS, _py

        for m1, m0 in _MATRIX_PAIRS[:50]:
            py = _py("mult_matrix", m1, m0)
            rs = _core.mult_matrix(m1, m0)
            assert self._close(py, rs), f"mult_matrix mismatch: {py} vs {rs}"

    def test_translate_matrix(self) -> None:
        from benchmarks.bench_matrix import _MATRIX_POINTS, _py

        for m, pt in _MATRIX_POINTS[:50]:
            py = _py("translate_matrix", m, pt)
            rs = _core.translate_matrix(m, pt)
            assert self._close(py, rs), f"translate_matrix mismatch"

    def test_apply_matrix_pt(self) -> None:
        from benchmarks.bench_matrix import _MATRIX_POINTS, _py

        for m, pt in _MATRIX_POINTS[:50]:
            py = _py("apply_matrix_pt", m, pt)
            rs = _core.apply_matrix_pt(m, pt)
            assert self._close(py, rs), f"apply_matrix_pt mismatch"

    def test_apply_matrix_norm(self) -> None:
        from benchmarks.bench_matrix import _MATRIX_POINTS, _py

        for m, pt in _MATRIX_POINTS[:50]:
            py = _py("apply_matrix_norm", m, pt)
            rs = _core.apply_matrix_norm(m, pt)
            assert self._close(py, rs), f"apply_matrix_norm mismatch"

    def test_apply_matrix_rect(self) -> None:
        from benchmarks.bench_matrix import _MATRIX_RECTS, _py

        for m, rect in _MATRIX_RECTS[:50]:
            py = _py("apply_matrix_rect", m, rect)
            rs = _core.apply_matrix_rect(m, rect)
            assert self._close(py, rs), f"apply_matrix_rect mismatch"


# ---------------------------------------------------------------------------
# Correctness: xref parsing
# ---------------------------------------------------------------------------

class TestXrefCorrectness:
    """Rust xref parsers must return the same entries as Python equivalents."""

    def test_parse_xref_table_small(self) -> None:
        from benchmarks.bench_xref import (
            _TABLE_SMALL,
            _py_parse_xref_table,
        )
        from pdfminer_core import parse_xref_table

        py = _py_parse_xref_table(_TABLE_SMALL)
        rs = [(objid, genno, offset) for objid, genno, offset, _in_use in parse_xref_table(_TABLE_SMALL)]
        assert py == rs

    def test_parse_xref_table_large(self) -> None:
        from benchmarks.bench_xref import (
            _TABLE_LARGE,
            _py_parse_xref_table,
        )
        from pdfminer_core import parse_xref_table

        py = _py_parse_xref_table(_TABLE_LARGE)
        rs = [(objid, genno, offset) for objid, genno, offset, _in_use in parse_xref_table(_TABLE_LARGE)]
        assert py == rs

    def test_parse_xref_stream_small(self) -> None:
        from benchmarks.bench_xref import (
            _IDX_S,
            _STREAM_SMALL,
            _STREAM_W,
            _py_parse_xref_stream,
        )
        from pdfminer_core import parse_xref_stream

        py = _py_parse_xref_stream(_STREAM_SMALL, _STREAM_W, _IDX_S)
        rs = list(parse_xref_stream(_STREAM_SMALL, _STREAM_W, _IDX_S))
        assert py == rs

    def test_scan_pdf_objects_small(self) -> None:
        from benchmarks.bench_xref import (
            _SCAN_SMALL,
            _py_scan_pdf_objects,
        )
        from pdfminer_core import scan_pdf_objects

        py = _py_scan_pdf_objects(_SCAN_SMALL)
        rs = [(int(o), int(g), int(p)) for o, g, p in scan_pdf_objects(_SCAN_SMALL)]
        assert py == rs


# ---------------------------------------------------------------------------
# Runnability: all benchmark cases execute without error
# ---------------------------------------------------------------------------

class TestBenchmarkRunnability:
    """Each registered benchmark case must run once without raising."""

    def _run_all(self, register_fn) -> None:  # type: ignore[no-untyped-def]
        from benchmarks.bench_utils import BenchmarkSuite
        suite = BenchmarkSuite(rounds=1, number=1)
        register_fn(suite)
        suite.run()
        assert suite.results, "No results collected"

    def test_predictor_runnability(self) -> None:
        from benchmarks.bench_predictor import register
        self._run_all(register)

    def test_lzw_runnability(self) -> None:
        from benchmarks.bench_lzw import register
        self._run_all(register)

    def test_matrix_runnability(self) -> None:
        from benchmarks.bench_matrix import register
        self._run_all(register)

    def test_xref_runnability(self) -> None:
        from benchmarks.bench_xref import register
        self._run_all(register)

    @pytest.mark.parametrize("pdf_name", [
        "simple1.pdf",
        "simple2.pdf",
        "simple3.pdf",
    ])
    def test_e2e_runnability(self, pdf_name: str) -> None:
        """End-to-end benchmark runs without error on small sample PDFs."""
        from benchmarks.bench_utils import BenchmarkSuite
        from benchmarks.bench_e2e import _REPO_ROOT, _extract_text, _set_has_rust

        path = _REPO_ROOT / "samples" / pdf_name
        if not path.exists():
            pytest.skip(f"{path} not found")

        # Rust path
        _set_has_rust(True)
        rs_text = _extract_text(path)

        # Python path
        _set_has_rust(False)
        try:
            py_text = _extract_text(path)
        finally:
            _set_has_rust(True)

        assert rs_text == py_text, f"Text mismatch for {pdf_name}"


# ---------------------------------------------------------------------------
# BenchmarkSuite infrastructure tests
# ---------------------------------------------------------------------------

class TestBenchmarkSuiteInfrastructure:
    """Unit tests for the BenchmarkSuite class itself."""

    def test_add_and_run(self) -> None:
        from benchmarks.bench_utils import BenchmarkSuite

        suite = BenchmarkSuite(rounds=2, number=10)
        called = {"n": 0}

        def _fn() -> int:
            called["n"] += 1
            return 42

        suite.add(group="test", variant="python", name="test fn", fn=_fn)
        suite.run()

        assert len(suite.results) == 1
        assert suite.results[0].group == "test"
        assert suite.results[0].variant == "python"
        assert suite.results[0].best_sec > 0
        # rounds=2, number=10 => 20 calls minimum
        assert called["n"] >= 20

    def test_report_returns_string(self) -> None:
        from benchmarks.bench_utils import BenchmarkSuite

        suite = BenchmarkSuite(rounds=1, number=1)
        suite.add(group="g", variant="python", name="py fn", fn=lambda: None)
        suite.add(group="g", variant="rust",   name="rs fn", fn=lambda: None)
        suite.run()
        report = suite.report()
        assert isinstance(report, str)
        assert "python" in report
        assert "rust" in report

    def test_speedup_map(self) -> None:
        from benchmarks.bench_utils import BenchmarkSuite

        suite = BenchmarkSuite(rounds=1, number=1)
        suite.add(group="g", variant="python", name="py", fn=lambda: None)
        suite.add(group="g", variant="rust",   name="rs", fn=lambda: None)
        suite.run()
        sm = suite.speedup_map()
        assert "g" in sm
        # Both implementations are trivially fast but the ratio must be a positive number.
        assert sm["g"] is not None and sm["g"] > 0

    def test_no_results_report(self) -> None:
        from benchmarks.bench_utils import BenchmarkSuite

        suite = BenchmarkSuite()
        report = suite.report()
        assert "no results" in report.lower()
