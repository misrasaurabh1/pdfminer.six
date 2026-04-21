"""Tests for Rust matrix math functions via pdfminer.utils."""

import math
import random

import pytest

from pdfminer.utils import (
    apply_matrix_norm,
    apply_matrix_pt,
    apply_matrix_rect,
    mult_matrix,
    translate_matrix,
)

# Standard test matrices
IDENTITY: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
TRANSLATE_10_20: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)
SCALE_2: tuple[float, float, float, float, float, float] = (2.0, 0.0, 0.0, 2.0, 0.0, 0.0)
ROTATE_90: tuple[float, float, float, float, float, float] = (0.0, 1.0, -1.0, 0.0, 0.0, 0.0)


# --- mult_matrix ---

def test_mult_identity_identity() -> None:
    """Identity × Identity = Identity."""
    assert mult_matrix(IDENTITY, IDENTITY) == IDENTITY


def test_mult_matrix_with_identity_is_noop() -> None:
    """Any matrix × Identity = that matrix."""
    m = (1.0, 2.0, 3.0, 2.0, -4.0, 1.0)
    assert mult_matrix(m, IDENTITY) == m


def test_mult_translate_combine() -> None:
    """Two translation matrices combine their translations.

    mult_matrix(m1, m0) computes m0 * m1 (m0 is left/outer).
    Translating by (5,0) then (3,0) = net (8,0).
    """
    t1 = (1.0, 0.0, 0.0, 1.0, 5.0, 0.0)
    t2 = (1.0, 0.0, 0.0, 1.0, 3.0, 0.0)
    result = mult_matrix(t1, t2)
    assert result == (1.0, 0.0, 0.0, 1.0, 8.0, 0.0)


def test_mult_matrix_known_values() -> None:
    """Spot-check against hand-computed result."""
    m0 = (1.0, 2.0, 3.0, 2.0, -4.0, 1.0)
    m1 = (3.0, 4.0, 1.0, 2.0, -2.0, 1.0)
    expected = (5.0, 8.0, 11.0, 16.0, -13.0, -13.0)
    assert mult_matrix(m0, m1) == expected


# --- translate_matrix ---

def test_translate_matrix_identity_zero() -> None:
    """Translating identity by (0,0) yields identity."""
    assert translate_matrix(IDENTITY, (0.0, 0.0)) == IDENTITY


def test_translate_matrix_known_values() -> None:
    m = (1.0, 0.0, 0.0, 1.0, 3.0, -3.0)
    result = translate_matrix(m, (12.0, -32.0))
    assert result == (1.0, 0.0, 0.0, 1.0, 15.0, -35.0)


def test_translate_matrix_preserves_linear_part() -> None:
    """Only the translation components (e, f) change."""
    m = (2.0, 1.0, 3.0, 4.0, 0.0, 0.0)
    result = translate_matrix(m, (1.0, 1.0))
    a, b, c, d, e, f = result
    assert (a, b, c, d) == (2.0, 1.0, 3.0, 4.0)
    assert e == pytest.approx(2.0 * 1 + 3.0 * 1 + 0.0)
    assert f == pytest.approx(1.0 * 1 + 4.0 * 1 + 0.0)


# --- apply_matrix_pt ---

def test_apply_matrix_pt_identity() -> None:
    assert apply_matrix_pt(IDENTITY, (1.0, 2.0)) == (1.0, 2.0)


def test_apply_matrix_pt_origin_gives_translation() -> None:
    assert apply_matrix_pt(TRANSLATE_10_20, (0.0, 0.0)) == (10.0, 20.0)


def test_apply_matrix_pt_translate() -> None:
    result = apply_matrix_pt(TRANSLATE_10_20, (1.0, 2.0))
    assert result == (11.0, 22.0)


def test_apply_matrix_pt_scale() -> None:
    result = apply_matrix_pt(SCALE_2, (3.0, 4.0))
    assert result == (6.0, 8.0)


def test_apply_matrix_pt_rotate_90() -> None:
    """ROTATE_90 = (0,1,-1,0,0,0): point (1,0) -> (0,1)."""
    x, y = apply_matrix_pt(ROTATE_90, (1.0, 0.0))
    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(1.0, abs=1e-12)


def test_apply_matrix_pt_known_values() -> None:
    m = (1.0, 2.0, 3.0, 2.0, -4.0, 1.0)
    x, y = apply_matrix_pt(m, (0.0, 0.0))
    assert (x, y) == (-4.0, 1.0)


# --- apply_matrix_norm ---

def test_apply_matrix_norm_identity() -> None:
    assert apply_matrix_norm(IDENTITY, (5.0, 7.0)) == (5.0, 7.0)


def test_apply_matrix_norm_no_translation() -> None:
    """apply_matrix_norm ignores (e, f): translating the matrix changes nothing."""
    m_translated = (1.0, 0.0, 0.0, 1.0, 100.0, 200.0)
    assert apply_matrix_norm(m_translated, (3.0, 4.0)) == (3.0, 4.0)


def test_apply_matrix_norm_is_pt_minus_origin() -> None:
    """apply_matrix_norm(m, v) == apply_matrix_pt(m, v) - apply_matrix_pt(m, (0,0))."""
    m = (2.0, 1.0, 3.0, 4.0, 5.0, 6.0)
    v = (7.0, 8.0)
    pt = apply_matrix_pt(m, v)
    origin = apply_matrix_pt(m, (0.0, 0.0))
    norm = apply_matrix_norm(m, v)
    assert norm[0] == pytest.approx(pt[0] - origin[0])
    assert norm[1] == pytest.approx(pt[1] - origin[1])


# --- apply_matrix_rect ---

def test_apply_matrix_rect_identity() -> None:
    assert apply_matrix_rect(IDENTITY, (0.0, 0.0, 100.0, 200.0)) == (
        0.0, 0.0, 100.0, 200.0
    )


def test_apply_matrix_rect_scale() -> None:
    result = apply_matrix_rect(SCALE_2, (1.0, 2.0, 3.0, 4.0))
    assert result == (2.0, 4.0, 6.0, 8.0)


def test_apply_matrix_rect_translate() -> None:
    result = apply_matrix_rect(TRANSLATE_10_20, (0.0, 0.0, 5.0, 3.0))
    assert result == (10.0, 20.0, 15.0, 23.0)


def test_apply_matrix_rect_rotate_180() -> None:
    """Rotation by 180 degrees swaps and negates coordinates."""
    rotate_180 = (-1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    result = apply_matrix_rect(rotate_180, (3.0, 4.0, 7.0, 6.0))
    assert result == (-7.0, -6.0, -3.0, -4.0)


def test_apply_matrix_rect_rotate_10_degrees() -> None:
    angle = math.radians(10)
    m = (math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), 0.0, 0.0)
    x0, y0, x1, y1 = apply_matrix_rect(m, (3.0, 4.0, 7.0, 6.0))
    assert x0 == pytest.approx(1.91253419)
    assert y0 == pytest.approx(4.46017555)
    assert x1 == pytest.approx(6.19906156)
    assert y1 == pytest.approx(7.12438376)


# --- Python/Rust equivalence ---

def _python_mult_matrix(
    m1: tuple[float, ...], m0: tuple[float, ...]
) -> tuple[float, ...]:
    a1, b1, c1, d1, e1, f1 = m1
    a0, b0, c0, d0, e0, f0 = m0
    return (
        a0 * a1 + c0 * b1,
        b0 * a1 + d0 * b1,
        a0 * c1 + c0 * d1,
        b0 * c1 + d0 * d1,
        a0 * e1 + c0 * f1 + e0,
        b0 * e1 + d0 * f1 + f0,
    )


def _python_apply_matrix_pt(
    m: tuple[float, ...], v: tuple[float, ...]
) -> tuple[float, float]:
    a, b, c, d, e, f = m
    x, y = v
    return a * x + c * y + e, b * x + d * y + f


@pytest.mark.parametrize("seed", [0, 42, 137, 999, 12345])
def test_mult_matrix_matches_python_reference(seed: int) -> None:
    """Verify mult_matrix output matches the reference Python formula on random inputs."""
    rng = random.Random(seed)
    for _ in range(200):
        m1 = tuple(rng.uniform(-10.0, 10.0) for _ in range(6))
        m0 = tuple(rng.uniform(-10.0, 10.0) for _ in range(6))
        expected = _python_mult_matrix(m1, m0)
        result = mult_matrix(m1, m0)  # type: ignore[arg-type]
        for r, e in zip(result, expected):
            assert r == pytest.approx(e, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 42, 137])
def test_apply_matrix_pt_matches_python_reference(seed: int) -> None:
    """Verify apply_matrix_pt output matches the reference Python formula."""
    rng = random.Random(seed)
    for _ in range(200):
        m = tuple(rng.uniform(-10.0, 10.0) for _ in range(6))
        v = (rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0))
        expected = _python_apply_matrix_pt(m, v)
        result = apply_matrix_pt(m, v)  # type: ignore[arg-type]
        assert result[0] == pytest.approx(expected[0], rel=1e-12)
        assert result[1] == pytest.approx(expected[1], rel=1e-12)
