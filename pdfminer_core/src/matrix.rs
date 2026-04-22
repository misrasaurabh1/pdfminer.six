use pyo3::prelude::*;
use pyo3::types::PySequence;

/// PDF transformation matrix: (a, b, c, d, e, f) representing
/// | a  b  0 |
/// | c  d  0 |
/// | e  f  1 |
///
/// All functions match the semantics of their Python counterparts in
/// pdfminer/utils.py exactly, including argument order and convention.
///
/// Input matrices accept both Python tuples and lists (sequences of 6 floats).
/// Input points accept both Python tuples and lists (sequences of 2 floats).
/// Return values are always tuples.

type M6 = (f64, f64, f64, f64, f64, f64);
type P2 = (f64, f64);

fn extract_matrix(m: &Bound<'_, PyAny>) -> PyResult<M6> {
    let seq = m.downcast::<PySequence>()?;
    if seq.len()? != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must have exactly 6 elements",
        ));
    }
    let a: f64 = seq.get_item(0)?.extract()?;
    let b: f64 = seq.get_item(1)?.extract()?;
    let c: f64 = seq.get_item(2)?.extract()?;
    let d: f64 = seq.get_item(3)?.extract()?;
    let e: f64 = seq.get_item(4)?.extract()?;
    let f: f64 = seq.get_item(5)?.extract()?;
    Ok((a, b, c, d, e, f))
}

fn extract_point(v: &Bound<'_, PyAny>) -> PyResult<P2> {
    let seq = v.downcast::<PySequence>()?;
    if seq.len()? != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "point must have exactly 2 elements",
        ));
    }
    let x: f64 = seq.get_item(0)?.extract()?;
    let y: f64 = seq.get_item(1)?.extract()?;
    Ok((x, y))
}

fn extract_rect(r: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64, f64)> {
    let seq = r.downcast::<PySequence>()?;
    if seq.len()? != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rect must have exactly 4 elements",
        ));
    }
    let x0: f64 = seq.get_item(0)?.extract()?;
    let y0: f64 = seq.get_item(1)?.extract()?;
    let x1: f64 = seq.get_item(2)?.extract()?;
    let y1: f64 = seq.get_item(3)?.extract()?;
    Ok((x0, y0, x1, y1))
}

/// Multiply two matrices.
///
/// Matches Python `mult_matrix(m1, m0)` which computes M0 * M1
/// (m0 is the left/outer matrix, m1 is the right/inner matrix).
/// Called here as `mult_matrix(m1, m2)` where m2 plays the role of m0.
#[pyfunction]
pub fn mult_matrix(m1: &Bound<'_, PyAny>, m2: &Bound<'_, PyAny>) -> PyResult<M6> {
    let (a1, b1, c1, d1, e1, f1) = extract_matrix(m1)?;
    let (a2, b2, c2, d2, e2, f2) = extract_matrix(m2)?;
    Ok((
        a2 * a1 + c2 * b1,
        b2 * a1 + d2 * b1,
        a2 * c1 + c2 * d1,
        b2 * c1 + d2 * d1,
        a2 * e1 + c2 * f1 + e2,
        b2 * e1 + d2 * f1 + f2,
    ))
}

/// Translate a matrix by a vector (x, y) in its own coordinate system.
///
/// Matches Python `translate_matrix(m, v)`.
#[pyfunction]
pub fn translate_matrix(m: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>) -> PyResult<M6> {
    let (a, b, c, d, e, f) = extract_matrix(m)?;
    let (x, y) = extract_point(v)?;
    Ok((a, b, c, d, x * a + y * c + e, x * b + y * d + f))
}

/// Apply a matrix to a 2D point (includes translation).
///
/// Matches Python `apply_matrix_pt(m, v)`.
#[pyfunction]
pub fn apply_matrix_pt(m: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>) -> PyResult<P2> {
    let (a, b, c, d, e, f) = extract_matrix(m)?;
    let (x, y) = extract_point(v)?;
    Ok((a * x + c * y + e, b * x + d * y + f))
}

/// Apply a matrix to a 2D vector without translation (for directions/normals).
///
/// Equivalent to `apply_matrix_pt(m, v) - apply_matrix_pt(m, (0, 0))`.
/// Matches Python `apply_matrix_norm(m, v)`.
#[pyfunction]
pub fn apply_matrix_norm(m: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>) -> PyResult<P2> {
    let (a, b, c, d, _, _) = extract_matrix(m)?;
    let (x, y) = extract_point(v)?;
    Ok((a * x + c * y, b * x + d * y))
}

/// Apply a matrix to an axis-aligned rectangle, returning the bounding box
/// of the transformed rectangle.
///
/// Matches Python `apply_matrix_rect(m, rect)`.
#[pyfunction]
pub fn apply_matrix_rect(
    m: &Bound<'_, PyAny>,
    rect: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64)> {
    let mat = extract_matrix(m)?;
    let (x0, y0, x1, y1) = extract_rect(rect)?;
    // Transform all four corners and take the bounding box, matching Python.
    let (lx0, ly0) = apply_matrix_pt_inner(mat, x0, y0);
    let (rx0, ry0) = apply_matrix_pt_inner(mat, x1, y0);
    let (rx1, ry1) = apply_matrix_pt_inner(mat, x1, y1);
    let (lx1, ly1) = apply_matrix_pt_inner(mat, x0, y1);
    Ok((
        lx0.min(lx1).min(rx0).min(rx1),
        ly0.min(ly1).min(ry0).min(ry1),
        lx0.max(lx1).max(rx0).max(rx1),
        ly0.max(ly1).max(ry0).max(ry1),
    ))
}

#[inline(always)]
pub(crate) fn apply_matrix_pt_inner(m: M6, x: f64, y: f64) -> P2 {
    let (a, b, c, d, e, f) = m;
    (a * x + c * y + e, b * x + d * y + f)
}

/// Native Rust version of apply_matrix_rect — no PyAny overhead.
pub(crate) fn apply_matrix_rect_impl(m: M6, rect: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    let (x0, y0, x1, y1) = rect;
    let (lx0, ly0) = apply_matrix_pt_inner(m, x0, y0);
    let (rx0, ry0) = apply_matrix_pt_inner(m, x1, y0);
    let (rx1, ry1) = apply_matrix_pt_inner(m, x1, y1);
    let (lx1, ly1) = apply_matrix_pt_inner(m, x0, y1);
    (
        lx0.min(lx1).min(rx0).min(rx1),
        ly0.min(ly1).min(ry0).min(ry1),
        lx0.max(lx1).max(rx0).max(rx1),
        ly0.max(ly1).max(ry0).max(ry1),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mult_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(translate_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_pt, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_norm, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_rect, m)?)?;
    Ok(())
}
