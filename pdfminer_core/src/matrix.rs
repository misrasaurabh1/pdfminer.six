use pyo3::prelude::*;

/// PDF transformation matrix: (a, b, c, d, e, f) representing
/// | a  b  0 |
/// | c  d  0 |
/// | e  f  1 |
///
/// All functions match the semantics of their Python counterparts in
/// pdfminer/utils.py exactly, including argument order and convention.

/// Multiply two matrices.
///
/// Matches Python `mult_matrix(m1, m0)` which computes M0 * M1
/// (m0 is the left/outer matrix, m1 is the right/inner matrix).
/// Called here as `mult_matrix(m1, m2)` where m2 plays the role of m0.
#[pyfunction]
pub fn mult_matrix(
    m1: (f64, f64, f64, f64, f64, f64),
    m2: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let (a1, b1, c1, d1, e1, f1) = m1;
    let (a2, b2, c2, d2, e2, f2) = m2;
    (
        a2 * a1 + c2 * b1,
        b2 * a1 + d2 * b1,
        a2 * c1 + c2 * d1,
        b2 * c1 + d2 * d1,
        a2 * e1 + c2 * f1 + e2,
        b2 * e1 + d2 * f1 + f2,
    )
}

/// Translate a matrix by a vector (x, y) in its own coordinate system.
///
/// Matches Python `translate_matrix(m, v)`.
#[pyfunction]
pub fn translate_matrix(
    m: (f64, f64, f64, f64, f64, f64),
    v: (f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let (a, b, c, d, e, f) = m;
    let (x, y) = v;
    (a, b, c, d, x * a + y * c + e, x * b + y * d + f)
}

/// Apply a matrix to a 2D point (includes translation).
///
/// Matches Python `apply_matrix_pt(m, v)`.
#[pyfunction]
pub fn apply_matrix_pt(
    m: (f64, f64, f64, f64, f64, f64),
    v: (f64, f64),
) -> (f64, f64) {
    let (a, b, c, d, e, f) = m;
    let (x, y) = v;
    (a * x + c * y + e, b * x + d * y + f)
}

/// Apply a matrix to a 2D vector without translation (for directions/normals).
///
/// Equivalent to `apply_matrix_pt(m, v) - apply_matrix_pt(m, (0, 0))`.
/// Matches Python `apply_matrix_norm(m, v)`.
#[pyfunction]
pub fn apply_matrix_norm(
    m: (f64, f64, f64, f64, f64, f64),
    v: (f64, f64),
) -> (f64, f64) {
    let (a, b, c, d, _, _) = m;
    let (x, y) = v;
    (a * x + c * y, b * x + d * y)
}

/// Apply a matrix to an axis-aligned rectangle, returning the bounding box
/// of the transformed rectangle.
///
/// Matches Python `apply_matrix_rect(m, rect)`.
#[pyfunction]
pub fn apply_matrix_rect(
    m: (f64, f64, f64, f64, f64, f64),
    rect: (f64, f64, f64, f64),
) -> (f64, f64, f64, f64) {
    let (x0, y0, x1, y1) = rect;
    // Transform all four corners and take the bounding box, matching Python.
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

#[inline(always)]
fn apply_matrix_pt_inner(
    m: (f64, f64, f64, f64, f64, f64),
    x: f64,
    y: f64,
) -> (f64, f64) {
    let (a, b, c, d, e, f) = m;
    (a * x + c * y + e, b * x + d * y + f)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mult_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(translate_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_pt, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_norm, m)?)?;
    m.add_function(wrap_pyfunction!(apply_matrix_rect, m)?)?;
    Ok(())
}
