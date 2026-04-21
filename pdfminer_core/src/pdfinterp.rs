use pyo3::prelude::*;

/// Affine matrix multiplication: returns `ma × mb`.
///
/// Both matrices are `(a, b, c, d, e, f)` representing:
/// ```text
/// | a  b  0 |
/// | c  d  0 |
/// | e  f  1 |
/// ```
#[inline]
fn mult_matrix_inner(
    ma: (f64, f64, f64, f64, f64, f64),
    mb: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let (a1, b1, c1, d1, e1, f1) = ma;
    let (a2, b2, c2, d2, e2, f2) = mb;
    (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )
}

/// Compute `(tx * a + ty * c + e, tx * b + ty * d + f)` — the new (e, f)
/// translation components after advancing by `(tx, ty)` in text space.
///
/// Called for every Td / TD / T* operator — the text-position hot path.
#[pyfunction]
#[pyo3(signature = (matrix, tx, ty))]
pub fn apply_text_advance(
    matrix: (f64, f64, f64, f64, f64, f64),
    tx: f64,
    ty: f64,
) -> (f64, f64) {
    let (a, b, c, d, e, f) = matrix;
    (tx * a + ty * c + e, tx * b + ty * d + f)
}

/// Multiply two affine matrices.
///
/// Equivalent to `pdfminer.utils.mult_matrix`, but operates on native
/// 6-tuples so the round-trip through Python sequences is avoided.
#[pyfunction]
#[pyo3(signature = (m1, m2))]
pub fn mult_matrix_rust(
    m1: (f64, f64, f64, f64, f64, f64),
    m2: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    mult_matrix_inner(m1, m2)
}

/// Combine the text matrix with the CTM.
///
/// Equivalent to `mult_matrix(textstate.matrix, ctm)` but takes native
/// tuples to avoid Python-sequence overhead per character.
#[pyfunction]
#[pyo3(signature = (textstate_matrix, ctm))]
pub fn text_state_to_matrix(
    textstate_matrix: (f64, f64, f64, f64, f64, f64),
    ctm: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    mult_matrix_inner(textstate_matrix, ctm)
}

/// Compute spacing displacements for a TJ array.
///
/// For each `(text_bytes_opt, spacing_adj_opt)` element:
/// - Numeric adjustments produce a `(tx, ty)` displacement in text space.
/// - Text elements emit `(0.0, 0.0)` — glyph-metric advances are font-dependent
///   and handled by the Python caller.
///
/// The returned vec is 1-to-1 with the input so callers can zip them together.
#[pyfunction]
#[pyo3(signature = (fontsize, scaling, text_sequence, is_horizontal))]
pub fn calculate_char_advances(
    fontsize: f64,
    scaling: f64,
    text_sequence: Vec<(Option<Vec<u8>>, Option<f64>)>,
    is_horizontal: bool,
) -> Vec<(f64, f64)> {
    let mut advances = Vec::with_capacity(text_sequence.len());
    let scale = scaling * 0.01;

    for (text_opt, adj_opt) in text_sequence {
        if let Some(adj) = adj_opt {
            let displacement = -adj * 0.001 * fontsize;
            let (tx, ty) = if is_horizontal {
                (displacement * scale, 0.0)
            } else {
                (0.0, displacement)
            };
            advances.push((tx, ty));
        }
        if text_opt.is_some() {
            advances.push((0.0, 0.0));
        }
    }

    advances
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_text_advance, m)?)?;
    m.add_function(wrap_pyfunction!(mult_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(text_state_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_char_advances, m)?)?;
    Ok(())
}
