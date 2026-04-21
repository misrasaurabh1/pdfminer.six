use pyo3::prelude::*;

/// Apply a spacing adjustment for a TJ array element.
///
/// Per PDF spec (§9.4.3), a number in a TJ array displaces the current
/// text position by `-element * 0.001 * fontsize` (horizontal text) or
/// `-element * 0.001 * fontsize` (vertical text on the y-axis).
///
/// Returns (tx, ty) — the displacement in text space.
#[inline]
fn tj_adjustment_advance(adj: f64, fontsize: f64, scaling: f64, is_horizontal: bool) -> (f64, f64) {
    if is_horizontal {
        let tx = -adj * 0.001 * fontsize * (scaling * 0.01);
        (tx, 0.0)
    } else {
        let ty = -adj * 0.001 * fontsize;
        (0.0, ty)
    }
}

/// Calculate character advances for a TJ sequence.
///
/// Each element is `(text_bytes_opt, spacing_adj_opt)`.
/// For spacing-adjustment elements (numeric), returns the (tx, ty) shift in
/// text space.  For text elements the caller is responsible for glyph-metric
/// advances; this function emits (0.0, 0.0) as a placeholder so the returned
/// vector has a 1-to-1 correspondence with the input vector.
///
/// Equivalent to the inner arithmetic in `PDFPageInterpreter.do_TJ()`.
#[pyfunction]
#[pyo3(signature = (matrix, textstate_scaling, textstate_charspace, textstate_wordspace,
                    fontsize, text_sequence, is_horizontal))]
pub fn calculate_char_advances(
    matrix: (f64, f64, f64, f64, f64, f64),
    textstate_scaling: f64,
    textstate_charspace: f64,
    textstate_wordspace: f64,
    fontsize: f64,
    text_sequence: Vec<(Option<Vec<u8>>, Option<f64>)>,
    is_horizontal: bool,
) -> PyResult<Vec<(f64, f64)>> {
    let _ = (matrix, textstate_charspace, textstate_wordspace); // consumed by caller
    let mut advances = Vec::with_capacity(text_sequence.len());

    for (text_opt, adj_opt) in text_sequence {
        if let Some(adj) = adj_opt {
            let (tx, ty) = tj_adjustment_advance(adj, fontsize, textstate_scaling, is_horizontal);
            advances.push((tx, ty));
        }
        if text_opt.is_some() {
            // Glyph-metric advances are font-dependent; caller handles them.
            advances.push((0.0, 0.0));
        }
    }

    Ok(advances)
}

/// Multiply two 2-D affine matrices stored as flat 6-tuples.
///
/// Both `a` and `b` are (a, b, c, d, e, f) representing the matrix:
/// ```text
/// | a  b  0 |
/// | c  d  0 |
/// | e  f  1 |
/// ```
/// Returns the product `a × b`.
#[inline]
fn mult_matrix(
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

/// Combine the text matrix with the current transformation matrix (CTM).
///
/// Equivalent to `mult_matrix(textstate_matrix, ctm)` in Python.
/// This is called for every rendered character and sits on the hottest path.
#[pyfunction]
#[pyo3(signature = (textstate_matrix, textstate_linematrix, ctm))]
pub fn text_state_to_matrix(
    textstate_matrix: (f64, f64, f64, f64, f64, f64),
    textstate_linematrix: (f64, f64),
    ctm: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    let _ = textstate_linematrix; // consumed by caller for position bookkeeping
    mult_matrix(textstate_matrix, ctm)
}

/// Multiply two affine matrices (Python-callable wrapper).
///
/// Exposed so Python code can replace `pdfminer.utils.mult_matrix` with this
/// faster implementation when the extension is available.
#[pyfunction]
#[pyo3(signature = (m1, m2))]
pub fn mult_matrix_rust(
    m1: (f64, f64, f64, f64, f64, f64),
    m2: (f64, f64, f64, f64, f64, f64),
) -> (f64, f64, f64, f64, f64, f64) {
    mult_matrix(m1, m2)
}

/// Apply a text-position advance (Td / linematrix update) in text space.
///
/// Returns `(new_e, new_f)` — the updated translation components of the text
/// matrix after moving by `(tx, ty)` in the *current* text coordinate system.
///
/// Equivalent to the body of `do_Td` in Python:
/// ```python
/// (a, b, c, d, e, f) = self.textstate.matrix
/// e_new = tx * a + ty * c + e
/// f_new = tx * b + ty * d + f
/// ```
#[pyfunction]
#[pyo3(signature = (matrix, tx, ty))]
pub fn apply_text_advance(
    matrix: (f64, f64, f64, f64, f64, f64),
    tx: f64,
    ty: f64,
) -> (f64, f64) {
    let (a, b, c, d, e, f) = matrix;
    let e_new = tx * a + ty * c + e;
    let f_new = tx * b + ty * d + f;
    (e_new, f_new)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_char_advances, m)?)?;
    m.add_function(wrap_pyfunction!(text_state_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mult_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(apply_text_advance, m)?)?;
    Ok(())
}
