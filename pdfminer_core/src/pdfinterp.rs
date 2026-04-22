use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

/// Split a flat list of `(pos, token)` pairs into `(operands, operator)` groups.
///
/// The input is the already-converted Python token list produced by
/// `PDFContentParser` / `_convert_rust_tokens`.  Each element is a 2-tuple
/// `(int_pos, token_object)`.
///
/// Tokens that are `PSKeyword` instances (identified by the Python-side
/// `PSKeyword` class) mark the end of an operand group and start the next.
/// All preceding non-keyword tokens become that operation's operand list.
///
/// Returns a `list` of `(operands, keyword_bytes_or_None)` pairs where:
/// - `operands` is a Python `list` of the accumulated operand objects.
/// - `keyword_bytes_or_None` is `bytes` (the raw keyword name) for a complete
///   operation, or `None` for any trailing operands without a following keyword.
///
/// Splitting in Rust removes the per-token `argstack.append()` / `pop()` calls
/// from the Python `execute()` hot loop, which is the dominant overhead there.
#[pyfunction]
#[pyo3(signature = (tokens, ps_keyword_class))]
pub fn split_ops_and_operands(
    py: Python<'_>,
    tokens: &Bound<'_, PyList>,
    ps_keyword_class: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let result = PyList::empty_bound(py);
    // Most PDF operators take ≤6 operands; pre-allocate to avoid small reallocations.
    let mut operands: Vec<PyObject> = Vec::with_capacity(8);

    for item in tokens.iter() {
        // Each item is a 2-tuple (pos, token).
        let tup = item.downcast::<PyTuple>()?;
        let token = tup.get_item(1)?;
        if token.is_instance(ps_keyword_class)? {
            // Extract the keyword name bytes from PSKeyword.name.
            let name_attr = token.getattr("name")?;
            let kw_bytes: Vec<u8> = if let Ok(b) = name_attr.downcast::<PyBytes>() {
                b.as_bytes().to_vec()
            } else {
                // PSKeyword.name is bytes in normal usage; str is a fallback.
                name_attr.extract::<String>()?.into_bytes()
            };
            let py_ops = PyList::new_bound(py, operands.iter().map(|o| o.bind(py)));
            let kw_obj: PyObject = PyBytes::new_bound(py, &kw_bytes).into();
            let pair = PyTuple::new_bound(py, [py_ops.into_any(), kw_obj.into_bound(py)]);
            result.append(pair)?;
            operands.clear();
        } else {
            operands.push(token.into());
        }
    }

    // Trailing operands without a closing keyword.
    if !operands.is_empty() {
        let py_ops = PyList::new_bound(py, operands.iter().map(|o| o.bind(py)));
        let none_obj = py.None().into_bound(py);
        let pair = PyTuple::new_bound(py, [py_ops.into_any(), none_obj]);
        result.append(pair)?;
    }

    Ok(result.into())
}

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

/// Compute per-character translated matrices for a text string (horizontal or vertical).
///
/// Each item is `(cid, advance, spacing_adj, needcharspace)`:
///   - `cid`: character code (used only to detect space for wordspace)
///   - `advance`: pre-computed glyph advance = `char_width(cid) * fontsize * scaling`
///   - `spacing_adj`: TJ-style spacing to subtract from the cursor before this glyph
///   - `needcharspace`: whether to add `charspace` before this glyph
///
/// `is_horizontal`: advances along x (horizontal) or y (vertical writing).
///
/// Returns `(matrices, final_x, final_y)` where `matrices` has one 6-tuple per item.
#[pyfunction]
#[pyo3(signature = (matrix, x, y, items, charspace, wordspace, is_horizontal))]
pub fn compute_char_matrices(
    matrix: (f64, f64, f64, f64, f64, f64),
    x: f64,
    y: f64,
    items: Vec<(u32, f64, f64, bool)>,
    charspace: f64,
    wordspace: f64,
    is_horizontal: bool,
) -> (Vec<(f64, f64, f64, f64, f64, f64)>, f64, f64) {
    let (a, b, c, d, e, f) = matrix;
    let mut cur_x = x;
    let mut cur_y = y;
    let mut matrices = Vec::with_capacity(items.len());

    for (cid, advance, spacing_adj, needcharspace) in items {
        if is_horizontal {
            cur_x -= spacing_adj;
            if needcharspace { cur_x += charspace; }
        } else {
            cur_y -= spacing_adj;
            if needcharspace { cur_y += charspace; }
        }
        let te = cur_x * a + cur_y * c + e;
        let tf = cur_x * b + cur_y * d + f;
        matrices.push((a, b, c, d, te, tf));
        if is_horizontal {
            cur_x += advance;
            if cid == 32 && wordspace != 0.0 { cur_x += wordspace; }
        } else {
            cur_y += advance;
            if cid == 32 && wordspace != 0.0 { cur_y += wordspace; }
        }
    }

    (matrices, cur_x, cur_y)
}

/// Decode bytes into CIDs using a flat 256-entry encoding table.
///
/// `encoding[byte]` is the CID; `u32::MAX` means unmapped (falls back to the byte
/// value itself, matching `bytearray(data)` for simple fonts).
/// Registered for use from Python where the encoding table is available as a list.
#[pyfunction]
pub fn decode_bytes_to_cids(encoding: Vec<u32>, data: &[u8]) -> Vec<u32> {
    data.iter()
        .map(|&b| {
            let mapped = encoding[b as usize];
            if mapped == u32::MAX { b as u32 } else { mapped }
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_char_advances, m)?)?;
    m.add_function(wrap_pyfunction!(text_state_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mult_matrix_rust, m)?)?;
    m.add_function(wrap_pyfunction!(apply_text_advance, m)?)?;
    m.add_function(wrap_pyfunction!(split_ops_and_operands, m)?)?;
    m.add_function(wrap_pyfunction!(compute_char_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bytes_to_cids, m)?)?;
    Ok(())
}
