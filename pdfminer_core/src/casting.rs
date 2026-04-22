use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

/// Internal helper: convert a single Python object to f64.
///
/// Fast path: direct extraction covers int and float (the common case).
/// Fast paths for str/bytes: parse directly in Rust without Python FFI overhead.
/// Final fallback: call `__float__()` for other types implementing the protocol.
#[inline]
fn to_f64(o: &Bound<'_, PyAny>) -> Option<f64> {
    if let Ok(v) = o.extract::<f64>() {
        return Some(v);
    }
    if let Ok(s) = o.downcast::<PyString>() {
        if let Ok(txt) = s.to_str() {
            return txt.trim().parse::<f64>().ok();
        }
    }
    if let Ok(b) = o.downcast::<PyBytes>() {
        if let Ok(s) = std::str::from_utf8(b.as_bytes()) {
            return s.trim().parse::<f64>().ok();
        }
    }
    o.call_method0("__float__")
        .ok()
        .and_then(|r| r.extract::<f64>().ok())
}

/// Convert any Python object to f64, returning None on failure.
///
/// Equivalent to Python's `float(o)` wrapped in try/except.
#[pyfunction]
pub fn safe_float(o: &Bound<'_, PyAny>) -> Option<f64> {
    to_f64(o)
}

/// Convert any Python object to i64, returning None on failure.
///
/// Handles int, bytes, and str the same way Python's `int()` does.
#[pyfunction]
pub fn safe_int(o: &Bound<'_, PyAny>) -> Option<i64> {
    if let Ok(v) = o.extract::<i64>() {
        return Some(v);
    }
    if let Ok(b) = o.downcast::<PyBytes>() {
        if let Ok(s) = std::str::from_utf8(b.as_bytes()) {
            return s.trim().parse::<i64>().ok();
        }
    }
    if let Ok(s) = o.downcast::<PyString>() {
        if let Ok(txt) = s.to_str() {
            return txt.trim().parse::<i64>().ok();
        }
    }
    None
}

/// Convert 6 args to a Matrix 6-tuple of f64, returning None if any fails.
#[pyfunction]
pub fn safe_matrix(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    c: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
    e: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    Some((to_f64(a)?, to_f64(b)?, to_f64(c)?, to_f64(d)?, to_f64(e)?, to_f64(f)?))
}

/// Convert 3 args to an RGB triple of f64, returning None if any fails.
#[pyfunction]
pub fn safe_rgb(
    r: &Bound<'_, PyAny>,
    g: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> Option<(f64, f64, f64)> {
    Some((to_f64(r)?, to_f64(g)?, to_f64(b)?))
}

/// Convert 4 args to a CMYK quadruple of f64, returning None if any fails.
#[pyfunction]
pub fn safe_cmyk(
    c: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    k: &Bound<'_, PyAny>,
) -> Option<(f64, f64, f64, f64)> {
    Some((to_f64(c)?, to_f64(m)?, to_f64(y)?, to_f64(k)?))
}

/// Convert 4 args to a Rect quadruple of f64, returning None if any fails.
#[pyfunction]
pub fn safe_rect(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    c: &Bound<'_, PyAny>,
    d: &Bound<'_, PyAny>,
) -> Option<(f64, f64, f64, f64)> {
    Some((to_f64(a)?, to_f64(b)?, to_f64(c)?, to_f64(d)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(safe_float, m)?)?;
    m.add_function(wrap_pyfunction!(safe_int, m)?)?;
    m.add_function(wrap_pyfunction!(safe_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(safe_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(safe_cmyk, m)?)?;
    m.add_function(wrap_pyfunction!(safe_rect, m)?)?;
    Ok(())
}
