/// PDF parser helpers: stream-length resolution and colorspace lookup.
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Attempt to read `/Length` from a stream attributes dict.
///
/// Returns `Some(length)` when the value is a plain Python `int`, `None`
/// otherwise (e.g. when it is an indirect reference that still needs
/// resolution — the Python caller must fall back to `int_value`).
///
/// This fast path avoids calling into Python's `int_value` / `resolve1` chain
/// in the common case where `Length` is already a literal integer.
#[pyfunction]
pub fn resolve_stream_length(attrs: &Bound<'_, PyDict>) -> PyResult<Option<usize>> {
    let length_item = attrs.get_item("Length")?;
    match length_item {
        None => Ok(None),
        Some(val) => match val.extract::<i64>() {
            Ok(n) if n >= 0 => Ok(Some(n as usize)),
            _ => Ok(None), // indirect ref or other type — let Python resolve
        },
    }
}

/// Return the number of color components for a predefined PDF color space name.
///
/// Exported for callers that need the raw component count without constructing
/// a `PDFColorSpace` object.  Returns `Some(n)` for known names, `None` for
/// unknown ones.
#[pyfunction]
pub fn get_predefined_colorspace_ncomponents(name: &str) -> Option<u32> {
    match name {
        "DeviceGray" | "CalGray" | "Separation" | "Indexed" | "Pattern" => Some(1),
        "CalRGB" | "Lab" | "DeviceRGB" => Some(3),
        "DeviceCMYK" => Some(4),
        _ => None,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resolve_stream_length, m)?)?;
    m.add_function(wrap_pyfunction!(get_predefined_colorspace_ncomponents, m)?)?;
    Ok(())
}
