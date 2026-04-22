/// PDF parser helpers: stream-length resolution, colorspace lookup,
/// and fast single-object parsing for _getobj_parse hot path.
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::psparser::{
    decode_hex_string, hex_nibble, is_delimiter, is_end_keyword, is_whitespace,
    parse_literal_string,
};

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

// ─── Local aliases for the shared PS helpers ─────────────────────────────────
// These are thin wrappers so callers in this module can use shorter names.

#[inline(always)]
fn is_ws(b: u8) -> bool {
    is_whitespace(b)
}

#[inline(always)]
fn is_delim(b: u8) -> bool {
    is_delimiter(b)
}

#[inline(always)]
fn is_end_tok(b: u8) -> bool {
    is_end_keyword(b)
}

// ─── Helpers used only within pdfparser ──────────────────────────────────────

/// Skip whitespace and comments; return new index.
fn skip_ws(data: &[u8], mut i: usize) -> usize {
    let n = data.len();
    loop {
        while i < n && is_ws(data[i]) {
            i += 1;
        }
        if i < n && data[i] == b'%' {
            while i < n && data[i] != b'\n' && data[i] != b'\r' {
                i += 1;
            }
        } else {
            break;
        }
    }
    i
}

/// Parse a /Name starting right after the '/'. Returns (name_bytes, new_i).
fn parse_name(data: &[u8], start: usize) -> (Vec<u8>, usize) {
    let n = data.len();
    let mut name: Vec<u8> = Vec::new();
    let mut i = start;
    loop {
        if i >= n {
            break;
        }
        let b = data[i];
        if b == b'#' {
            if i + 2 < n && data[i + 1].is_ascii_hexdigit() && data[i + 2].is_ascii_hexdigit() {
                name.push((hex_nibble(data[i + 1]) << 4) | hex_nibble(data[i + 2]));
                i += 3;
            } else {
                break;
            }
        } else if is_ws(b) || is_delim(b) {
            break;
        } else {
            name.push(b);
            i += 1;
        }
    }
    (name, i)
}

/// Parse a number starting at `start`.
/// Returns `Some((number_bytes, new_i, is_float))` or `None`.
fn parse_num(data: &[u8], start: usize) -> Option<(&[u8], usize, bool)> {
    let n = data.len();
    let mut i = start;
    let mut is_float = false;

    if i < n && (data[i] == b'+' || data[i] == b'-') {
        i += 1;
    }
    let digit_start = i;
    while i < n && data[i].is_ascii_digit() {
        i += 1;
    }
    if i < n && data[i] == b'.' {
        is_float = true;
        i += 1;
        while i < n && data[i].is_ascii_digit() {
            i += 1;
        }
    }
    // Must be terminated by whitespace or delimiter
    if i < n && !is_ws(data[i]) && !is_delim(data[i]) {
        return None;
    }
    // Must have at least one digit
    if i == digit_start {
        return None;
    }
    Some((&data[start..i], i, is_float))
}

/// A simple parsed PDF value (no streams).
#[derive(Debug)]
enum PdfVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    Bytes(Vec<u8>),
    Name(Vec<u8>),
    Array(Vec<PdfVal>),
    Dict(Vec<(Vec<u8>, PdfVal)>),
    ObjRef(i64, i64),
}

/// Parse a single PDF value at index `start` in `data`.
/// Returns `Some((val, new_i))` or `None`.
fn parse_val(data: &[u8], start: usize) -> Option<(PdfVal, usize)> {
    let n = data.len();
    let i = skip_ws(data, start);
    if i >= n {
        return None;
    }
    let c = data[i];
    match c {
        // Boolean / null — alphabetic-starting tokens
        b if b.is_ascii_alphabetic() => {
            let mut end = i;
            while end < n && !is_end_tok(data[end]) {
                end += 1;
            }
            if end < n && !is_ws(data[end]) && !is_delim(data[end]) {
                return None;
            }
            match &data[i..end] {
                b"true" => Some((PdfVal::Bool(true), end)),
                b"false" => Some((PdfVal::Bool(false), end)),
                b"null" => Some((PdfVal::Null, end)),
                _ => None,
            }
        }
        // Number (or indirect ref N N R)
        b'0'..=b'9' | b'+' | b'-' | b'.' => {
            let (num1, end1, float1) = parse_num(data, i)?;
            let txt1 = std::str::from_utf8(num1).ok()?;
            if float1 {
                let v = txt1.parse::<f64>().ok()?;
                return Some((PdfVal::Float(v), end1));
            }
            let v1 = txt1.parse::<i64>().ok()?;
            // Peek for indirect ref: N N R
            let j = skip_ws(data, end1);
            if j < n && data[j].is_ascii_digit() {
                if let Some((num2, end2, false)) = parse_num(data, j) {
                    let k = skip_ws(data, end2);
                    if k < n && data[k] == b'R' {
                        let after_r = k + 1;
                        if after_r >= n || is_ws(data[after_r]) || is_delim(data[after_r]) {
                            let genno = std::str::from_utf8(num2).ok()?.parse::<i64>().ok()?;
                            return Some((PdfVal::ObjRef(v1, genno), after_r));
                        }
                    }
                }
            }
            Some((PdfVal::Int(v1), end1))
        }
        b'/' => {
            let (name, end) = parse_name(data, i + 1);
            Some((PdfVal::Name(name), end))
        }
        b'(' => {
            let (bytes, end) = parse_literal_string(data, i + 1)?;
            Some((PdfVal::Bytes(bytes), end))
        }
        b'<' => {
            if i + 1 < n && data[i + 1] == b'<' {
                let (dict, end) = parse_dict_inner(data, i + 2)?;
                Some((PdfVal::Dict(dict), end))
            } else {
                let hex_start = i + 1;
                let mut j = hex_start;
                while j < n && data[j] != b'>' {
                    if !data[j].is_ascii_hexdigit() && !is_ws(data[j]) {
                        return None;
                    }
                    j += 1;
                }
                if j >= n {
                    return None;
                }
                let bytes = decode_hex_string(&data[hex_start..j]);
                Some((PdfVal::Bytes(bytes), j + 1))
            }
        }
        b'[' => {
            let (arr, end) = parse_array_inner(data, i + 1)?;
            Some((PdfVal::Array(arr), end))
        }
        _ => None,
    }
}

fn parse_array_inner(data: &[u8], start: usize) -> Option<(Vec<PdfVal>, usize)> {
    let n = data.len();
    let mut items = Vec::new();
    let mut i = start;
    loop {
        i = skip_ws(data, i);
        if i >= n {
            return None;
        }
        if data[i] == b']' {
            return Some((items, i + 1));
        }
        let (val, new_i) = parse_val(data, i)?;
        items.push(val);
        i = new_i;
    }
}

fn parse_dict_inner(data: &[u8], start: usize) -> Option<(Vec<(Vec<u8>, PdfVal)>, usize)> {
    let n = data.len();
    let mut pairs = Vec::new();
    let mut i = start;
    loop {
        i = skip_ws(data, i);
        if i >= n {
            return None;
        }
        if data[i] == b'>' && i + 1 < n && data[i + 1] == b'>' {
            return Some((pairs, i + 2));
        }
        if data[i] != b'/' {
            return None;
        }
        let (key_name, after_key) = parse_name(data, i + 1);
        i = after_key;
        let (val, new_i) = parse_val(data, i)?;
        pairs.push((key_name, val));
        i = new_i;
    }
}

/// Convert a `PdfVal` to a Python object.
fn val_to_py<'py>(
    py: Python<'py>,
    val: PdfVal,
    lit_fn: &Bound<'py, PyAny>,
    objref_fn: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match val {
        PdfVal::Int(v) => Ok(v.to_object(py).into_bound(py)),
        PdfVal::Float(v) => Ok(v.to_object(py).into_bound(py)),
        PdfVal::Bool(b) => Ok(b.to_object(py).into_bound(py)),
        PdfVal::Null => Ok(py.None().into_bound(py)),
        PdfVal::Bytes(v) => Ok(PyBytes::new_bound(py, &v).into_any()),
        PdfVal::Name(v) => {
            let name_obj: Bound<'_, PyAny> = match std::str::from_utf8(&v) {
                Ok(s) => s.to_object(py).into_bound(py),
                Err(_) => PyBytes::new_bound(py, &v).into_any(),
            };
            Ok(lit_fn.call1((name_obj,))?)
        }
        PdfVal::ObjRef(objid, _genno) => Ok(objref_fn.call1((objid,))?),
        PdfVal::Array(items) => {
            let py_list = PyList::empty_bound(py);
            for item in items {
                let py_val = val_to_py(py, item, lit_fn, objref_fn)?;
                py_list.append(py_val)?;
            }
            Ok(py_list.into_any())
        }
        PdfVal::Dict(pairs) => {
            let d = PyDict::new_bound(py);
            for (key_bytes, v) in pairs {
                let key_str: Bound<'_, PyAny> = match std::str::from_utf8(&key_bytes) {
                    Ok(s) => s.to_object(py).into_bound(py),
                    Err(_) => PyBytes::new_bound(py, &key_bytes).into_any(),
                };
                let py_val = val_to_py(py, v, lit_fn, objref_fn)?;
                d.set_item(key_str, py_val)?;
            }
            Ok(d.into_any())
        }
    }
}

/// Parse a single PDF object from a raw bytes chunk.
///
/// `data` must begin at the object header:
///   `<objid> <genno> obj <value> [endobj]`
///
/// Returns `Some((object, bytes_consumed))` on success, `None` on parse failure
/// (caller falls back to the Python PSStackParser path).
///
/// Handles: integers, floats, booleans, null, strings, names, arrays, dicts,
/// and indirect references.  Does NOT handle streams.
///
/// Arguments:
///   `expected_objid` — the object id we expect to find; returns None on mismatch.
///   `lit_fn`         — `LIT(name)` callable for creating PSLiteral objects.
///   `objref_fn`      — callable `f(objid)` that creates a PDFObjRef.
#[pyfunction]
pub fn parse_pdf_object_at(
    py: Python<'_>,
    data: &[u8],
    expected_objid: i64,
    lit_fn: &Bound<'_, PyAny>,
    objref_fn: &Bound<'_, PyAny>,
) -> PyResult<Option<(PyObject, usize)>> {
    let n = data.len();
    if n == 0 {
        return Ok(None);
    }

    let mut i = skip_ws(data, 0);

    // --- objid ---
    let (num1_bytes, after1, float1) = match parse_num(data, i) {
        Some(r) => r,
        None => return Ok(None),
    };
    if float1 {
        return Ok(None);
    }
    let parsed_objid = match std::str::from_utf8(num1_bytes)
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
    {
        Some(v) => v,
        None => return Ok(None),
    };
    if parsed_objid != expected_objid {
        return Ok(None);
    }
    i = skip_ws(data, after1);

    // --- genno ---
    let (_genno_bytes, after2, float2) = match parse_num(data, i) {
        Some(r) => r,
        None => return Ok(None),
    };
    if float2 {
        return Ok(None);
    }
    i = skip_ws(data, after2);

    // --- "obj" keyword ---
    if i + 3 > n {
        return Ok(None);
    }
    if &data[i..i + 3] != b"obj" {
        return Ok(None);
    }
    let obj_end = i + 3;
    if obj_end < n && !is_ws(data[obj_end]) && !is_delim(data[obj_end]) {
        return Ok(None); // e.g. "objstm" — not the plain "obj" keyword
    }
    i = skip_ws(data, obj_end);

    // --- value ---
    let (val, consumed) = match parse_val(data, i) {
        Some(r) => r,
        None => return Ok(None),
    };

    // Stream objects need the file handle for body reading; fall back to Python.
    let after_val = skip_ws(data, consumed);
    if after_val + 6 <= n && &data[after_val..after_val + 6] == b"stream" {
        let stream_end = after_val + 6;
        if stream_end >= n || is_ws(data[stream_end]) || is_delim(data[stream_end]) {
            return Ok(None);
        }
    }

    let py_val = val_to_py(py, val, lit_fn, objref_fn)?;
    Ok(Some((py_val.into(), consumed)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resolve_stream_length, m)?)?;
    m.add_function(wrap_pyfunction!(get_predefined_colorspace_ncomponents, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pdf_object_at, m)?)?;
    Ok(())
}
