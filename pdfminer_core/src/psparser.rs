use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple, PyType};

// ─── Token types (must match Python side) ────────────────────────────────────
// 0 = Integer   (i64)
// 1 = Float     (f64)
// 2 = String    (bytes)  – parsed literal string  (...)
// 3 = Literal   (bytes)  – /Name
// 4 = Keyword   (bytes)
// 5 = HexString (bytes)  – <...>
// 6 = Bool true
// 7 = Bool false

#[derive(Debug)]
enum PsToken {
    Integer(i64),
    Float(f64),
    Bytes(Vec<u8>),    // string literal
    Literal(Vec<u8>),  // /Name
    Keyword(Vec<u8>),  // keyword
    HexString(Vec<u8>),
    Bool(bool),
}

// ─── Byte classification helpers ─────────────────────────────────────────────

#[inline(always)]
pub(crate) fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'\x0c' | b'\x00')
}

#[inline(always)]
pub(crate) fn is_delimiter(b: u8) -> bool {
    matches!(b, b'%' | b'/' | b'[' | b']' | b'(' | b')' | b'<' | b'>' | b'{' | b'}')
}

#[inline(always)]
pub(crate) fn is_end_keyword(b: u8) -> bool {
    is_whitespace(b) || is_delimiter(b) || b == b'#'
}

#[inline(always)]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

// ─── Core tokenizer ──────────────────────────────────────────────────────────

/// Parse as many complete tokens as possible from `data`, starting at `offset`.
///
/// Returns `(tokens, bytes_consumed)` where:
///   - each token is `(absolute_position, PsToken)`
///   - `bytes_consumed` is how many bytes of `data` were consumed
///
/// Stops and returns the current position when a token would need more data
/// (i.e. could be incomplete at the buffer boundary).
fn tokenize_buffer(data: &[u8], base_offset: usize) -> (Vec<(usize, PsToken)>, usize) {
    let mut tokens: Vec<(usize, PsToken)> = Vec::new();
    let mut i = 0usize;
    let n = data.len();

    macro_rules! need {
        ($count:expr) => {
            if i + $count > n {
                return (tokens, i);
            }
        };
    }

    'outer: while i < n {
        // ── Skip whitespace ───────────────────────────────────────────────
        while i < n && is_whitespace(data[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }

        let tok_start = i;
        let abs_pos = base_offset + tok_start;
        let c = data[i];

        match c {
            // ── Comment: % … EOL ─────────────────────────────────────────
            b'%' => {
                i += 1;
                while i < n && data[i] != b'\n' && data[i] != b'\r' {
                    i += 1;
                }
                // No token emitted – comments are discarded.
                continue 'outer;
            }

            // ── Literal name: /…  ─────────────────────────────────────────
            b'/' => {
                i += 1;
                let mut name: Vec<u8> = Vec::new();
                // Loop until we hit a delimiter/whitespace that is NOT '#'.
                // '#' is the hex-escape prefix inside a literal name (#XY).
                loop {
                    if i >= n {
                        // End of buffer in the middle of a literal name.
                        // Can't tell if complete – bail to Python.
                        return (tokens, tok_start);
                    }
                    let b = data[i];
                    if b == b'#' {
                        // Hex escape: #XY
                        if i + 2 >= n {
                            // Not enough bytes to complete the escape.
                            return (tokens, tok_start);
                        }
                        let hi = data[i + 1];
                        let lo = data[i + 2];
                        if is_hex_digit(hi) && is_hex_digit(lo) {
                            let byte_val = (hex_nibble(hi) << 4) | hex_nibble(lo);
                            name.push(byte_val);
                            i += 3;
                        } else {
                            // '#' not followed by two hex digits – end of name.
                            break;
                        }
                    } else if is_whitespace(b) || is_delimiter(b) {
                        // Standard end-of-literal character.
                        break;
                    } else {
                        name.push(b);
                        i += 1;
                    }
                }
                tokens.push((abs_pos, PsToken::Literal(name)));
            }

            // ── Number: digit, sign, or dot ──────────────────────────────
            b'0'..=b'9' | b'+' | b'-' => {
                i += 1;
                while i < n && data[i].is_ascii_digit() {
                    i += 1;
                }
                if i >= n {
                    // Could be incomplete integer at end of buffer.
                    return (tokens, tok_start);
                }
                if data[i] == b'.' {
                    // Float
                    i += 1;
                    while i < n && data[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i >= n {
                        return (tokens, tok_start);
                    }
                    // Verify termination
                    if !is_whitespace(data[i]) && !is_delimiter(data[i]) {
                        // Not terminated yet – ambiguous, fall back
                        return (tokens, tok_start);
                    }
                    let s = &data[tok_start..i];
                    if let Ok(txt) = std::str::from_utf8(s) {
                        if let Ok(v) = txt.parse::<f64>() {
                            tokens.push((abs_pos, PsToken::Float(v)));
                        }
                    }
                } else {
                    // Verify termination
                    if !is_whitespace(data[i]) && !is_delimiter(data[i]) {
                        // Could be a keyword starting with a digit... rare but
                        // fall back to Python for safety.
                        return (tokens, tok_start);
                    }
                    let s = &data[tok_start..i];
                    if let Ok(txt) = std::str::from_utf8(s) {
                        if let Ok(v) = txt.parse::<i64>() {
                            tokens.push((abs_pos, PsToken::Integer(v)));
                        } else if let Ok(v) = txt.parse::<f64>() {
                            tokens.push((abs_pos, PsToken::Float(v)));
                        }
                    }
                }
            }

            // ── Float starting with dot: .NNN ─────────────────────────────
            b'.' => {
                i += 1;
                while i < n && data[i].is_ascii_digit() {
                    i += 1;
                }
                if i >= n {
                    return (tokens, tok_start);
                }
                if !is_whitespace(data[i]) && !is_delimiter(data[i]) {
                    return (tokens, tok_start);
                }
                let s = &data[tok_start..i];
                if let Ok(txt) = std::str::from_utf8(s) {
                    if let Ok(v) = txt.parse::<f64>() {
                        tokens.push((abs_pos, PsToken::Float(v)));
                    }
                }
            }

            // ── Keyword / boolean ─────────────────────────────────────────
            b if b.is_ascii_alphabetic() || b == b'*' => {
                i += 1;
                while i < n && !is_end_keyword(data[i]) {
                    i += 1;
                }
                if i >= n {
                    // Might be incomplete keyword at end of buffer
                    return (tokens, tok_start);
                }
                let kw = &data[tok_start..i];
                match kw {
                    b"true" => tokens.push((abs_pos, PsToken::Bool(true))),
                    b"false" => tokens.push((abs_pos, PsToken::Bool(false))),
                    _ => tokens.push((abs_pos, PsToken::Keyword(kw.to_vec()))),
                }
            }

            // ── Single-char keywords: [ ] { } ─────────────────────────────
            b'[' | b']' | b'{' | b'}' => {
                i += 1;
                tokens.push((abs_pos, PsToken::Keyword(vec![c])));
            }

            // ── << or < hex-string > ─────────────────────────────────────
            b'<' => {
                need!(2);
                if data[i + 1] == b'<' {
                    // Dictionary begin
                    i += 2;
                    tokens.push((abs_pos, PsToken::Keyword(b"<<".to_vec())));
                } else {
                    // Hex string: <…>
                    i += 1; // consume '<'
                    let hex_start = i;
                    // Find closing '>'
                    let mut j = i;
                    while j < n && data[j] != b'>' {
                        // Non-hex, non-whitespace: break (invalid or end-of-buffer)
                        if !is_hex_digit(data[j]) && !is_whitespace(data[j]) {
                            // Stop – return partial to Python
                            return (tokens, tok_start);
                        }
                        j += 1;
                    }
                    if j >= n {
                        // Closing '>' not found – incomplete buffer
                        return (tokens, tok_start);
                    }
                    // data[j] == b'>'
                    let hex_bytes = &data[hex_start..j];
                    let result = decode_hex_string(hex_bytes);
                    tokens.push((abs_pos, PsToken::HexString(result)));
                    i = j + 1; // consume '>'
                }
            }

            // ── >> ───────────────────────────────────────────────────────
            b'>' => {
                need!(2);
                if data[i + 1] == b'>' {
                    i += 2;
                    tokens.push((abs_pos, PsToken::Keyword(b">>".to_vec())));
                } else {
                    // Lone '>' – treated as a keyword by Python
                    i += 1;
                    tokens.push((abs_pos, PsToken::Keyword(vec![b'>'])));
                }
            }

            // ── Literal string: (…) ───────────────────────────────────────
            b'(' => {
                i += 1; // consume '('
                let result = match parse_literal_string(data, i) {
                    Some((s, end_pos)) => {
                        i = end_pos;
                        s
                    }
                    None => {
                        // Incomplete – return to Python
                        return (tokens, tok_start);
                    }
                };
                tokens.push((abs_pos, PsToken::Bytes(result)));
            }

            // ── Null byte – skip ──────────────────────────────────────────
            b'\x00' => {
                i += 1;
            }

            // ── Any other single character becomes a keyword ──────────────
            _ => {
                i += 1;
                tokens.push((abs_pos, PsToken::Keyword(vec![c])));
            }
        }
    }

    (tokens, i)
}

/// Decode a PS hex string (whitespace allowed between hex digits).
///
/// Single-pass: no intermediate digit buffer is allocated.
pub(crate) fn decode_hex_string(hex_bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity((hex_bytes.len() + 1) / 2);
    let mut hi: Option<u8> = None;
    for &b in hex_bytes {
        if !is_hex_digit(b) {
            continue; // skip whitespace
        }
        match hi {
            None => hi = Some(hex_nibble(b)),
            Some(h) => {
                result.push((h << 4) | hex_nibble(b));
                hi = None;
            }
        }
    }
    // Odd trailing nibble: treat as if followed by '0'
    if let Some(h) = hi {
        result.push(h << 4);
    }
    result
}

#[inline(always)]
pub(crate) fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Parse a PS literal string starting right after the opening '('.
///
/// Returns `Some((bytes, new_pos))` where `new_pos` is past the closing ')`.
/// Returns `None` if the string is incomplete (buffer boundary).
pub(crate) fn parse_literal_string(data: &[u8], start: usize) -> Option<(Vec<u8>, usize)> {
    let n = data.len();
    let mut result = Vec::new();
    let mut i = start;
    let mut paren_depth = 1i32; // we already consumed the outer '('

    while i < n {
        match data[i] {
            b'\\' => {
                i += 1;
                if i >= n {
                    return None; // incomplete escape
                }
                match data[i] {
                    // Octal escape: \NNN (1-3 octal digits)
                    d @ b'0'..=b'7' => {
                        let mut oct_str = [d, 0u8, 0u8];
                        let mut oct_len = 1usize;
                        let mut j = i + 1;
                        while j < n && oct_len < 3 {
                            if matches!(data[j], b'0'..=b'7') {
                                oct_str[oct_len] = data[j];
                                oct_len += 1;
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        let oct_val = oct_str[..oct_len]
                            .iter()
                            .fold(0u32, |acc, &b| acc * 8 + (b - b'0') as u32);
                        result.push((oct_val & 0xFF) as u8);
                        i = j;
                        continue;
                    }
                    b'b' => result.push(8),
                    b't' => result.push(9),
                    b'n' => result.push(10),
                    b'f' => result.push(12),
                    b'r' => result.push(13),
                    b'(' => result.push(b'('),
                    b')' => result.push(b')'),
                    b'\\' => result.push(b'\\'),
                    b'\r' => {
                        // \r or \r\n – line continuation, skip
                        i += 1;
                        if i < n && data[i] == b'\n' {
                            i += 1;
                        }
                        continue;
                    }
                    b'\n' => {
                        // \n – line continuation, skip
                        i += 1;
                        continue;
                    }
                    other => {
                        // Unknown escape – just emit the character
                        result.push(other);
                    }
                }
                i += 1;
            }
            b'(' => {
                paren_depth += 1;
                result.push(b'(');
                i += 1;
            }
            b')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    return Some((result, i + 1));
                }
                result.push(b')');
                i += 1;
            }
            other => {
                result.push(other);
                i += 1;
            }
        }
    }
    None // ran out of data
}

// ─── PyO3 bridge ─────────────────────────────────────────────────────────────

/// Convert a `PsToken` to a Python value.
///
/// - `Integer`, `Float`, `Bool`, `Bytes`, `HexString` → native Python objects
/// - `Literal` → `(3, name_bytes)` tuple  (caller must call `LIT()`)
/// - `Keyword` → `(4, kw_bytes)` tuple    (caller must call `KWD()`)
#[inline]
fn token_to_pyobject(py: Python<'_>, tok: PsToken) -> PyResult<PyObject> {
    match tok {
        PsToken::Integer(v) => Ok(v.to_object(py)),
        PsToken::Float(v) => Ok(v.to_object(py)),
        PsToken::Bool(b) => Ok(b.to_object(py)),
        PsToken::Bytes(v) => Ok(PyBytes::new_bound(py, &v).into()),
        PsToken::HexString(v) => Ok(PyBytes::new_bound(py, &v).into()),
        PsToken::Literal(v) => {
            let type_obj: PyObject = 3u8.to_object(py);
            let val_obj: PyObject = PyBytes::new_bound(py, &v).into();
            Ok(PyTuple::new_bound(py, [type_obj, val_obj]).into())
        }
        PsToken::Keyword(v) => {
            let type_obj: PyObject = 4u8.to_object(py);
            let val_obj: PyObject = PyBytes::new_bound(py, &v).into();
            Ok(PyTuple::new_bound(py, [type_obj, val_obj]).into())
        }
    }
}

/// Python-callable: tokenize a bytes buffer and return
/// `(list_of_(pos, value), bytes_consumed)`.
///
/// Each element is a 2-tuple `(position, value)` where `value` is:
/// - a native Python `int`, `float`, `bool`, or `bytes` for those token types
/// - a `(3, name_bytes)` tuple for PS literals (caller must call `LIT()`)
/// - a `(4, kw_bytes)` tuple for PS keywords (caller must call `KWD()`)
#[pyfunction]
pub fn tokenize_ps_buffer(
    py: Python<'_>,
    data: &[u8],
    base_offset: usize,
) -> PyResult<(PyObject, usize)> {
    let (tok_list, consumed) = tokenize_buffer(data, base_offset);

    let py_list = PyList::empty_bound(py);
    for (pos, tok) in tok_list {
        let pos_obj: PyObject = pos.to_object(py);
        let val_obj = token_to_pyobject(py, tok)?;
        let tup = PyTuple::new_bound(py, [pos_obj, val_obj]);
        py_list.append(tup)?;
    }

    Ok((py_list.into(), consumed))
}

/// Tokenize an entire content stream at once, returning all tokens.
///
/// Unlike `tokenize_ps_buffer` which stops at buffer boundaries, this function
/// processes the whole input in one shot.  Because we have the complete data,
/// there are no partial-token boundary concerns.
///
/// Returns a list of `(position, value)` 2-tuples using the same format as
/// `tokenize_ps_buffer`.  The caller converts literals/keywords using
/// `_convert_rust_tokens`.
#[pyfunction]
pub fn tokenize_full_stream(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<PyObject> {
    let n = data.len();
    let (mut all_tokens, consumed) = tokenize_buffer(data, 0);

    // tokenize_buffer stops when a token at the end of the slice lacks a
    // terminating whitespace.  Appending a newline and re-running flushes it.
    if consumed < n {
        let mut tail = data[consumed..].to_vec();
        tail.push(b'\n');
        let (extra_tokens, _) = tokenize_buffer(&tail, consumed);
        all_tokens.extend(extra_tokens);
    }

    let py_list = PyList::empty_bound(py);
    for (pos, tok) in all_tokens {
        let pos_obj: PyObject = pos.to_object(py);
        let val_obj = token_to_pyobject(py, tok)?;
        let tup = PyTuple::new_bound(py, [pos_obj, val_obj]);
        py_list.append(tup)?;
    }

    Ok(py_list.into())
}

// ─── next_object_from_tokens ─────────────────────────────────────────────────

/// Extract the `.name` bytes from a PSKeyword object without heap-allocating
/// when the name is already a `bytes` object (the common case).
#[inline]
fn kw_name_bytes(tok: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let attr = tok.getattr("name")?;
    if let Ok(b) = attr.downcast::<PyBytes>() {
        Ok(b.as_bytes().to_vec())
    } else {
        attr.extract::<Vec<u8>>()
    }
}

/// Build a Python dict from a flat slice of values taken pairwise as (key, value).
/// Keys that are PSLiteral objects are converted to str via their `.name` attribute
/// (UTF-8 decoded), matching what `literal_name()` does on the Python side.
/// None values are skipped (matches the `if v is not None` guard in PSStackParser).
#[inline]
fn build_dict<'py>(
    py: Python<'py>,
    pairs: &[PyObject],
    lit_type: &Bound<'_, PyType>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for chunk in pairs.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let key = chunk[0].bind(py);
        let val = chunk[1].bind(py);
        if val.is_none() {
            continue;
        }
        // Derive string key: PSLiteral.name decoded as UTF-8 (str fallback for non-UTF-8).
        let str_key: PyObject = if key.is_instance(lit_type).unwrap_or(false) {
            match key.getattr("name") {
                Ok(attr) => match attr.downcast::<PyBytes>() {
                    Ok(b) => match std::str::from_utf8(b.as_bytes()) {
                        Ok(s) => s.to_object(py),
                        Err(_) => attr.to_object(py),
                    },
                    Err(_) => attr.to_object(py), // already a str
                },
                Err(_) => key.to_object(py),
            }
        } else {
            key.to_object(py)
        };
        d.set_item(str_key, val)?;
    }
    Ok(d)
}

/// Process tokens from a pre-tokenized list to build the next complete object.
///
/// `tokens` is the `_pretokenized_list` — a Python list of `(pos, token)` 2-tuples
/// where `token` is already a Python object (int, float, bool, bytes, PSLiteral,
/// PSKeyword, …).
///
/// Returns `Some((object, new_pos))` where `new_pos` is the index *after* the
/// last consumed token, or `None` if `start_pos >= len(tokens)`.
///
/// Handles nested `[ … ]` (array) and `<< … >>` (dict) construction.
/// For PSKeyword tokens that are not structural (`[`, `]`, `<<`, `>>`), the
/// keyword object is returned directly so the caller handles dispatch.
#[pyfunction]
#[pyo3(signature = (tokens, start_pos, keyword_type, literal_type))]
pub fn next_object_from_tokens(
    py: Python<'_>,
    tokens: &Bound<'_, PyList>,
    start_pos: usize,
    keyword_type: &Bound<'_, PyType>,
    literal_type: &Bound<'_, PyType>,
) -> PyResult<Option<(PyObject, usize)>> {
    let n = tokens.len();
    if start_pos >= n {
        return Ok(None);
    }

    let first = tokens.get_item(start_pos)?;
    let first_tup = first.downcast::<PyTuple>()?;
    let pos_obj: PyObject = first_tup.get_item(0)?.into();
    let first_token = first_tup.get_item(1)?;

    // int, float, bool, bytes, PSLiteral — return immediately.
    if !first_token.is_instance(keyword_type)? {
        let obj_tuple = PyTuple::new_bound(py, [pos_obj, first_token.into()]);
        return Ok(Some((obj_tuple.into(), start_pos + 1)));
    }

    let kw_name = kw_name_bytes(&first_token)?;

    match kw_name.as_slice() {
        b"[" => {
            // Collect tokens until the matching `]`, handling nesting.
            let mut items: Vec<PyObject> = Vec::new();
            let mut depth: usize = 1;
            let mut idx = start_pos + 1;

            while idx < n {
                let entry = tokens.get_item(idx)?;
                let tup = entry.downcast::<PyTuple>()?;
                let tok = tup.get_item(1)?;
                idx += 1;

                if tok.is_instance(keyword_type)? {
                    match kw_name_bytes(&tok)?.as_slice() {
                        b"[" => { depth += 1; items.push(tok.into()); }
                        b"]" => {
                            depth -= 1;
                            if depth == 0 {
                                let list = PyList::new_bound(py, items.iter().map(|o| o.bind(py)));
                                let obj_tuple = PyTuple::new_bound(py, [pos_obj, list.into_any().into()]);
                                return Ok(Some((obj_tuple.into(), idx)));
                            }
                            items.push(tok.into());
                        }
                        _ => { items.push(tok.into()); }
                    }
                } else {
                    items.push(tok.into());
                }
            }
            // Unterminated array — lenient fallback, matches PSStackParser.
            let list = PyList::new_bound(py, items.iter().map(|o| o.bind(py)));
            let obj_tuple = PyTuple::new_bound(py, [pos_obj, list.into_any().into()]);
            Ok(Some((obj_tuple.into(), idx)))
        }

        b"<<" => {
            // Collect token pairs until the matching `>>`, handling nesting.
            let mut pairs: Vec<PyObject> = Vec::new();
            let mut depth: usize = 1;
            let mut idx = start_pos + 1;

            while idx < n {
                let entry = tokens.get_item(idx)?;
                let tup = entry.downcast::<PyTuple>()?;
                let tok = tup.get_item(1)?;
                idx += 1;

                if tok.is_instance(keyword_type)? {
                    match kw_name_bytes(&tok)?.as_slice() {
                        b"<<" => { depth += 1; pairs.push(tok.into()); }
                        b">>" => {
                            depth -= 1;
                            if depth == 0 {
                                let d = build_dict(py, &pairs, literal_type)?;
                                let obj_tuple = PyTuple::new_bound(py, [pos_obj, d.into_any().into()]);
                                return Ok(Some((obj_tuple.into(), idx)));
                            }
                            pairs.push(tok.into());
                        }
                        _ => { pairs.push(tok.into()); }
                    }
                } else {
                    pairs.push(tok.into());
                }
            }
            // Unterminated dict — lenient fallback.
            let d = build_dict(py, &pairs, literal_type)?;
            let obj_tuple = PyTuple::new_bound(py, [pos_obj, d.into_any().into()]);
            Ok(Some((obj_tuple.into(), idx)))
        }

        // `]`, `>>` at top level, and all PDF operator keywords — pass through.
        _ => {
            let obj_tuple = PyTuple::new_bound(py, [pos_obj, first_token.into()]);
            Ok(Some((obj_tuple.into(), start_pos + 1)))
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_ps_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_full_stream, m)?)?;
    m.add_function(wrap_pyfunction!(next_object_from_tokens, m)?)?;
    Ok(())
}
