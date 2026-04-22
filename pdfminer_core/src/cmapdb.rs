use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::psparser::{
    decode_hex_string, hex_nibble, is_delimiter, is_whitespace, parse_literal_string,
};

/// Batch decode a list of CIDs to unicode strings using a cid2unichr dict.
///
/// Returns a list where each element is the unicode string for the CID,
/// or None if the CID is not in the map.
#[pyfunction]
pub fn unicodemap_decode_batch(
    py: Python<'_>,
    cid2unichr: &Bound<'_, PyDict>,
    cids: Vec<u32>,
) -> PyResult<Vec<Option<String>>> {
    let mut result = Vec::with_capacity(cids.len());
    for cid in cids {
        let key = cid.to_object(py);
        match cid2unichr.get_item(key.bind(py))? {
            Some(val) => result.push(val.extract::<Option<String>>()?),
            None => result.push(None),
        }
    }
    Ok(result)
}

/// Decode a byte sequence using a code2cid nested dict, returning CIDs.
///
/// Values are either an int CID (leaf) or a nested dict (multi-byte prefix).
#[pyfunction]
pub fn cmap_decode(
    py: Python<'_>,
    code2cid: &Bound<'_, PyDict>,
    code: &[u8],
) -> PyResult<Vec<u32>> {
    let mut result = Vec::with_capacity(code.len());
    let mut current = code2cid.clone().into_any();

    for &byte in code {
        let current_dict = current.downcast::<PyDict>()?;

        match current_dict.get_item(byte.to_object(py).bind(py))? {
            Some(value) => {
                if let Ok(cid) = value.extract::<u32>() {
                    result.push(cid);
                    current = code2cid.clone().into_any();
                } else if let Ok(nested) = value.downcast::<PyDict>() {
                    current = nested.clone().into_any();
                } else {
                    current = code2cid.clone().into_any();
                }
            }
            None => {
                current = code2cid.clone().into_any();
            }
        }
    }

    Ok(result)
}

// ─── CMap stream parser ───────────────────────────────────────────────────────

#[inline(always)]
fn is_end_kw(b: u8) -> bool {
    is_whitespace(b) || is_delimiter(b) || b == b'#'
}

#[derive(Debug, Clone)]
enum CMapToken {
    Integer(i64),
    Float(f64),
    Bytes(Vec<u8>),
    HexBytes(Vec<u8>),
    Literal(Vec<u8>), // /Name
    Keyword(Vec<u8>),
    Bool(bool),
    ArrayStart,
    ArrayEnd,
}

fn tokenize_cmap(data: &[u8]) -> Vec<CMapToken> {
    let mut tokens = Vec::new();
    let n = data.len();
    let mut i = 0usize;

    'outer: while i < n {
        // skip whitespace
        while i < n && is_whitespace(data[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }

        let c = data[i];
        let tok_start = i;

        match c {
            b'%' => {
                i += 1;
                while i < n && data[i] != b'\n' && data[i] != b'\r' {
                    i += 1;
                }
                continue 'outer;
            }

            b'/' => {
                i += 1;
                let mut name: Vec<u8> = Vec::new();
                loop {
                    if i >= n {
                        break;
                    }
                    let b = data[i];
                    if b == b'#' {
                        if i + 2 < n
                            && data[i + 1].is_ascii_hexdigit()
                            && data[i + 2].is_ascii_hexdigit()
                        {
                            name.push((hex_nibble(data[i + 1]) << 4) | hex_nibble(data[i + 2]));
                            i += 3;
                        } else {
                            break;
                        }
                    } else if is_whitespace(b) || is_delimiter(b) {
                        break;
                    } else {
                        name.push(b);
                        i += 1;
                    }
                }
                tokens.push(CMapToken::Literal(name));
            }

            b'0'..=b'9' | b'+' | b'-' => {
                i += 1;
                while i < n && data[i].is_ascii_digit() {
                    i += 1;
                }
                // Check for trailing decimal
                let is_float = i < n && data[i] == b'.';
                if is_float {
                    i += 1;
                    while i < n && data[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                // Flush trailing token without whitespace by treating end-of-data as terminated.
                let s = &data[tok_start..i];
                if let Ok(txt) = std::str::from_utf8(s) {
                    if is_float {
                        if let Ok(v) = txt.parse::<f64>() {
                            tokens.push(CMapToken::Float(v));
                        }
                    } else if let Ok(v) = txt.parse::<i64>() {
                        tokens.push(CMapToken::Integer(v));
                    } else if let Ok(v) = txt.parse::<f64>() {
                        tokens.push(CMapToken::Float(v));
                    }
                }
            }

            b'.' => {
                i += 1;
                while i < n && data[i].is_ascii_digit() {
                    i += 1;
                }
                let s = &data[tok_start..i];
                if let Ok(txt) = std::str::from_utf8(s) {
                    if let Ok(v) = txt.parse::<f64>() {
                        tokens.push(CMapToken::Float(v));
                    }
                }
            }

            b if b.is_ascii_alphabetic() || b == b'*' => {
                i += 1;
                while i < n && !is_end_kw(data[i]) {
                    i += 1;
                }
                // Flush at end-of-data even without a terminating character.
                let kw = &data[tok_start..i];
                match kw {
                    b"true" => tokens.push(CMapToken::Bool(true)),
                    b"false" => tokens.push(CMapToken::Bool(false)),
                    _ => tokens.push(CMapToken::Keyword(kw.to_vec())),
                }
            }

            b'[' => {
                i += 1;
                tokens.push(CMapToken::ArrayStart);
            }
            b']' => {
                i += 1;
                tokens.push(CMapToken::ArrayEnd);
            }
            b'{' | b'}' => {
                i += 1;
                tokens.push(CMapToken::Keyword(vec![c]));
            }

            b'<' => {
                if i + 1 < n && data[i + 1] == b'<' {
                    i += 2;
                    tokens.push(CMapToken::Keyword(b"<<".to_vec()));
                } else {
                    i += 1;
                    let hex_start = i;
                    while i < n && data[i] != b'>' {
                        i += 1;
                    }
                    let result = decode_hex_string(&data[hex_start..i]);
                    tokens.push(CMapToken::HexBytes(result));
                    if i < n {
                        i += 1; // consume '>'
                    }
                }
            }

            b'>' => {
                if i + 1 < n && data[i + 1] == b'>' {
                    i += 2;
                    tokens.push(CMapToken::Keyword(b">>".to_vec()));
                } else {
                    i += 1;
                    tokens.push(CMapToken::Keyword(vec![b'>']));
                }
            }

            b'(' => {
                i += 1;
                if let Some((s, end)) = parse_literal_string(data, i) {
                    tokens.push(CMapToken::Bytes(s));
                    i = end;
                }
            }

            _ => {
                i += 1;
                tokens.push(CMapToken::Keyword(vec![c]));
            }
        }
    }

    tokens
}

/// Unpack a big-endian byte slice into a u64.
fn nunpack(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |acc, &x| (acc << 8) | x as u64)
}

/// Decode bytes as UTF-16BE to a Rust String (lossy).
fn decode_utf16be(b: &[u8]) -> String {
    let u16s: Vec<u16> = if b.len() % 2 != 0 {
        let mut padded = b.to_vec();
        padded.push(0);
        padded
            .chunks(2)
            .map(|c| ((c[0] as u16) << 8) | c[1] as u16)
            .collect()
    } else {
        b.chunks(2)
            .map(|c| ((c[0] as u16) << 8) | c[1] as u16)
            .collect()
    };
    String::from_utf16_lossy(&u16s).to_string()
}

/// Parse a CMap stream and return `(cid2unichr, attrs)` Python dicts.
///
/// Fast-path for `CMapParser.run()`. Handles bfchar/bfrange/cidchar/cidrange,
/// attribute `def` pairs, and signals `usecmap` via `attrs["_usecmap"]`.
#[pyfunction]
pub fn parse_cmap_stream(py: Python<'_>, data: &[u8]) -> PyResult<(PyObject, PyObject)> {
    let tokens = tokenize_cmap(data);
    let n = tokens.len();

    let cid2unichr = PyDict::new_bound(py);
    let attrs = PyDict::new_bound(py);

    // Set cid2unichr[cid] = unicode_str, skipping   that would overwrite " ".
    let set_cid = |cid2unichr: &Bound<'_, PyDict>, cid: u32, unichr: &str| -> PyResult<()> {
        if unichr == "\u{00a0}" {
            let key = cid.to_object(py);
            if let Some(existing) = cid2unichr.get_item(key.bind(py))? {
                if existing.extract::<String>().ok().as_deref() == Some(" ") {
                    return Ok(());
                }
            }
        }
        cid2unichr.set_item(cid, unichr)?;
        Ok(())
    };

    let mut i = 0usize;
    let mut in_cmap = true; // some CMaps omit begincmap
    let mut stack: Vec<CMapToken> = Vec::new();

    while i < n {
        match &tokens[i] {
            CMapToken::Keyword(kw) => {
                let kw = kw.clone();
                i += 1;
                match kw.as_slice() {
                    b"begincmap" => {
                        in_cmap = true;
                        stack.clear();
                    }
                    b"endcmap" => {
                        in_cmap = false;
                    }
                    b"def" if in_cmap => {
                        if stack.len() >= 2 {
                            let v = stack.pop().unwrap();
                            let k = stack.pop().unwrap();
                            if let CMapToken::Literal(name) = k {
                                let key_str = String::from_utf8_lossy(&name).into_owned();
                                match v {
                                    CMapToken::Integer(n) => attrs.set_item(key_str, n)?,
                                    CMapToken::Float(f) => attrs.set_item(key_str, f)?,
                                    CMapToken::Bytes(b) | CMapToken::HexBytes(b) => {
                                        attrs.set_item(
                                            key_str,
                                            String::from_utf8_lossy(&b).into_owned(),
                                        )?;
                                    }
                                    CMapToken::Literal(b) => {
                                        attrs.set_item(
                                            key_str,
                                            String::from_utf8_lossy(&b).into_owned(),
                                        )?;
                                    }
                                    CMapToken::Bool(b) => attrs.set_item(key_str, b)?,
                                    _ => {}
                                }
                            }
                        }
                    }
                    b"begincodespacerange"
                    | b"endcodespacerange"
                    | b"beginnotdefrange"
                    | b"endnotdefrange"
                    | b"begincidrange"
                    | b"begincidchar"
                    | b"beginbfrange"
                    | b"beginbfchar" => {
                        stack.clear();
                    }
                    b"endcidrange" if in_cmap => {
                        let objs: Vec<CMapToken> = stack.drain(..).collect();
                        let mut j = 0;
                        while j + 2 < objs.len() {
                            let start_bytes = match &objs[j] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 3;
                                    continue;
                                }
                            };
                            let end_bytes = match &objs[j + 1] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 3;
                                    continue;
                                }
                            };
                            let base_cid = match &objs[j + 2] {
                                CMapToken::Integer(n) => *n,
                                _ => {
                                    j += 3;
                                    continue;
                                }
                            };
                            j += 3;
                            if start_bytes.len() != end_bytes.len() {
                                continue;
                            }
                            let vlen = start_bytes.len().min(4);
                            let prefix_len = start_bytes.len().saturating_sub(4);
                            if start_bytes[..prefix_len] != end_bytes[..prefix_len] {
                                continue;
                            }
                            let start = nunpack(&start_bytes[prefix_len..]);
                            let end = nunpack(&end_bytes[prefix_len..]);
                            for k in 0..=(end.saturating_sub(start)) {
                                let cid = base_cid + k as i64;
                                if cid < 0 {
                                    continue;
                                }
                                let packed = (start + k) as u32;
                                let packed_bytes = packed.to_be_bytes();
                                let mut code_bytes = start_bytes[..prefix_len].to_vec();
                                code_bytes.extend_from_slice(&packed_bytes[4 - vlen..]);
                                set_cid(&cid2unichr, cid as u32, &decode_utf16be(&code_bytes))?;
                            }
                        }
                    }
                    b"endcidchar" if in_cmap => {
                        let objs: Vec<CMapToken> = stack.drain(..).collect();
                        let mut j = 0;
                        while j + 1 < objs.len() {
                            let code_bytes = match &objs[j] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 2;
                                    continue;
                                }
                            };
                            let cid = match &objs[j + 1] {
                                CMapToken::Integer(n) => *n,
                                _ => {
                                    j += 2;
                                    continue;
                                }
                            };
                            j += 2;
                            if cid >= 0 {
                                set_cid(
                                    &cid2unichr,
                                    cid as u32,
                                    &decode_utf16be(&code_bytes),
                                )?;
                            }
                        }
                    }
                    b"endbfrange" if in_cmap => {
                        let objs: Vec<CMapToken> = stack.drain(..).collect();
                        let mut j = 0;
                        while j + 2 < objs.len() {
                            let start_bytes = match &objs[j] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 3;
                                    continue;
                                }
                            };
                            let end_bytes = match &objs[j + 1] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 3;
                                    continue;
                                }
                            };
                            let code_tok = objs[j + 2].clone();
                            j += 3;
                            if start_bytes.len() != end_bytes.len() {
                                continue;
                            }
                            let start = nunpack(&start_bytes) as u32;
                            let end = nunpack(&end_bytes) as u32;
                            match code_tok {
                                CMapToken::HexBytes(code) => {
                                    let vlen = code.len().min(4);
                                    let prefix_len = code.len().saturating_sub(4);
                                    let prefix = code[..prefix_len].to_vec();
                                    let base = nunpack(&code[prefix_len..]) as u32;
                                    for k in 0..=(end.saturating_sub(start)) {
                                        let cid = start + k;
                                        let code_val = base + k;
                                        let packed = code_val.to_be_bytes();
                                        let mut x = prefix.clone();
                                        x.extend_from_slice(&packed[4 - vlen..]);
                                        set_cid(&cid2unichr, cid, &decode_utf16be(&x))?;
                                    }
                                }
                                CMapToken::Bytes(code) => {
                                    let unichr = String::from_utf8_lossy(&code).into_owned();
                                    for k in 0..=(end.saturating_sub(start)) {
                                        set_cid(&cid2unichr, start + k, &unichr)?;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    b"endbfchar" if in_cmap => {
                        let objs: Vec<CMapToken> = stack.drain(..).collect();
                        let mut j = 0;
                        while j + 1 < objs.len() {
                            let cid_bytes = match &objs[j] {
                                CMapToken::HexBytes(b) => b.clone(),
                                _ => {
                                    j += 2;
                                    continue;
                                }
                            };
                            let code_tok = objs[j + 1].clone();
                            j += 2;
                            let cid = nunpack(&cid_bytes) as u32;
                            match code_tok {
                                CMapToken::HexBytes(unicode_bytes) => {
                                    set_cid(&cid2unichr, cid, &decode_utf16be(&unicode_bytes))?;
                                }
                                CMapToken::Bytes(s) => {
                                    set_cid(
                                        &cid2unichr,
                                        cid,
                                        &String::from_utf8_lossy(&s).into_owned(),
                                    )?;
                                }
                                CMapToken::Literal(name) => {
                                    // PSLiteral glyph names — name2unicode is Python-only.
                                    // Store the raw name; callers that need proper glyph
                                    // resolution will use the Python fallback.
                                    set_cid(
                                        &cid2unichr,
                                        cid,
                                        &String::from_utf8_lossy(&name).into_owned(),
                                    )?;
                                }
                                _ => {}
                            }
                        }
                    }
                    b"usecmap" => {
                        // Signal to Python caller via a reserved attrs key.
                        if let Some(CMapToken::Literal(name)) = stack.last() {
                            let cmap_name = String::from_utf8_lossy(name).into_owned();
                            attrs.set_item("_usecmap", cmap_name)?;
                        }
                        stack.clear();
                    }
                    _ => {
                        stack.push(CMapToken::Keyword(kw));
                    }
                }
            }

            CMapToken::ArrayStart => {
                // Eagerly collect array elements and immediately apply to the
                // preceding bfrange <start> <end> pair already on the stack.
                i += 1;
                let mut items: Vec<Vec<u8>> = Vec::new();
                while i < n {
                    match &tokens[i] {
                        CMapToken::ArrayEnd => {
                            i += 1;
                            break;
                        }
                        CMapToken::HexBytes(b) => {
                            items.push(b.clone());
                            i += 1;
                        }
                        CMapToken::Bytes(b) => {
                            items.push(b.clone());
                            i += 1;
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }
                // Stack must have <start_bytes> <end_bytes> from the enclosing bfrange.
                if stack.len() >= 2 {
                    let end_tok = stack.pop().unwrap();
                    let start_tok = stack.pop().unwrap();
                    if let (
                        CMapToken::HexBytes(start_bytes),
                        CMapToken::HexBytes(end_bytes),
                    ) = (start_tok, end_tok)
                    {
                        if start_bytes.len() == end_bytes.len() {
                            let start = nunpack(&start_bytes) as u32;
                            let end = nunpack(&end_bytes) as u32;
                            for (k, unicode_bytes) in (0u32..).zip(items.iter()) {
                                let cid = start + k;
                                if cid > end {
                                    break;
                                }
                                set_cid(&cid2unichr, cid, &decode_utf16be(unicode_bytes))?;
                            }
                        }
                    }
                }
                continue;
            }

            other => {
                stack.push(other.clone());
                i += 1;
            }
        }
    }

    Ok((cid2unichr.into(), attrs.into()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unicodemap_decode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cmap_decode, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cmap_stream, m)?)?;
    Ok(())
}
