use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Precomputed 85-base positional weights.
const W4: u32 = 52_200_625; // 85^4
const W3: u32 = 614_125;    // 85^3
const W2: u32 = 7_225;      // 85^2
const W1: u32 = 85;

/// Decode ASCII85-encoded data, mirroring the Python `ascii85decode` function.
///
/// Strips leading `<~` / `~` and trailing `~>` / `~` (with surrounding
/// whitespace), then decodes the Adobe ASCII85 body.
pub fn ascii85_decode_impl(data: &[u8]) -> Result<Vec<u8>, String> {
    decode_a85(strip_trailing(strip_leading(data)))
}

/// Strip `^\s*<?\s*~\s*` — mirrors the Python `start_re`.
fn strip_leading(data: &[u8]) -> &[u8] {
    let mut i = 0;
    while i < data.len() && data[i].is_ascii_whitespace() {
        i += 1;
    }
    let ws_end = i; // remember how far whitespace reached
    if i < data.len() && data[i] == b'<' {
        i += 1;
    }
    while i < data.len() && data[i].is_ascii_whitespace() {
        i += 1;
    }
    if i < data.len() && data[i] == b'~' {
        i += 1;
        while i < data.len() && data[i].is_ascii_whitespace() {
            i += 1;
        }
        &data[i..]
    } else {
        // No `~` marker found — strip only the leading whitespace.
        &data[ws_end..]
    }
}

/// Strip `\s*~\s*>?\s*$` — mirrors the Python `end_re`.
fn strip_trailing(data: &[u8]) -> &[u8] {
    let mut i = data.len();
    while i > 0 && data[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    let ws_start = i; // remember where trailing whitespace began
    if i > 0 && data[i - 1] == b'>' {
        i -= 1;
    }
    while i > 0 && data[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    if i > 0 && data[i - 1] == b'~' {
        i -= 1;
        while i > 0 && data[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        &data[..i]
    } else {
        // No `~` marker found — keep everything up to the trailing whitespace.
        &data[..ws_start]
    }
}

/// Emit four bytes from a fully-padded 5-element group array, writing
/// only the first `n` bytes into `out`.
#[inline]
fn emit_group(group: &[u32; 5], n: usize, out: &mut Vec<u8>) {
    let v = group[0] * W4 + group[1] * W3 + group[2] * W2 + group[3] * W1 + group[4];
    let bytes = [(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, v as u8];
    out.extend_from_slice(&bytes[..n]);
}

/// Decode Adobe ASCII85-encoded bytes (without markers).
///
/// Matches Python's `base64.a85decode()` behaviour: `!`–`u` are the 85
/// digits, `z` is shorthand for four zero bytes, groups of 5 chars →
/// 4 bytes, partial final group of n chars → n-1 bytes, whitespace ignored.
fn decode_a85(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut group = [0u32; 5];
    let mut group_len = 0usize;

    for &b in data {
        if b.is_ascii_whitespace() {
            continue;
        }
        if b == b'z' {
            if group_len != 0 {
                return Err("'z' inside ASCII85 group".to_string());
            }
            out.extend_from_slice(&[0u8; 4]);
            continue;
        }
        if b < b'!' || b > b'u' {
            return Err(format!("invalid ASCII85 character: 0x{b:02x}"));
        }
        group[group_len] = (b - b'!') as u32;
        group_len += 1;
        if group_len == 5 {
            emit_group(&group, 4, &mut out);
            group_len = 0;
        }
    }

    // Partial final group: n chars → n-1 bytes.
    // Pad missing positions with 84 (`u`), matching Python's a85decode.
    if group_len > 0 {
        for slot in &mut group[group_len..5] {
            *slot = 84;
        }
        emit_group(&group, group_len - 1, &mut out);
    }

    Ok(out)
}

/// Decode ASCIIHex-encoded data, mirroring the Python `asciihexdecode` function.
///
/// Strips whitespace, stops at `>` (EOD), pads with `0` when an odd number
/// of hex digits precedes `>`, then decodes hex pairs.
pub fn asciihex_decode_impl(data: &[u8]) -> Result<Vec<u8>, String> {
    // Collect non-whitespace hex digits; stop at `>` and record whether the
    // count was odd so we can pad — matching Python's `idx % 2 == 1` check.
    let mut hex_bytes: Vec<u8> = Vec::with_capacity(data.len() / 2 + 1);
    let mut hit_eod = false;

    'outer: for &b in data {
        match b {
            b if b.is_ascii_whitespace() => {}
            b'>' => {
                if hex_bytes.len() % 2 == 1 {
                    hex_bytes.push(b'0');
                }
                hit_eod = true;
                break 'outer;
            }
            _ => hex_bytes.push(b),
        }
    }

    // Without `>`, an odd count is an error (same as Python's unhexlify).
    if !hit_eod && hex_bytes.len() % 2 != 0 {
        return Err(format!(
            "Odd-length hex string: {} bytes",
            hex_bytes.len()
        ));
    }

    let mut out = Vec::with_capacity(hex_bytes.len() / 2);
    for chunk in hex_bytes.chunks(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }

    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex character: 0x{b:02x}")),
    }
}

#[pyfunction]
pub fn ascii85decode(data: &[u8]) -> PyResult<Vec<u8>> {
    ascii85_decode_impl(data).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn asciihexdecode(data: &[u8]) -> PyResult<Vec<u8>> {
    asciihex_decode_impl(data).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ascii85decode, m)?)?;
    m.add_function(wrap_pyfunction!(asciihexdecode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ASCII85 tests ────────────────────────────────────────────────────────

    #[test]
    fn test_a85_basic() {
        // "9jqo^" encodes "Man " in ASCII85
        assert_eq!(ascii85_decode_impl(b"9jqo^~>").unwrap(), b"Man ");
    }

    #[test]
    fn test_a85_with_start_delimiter() {
        assert_eq!(ascii85_decode_impl(b"<~9jqo^~>").unwrap(), b"Man ");
    }

    #[test]
    fn test_a85_z_shorthand() {
        // 'z' = four zero bytes
        assert_eq!(
            ascii85_decode_impl(b"z~>").unwrap(),
            b"\x00\x00\x00\x00"
        );
    }

    #[test]
    fn test_a85_partial_1_byte() {
        // 1 byte "\x00" → "!!" (2 chars)
        assert_eq!(ascii85_decode_impl(b"!!~>").unwrap(), b"\x00");
    }

    #[test]
    fn test_a85_partial_2_bytes() {
        // "\x00\x01" → "!!*" (3 chars)
        assert_eq!(ascii85_decode_impl(b"!!*~>").unwrap(), b"\x00\x01");
    }

    #[test]
    fn test_a85_partial_3_bytes() {
        assert_eq!(ascii85_decode_impl(b"!!*-~>").unwrap(), b"\x00\x01\x02");
    }

    #[test]
    fn test_a85_hello_world() {
        // "Hello, World!" — encoded by base64.a85encode
        assert_eq!(
            ascii85_decode_impl(b"87cURD_*#4DfTZ)+T~>").unwrap(),
            b"Hello, World!"
        );
    }

    #[test]
    fn test_a85_no_end_marker() {
        // Works without ~>
        assert_eq!(ascii85_decode_impl(b"9jqo^").unwrap(), b"Man ");
    }

    // ── ASCIIHex tests ───────────────────────────────────────────────────────

    #[test]
    fn test_hex_basic() {
        assert_eq!(asciihex_decode_impl(b"48656c6c6f").unwrap(), b"Hello");
    }

    #[test]
    fn test_hex_with_spaces() {
        assert_eq!(
            asciihex_decode_impl(b"48 65 6c 6c 6f").unwrap(),
            b"Hello"
        );
    }

    #[test]
    fn test_hex_with_terminator() {
        assert_eq!(asciihex_decode_impl(b"48656c6c6f>").unwrap(), b"Hello");
    }

    #[test]
    fn test_hex_uppercase() {
        assert_eq!(asciihex_decode_impl(b"48656C6C6F").unwrap(), b"Hello");
    }

    #[test]
    fn test_hex_odd_with_terminator() {
        // 9 hex chars before '>' → pads with '0'
        // "48656c6c6" + "0" = b"Hello`" (0x60 = `)
        let result = asciihex_decode_impl(b"48656c6c6>").unwrap();
        assert_eq!(result, b"\x48\x65\x6c\x6c\x60");
    }

    #[test]
    fn test_hex_stop_at_terminator() {
        // Data after '>' is ignored
        assert_eq!(
            asciihex_decode_impl(b"4865>6c6c6f").unwrap(),
            b"\x48\x65"
        );
    }

    #[test]
    fn test_hex_empty() {
        assert_eq!(asciihex_decode_impl(b"").unwrap(), b"");
        assert_eq!(asciihex_decode_impl(b">").unwrap(), b"");
    }
}
