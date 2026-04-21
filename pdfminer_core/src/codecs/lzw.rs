use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Decode PDF LZW-compressed data.
///
/// Mirrors the behaviour of `LZWDecoder` in `pdfminer/lzw.py` exactly:
/// - 9-bit initial code width, growing to 10/11/12 bits.
/// - Code 256 = clear/reset, code 257 = EOD.
/// - earlychange = 1 (code width bumps when table length hits 511/1023/2047).
/// - Corrupt data is silently truncated (same as `except CorruptDataError: break`).
pub fn lzw_decode_impl(data: &[u8]) -> Result<Vec<u8>, String> {
    const CLEAR: u32 = 256;
    const EOD: u32 = 257;

    // ── bit-stream state ──────────────────────────────────────────────────────
    let mut buf: u32 = 0; // current byte from the input stream
    let mut bpos: u32 = 8; // bits consumed from `buf` (8 = buffer empty)
    let mut data_pos: usize = 0; // next byte to consume from `data`

    // Read `bits` bits MSB-first.  Returns None on EOF.
    macro_rules! readbits {
        ($bits:expr) => {{
            let mut remaining: u32 = $bits;
            let mut v: u32 = 0;
            let result: Option<u32> = loop {
                let r = 8 - bpos; // bits still available in current byte
                if remaining <= r {
                    v = (v << remaining)
                        | ((buf >> (r - remaining)) & ((1 << remaining) - 1));
                    bpos += remaining;
                    break Some(v);
                } else {
                    v = (v << r) | (buf & ((1 << r) - 1));
                    remaining -= r;
                    if data_pos >= data.len() {
                        break None; // EOF
                    }
                    buf = data[data_pos] as u32;
                    data_pos += 1;
                    bpos = 0;
                }
            };
            result
        }};
    }

    // ── LZW string table ──────────────────────────────────────────────────────
    // Each entry is a contiguous slice stored in `strings`.
    // We record the (start, end) byte offsets in two parallel Vecs.
    // Entries 0-255 are single-byte literals; 256/257 are sentinels.
    let mut string_starts: Vec<u32> = Vec::with_capacity(4096);
    let mut string_ends: Vec<u32> = Vec::with_capacity(4096);
    let mut strings: Vec<u8> = Vec::with_capacity(1 << 16);

    // Initialise (or re-initialise after a CLEAR code).
    macro_rules! reset_table {
        () => {{
            string_starts.clear();
            string_ends.clear();
            strings.clear();
            for b in 0u8..=255u8 {
                let start = strings.len() as u32;
                strings.push(b);
                string_starts.push(start);
                string_ends.push(start + 1);
            }
            // Sentinel slots for CLEAR (256) and EOD (257) – never dereferenced.
            string_starts.push(0);
            string_ends.push(0);
            string_starts.push(0);
            string_ends.push(0);
            9u32 // initial nbits
        }};
    }

    let mut nbits: u32 = reset_table!();
    // prevbuf: (start, end) of the previous output sequence in `strings`.
    // None means "we just reset – next code is the first after a CLEAR".
    let mut prevbuf: Option<(usize, usize)> = None;

    let mut output: Vec<u8> = Vec::new();

    loop {
        let code = match readbits!(nbits) {
            Some(c) => c,
            None => break, // EOF → stop (same as PDFEOFError → break)
        };

        if code == CLEAR {
            nbits = reset_table!();
            prevbuf = None;
            continue;
        }

        if code == EOD {
            break;
        }

        let table_len = string_starts.len();
        let code_idx = code as usize;

        // ── decode the entry for `code` ────────────────────────────────────
        // We need to know entry[0] (first byte) and entry[..] (whole string)
        // to build the new table entry and write output.  To avoid borrow
        // conflicts we materialise them into a small local buffer.

        let (entry_first_byte, entry): (u8, Vec<u8>) = match prevbuf {
            None => {
                // First code after reset: just emit the literal.
                if code_idx >= table_len {
                    break; // corrupt
                }
                let (s, e) = (
                    string_starts[code_idx] as usize,
                    string_ends[code_idx] as usize,
                );
                let slice = strings[s..e].to_vec();
                output.extend_from_slice(&slice);
                prevbuf = Some((s, e));
                continue; // no new table entry on first code after CLEAR
            }
            Some((ps, pe)) => {
                if code_idx < table_len {
                    // Known code.
                    let (s, e) = (
                        string_starts[code_idx] as usize,
                        string_ends[code_idx] as usize,
                    );
                    let slice = strings[s..e].to_vec();
                    let first = slice[0];
                    (first, slice)
                } else if code_idx == table_len {
                    // Special case: code == next table index.
                    // entry = prev + prev[0]
                    let prev_slice = strings[ps..pe].to_vec();
                    let first = prev_slice[0];
                    let mut entry = prev_slice;
                    entry.push(first);
                    (first, entry)
                } else {
                    break; // corrupt
                }
            }
        };

        // ── add new entry to the table: prev + entry[0] ───────────────────
        let (ps, pe) = prevbuf.unwrap();
        let prev_slice = strings[ps..pe].to_vec();
        let new_start = strings.len() as u32;
        strings.extend_from_slice(&prev_slice);
        strings.push(entry_first_byte);
        let new_end = strings.len() as u32;
        string_starts.push(new_start);
        string_ends.push(new_end);

        // ── output the decoded entry ──────────────────────────────────────
        output.extend_from_slice(&entry);

        // Store start/end of the just-output string so the *next* iteration
        // can use it as `prev`.  If this was the special `code == table_len`
        // case the entry was just appended, otherwise it was a pre-existing
        // entry.
        let entry_code_idx = if code_idx < table_len {
            code_idx
        } else {
            // code_idx == old table_len, which is now the last element.
            string_starts.len() - 1
        };
        prevbuf = Some((
            string_starts[entry_code_idx] as usize,
            string_ends[entry_code_idx] as usize,
        ));

        // ── adjust nbits (earlychange = 1) ────────────────────────────────
        let new_table_len = string_starts.len();
        if new_table_len == 511 {
            nbits = 10;
        } else if new_table_len == 1023 {
            nbits = 11;
        } else if new_table_len == 2047 {
            nbits = 12;
        }
    }

    Ok(output)
}

#[pyfunction]
pub fn lzw_decode(data: &[u8]) -> PyResult<Vec<u8>> {
    lzw_decode_impl(data).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lzw_decode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LZW encoder for round-trip tests (no compression, just packing).
    fn lzw_encode_literals(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bit_buf: u64 = 0;
        let mut bit_count: u32 = 0;

        let mut push_code = |code: u32, nbits: u32| {
            bit_buf = (bit_buf << nbits) | code as u64;
            bit_count += nbits;
            while bit_count >= 8 {
                bit_count -= 8;
                out.push((bit_buf >> bit_count) as u8);
            }
        };

        push_code(256, 9); // CLEAR
        for &b in input {
            push_code(b as u32, 9);
        }
        push_code(257, 9); // EOD

        if bit_count > 0 {
            out.push((bit_buf << (8 - bit_count)) as u8);
        }
        out
    }

    #[test]
    fn test_empty_input() {
        let decoded = lzw_decode_impl(&[]).unwrap();
        assert_eq!(decoded, b"");
    }

    #[test]
    fn test_empty_payload() {
        let encoded = lzw_encode_literals(b"");
        let decoded = lzw_decode_impl(&encoded).unwrap();
        assert_eq!(decoded, b"");
    }

    #[test]
    fn test_single_byte() {
        let encoded = lzw_encode_literals(b"A");
        let decoded = lzw_decode_impl(&encoded).unwrap();
        assert_eq!(decoded, b"A");
    }

    #[test]
    fn test_hello_world() {
        let encoded = lzw_encode_literals(b"Hello, world!");
        let decoded = lzw_decode_impl(&encoded).unwrap();
        assert_eq!(decoded, b"Hello, world!");
    }

    #[test]
    fn test_all_byte_values() {
        let input: Vec<u8> = (0u8..=255u8).collect();
        let encoded = lzw_encode_literals(&input);
        let decoded = lzw_decode_impl(&encoded).unwrap();
        assert_eq!(decoded, input);
    }
}
