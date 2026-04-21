use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Decode PDF LZW-compressed data.
///
/// Mirrors `LZWDecoder` in `pdfminer/lzw.py`:
/// - 9-bit initial code width, growing to 10/11/12 bits.
/// - Code 256 = clear/reset, code 257 = EOD.
/// - earlychange = 1 (width bumps when table length reaches 511/1023/2047).
/// - Corrupt codes are silently truncated (mirrors `except CorruptDataError: break`).
pub fn lzw_decode_impl(data: &[u8]) -> Result<Vec<u8>, String> {
    const CLEAR: u32 = 256;
    const EOD: u32 = 257;

    // Bit-stream state – MSB-first, one byte lookahead.
    let mut buf: u32 = 0;
    let mut bpos: u32 = 8; // bits already consumed from `buf` (8 = empty)
    let mut data_pos: usize = 0;

    // Reads `bits` bits MSB-first; returns None on EOF.
    macro_rules! readbits {
        ($bits:expr) => {{
            let mut remaining: u32 = $bits;
            let mut v: u32 = 0;
            let result: Option<u32> = loop {
                let r = 8 - bpos;
                if remaining <= r {
                    v = (v << remaining)
                        | ((buf >> (r - remaining)) & ((1 << remaining) - 1));
                    bpos += remaining;
                    break Some(v);
                } else {
                    v = (v << r) | (buf & ((1 << r) - 1));
                    remaining -= r;
                    if data_pos >= data.len() {
                        break None;
                    }
                    buf = data[data_pos] as u32;
                    data_pos += 1;
                    bpos = 0;
                }
            };
            result
        }};
    }

    // String table stored as a flat byte slab.
    // `string_starts[i]` and `string_ends[i]` index into `strings`.
    let mut string_starts: Vec<u32> = Vec::with_capacity(4096);
    let mut string_ends: Vec<u32> = Vec::with_capacity(4096);
    let mut strings: Vec<u8> = Vec::with_capacity(1 << 16);

    // Reset to the initial 258-entry table (codes 0-255 + CLEAR + EOD sentinels).
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
            string_starts.push(0); // 256 = CLEAR sentinel
            string_ends.push(0);
            string_starts.push(0); // 257 = EOD sentinel
            string_ends.push(0);
            9u32
        }};
    }

    let mut nbits: u32 = reset_table!();
    // (start, end) of the previous output run in `strings`; None after CLEAR.
    let mut prev: Option<(usize, usize)> = None;
    let mut output: Vec<u8> = Vec::new();

    loop {
        let code = match readbits!(nbits) {
            Some(c) => c,
            None => break,
        };

        if code == CLEAR {
            nbits = reset_table!();
            prev = None;
            continue;
        }
        if code == EOD {
            break;
        }

        let table_len = string_starts.len();
        let code_idx = code as usize;

        let (ps, pe) = match prev {
            None => {
                // First code after reset: emit literal, no table addition.
                if code_idx >= table_len {
                    break; // corrupt
                }
                let s = string_starts[code_idx] as usize;
                let e = string_ends[code_idx] as usize;
                output.extend_from_slice(&strings[s..e]);
                prev = Some((s, e));
                continue;
            }
            Some(p) => p,
        };

        // Determine (entry_start, entry_end) in the `strings` slab, appending
        // a new entry when necessary.  We also record `entry_first` for building
        // the new table entry (prev + entry[0]).
        let (entry_s, entry_e, entry_first) = if code_idx < table_len {
            let s = string_starts[code_idx] as usize;
            let e = string_ends[code_idx] as usize;
            (s, e, strings[s])
        } else if code_idx == table_len {
            // Special case: code equals the *next* slot – entry is prev + prev[0].
            // We append it now so the entry and the new table entry are the same.
            let first = strings[ps];
            let new_s = strings.len();
            // Copy prev bytes, then append first byte.
            strings.extend_from_slice(&strings[ps..pe].to_vec());
            strings.push(first);
            let new_e = strings.len();
            string_starts.push(new_s as u32);
            string_ends.push(new_e as u32);
            // Output and update prev to this new entry, then skip the normal
            // table-append below.
            output.extend_from_slice(&strings[new_s..new_e]);
            let new_table_len = string_starts.len();
            if new_table_len == 511 {
                nbits = 10;
            } else if new_table_len == 1023 {
                nbits = 11;
            } else if new_table_len == 2047 {
                nbits = 12;
            }
            prev = Some((new_s, new_e));
            continue;
        } else {
            break; // corrupt
        };

        // Append new table entry: prev + entry[0].
        let new_start = strings.len() as u32;
        strings.extend_from_slice(&strings[ps..pe].to_vec());
        strings.push(entry_first);
        let new_end = strings.len() as u32;
        string_starts.push(new_start);
        string_ends.push(new_end);

        output.extend_from_slice(&strings[entry_s..entry_e]);

        prev = Some((entry_s, entry_e));

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

    /// Minimal PDF-compatible LZW encoder for round-trip tests (no compression).
    /// Correctly tracks table growth and upgrades code width at the same
    /// thresholds as the decoder.
    fn lzw_encode_literals(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut bit_buf: u64 = 0;
        let mut bit_count: u32 = 0;

        let push_code = |code: u32, nbits: u32, out: &mut Vec<u8>, bit_buf: &mut u64, bit_count: &mut u32| {
            *bit_buf = (*bit_buf << nbits) | code as u64;
            *bit_count += nbits;
            while *bit_count >= 8 {
                *bit_count -= 8;
                out.push((*bit_buf >> *bit_count) as u8);
            }
        };

        let mut nbits: u32 = 9;
        let mut table_len: u32 = 258; // 0-255 + CLEAR + EOD

        push_code(256, 9, &mut out, &mut bit_buf, &mut bit_count); // CLEAR

        for (i, &b) in input.iter().enumerate() {
            push_code(b as u32, nbits, &mut out, &mut bit_buf, &mut bit_count);
            // First literal after CLEAR doesn't add a table entry.
            if i > 0 {
                table_len += 1;
                if table_len == 511 {
                    nbits = 10;
                } else if table_len == 1023 {
                    nbits = 11;
                } else if table_len == 2047 {
                    nbits = 12;
                }
            }
        }

        push_code(257, nbits, &mut out, &mut bit_buf, &mut bit_count); // EOD

        if bit_count > 0 {
            out.push((bit_buf << (8 - bit_count)) as u8);
        }
        out
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(lzw_decode_impl(&[]).unwrap(), b"");
    }

    #[test]
    fn test_empty_payload() {
        let encoded = lzw_encode_literals(b"");
        assert_eq!(lzw_decode_impl(&encoded).unwrap(), b"");
    }

    #[test]
    fn test_single_byte() {
        let encoded = lzw_encode_literals(b"A");
        assert_eq!(lzw_decode_impl(&encoded).unwrap(), b"A");
    }

    #[test]
    fn test_hello_world() {
        let encoded = lzw_encode_literals(b"Hello, world!");
        assert_eq!(lzw_decode_impl(&encoded).unwrap(), b"Hello, world!");
    }

    #[test]
    fn test_all_byte_values() {
        // Exercises the 9→10 bit-width transition.
        let input: Vec<u8> = (0u8..=255u8).collect();
        let encoded = lzw_encode_literals(&input);
        assert_eq!(lzw_decode_impl(&encoded).unwrap(), input);
    }

    #[test]
    fn test_known_vector_a() {
        // pack_codes([256, 65, 257], [9, 9, 9]) → b"A"
        let data: &[u8] = &[0x80, 0x10, 0x60, 0x20];
        assert_eq!(lzw_decode_impl(data).unwrap(), b"A");
    }
}
