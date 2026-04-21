/// PDF stream filter-chain decode pipeline.
///
/// `decode_stream_filters` applies a list of (filter_name, params_dict) pairs
/// to raw stream bytes in sequence, returning the fully decoded data.
///
/// Filters handled in Rust:
///   FlateDecode / Fl      — zlib inflate + optional PNG/TIFF predictor
///   LZWDecode / LZW       — LZW inflate + optional PNG/TIFF predictor
///   ASCII85Decode / A85   — ASCII-85 decode
///   ASCIIHexDecode / AHx  — ASCII-hex decode
///   RunLengthDecode / RL  — PDF run-length decode
///   DCTDecode / DCT       — pass-through (JPEG, returned as-is)
///   JPXDecode             — pass-through
///   JBIG2Decode           — pass-through
///
/// Unsupported filters (CCITTFaxDecode, Crypt) raise `PyValueError` so that
/// the Python caller can fall back to the pure-Python path.
use flate2::read::ZlibDecoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::io::Read;

use crate::codecs::lzw::lzw_decode_impl;
use crate::predictor::{apply_png_predictor_impl, apply_tiff_predictor_impl};

// ---------------------------------------------------------------------------
// Individual filter decoders
// ---------------------------------------------------------------------------

fn flate_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| e.to_string())?;
    Ok(output)
}

/// Decompress deflate data byte-by-byte, stopping at CRC/checksum errors.
/// Mirrors Python's `decompress_corrupted` — used only when the fast path fails.
fn flate_decode_tolerant(data: &[u8]) -> Vec<u8> {
    // Read as much decompressed data as possible before hitting a CRC error.
    // ZlibDecoder's `read` will return an error at the first bad byte,
    // but everything decompressed so far is in `output`.
    let mut dec = ZlibDecoder::new(data);
    let mut output = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match dec.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => output.extend_from_slice(&buf[..n]),
            Err(_) => {
                // Stop at CRC checksum error (same as Python's decompress_corrupted).
                // Data already in `output` is retained.
                break;
            }
        }
    }
    output
}

fn ascii85_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    // Strip leading "<~" / "~" and trailing "~>" / "~"
    let data = strip_ascii85_markers(data);
    let mut output = Vec::with_capacity(data.len() / 5 * 4 + 4);
    let mut group: [u8; 5] = [0; 5];
    let mut group_len = 0usize;

    for &byte in data {
        match byte {
            b'z' if group_len == 0 => {
                output.extend_from_slice(&[0u8; 4]);
            }
            b'~' => break, // end marker
            b if b < b'!' => {
                // ignore whitespace, accept printable ASCII85 chars
                if (b as char).is_ascii_whitespace() {
                    continue;
                }
                return Err(format!("Invalid ASCII85 character: {byte}"));
            }
            b => {
                group[group_len] = b - b'!';
                group_len += 1;
                if group_len == 5 {
                    let val: u32 = (group[0] as u32) * 85_u32.pow(4)
                        + (group[1] as u32) * 85_u32.pow(3)
                        + (group[2] as u32) * 85_u32.pow(2)
                        + (group[3] as u32) * 85
                        + (group[4] as u32);
                    output.extend_from_slice(&val.to_be_bytes());
                    group_len = 0;
                }
            }
        }
    }
    // Handle a partial group at the end
    if group_len > 0 {
        // Pad with 'u' values (84)
        for i in group_len..5 {
            group[i] = 84;
        }
        let val: u32 = (group[0] as u32) * 85_u32.pow(4)
            + (group[1] as u32) * 85_u32.pow(3)
            + (group[2] as u32) * 85_u32.pow(2)
            + (group[3] as u32) * 85
            + (group[4] as u32);
        let bytes = val.to_be_bytes();
        output.extend_from_slice(&bytes[..group_len - 1]);
    }
    Ok(output)
}

fn strip_ascii85_markers(data: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = data.len();

    // Strip leading whitespace
    while start < end && (data[start] as char).is_ascii_whitespace() {
        start += 1;
    }
    // Strip leading "<~" or "~"
    if data[start..end].starts_with(b"<~") {
        start += 2;
    } else if data[start..end].starts_with(b"~") {
        start += 1;
    }
    // Strip trailing whitespace
    while end > start && (data[end - 1] as char).is_ascii_whitespace() {
        end -= 1;
    }
    // Strip trailing "~>" or "~"
    if data[start..end].ends_with(b"~>") {
        end -= 2;
    } else if data[start..end].ends_with(b"~") {
        end -= 1;
    }
    // Strip any remaining trailing ">"
    if end > start && data[end - 1] == b'>' {
        end -= 1;
    }
    &data[start..end]
}

fn asciihex_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    // Remove whitespace, stop at '>'
    let filtered: Vec<u8> = data
        .iter()
        .copied()
        .take_while(|&b| b != b'>')
        .filter(|b| !b.is_ascii_whitespace())
        .collect();

    let padded = if filtered.len() % 2 == 1 {
        let mut v = filtered.clone();
        v.push(b'0');
        v
    } else {
        filtered
    };

    (0..padded.len())
        .step_by(2)
        .map(|i| {
            let hi = hex_nibble(padded[i])?;
            let lo = hex_nibble(padded[i + 1])?;
            Ok((hi << 4) | lo)
        })
        .collect()
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("Invalid hex digit: {b}")),
    }
}

fn runlength_decode(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let length = data[i] as usize;
        i += 1;
        if length == 128 {
            break; // EOD
        } else if length < 128 {
            // Copy next (length+1) bytes literally
            let end = (i + length + 1).min(data.len());
            output.extend_from_slice(&data[i..end]);
            i += length + 1;
        } else {
            // length in 129..255: repeat the next byte (257-length) times
            let count = 257 - length;
            if i < data.len() {
                let byte = data[i];
                i += 1;
                output.extend(std::iter::repeat(byte).take(count));
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Predictor application
// ---------------------------------------------------------------------------

/// Extract predictor parameters from a params dict and apply them.
fn apply_predictor(params: &Bound<'_, PyDict>, data: Vec<u8>) -> Result<Vec<u8>, String> {
    let pred_obj = params
        .get_item("Predictor")
        .map_err(|e| e.to_string())?;
    let pred_obj = match pred_obj {
        Some(p) => p,
        None => return Ok(data),
    };
    let pred: i64 = pred_obj.extract().map_err(|e| e.to_string())?;

    if pred == 1 {
        return Ok(data); // no predictor
    }

    let colors: usize = get_int_param(params, "Colors", 1)?;
    let columns: usize = get_int_param(params, "Columns", 1)?;
    let bpc: usize = get_int_param(params, "BitsPerComponent", 8)?;

    if pred == 2 {
        apply_tiff_predictor_impl(colors, columns, bpc, &data)
    } else if pred >= 10 {
        apply_png_predictor_impl(pred as u8, colors, columns, bpc, &data)
    } else {
        Err(format!("Unsupported predictor: {pred}"))
    }
}

fn get_int_param(params: &Bound<'_, PyDict>, key: &str, default: usize) -> Result<usize, String> {
    let item = params.get_item(key).map_err(|e| e.to_string())?;
    match item {
        None => Ok(default),
        Some(v) => {
            let n: i64 = v.extract().map_err(|e| e.to_string())?;
            Ok(n as usize)
        }
    }
}

// ---------------------------------------------------------------------------
// Public function: apply a single filter
// ---------------------------------------------------------------------------

fn apply_single_filter(
    filter_name: &str,
    params: Option<&Bound<'_, PyDict>>,
    data: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let decoded = match filter_name {
        // FlateDecode — on error, raise so the Python path can use its
        // `decompress_corrupted` logic which correctly handles partial data.
        "FlateDecode" | "Fl" => {
            flate_decode(&data).map_err(|e| e)?
        }
        // LZWDecode
        "LZWDecode" | "LZW" => lzw_decode_impl(&data)?,
        // ASCII85Decode
        "ASCII85Decode" | "A85" => ascii85_decode(&data)?,
        // ASCIIHexDecode
        "ASCIIHexDecode" | "AHx" => asciihex_decode(&data)?,
        // RunLengthDecode
        "RunLengthDecode" | "RL" => runlength_decode(&data)?,
        // Pass-through filters
        "DCTDecode" | "DCT" | "JPXDecode" | "JBIG2Decode" => data,
        // Unsupported — signal the Python fallback
        "CCITTFaxDecode" | "CCF" => {
            return Err(format!("CCITTFaxDecode not supported in Rust path"));
        }
        "Crypt" => {
            return Err("/Crypt filter is not supported in Rust path".to_string());
        }
        other => {
            return Err(format!("Unknown filter: {other}"));
        }
    };

    // Apply predictor if present
    if let Some(p) = params {
        apply_predictor(p, decoded)
    } else {
        Ok(decoded)
    }
}

// ---------------------------------------------------------------------------
// Exported pyfunction
// ---------------------------------------------------------------------------

/// Apply a full PDF filter chain to raw stream bytes.
///
/// `filters` is a list of `(filter_name: str, params: dict | None)` tuples.
/// Raises `ValueError` for unsupported filters so the Python caller can fall
/// back to the pure-Python decode path.
///
/// Returns `bytes`.
#[pyfunction]
pub fn decode_stream_filters<'py>(
    py: Python<'py>,
    rawdata: &[u8],
    filters: Vec<(String, PyObject)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut data = rawdata.to_vec();
    for (filter_name, params_obj) in filters {
        // params may be None or a dict
        let params_dict: Option<Bound<'_, PyDict>> =
            if params_obj.is_none(py) {
                None
            } else {
                Some(
                    params_obj
                        .downcast_bound::<PyDict>(py)
                        .map_err(|_| {
                            PyValueError::new_err(format!(
                                "params for filter {} is not a dict",
                                filter_name
                            ))
                        })?
                        .clone(),
                )
            };

        data = apply_single_filter(&filter_name, params_dict.as_ref(), data)
            .map_err(PyValueError::new_err)?;
    }
    Ok(PyBytes::new_bound(py, &data))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_stream_filters, m)?)?;
    Ok(())
}
