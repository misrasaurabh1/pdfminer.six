use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parse a standard xref table from bytes.
/// Returns list of (objid, genno, offset, is_in_use) tuples.
/// Only in-use entries (type "n") are returned.
#[pyfunction]
pub fn parse_xref_table(data: &[u8]) -> PyResult<Vec<(u64, u64, u64, bool)>> {
    let text = std::str::from_utf8(data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut entries = Vec::new();
    let mut current_objid = 0u64;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line == "xref" {
            continue;
        }
        if line == "trailer" || line.starts_with("trailer") {
            break;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            // Section header: start_objid count
            current_objid = parts[0]
                .parse()
                .map_err(|e: std::num::ParseIntError| PyValueError::new_err(e.to_string()))?;
            // count is parts[1]; we iterate entries one-by-one as they come
        } else if parts.len() == 3 {
            // Entry: offset genno n/f
            let offset: u64 = parts[0]
                .parse()
                .map_err(|e: std::num::ParseIntError| PyValueError::new_err(e.to_string()))?;
            let genno: u64 = parts[1]
                .parse()
                .map_err(|e: std::num::ParseIntError| PyValueError::new_err(e.to_string()))?;
            let in_use = parts[2] == "n";
            entries.push((current_objid, genno, offset, in_use));
            current_objid += 1;
        }
    }

    Ok(entries)
}

/// Scan entire PDF data for object markers (fallback for corrupt PDFs).
/// Matches lines of the form: `<digits> <digits> obj`
/// Returns list of (objid, genno, byte_offset) tuples.
#[pyfunction]
pub fn scan_pdf_objects(data: &[u8]) -> PyResult<Vec<(u64, u64, u64)>> {
    let mut results = Vec::new();
    let mut pos: usize = 0;

    while pos < data.len() {
        // Find end of this line (LF-terminated; handle CR+LF too)
        let line_end = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| pos + p)
            .unwrap_or(data.len());

        let line_bytes = &data[pos..line_end];

        if let Some((objid, genno)) = try_match_obj_header(line_bytes) {
            results.push((objid, genno, pos as u64));
        }

        pos = if line_end < data.len() {
            line_end + 1
        } else {
            data.len()
        };
    }

    Ok(results)
}

/// Try to parse `<digits> <digits> obj[<non-word>|EOF]` from the beginning of a line.
fn try_match_obj_header(line: &[u8]) -> Option<(u64, u64)> {
    // Trim leading whitespace
    let line = trim_start(line);

    // Parse first integer (objid)
    let (objid_bytes, rest) = split_digits(line)?;
    let objid: u64 = parse_u64(objid_bytes)?;

    // Skip whitespace between objid and genno
    let rest = trim_start(rest);
    if rest.is_empty() {
        return None;
    }

    // Parse second integer (genno)
    let (genno_bytes, rest) = split_digits(rest)?;
    let genno: u64 = parse_u64(genno_bytes)?;

    // Skip whitespace between genno and keyword
    let rest = trim_start(rest);

    // Must start with "obj" followed by a non-word character or end-of-line
    if !rest.starts_with(b"obj") {
        return None;
    }
    let after_obj = &rest[3..];
    // Validate word boundary: next char (if any) must not be alphanumeric or '_'
    if let Some(&next) = after_obj.first() {
        if next.is_ascii_alphanumeric() || next == b'_' {
            return None;
        }
    }

    Some((objid, genno))
}

#[inline]
fn trim_start(data: &[u8]) -> &[u8] {
    let n = data.iter().take_while(|&&b| b == b' ' || b == b'\t' || b == b'\r').count();
    &data[n..]
}

#[inline]
fn split_digits(data: &[u8]) -> Option<(&[u8], &[u8])> {
    let n = data.iter().take_while(|&&b| b.is_ascii_digit()).count();
    if n == 0 {
        None
    } else {
        Some((&data[..n], &data[n..]))
    }
}

#[inline]
fn parse_u64(digits: &[u8]) -> Option<u64> {
    std::str::from_utf8(digits).ok()?.parse().ok()
}

/// Parse a compressed xref stream (PDF 1.5+).
///
/// * `data`  — raw (decompressed) stream bytes
/// * `w`     — field widths [w1, w2, w3] as specified by the /W array
/// * `index` — flat list [start1, count1, start2, count2, …] from /Index
///
/// Returns a list of `(entry_type, objid, field2, field3)` tuples.
#[pyfunction]
pub fn parse_xref_stream(
    data: &[u8],
    w: Vec<usize>,
    index: Vec<u64>,
) -> PyResult<Vec<(u8, u64, u64, u64)>> {
    if w.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "w must have exactly 3 elements, got {}",
            w.len()
        )));
    }
    if index.len() % 2 != 0 {
        return Err(PyValueError::new_err("index length must be even"));
    }

    let (w1, w2, w3) = (w[0], w[1], w[2]);
    let entry_size = w1 + w2 + w3;

    if entry_size == 0 {
        return Err(PyValueError::new_err("xref stream entry size is zero"));
    }
    if data.len() % entry_size != 0 {
        return Err(PyValueError::new_err(format!(
            "XRef stream data length {} not divisible by entry size {}",
            data.len(),
            entry_size
        )));
    }

    let mut results = Vec::new();
    let mut data_pos: usize = 0;

    for chunk in index.chunks(2) {
        let start_objid = chunk[0];
        let count = chunk[1];

        for i in 0..count {
            if data_pos + entry_size > data.len() {
                break;
            }

            // w1 field: entry type (default = 1 if w1 == 0)
            let entry_type = if w1 == 0 {
                1u8
            } else {
                let val = read_be_field(&data[data_pos..data_pos + w1]);
                val as u8
            };

            let f2 = read_be_field(&data[data_pos + w1..data_pos + w1 + w2]);
            let f3 = read_be_field(&data[data_pos + w1 + w2..data_pos + entry_size]);

            results.push((entry_type, start_objid + i, f2, f3));
            data_pos += entry_size;
        }
    }

    Ok(results)
}

/// Read a big-endian unsigned integer from a byte slice of any width (0–8 bytes).
#[inline]
fn read_be_field(data: &[u8]) -> u64 {
    data.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_xref_table, m)?)?;
    m.add_function(wrap_pyfunction!(scan_pdf_objects, m)?)?;
    m.add_function(wrap_pyfunction!(parse_xref_stream, m)?)?;
    Ok(())
}
