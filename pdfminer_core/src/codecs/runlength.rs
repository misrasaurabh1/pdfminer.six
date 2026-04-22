use pyo3::prelude::*;

/// Decode PDF RunLength-encoded data.
///
/// Mirrors `rldecode` in `pdfminer/runlength.py` (PDF Reference 1.4 §3.3.4):
/// - length 0–127: copy the following (length+1) literal bytes.
/// - length 128: EOD.
/// - length 129–255: repeat the following byte (257-length) times.
#[pyfunction]
pub fn runlength_decode(data: &[u8]) -> PyResult<Vec<u8>> {
    let mut output = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let length = data[i] as i32;
        i += 1;
        if length == 128 {
            break;
        } else if length < 128 {
            let count = (length + 1) as usize;
            let end = i + count;
            if end > data.len() {
                // Truncated data: copy what we have
                output.extend_from_slice(&data[i..]);
                break;
            }
            output.extend_from_slice(&data[i..end]);
            i = end;
        } else {
            // length > 128
            if i >= data.len() {
                break;
            }
            let byte = data[i];
            i += 1;
            let count = (257 - length) as usize;
            output.resize(output.len() + count, byte);
        }
    }
    Ok(output)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(runlength_decode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(runlength_decode(&[]).unwrap(), b"");
    }

    #[test]
    fn test_eod_immediate() {
        assert_eq!(runlength_decode(&[128]).unwrap(), b"");
    }

    #[test]
    fn test_literal_run() {
        // length=2 means copy next 3 bytes
        let data = &[2u8, b'A', b'B', b'C', 128];
        assert_eq!(runlength_decode(data).unwrap(), b"ABC");
    }

    #[test]
    fn test_repeat_run() {
        // length=255 means repeat next byte (257-255)=2 times
        let data = &[255u8, b'X', 128];
        assert_eq!(runlength_decode(data).unwrap(), b"XX");
    }

    #[test]
    fn test_mixed() {
        // literal 2 bytes ("AB"), then repeat 'C' 3 times (257-254=3)
        let data = &[1u8, b'A', b'B', 254, b'C', 128];
        assert_eq!(runlength_decode(data).unwrap(), b"ABCCC");
    }
}
