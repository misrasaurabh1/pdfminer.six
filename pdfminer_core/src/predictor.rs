use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
fn paeth_predictor(left: u8, above: u8, upper_left: u8) -> u8 {
    let l = left as i32;
    let a = above as i32;
    let ul = upper_left as i32;
    let p = l + a - ul;
    let pa = (p - l).abs();
    let pb = (p - a).abs();
    let pc = (p - ul).abs();
    if pa <= pb && pa <= pc {
        left
    } else if pb <= pc {
        above
    } else {
        upper_left
    }
}

pub fn apply_png_predictor_impl(
    _pred: u8,
    colors: usize,
    columns: usize,
    bitspercomponent: usize,
    data: &[u8],
) -> Result<Vec<u8>, String> {
    if bitspercomponent != 8 && bitspercomponent != 1 {
        return Err(format!(
            "Unsupported `bitspercomponent': {}",
            bitspercomponent
        ));
    }

    let nbytes = colors * columns * bitspercomponent / 8;
    // bytes per complete pixel (minimum 1 for sub-byte depths)
    let bpp = (colors * bitspercomponent / 8).max(1);
    let stride = nbytes + 1; // 1 filter byte + nbytes data bytes

    let mut output = Vec::with_capacity(data.len());
    let mut line_above = vec![0u8; nbytes];

    let mut offset = 0;
    while offset + stride <= data.len() {
        let filter_type = data[offset];
        let line_encoded = &data[offset + 1..offset + stride];
        let mut raw = vec![0u8; nbytes];

        match filter_type {
            0 => {
                raw.copy_from_slice(line_encoded);
            }
            1 => {
                for j in 0..nbytes {
                    let left = if j < bpp { 0 } else { raw[j - bpp] };
                    raw[j] = line_encoded[j].wrapping_add(left);
                }
            }
            2 => {
                for j in 0..nbytes {
                    raw[j] = line_encoded[j].wrapping_add(line_above[j]);
                }
            }
            3 => {
                for j in 0..nbytes {
                    let left = if j < bpp { 0u32 } else { raw[j - bpp] as u32 };
                    let above = line_above[j] as u32;
                    raw[j] = line_encoded[j].wrapping_add(((left + above) / 2) as u8);
                }
            }
            4 => {
                for j in 0..nbytes {
                    let (left, upper_left) = if j < bpp {
                        (0u8, 0u8)
                    } else {
                        (raw[j - bpp], line_above[j - bpp])
                    };
                    let above = line_above[j];
                    raw[j] = line_encoded[j].wrapping_add(paeth_predictor(left, above, upper_left));
                }
            }
            _ => {
                return Err(format!("Unsupported predictor value: {}", filter_type));
            }
        }

        output.extend_from_slice(&raw);
        line_above[..nbytes].copy_from_slice(&raw);
        offset += stride;
    }

    Ok(output)
}

pub fn apply_tiff_predictor_impl(
    colors: usize,
    columns: usize,
    bitspercomponent: usize,
    data: &[u8],
) -> Result<Vec<u8>, String> {
    if bitspercomponent != 8 {
        return Err(format!(
            "Unsupported `bitspercomponent': {}",
            bitspercomponent
        ));
    }

    let bpp = colors * (bitspercomponent / 8);
    let nbytes = columns * bpp;
    let mut buf = data.to_vec();

    let mut row_start = 0;
    while row_start + nbytes <= buf.len() {
        for i in bpp..nbytes {
            buf[row_start + i] = buf[row_start + i].wrapping_add(buf[row_start + i - bpp]);
        }
        row_start += nbytes;
    }

    Ok(buf)
}

#[pyfunction]
#[pyo3(signature = (pred, colors, columns, bitspercomponent, data))]
pub fn apply_png_predictor(
    pred: u8,
    colors: usize,
    columns: usize,
    bitspercomponent: usize,
    data: &[u8],
) -> PyResult<Vec<u8>> {
    apply_png_predictor_impl(pred, colors, columns, bitspercomponent, data)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (colors, columns, bitspercomponent, data))]
pub fn apply_tiff_predictor(
    colors: usize,
    columns: usize,
    bitspercomponent: usize,
    data: &[u8],
) -> PyResult<Vec<u8>> {
    apply_tiff_predictor_impl(colors, columns, bitspercomponent, data)
        .map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_png_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(apply_tiff_predictor, m)?)?;
    Ok(())
}
