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
    let mut line_above = vec![0u8; nbytes.max(columns)];

    let mut offset = 0;
    while offset + stride <= data.len() {
        let filter_type = data[offset];
        let line_encoded = &data[offset + 1..offset + stride];
        let mut raw = vec![0u8; nbytes];

        match filter_type {
            0 => {
                // Filter type 0: None
                raw.copy_from_slice(line_encoded);
            }
            1 => {
                // Filter type 1: Sub
                // Raw(x) = Sub(x) + Raw(x - bpp)
                for j in 0..nbytes {
                    let raw_x_bpp = if j < bpp { 0 } else { raw[j - bpp] };
                    raw[j] = line_encoded[j].wrapping_add(raw_x_bpp);
                }
            }
            2 => {
                // Filter type 2: Up
                // Raw(x) = Up(x) + Prior(x)
                for j in 0..nbytes {
                    raw[j] = line_encoded[j].wrapping_add(line_above[j]);
                }
            }
            3 => {
                // Filter type 3: Average
                // Raw(x) = Average(x) + floor((Raw(x-bpp) + Prior(x)) / 2)
                for j in 0..nbytes {
                    let raw_x_bpp = if j < bpp { 0u32 } else { raw[j - bpp] as u32 };
                    let prior_x = line_above[j] as u32;
                    raw[j] = line_encoded[j].wrapping_add(((raw_x_bpp + prior_x) / 2) as u8);
                }
            }
            4 => {
                // Filter type 4: Paeth
                // Raw(x) = Paeth(x) + PaethPredictor(Raw(x-bpp), Prior(x), Prior(x-bpp))
                for j in 0..nbytes {
                    let (raw_x_bpp, prior_x_bpp) = if j < bpp {
                        (0u8, 0u8)
                    } else {
                        (raw[j - bpp], line_above[j - bpp])
                    };
                    let prior_x = line_above[j];
                    let paeth = paeth_predictor(raw_x_bpp, prior_x, prior_x_bpp);
                    raw[j] = line_encoded[j].wrapping_add(paeth);
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
    let mut buf = Vec::with_capacity(data.len());

    let mut scanline_i = 0;
    while scanline_i + nbytes <= data.len() {
        let mut raw = data[scanline_i..scanline_i + nbytes].to_vec();
        for i in bpp..nbytes {
            raw[i] = raw[i].wrapping_add(raw[i - bpp]);
        }
        buf.extend_from_slice(&raw);
        scanline_i += nbytes;
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
