use pyo3::prelude::*;
use pyo3::types::PyDict;

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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cmap_decode, m)?)?;
    Ok(())
}
