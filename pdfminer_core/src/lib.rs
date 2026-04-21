mod codecs;

use pyo3::prelude::*;

#[pymodule]
fn pdfminer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    codecs::ccitt::register_module(m)?;
    Ok(())
}
