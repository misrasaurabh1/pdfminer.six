use pyo3::prelude::*;

mod matrix;

#[pymodule]
fn pdfminer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    matrix::register(m)?;
    Ok(())
}
