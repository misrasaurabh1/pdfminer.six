use pyo3::prelude::*;

mod pdfinterp;

#[pymodule]
fn pdfminer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pdfinterp::register(m)?;
    Ok(())
}
