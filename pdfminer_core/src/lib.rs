use pyo3::prelude::*;

mod xref;

#[pymodule]
fn pdfminer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    xref::register(m)?;
    Ok(())
}
