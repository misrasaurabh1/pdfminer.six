use pyo3::prelude::*;

mod cmapdb;
mod codecs;
mod converter;
mod layout;
mod pdffont;
mod pdfinterp;
mod matrix;
mod pdfdocument;
mod predictor;
mod psparser;
mod xref;

#[pymodule]
fn pdfminer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    codecs::register(m)?;
    predictor::register(m)?;
    matrix::register(m)?;
    psparser::register(m)?;
    cmapdb::register(m)?;
    pdfdocument::register(m)?;
    xref::register(m)?;
    converter::register(m)?;
    layout::register(m)?;
    pdffont::register(m)?;
    pdfinterp::register(m)?;
    Ok(())
}
