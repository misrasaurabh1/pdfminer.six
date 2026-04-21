use pyo3::prelude::*;

mod ascii85;
mod ccitt;
pub mod lzw;
mod runlength;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    lzw::register(m)?;
    ccitt::register(m)?;
    runlength::register(m)?;
    ascii85::register(m)?;
    Ok(())
}
