use pyo3::prelude::*;

mod regex;
use regex::pre_tokenize_rust;

#[pymodule]
fn bpe_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pre_tokenize_rust, m)?)?;
    Ok(())
}