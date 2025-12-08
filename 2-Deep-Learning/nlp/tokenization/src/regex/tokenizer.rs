use pyo3::prelude::*;
use fancy_regex::Regex;

#[pyfunction]
pub fn pre_tokenize_rust(pattern: &str, text: &str) -> PyResult<Vec<String>> {
    let re = Regex::new(pattern)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Regex inválida: {}", e)))?;
    
    let mut results = Vec::new();
    
    for match_result in re.find_iter(text) {
        match match_result {
            Ok(mat) => results.push(mat.as_str().to_string()),
            Err(e) => {
                // Log do erro se necessário, mas continua
                eprintln!("Erro no matching: {}", e);
                continue;
            }
        }
    }
    
    Ok(results)
}