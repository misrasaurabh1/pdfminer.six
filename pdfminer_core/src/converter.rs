use pyo3::prelude::*;

/// Escape XML/HTML special characters in text.
/// Equivalent to `html.escape()` in Python (which pdfminer uses via `enc()`).
#[pyfunction]
pub fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

/// Format a bounding box as a string with 3 decimal places.
/// Equivalent to `bbox2str()` in pdfminer/utils.py:
///   f"{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}"
#[pyfunction]
pub fn format_bbox(bbox: (f64, f64, f64, f64)) -> String {
    format!(
        "{:.3},{:.3},{:.3},{:.3}",
        bbox.0, bbox.1, bbox.2, bbox.3
    )
}

/// Convert a list of (text, bbox) pairs to a text string.
/// Sorts items by descending Y then ascending X, then joins.
/// This mirrors the positional ordering TextConverter relies on
/// after layout analysis has already grouped lines.
#[pyfunction]
pub fn build_text_output(
    text_items: Vec<(String, (f64, f64, f64, f64))>,
    _page_width: f64,
    _page_height: f64,
) -> String {
    let mut sorted_items = text_items;
    // Sort by descending y0 (top of page first), then ascending x0
    sorted_items.sort_by(|a, b| {
        let ya = (a.1).1;
        let yb = (b.1).1;
        yb.partial_cmp(&ya)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let xa = (a.1).0;
                let xb = (b.1).0;
                xa.partial_cmp(&xb).unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let total: usize = sorted_items.iter().map(|(t, _)| t.len()).sum();
    let mut out = String::with_capacity(total);
    for (text, _) in sorted_items {
        out.push_str(&text);
    }
    out
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xml_escape, m)?)?;
    m.add_function(wrap_pyfunction!(format_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(build_text_output, m)?)?;
    Ok(())
}
