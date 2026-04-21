use pyo3::prelude::*;
use std::collections::HashMap;

/// Build an xref lookup table from a list of (objid, offset) pairs.
#[pyfunction]
pub fn build_xref_lookup(entries: Vec<(u64, u64)>) -> HashMap<u64, u64> {
    entries.into_iter().collect()
}

/// Traverse a pre-fetched page tree to collect all page node IDs in order.
///
/// `nodes` is a list of `(node_type, kids_or_none)` pairs where:
/// - `node_type` is either `"Pages"` or `"Page"`
/// - `kids_or_none` is `Some(vec_of_ids)` for Pages nodes, `None` for Page nodes
///
/// The first element of `nodes` is the root.  IDs in `kids` index into
/// `nodes` by position.  Returns leaf page IDs (type == "Page") in reading
/// order, up to `target_count` pages.
#[pyfunction]
pub fn flatten_page_tree(
    nodes: Vec<(String, Option<Vec<u64>>)>,
    target_count: usize,
) -> PyResult<Vec<u64>> {
    if nodes.is_empty() {
        return Ok(vec![]);
    }

    // Iterative DFS — avoids Python recursion-limit issues.
    let cap = if target_count > 0 { target_count } else { nodes.len() };
    let mut result: Vec<u64> = Vec::with_capacity(cap);
    let mut stack: Vec<usize> = vec![0];

    while let Some(idx) = stack.pop() {
        if idx >= nodes.len() {
            continue;
        }
        let (ref node_type, ref kids_opt) = nodes[idx];
        if node_type == "Page" {
            result.push(idx as u64);
            if target_count > 0 && result.len() >= target_count {
                break;
            }
        } else if node_type == "Pages" {
            if let Some(kids) = kids_opt {
                // Reverse so left-most child is popped first.
                for &kid_idx in kids.iter().rev() {
                    stack.push(kid_idx as usize);
                }
            }
        }
    }

    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_xref_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_page_tree, m)?)?;
    Ok(())
}
