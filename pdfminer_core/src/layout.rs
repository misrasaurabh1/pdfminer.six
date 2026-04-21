use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Spatial index for fast rectangle overlap queries.
/// Replaces Python's Plane class for hot-path operations.
#[pyclass]
pub struct Plane {
    bbox: (f64, f64, f64, f64),
    gridsize: f64,
    grid: HashMap<(i64, i64), Vec<usize>>,
    // (id, bbox) stored by insertion index
    objects: Vec<Option<(f64, f64, f64, f64)>>,
    // maps object-id back to slot index
    id_to_slot: HashMap<usize, usize>,
    // free slots for reuse
    free_slots: Vec<usize>,
}

impl Plane {
    fn grid_range(
        &self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> impl Iterator<Item = (i64, i64)> {
        let (bx0, by0, bx1, by1) = self.bbox;
        // clamp to plane bounds
        let qx0 = x0.max(bx0);
        let qy0 = y0.max(by0);
        let qx1 = x1.min(bx1);
        let qy1 = y1.min(by1);

        if qx1 <= bx0 || bx1 <= qx0 || qy1 <= by0 || by1 <= qy0 {
            return vec![].into_iter();
        }

        let gs = self.gridsize;
        let gx0 = (qx0 / gs).floor() as i64;
        let gy0 = (qy0 / gs).floor() as i64;
        let gx1 = ((qx1 + gs) / gs).floor() as i64;
        let gy1 = ((qy1 + gs) / gs).floor() as i64;

        let mut cells = Vec::new();
        for gy in gy0..gy1 {
            for gx in gx0..gx1 {
                cells.push((gx, gy));
            }
        }
        cells.into_iter()
    }
}

#[pymethods]
impl Plane {
    #[new]
    #[pyo3(signature = (bbox, gridsize=50.0))]
    pub fn new(bbox: (f64, f64, f64, f64), gridsize: f64) -> Self {
        Plane {
            bbox,
            gridsize,
            grid: HashMap::new(),
            objects: Vec::new(),
            id_to_slot: HashMap::new(),
            free_slots: Vec::new(),
        }
    }

    /// Add object by its Python object id and bounding box.
    pub fn add_bbox(&mut self, obj_id: usize, obj_bbox: (f64, f64, f64, f64)) {
        let slot = if let Some(s) = self.free_slots.pop() {
            self.objects[s] = Some(obj_bbox);
            s
        } else {
            let s = self.objects.len();
            self.objects.push(Some(obj_bbox));
            s
        };
        self.id_to_slot.insert(obj_id, slot);

        let (x0, y0, x1, y1) = obj_bbox;
        for cell in self.grid_range(x0, y0, x1, y1) {
            self.grid.entry(cell).or_default().push(slot);
        }
    }

    /// Remove object by its Python object id.
    pub fn remove_bbox(&mut self, obj_id: usize) {
        if let Some(slot) = self.id_to_slot.remove(&obj_id) {
            if let Some(bbox) = self.objects[slot].take() {
                let (x0, y0, x1, y1) = bbox;
                for cell in self.grid_range(x0, y0, x1, y1) {
                    if let Some(list) = self.grid.get_mut(&cell) {
                        list.retain(|&s| s != slot);
                    }
                }
                self.free_slots.push(slot);
            }
        }
    }

    /// Find all object ids whose bboxes overlap with the given query bbox.
    pub fn find_overlapping(&self, query: (f64, f64, f64, f64)) -> Vec<usize> {
        let (qx0, qy0, qx1, qy1) = query;
        let mut seen: HashSet<usize> = HashSet::new();
        let mut result: Vec<usize> = Vec::new();

        for cell in self.grid_range(qx0, qy0, qx1, qy1) {
            if let Some(list) = self.grid.get(&cell) {
                for &slot in list {
                    if seen.contains(&slot) {
                        continue;
                    }
                    seen.insert(slot);
                    if let Some((ox0, oy0, ox1, oy1)) = self.objects[slot] {
                        // overlap: obj does NOT miss the query rect
                        if ox1 <= qx0 || qx1 <= ox0 || oy1 <= qy0 || qy1 <= oy0 {
                            continue;
                        }
                        // find the obj_id for this slot
                        for (&id, &s) in &self.id_to_slot {
                            if s == slot {
                                result.push(id);
                                break;
                            }
                        }
                    }
                }
            }
        }
        result
    }

    pub fn __len__(&self) -> usize {
        self.id_to_slot.len()
    }
}

/// Fast overlap detection between two bounding boxes.
/// Returns True if the boxes strictly overlap (touching edges do not count).
#[pyfunction]
pub fn bbox_overlap(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// Calculate overlap area between two bounding boxes.
#[pyfunction]
pub fn bbox_overlap_area(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> f64 {
    let x_overlap = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let y_overlap = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    x_overlap * y_overlap
}

/// Group characters into text lines based on vertical position overlap.
///
/// Each element of `char_bboxes` is (x0, y0, x1, y1) for a character.
/// Characters on the "same line" must have vertical overlap greater than
/// `line_overlap * min_height`.
///
/// Returns a list of groups, where each group is a list of (sorted) char
/// indices that belong to the same horizontal text line.
///
/// This mirrors the core halign logic of LTLayoutContainer.group_objects().
/// It does NOT handle vertical text (detect_vertical) — that stays in Python.
#[pyfunction]
pub fn group_chars_into_lines(
    char_bboxes: Vec<(f64, f64, f64, f64)>,
    line_overlap: f64,
    char_margin: f64,
) -> Vec<Vec<usize>> {
    if char_bboxes.is_empty() {
        return vec![];
    }

    // We group adjacent characters (in left-to-right order) just like the
    // Python code does: iterate in sequence, check halign with previous char.
    // Indices are 0-based positions into char_bboxes.

    let n = char_bboxes.len();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = vec![0];

    for i in 1..n {
        let (x0_prev, y0_prev, x1_prev, y1_prev) = char_bboxes[i - 1];
        let (x0_curr, y0_curr, x1_curr, y1_curr) = char_bboxes[i];

        let h_prev = y1_prev - y0_prev;
        let h_curr = y1_curr - y0_curr;
        let w_prev = x1_prev - x0_prev;
        let w_curr = x1_curr - x0_curr;

        // Vertical overlap: do the y-ranges overlap?
        let voverlap_amount = {
            let lo = y0_prev.max(y0_curr);
            let hi = y1_prev.min(y1_curr);
            if hi > lo { hi - lo } else { 0.0 }
        };
        let is_voverlap = y1_prev >= y0_curr && y0_prev <= y1_curr;

        // Horizontal distance
        let is_hoverlap = x1_prev >= x0_curr && x0_prev <= x1_curr;
        let hdist = if is_hoverlap {
            0.0
        } else {
            (x0_curr - x1_prev).abs().min((x1_curr - x0_prev).abs())
        };

        let halign = is_voverlap
            && h_prev.min(h_curr) * line_overlap < voverlap_amount
            && hdist < w_prev.max(w_curr) * char_margin;

        if halign {
            current_group.push(i);
        } else {
            groups.push(current_group.clone());
            current_group = vec![i];
        }
    }
    groups.push(current_group);
    groups
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Plane>()?;
    m.add_function(wrap_pyfunction!(bbox_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_overlap_area, m)?)?;
    m.add_function(wrap_pyfunction!(group_chars_into_lines, m)?)?;
    Ok(())
}
