// CCITT Fax decoder
//
// Implements:
//   T.6 Group 4 (k < 0)
//   T.4 Group 3 1-D (k == 0)
//   T.4 Group 3 2-D (k > 0) — treated as 1-D
//
// Reference: ITU-T T.4, T.6

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Trie node for Huffman decoding
// ---------------------------------------------------------------------------

struct TrieNode {
    value: Option<i32>,
    children: [Option<Box<TrieNode>>; 2],
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            value: None,
            children: [None, None],
        }
    }

    fn insert(&mut self, bits: &[u8], value: i32) {
        if bits.is_empty() {
            self.value = Some(value);
            return;
        }
        let bit = if bits[0] == b'1' { 1 } else { 0 };
        if self.children[bit].is_none() {
            self.children[bit] = Some(Box::new(TrieNode::new()));
        }
        self.children[bit].as_mut().unwrap().insert(&bits[1..], value);
    }
}

fn build_trie(table: &[(&str, i32)]) -> TrieNode {
    let mut root = TrieNode::new();
    for &(bits, value) in table {
        root.insert(bits.as_bytes(), value);
    }
    root
}

// ---------------------------------------------------------------------------
// Mode code sentinels (not T.6 wire values — internal dispatch tags)
// ---------------------------------------------------------------------------

const MODE_HORIZONTAL: i32 = 100;
const MODE_PASS: i32 = 200;
const MODE_EOFB: i32 = 999;

// ---------------------------------------------------------------------------
// Static Huffman tables
// ---------------------------------------------------------------------------

// Mode codes (T.6 / G4)
// Vertical offset: -3..=3 stored as-is.
// Special: MODE_HORIZONTAL, MODE_PASS, MODE_EOFB.
static MODE_TABLE: &[(&str, i32)] = &[
    ("1",                         0),
    ("011",                       1),
    ("010",                      -1),
    ("001",                     MODE_HORIZONTAL),
    ("0001",                    MODE_PASS),
    ("000011",                    2),
    ("000010",                   -2),
    ("0000011",                   3),
    ("0000010",                  -3),
    ("000000000001000000000001", MODE_EOFB),
];

// T.4 white run-length codes
static WHITE_TABLE: &[(&str, i32)] = &[
    ("00110101", 0),  ("000111", 1),   ("0111", 2),      ("1000", 3),
    ("1011", 4),      ("1100", 5),     ("1110", 6),      ("1111", 7),
    ("10011", 8),     ("10100", 9),    ("00111", 10),    ("01000", 11),
    ("001000", 12),   ("000011", 13),  ("110100", 14),   ("110101", 15),
    ("101010", 16),   ("101011", 17),  ("0100111", 18),  ("0001100", 19),
    ("0001000", 20),  ("0010111", 21), ("0000011", 22),  ("0000100", 23),
    ("0101000", 24),  ("0101011", 25), ("0010011", 26),  ("0100100", 27),
    ("0011000", 28),  ("00000010", 29), ("00000011", 30), ("00011010", 31),
    ("00011011", 32), ("00010010", 33), ("00010011", 34), ("00010100", 35),
    ("00010101", 36), ("00010110", 37), ("00010111", 38), ("00101000", 39),
    ("00101001", 40), ("00101010", 41), ("00101011", 42), ("00101100", 43),
    ("00101101", 44), ("00000100", 45), ("00000101", 46), ("00001010", 47),
    ("00001011", 48), ("01010010", 49), ("01010011", 50), ("01010100", 51),
    ("01010101", 52), ("00100100", 53), ("00100101", 54), ("01011000", 55),
    ("01011001", 56), ("01011010", 57), ("01011011", 58), ("01001010", 59),
    ("01001011", 60), ("00110010", 61), ("00110011", 62), ("00110100", 63),
    // make-up
    ("11011", 64),       ("10010", 128),      ("010111", 192),
    ("0110111", 256),    ("00110110", 320),    ("00110111", 384),
    ("01100100", 448),   ("01100101", 512),    ("01101000", 576),
    ("01100111", 640),   ("011001100", 704),   ("011001101", 768),
    ("011010010", 832),  ("011010011", 896),   ("011010100", 960),
    ("011010101", 1024), ("011010110", 1088),  ("011010111", 1152),
    ("011011000", 1216), ("011011001", 1280),  ("011011010", 1344),
    ("011011011", 1408), ("010011000", 1472),  ("010011001", 1536),
    ("010011010", 1600), ("011000", 1664),     ("010011011", 1728),
    // additional make-up (shared)
    ("00000001000", 1792),   ("00000001100", 1856),  ("00000001101", 1920),
    ("000000010010", 1984),  ("000000010011", 2048), ("000000010100", 2112),
    ("000000010101", 2176),  ("000000010110", 2240), ("000000010111", 2304),
    ("000000011100", 2368),  ("000000011101", 2432), ("000000011110", 2496),
    ("000000011111", 2560),
];

// T.4 black run-length codes
static BLACK_TABLE: &[(&str, i32)] = &[
    ("0000110111", 0),  ("010", 1),    ("11", 2),     ("10", 3),
    ("011", 4),         ("0011", 5),   ("0010", 6),   ("00011", 7),
    ("000101", 8),      ("000100", 9), ("0000100", 10), ("0000101", 11),
    ("0000111", 12),    ("00000100", 13), ("00000111", 14), ("000011000", 15),
    ("0000010111", 16), ("0000011000", 17), ("0000001000", 18),
    ("00001100111", 19), ("00001101000", 20), ("00001101100", 21),
    ("00000110111", 22), ("00000101000", 23), ("00000010111", 24),
    ("00000011000", 25), ("000011001010", 26), ("000011001011", 27),
    ("000011001100", 28), ("000011001101", 29), ("000001101000", 30),
    ("000001101001", 31), ("000001101010", 32), ("000001101011", 33),
    ("000011010010", 34), ("000011010011", 35), ("000011010100", 36),
    ("000011010101", 37), ("000011010110", 38), ("000011010111", 39),
    ("000001101100", 40), ("000001101101", 41), ("000011011010", 42),
    ("000011011011", 43), ("000001010100", 44), ("000001010101", 45),
    ("000001010110", 46), ("000001010111", 47), ("000001100100", 48),
    ("000001100101", 49), ("000001010010", 50), ("000001010011", 51),
    ("000000100100", 52), ("000000110111", 53), ("000000111000", 54),
    ("000000100111", 55), ("000000101000", 56), ("000001011000", 57),
    ("000001011001", 58), ("000000101011", 59), ("000000101100", 60),
    ("000001011010", 61), ("000001100110", 62), ("000001100111", 63),
    // make-up
    ("0000001111", 64),      ("000011001000", 128),  ("000011001001", 192),
    ("000001011011", 256),   ("000000110011", 320),  ("000000110100", 384),
    ("000000110101", 448),   ("0000001101100", 512), ("0000001101101", 576),
    ("0000001001010", 640),  ("0000001001011", 704), ("0000001001100", 768),
    ("0000001001101", 832),  ("0000001110010", 896), ("0000001110011", 960),
    ("0000001110100", 1024), ("0000001110101", 1088), ("0000001110110", 1152),
    ("0000001110111", 1216), ("0000001010010", 1280), ("0000001010011", 1344),
    ("0000001010100", 1408), ("0000001010101", 1472), ("0000001011010", 1536),
    ("0000001011011", 1600), ("0000001100100", 1664), ("0000001100101", 1728),
    // additional make-up (shared)
    ("00000001000", 1792),   ("00000001100", 1856),  ("00000001101", 1920),
    ("000000010010", 1984),  ("000000010011", 2048), ("000000010100", 2112),
    ("000000010101", 2176),  ("000000010110", 2240), ("000000010111", 2304),
    ("000000011100", 2368),  ("000000011101", 2432), ("000000011110", 2496),
    ("000000011111", 2560),
];

// ---------------------------------------------------------------------------
// Static trie instances
// ---------------------------------------------------------------------------

static WHITE_TRIE: OnceLock<TrieNode> = OnceLock::new();
static BLACK_TRIE: OnceLock<TrieNode> = OnceLock::new();
static MODE_TRIE: OnceLock<TrieNode> = OnceLock::new();

fn white_trie() -> &'static TrieNode {
    WHITE_TRIE.get_or_init(|| build_trie(WHITE_TABLE))
}

fn black_trie() -> &'static TrieNode {
    BLACK_TRIE.get_or_init(|| build_trie(BLACK_TABLE))
}

fn mode_trie() -> &'static TrieNode {
    MODE_TRIE.get_or_init(|| build_trie(MODE_TABLE))
}

/// Return the run-length trie for the given colour (1=white, 0=black).
#[inline]
fn run_trie(color: u8) -> &'static TrieNode {
    if color == 1 { white_trie() } else { black_trie() }
}

// ---------------------------------------------------------------------------
// Bit reader (MSB-first)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    fn next_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    fn is_exhausted(&self) -> bool {
        self.byte_pos >= self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Huffman decoder
// ---------------------------------------------------------------------------

fn decode_huffman(reader: &mut BitReader<'_>, trie: &TrieNode) -> Option<i32> {
    let mut node = trie;
    loop {
        if node.children[0].is_none() && node.children[1].is_none() {
            return node.value;
        }
        let bit = reader.next_bit()? as usize;
        match &node.children[bit] {
            None => {
                // Dead branch — should not happen with valid data
                return node.value;
            }
            Some(child) => {
                node = child;
            }
        }
    }
}

/// Decode a complete run (make-up codes chain into terminating code).
fn decode_run(reader: &mut BitReader<'_>, trie: &TrieNode) -> Option<i32> {
    let mut total = 0i32;
    loop {
        let run = decode_huffman(reader, trie)?;
        total += run;
        if run < 64 {
            return Some(total);
        }
        // make-up code — accumulate and continue
    }
}

// ---------------------------------------------------------------------------
// Group 4 (T.6) row decoder
// ---------------------------------------------------------------------------

/// Find b1: first pixel on refline at or after `start` with colour != `a0_color`.
fn find_b1(refline: &[u8], start: usize, a0_color: u8) -> usize {
    let width = refline.len();
    let mut pos = start;
    while pos < width && refline[pos] == a0_color {
        pos += 1;
    }
    pos
}

/// Find b2: first changing element on refline after b1 (next colour transition).
fn find_b2(refline: &[u8], b1: usize) -> usize {
    let width = refline.len();
    if b1 >= width {
        return width;
    }
    let b1_color = refline[b1];
    let mut pos = b1 + 1;
    while pos < width && refline[pos] == b1_color {
        pos += 1;
    }
    pos
}

fn decode_g4_row(
    reader: &mut BitReader<'_>,
    refline: &[u8],
    curline: &mut Vec<u8>,
    width: usize,
) -> bool {
    let mut a0: usize = 0;
    let mut a0_color: u8 = 1;

    curline.clear();
    curline.resize(width, 1);

    while a0 < width {
        let mode_val = match decode_huffman(reader, mode_trie()) {
            Some(v) => v,
            None => return false,
        };

        match mode_val {
            MODE_EOFB => return false,
            MODE_PASS => {
                let b1 = find_b1(refline, a0, a0_color);
                let b2 = find_b2(refline, b1);
                let end = b2.min(width);
                curline[a0..end].fill(a0_color);
                a0 = b2;
            }
            MODE_HORIZONTAL => {
                let n1 = match decode_run(reader, run_trie(a0_color)) {
                    Some(n) => n as usize,
                    None => return false,
                };
                let n2 = match decode_run(reader, run_trie(1 - a0_color)) {
                    Some(n) => n as usize,
                    None => return false,
                };
                let end1 = (a0 + n1).min(width);
                curline[a0..end1].fill(a0_color);
                let end2 = (end1 + n2).min(width);
                curline[end1..end2].fill(1 - a0_color);
                a0 = end2;
            }
            v => {
                // Vertical mode: v in -3..=3
                let b1 = find_b1(refline, a0, a0_color);
                let a1_signed = b1 as i32 + v;
                let a1 = if a1_signed < 0 {
                    0
                } else {
                    (a1_signed as usize).min(width)
                };
                curline[a0..a1].fill(a0_color);
                a0 = a1;
                a0_color = 1 - a0_color;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Group 3 1-D row decoder
// ---------------------------------------------------------------------------

fn decode_g3_1d_row(
    reader: &mut BitReader<'_>,
    curline: &mut Vec<u8>,
    width: usize,
    end_of_line: bool,
) -> bool {
    if end_of_line {
        // Consume EOL: 11 zero bits then a 1 bit (= 000000000001)
        let mut zeros = 0u32;
        loop {
            match reader.next_bit() {
                None => return false,
                Some(0) => zeros += 1,
                Some(_) => {
                    if zeros >= 11 {
                        break;
                    }
                    zeros = 0;
                }
            }
        }
    }

    curline.clear();
    curline.resize(width, 1);

    let mut pos = 0usize;
    let mut color = 1u8;

    while pos < width {
        let run = match decode_run(reader, run_trie(color)) {
            Some(r) => r as usize,
            None => return false,
        };
        let end = (pos + run).min(width);
        curline[pos..end].fill(color);
        pos = end;
        color = 1 - color;
    }
    true
}

// ---------------------------------------------------------------------------
// Output helper
// ---------------------------------------------------------------------------

fn pack_row(bits: &[u8], reversed: bool, output: &mut Vec<u8>) {
    let nbytes = (bits.len() + 7) / 8;
    let start = output.len();
    output.resize(start + nbytes, 0u8);
    for (i, &b) in bits.iter().enumerate() {
        let pixel = if reversed { 1 - b } else { b };
        if pixel != 0 {
            output[start + i / 8] |= 1u8 << (7 - (i % 8));
        }
    }
}

// ---------------------------------------------------------------------------
// Public pyfunction
// ---------------------------------------------------------------------------

/// Decode CCITT Fax (Group 3/4) compressed PDF image data.
///
/// k: compression mode (negative = Group 4, 0 = Group 3 1-D, positive = Group 3 2-D).
/// columns: image width in pixels (default 1728).
/// rows: number of rows to decode (0 = until EOD).
/// end_of_line: whether Group 3 EOL codes are present.
/// black_is_1: invert pixel values.
/// damaged_rows_before_error: tolerance for bad rows (currently ignored).
#[pyfunction]
#[pyo3(signature = (data, k=0, columns=1728, rows=0, end_of_line=false, black_is_1=false, damaged_rows_before_error=0))]
pub fn ccitt_fax_decode(
    data: &[u8],
    k: i32,
    columns: u32,
    rows: u32,
    end_of_line: bool,
    black_is_1: bool,
    damaged_rows_before_error: u32,
) -> PyResult<Vec<u8>> {
    let _ = damaged_rows_before_error;
    let width = columns as usize;
    if width == 0 {
        return Err(PyValueError::new_err("columns must be > 0"));
    }

    let mut reader = BitReader::new(data);
    let mut output: Vec<u8> = Vec::new();
    let mut row_count = 0u32;
    let max_rows = if rows == 0 { u32::MAX } else { rows };

    if k < 0 {
        // Group 4 (T.6)
        let mut refline: Vec<u8> = vec![1u8; width];
        let mut curline: Vec<u8> = Vec::with_capacity(width);

        while row_count < max_rows && !reader.is_exhausted() {
            if !decode_g4_row(&mut reader, &refline, &mut curline, width) {
                break;
            }
            pack_row(&curline, black_is_1, &mut output);
            row_count += 1;
            std::mem::swap(&mut refline, &mut curline);
        }
    } else {
        // Group 3 1-D (k == 0) or 2-D (k > 0, treated as 1-D)
        let mut curline: Vec<u8> = Vec::with_capacity(width);

        while row_count < max_rows && !reader.is_exhausted() {
            if !decode_g3_1d_row(&mut reader, &mut curline, width, end_of_line) {
                break;
            }
            pack_row(&curline, black_is_1, &mut output);
            row_count += 1;
        }
    }

    Ok(output)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ccitt_fax_decode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tries_build() {
        let _w = white_trie();
        let _b = black_trie();
        let _m = mode_trie();
    }

    #[test]
    fn test_white_run_zero() {
        // White run=0 code is "00110101" (8 bits)
        let data = &[0b00110101u8, 0];
        let mut reader = BitReader::new(data);
        let run = decode_run(&mut reader, white_trie());
        assert_eq!(run, Some(0));
    }

    #[test]
    fn test_empty_input() {
        let result = ccitt_fax_decode(&[], 0, 1, 0, false, false, 0).unwrap();
        assert_eq!(result, b"");
    }
}
