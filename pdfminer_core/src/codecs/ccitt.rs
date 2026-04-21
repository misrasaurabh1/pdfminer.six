/// CCITT Fax decoder (T.4/T.6) implementation in Rust.
///
/// Implements T.6 Group 4 2D fax decompression as used in PDF CCITTFaxDecode
/// filter. The algorithm decodes bit-by-bit using Huffman tables for run-length
/// coding of white and black pixel runs.
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ────────────────────────────────────────────────────────────────────────────
// Huffman tables – fixed T.4 code tables
// Each entry is (code_bits, code_len, run_length)
// ────────────────────────────────────────────────────────────────────────────

static WHITE_TERM: &[(u16, u8, u16)] = &[
    (0b00110101, 8, 0),
    (0b000111, 6, 1),
    (0b0111, 4, 2),
    (0b1000, 4, 3),
    (0b1011, 4, 4),
    (0b1100, 4, 5),
    (0b1110, 4, 6),
    (0b1111, 4, 7),
    (0b10011, 5, 8),
    (0b10100, 5, 9),
    (0b00111, 5, 10),
    (0b01000, 5, 11),
    (0b001000, 6, 12),
    (0b000011, 6, 13),
    (0b110100, 6, 14),
    (0b110101, 6, 15),
    (0b101010, 6, 16),
    (0b101011, 6, 17),
    (0b0100111, 7, 18),
    (0b0001100, 7, 19),
    (0b0001000, 7, 20),
    (0b0010111, 7, 21),
    (0b0000011, 7, 22),
    (0b0000100, 7, 23),
    (0b0101000, 7, 24),
    (0b0101011, 7, 25),
    (0b0010011, 7, 26),
    (0b0100100, 7, 27),
    (0b0011000, 7, 28),
    (0b00000010, 8, 29),
    (0b00000011, 8, 30),
    (0b00011010, 8, 31),
    (0b00011011, 8, 32),
    (0b00010010, 8, 33),
    (0b00010011, 8, 34),
    (0b00010100, 8, 35),
    (0b00010101, 8, 36),
    (0b00010110, 8, 37),
    (0b00010111, 8, 38),
    (0b00101000, 8, 39),
    (0b00101001, 8, 40),
    (0b00101010, 8, 41),
    (0b00101011, 8, 42),
    (0b00101100, 8, 43),
    (0b00101101, 8, 44),
    (0b00000100, 8, 45),
    (0b00000101, 8, 46),
    (0b00001010, 8, 47),
    (0b00001011, 8, 48),
    (0b01010010, 8, 49),
    (0b01010011, 8, 50),
    (0b01010100, 8, 51),
    (0b01010101, 8, 52),
    (0b00100100, 8, 53),
    (0b00100101, 8, 54),
    (0b01011000, 8, 55),
    (0b01011001, 8, 56),
    (0b01011010, 8, 57),
    (0b01011011, 8, 58),
    (0b01001010, 8, 59),
    (0b01001011, 8, 60),
    (0b00110010, 8, 61),
    (0b00110011, 8, 62),
    (0b00110100, 8, 63),
];

static WHITE_MAKEUP: &[(u16, u8, u16)] = &[
    (0b11011, 5, 64),
    (0b10010, 5, 128),
    (0b010111, 6, 192),
    (0b0110111, 7, 256),
    (0b00110110, 8, 320),
    (0b00110111, 8, 384),
    (0b01100100, 8, 448),
    (0b01100101, 8, 512),
    (0b01101000, 8, 576),
    (0b01100111, 8, 640),
    (0b011001100, 9, 704),
    (0b011001101, 9, 768),
    (0b011010010, 9, 832),
    (0b011010011, 9, 896),
    (0b011010100, 9, 960),
    (0b011010101, 9, 1024),
    (0b011010110, 9, 1088),
    (0b011010111, 9, 1152),
    (0b011011000, 9, 1216),
    (0b011011001, 9, 1280),
    (0b011011010, 9, 1344),
    (0b011011011, 9, 1408),
    (0b010011000, 9, 1472),
    (0b010011001, 9, 1536),
    (0b010011010, 9, 1600),
    (0b011000, 6, 1664),
    (0b010011011, 9, 1728),
];

// Shared by white and black; run lengths >= 1792.
// These codes are self-terminating (no following terminating code needed).
static EXTENDED_MAKEUP: &[(u32, u8, u16)] = &[
    (0b00000001000, 11, 1792),
    (0b00000001100, 11, 1856),
    (0b00000001101, 11, 1920),
    (0b000000010010, 12, 1984),
    (0b000000010011, 12, 2048),
    (0b000000010100, 12, 2112),
    (0b000000010101, 12, 2176),
    (0b000000010110, 12, 2240),
    (0b000000010111, 12, 2304),
    (0b000000011100, 12, 2368),
    (0b000000011101, 12, 2432),
    (0b000000011110, 12, 2496),
    (0b000000011111, 12, 2560),
];

static BLACK_TERM: &[(u32, u8, u16)] = &[
    (0b0000110111, 10, 0),
    (0b010, 3, 1),
    (0b11, 2, 2),
    (0b10, 2, 3),
    (0b011, 3, 4),
    (0b0011, 4, 5),
    (0b0010, 4, 6),
    (0b00011, 5, 7),
    (0b000101, 6, 8),
    (0b000100, 6, 9),
    (0b0000100, 7, 10),
    (0b0000101, 7, 11),
    (0b0000111, 7, 12),
    (0b00000100, 8, 13),
    (0b00000111, 8, 14),
    (0b000011000, 9, 15),
    (0b0000010111, 10, 16),
    (0b0000011000, 10, 17),
    (0b0000001000, 10, 18),
    (0b00001100111, 11, 19),
    (0b00001101000, 11, 20),
    (0b00001101100, 11, 21),
    (0b00000110111, 11, 22),
    (0b00000101000, 11, 23),
    (0b00000010111, 11, 24),
    (0b00000011000, 11, 25),
    (0b000011001010, 12, 26),
    (0b000011001011, 12, 27),
    (0b000011001100, 12, 28),
    (0b000011001101, 12, 29),
    (0b000001101000, 12, 30),
    (0b000001101001, 12, 31),
    (0b000001101010, 12, 32),
    (0b000001101011, 12, 33),
    (0b000011010010, 12, 34),
    (0b000011010011, 12, 35),
    (0b000011010100, 12, 36),
    (0b000011010101, 12, 37),
    (0b000011010110, 12, 38),
    (0b000011010111, 12, 39),
    (0b000001101100, 12, 40),
    (0b000001101101, 12, 41),
    (0b000011011010, 12, 42),
    (0b000011011011, 12, 43),
    (0b000001010100, 12, 44),
    (0b000001010101, 12, 45),
    (0b000001010110, 12, 46),
    (0b000001010111, 12, 47),
    (0b000001100100, 12, 48),
    (0b000001100101, 12, 49),
    (0b000001010010, 12, 50),
    (0b000001010011, 12, 51),
    (0b000000100100, 12, 52),
    (0b000000110111, 12, 53),
    (0b000000111000, 12, 54),
    (0b000000100111, 12, 55),
    (0b000000101000, 12, 56),
    (0b000001011000, 12, 57),
    (0b000001011001, 12, 58),
    (0b000000101011, 12, 59),
    (0b000000101100, 12, 60),
    (0b000001011010, 12, 61),
    (0b000001100110, 12, 62),
    (0b000001100111, 12, 63),
];

static BLACK_MAKEUP: &[(u32, u8, u16)] = &[
    (0b0000001111, 10, 64),
    (0b000011001000, 12, 128),
    (0b000011001001, 12, 192),
    (0b000001011011, 12, 256),
    (0b000000110011, 12, 320),
    (0b000000110100, 12, 384),
    (0b000000110101, 12, 448),
    (0b0000001101100, 13, 512),
    (0b0000001101101, 13, 576),
    (0b0000001001010, 13, 640),
    (0b0000001001011, 13, 704),
    (0b0000001001100, 13, 768),
    (0b0000001001101, 13, 832),
    (0b0000001110010, 13, 896),
    (0b0000001110011, 13, 960),
    (0b0000001110100, 13, 1024),
    (0b0000001110101, 13, 1088),
    (0b0000001110110, 13, 1152),
    (0b0000001110111, 13, 1216),
    (0b0000001010010, 13, 1280),
    (0b0000001010011, 13, 1344),
    (0b0000001010100, 13, 1408),
    (0b0000001010101, 13, 1472),
    (0b0000001011010, 13, 1536),
    (0b0000001011011, 13, 1600),
    (0b0000001100100, 13, 1664),
    (0b0000001100101, 13, 1728),
];

// ────────────────────────────────────────────────────────────────────────────
// Huffman tree
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Node {
    Branch(usize, usize), // (index of 0-child, index of 1-child)
    Leaf(i32),
}

struct HuffmanTree {
    nodes: Vec<Node>,
}

impl HuffmanTree {
    fn new() -> Self {
        HuffmanTree {
            nodes: vec![Node::Branch(0, 0)],
        }
    }

    fn add(&mut self, code: u64, len: u8, value: i32) {
        let mut idx = 0;
        for i in (0..len).rev() {
            let bit = ((code >> i) & 1) as usize;
            let (c0, c1) = match &self.nodes[idx] {
                Node::Branch(c0, c1) => (*c0, *c1),
                Node::Leaf(_) => panic!("trying to extend a leaf"),
            };
            let child = if bit == 0 { c0 } else { c1 };
            if i == 0 {
                let new_idx = self.nodes.len();
                self.nodes.push(Node::Leaf(value));
                match &mut self.nodes[idx] {
                    Node::Branch(c0, c1) => {
                        if bit == 0 {
                            *c0 = new_idx;
                        } else {
                            *c1 = new_idx;
                        }
                    }
                    _ => unreachable!(),
                }
            } else if child == 0 {
                let new_idx = self.nodes.len();
                self.nodes.push(Node::Branch(0, 0));
                match &mut self.nodes[idx] {
                    Node::Branch(c0, c1) => {
                        if bit == 0 {
                            *c0 = new_idx;
                        } else {
                            *c1 = new_idx;
                        }
                    }
                    _ => unreachable!(),
                }
                idx = new_idx;
            } else {
                idx = child;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Bit reader
// ────────────────────────────────────────────────────────────────────────────

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 7 = MSB, 0 = LSB of current byte
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 7,
        }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let byte = self.data[self.byte_pos];
        let bit = (byte >> self.bit_pos) & 1;
        if self.bit_pos == 0 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        } else {
            self.bit_pos -= 1;
        }
        Some(bit)
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos != 7 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        }
    }

    fn is_exhausted(&self) -> bool {
        self.byte_pos >= self.data.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Huffman tables (built once, shared across all decode calls)
// ────────────────────────────────────────────────────────────────────────────

struct Tables {
    white_run: HuffmanTree,
    black_run: HuffmanTree,
    mode: HuffmanTree,
}

fn build_run_tree(
    term: &[(impl Into<u64> + Copy, u8, u16)],
    makeup: &[(impl Into<u64> + Copy, u8, u16)],
) -> HuffmanTree {
    let mut tree = HuffmanTree::new();
    for &(code, len, run) in term {
        tree.add(code.into(), len, run as i32);
    }
    for &(code, len, run) in makeup {
        tree.add(code.into(), len, run as i32);
    }
    for &(code, len, run) in EXTENDED_MAKEUP {
        tree.add(code as u64, len, run as i32);
    }
    tree
}

impl Tables {
    fn build() -> Self {
        let white_run = build_run_tree(WHITE_TERM, WHITE_MAKEUP);
        let black_run = build_run_tree(BLACK_TERM, BLACK_MAKEUP);

        let mut mode = HuffmanTree::new();
        mode.add(0b0001, 4, MODE_PASS);
        mode.add(0b001, 3, MODE_HORIZ);
        mode.add(0b1, 1, 0);   // v(0)
        mode.add(0b011, 3, 1); // v(+1)
        mode.add(0b010, 3, -1); // v(-1)
        mode.add(0b000011, 6, 2); // v(+2)
        mode.add(0b000010, 6, -2); // v(-2)
        mode.add(0b0000011, 7, 3); // v(+3)
        mode.add(0b0000010, 7, -3); // v(-3)
        mode.add(0b000000000001000000000001, 24, MODE_EOFB);
        mode.add(0b0000001111, 10, MODE_UNCOMPRESSED);

        Tables {
            white_run,
            black_run,
            mode,
        }
    }
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn get_tables() -> &'static Tables {
    TABLES.get_or_init(Tables::build)
}

const MODE_PASS: i32 = i32::MAX;
const MODE_HORIZ: i32 = i32::MAX - 1;
const MODE_EOFB: i32 = i32::MAX - 2;
const MODE_UNCOMPRESSED: i32 = i32::MAX - 3;

// ────────────────────────────────────────────────────────────────────────────
// Core decode helpers
// ────────────────────────────────────────────────────────────────────────────

fn decode_run(reader: &mut BitReader, tree: &HuffmanTree) -> Option<i32> {
    let mut idx = 0usize;
    loop {
        let bit = reader.read_bit()? as usize;
        match &tree.nodes[idx] {
            Node::Branch(c0, c1) => {
                idx = if bit == 0 { *c0 } else { *c1 };
                if idx == 0 {
                    return None;
                }
            }
            Node::Leaf(_) => return None,
        }
        match &tree.nodes[idx] {
            Node::Leaf(v) => return Some(*v),
            Node::Branch(_, _) => {}
        }
    }
}

/// Decode a complete run length: one or more makeup codes followed by a
/// terminating code.  Extended makeup codes (run >= 1792) are self-terminating.
fn decode_full_run(reader: &mut BitReader, tree: &HuffmanTree) -> Option<u32> {
    let mut total: u32 = 0;
    loop {
        let run = decode_run(reader, tree)?;
        if run < 0 {
            return None;
        }
        total += run as u32;
        // A run < 64 or >= 1792 terminates the sequence; 64..1791 are makeup
        // codes that must be followed by another code.
        if run < 64 || run >= 1792 {
            return Some(total);
        }
    }
}

/// Return the position of the first changing element on `ref_line` at or after
/// `start` whose pixel value equals `target_color`.
/// A changing element at position `i` satisfies `ref_line[i-1] != ref_line[i]`
/// (position 0 has an implicit white pixel to its left).
fn find_ce(ref_line: &[u8], start: usize, width: usize, target_color: u8) -> u32 {
    let mut i = start;
    while i < width {
        // Pixel to the left (implicit white before position 0)
        let left = if i == 0 { 1u8 } else { ref_line[i - 1] };
        if left != ref_line[i] && ref_line[i] == target_color {
            return i as u32;
        }
        i += 1;
    }
    width as u32
}

/// Find b1: first CE on `ref_line` strictly after `a0` with color `!current_color`.
fn find_b1(ref_line: &[u8], a0: i32, width: u32, current_color: u8) -> u32 {
    let start = (a0 + 1).max(0) as usize;
    find_ce(ref_line, start, width as usize, 1 - current_color)
}

/// Find b2: first CE on `ref_line` strictly after `b1` with color `current_color`.
fn find_b2(ref_line: &[u8], b1: u32, width: u32, current_color: u8) -> u32 {
    let start = (b1 + 1) as usize;
    find_ce(ref_line, start, width as usize, current_color)
}

// ────────────────────────────────────────────────────────────────────────────
// T.6 Group 4 2D decoder
// ────────────────────────────────────────────────────────────────────────────

/// Decode one T.6 scan line.  Returns `Some(pixels)` on success, `None` on
/// EOFB or data exhaustion before the row is complete.  Sets `*eofb_seen`
/// when the end-of-block marker is consumed.
fn decode_g4_row(
    reader: &mut BitReader,
    tables: &Tables,
    ref_line: &[u8],
    columns: u32,
    eofb_seen: &mut bool,
) -> Option<Vec<u8>> {
    let w = columns as usize;
    let mut cur_line: Vec<u8> = vec![1u8; w];
    let mut a0: i32 = -1;
    let mut color: u8 = 1; // lines start white

    loop {
        let mode = decode_run(reader, &tables.mode)?;

        if mode == MODE_EOFB {
            let _ = decode_run(reader, &tables.mode); // consume second EOFB
            *eofb_seen = true;
            return None;
        }

        if mode == MODE_UNCOMPRESSED {
            // Rare in PDFs; treat as end of stream
            return None;
        }

        if mode == MODE_PASS {
            let b1 = find_b1(ref_line, a0, columns, color);
            let b2 = find_b2(ref_line, b1, columns, color);
            // Fill [max(0, a0), b2) with current color.
            // Mirrors Python _do_pass: for x in range(self._curpos, x1)
            let start = a0.max(0) as usize;
            let end = (b2 as usize).min(w);
            cur_line[start..end].fill(color);
            a0 = b2 as i32;
        } else if mode == MODE_HORIZ {
            let run1_tree = if color == 1 {
                &tables.white_run
            } else {
                &tables.black_run
            };
            let n1 = decode_full_run(reader, run1_tree)?;

            let run2_tree = if color == 0 {
                &tables.white_run
            } else {
                &tables.black_run
            };
            let n2 = decode_full_run(reader, run2_tree)?;

            // Fill n1 of current color then n2 of the opposite color.
            // Start at max(0, a0) — mirrors Python: x = max(0, _curpos)
            let x = a0.max(0) as usize;
            let end1 = (x + n1 as usize).min(w);
            cur_line[x..end1].fill(color);
            let end2 = (end1 + n2 as usize).min(w);
            cur_line[end1..end2].fill(1 - color);
            a0 = (x + n1 as usize + n2 as usize) as i32;
        } else {
            // Vertical mode: dx ∈ {-3..=+3}
            let dx = mode;
            let b1 = find_b1(ref_line, a0, columns, color);
            let a1 = (b1 as i32 + dx).clamp(0, columns as i32) as u32;

            // Fill [max(0, a0), a1) with current color, handling reversed ranges.
            // Mirrors Python _do_vertical: x0 = max(0, self._curpos)
            let x0 = a0.max(0) as usize;
            let x1 = (a1 as usize).min(w);
            if x1 >= x0 {
                cur_line[x0..x1].fill(color);
            } else {
                cur_line[x1..x0].fill(color);
            }
            a0 = a1 as i32;
            color = 1 - color;
        }

        if a0 >= columns as i32 {
            return Some(cur_line);
        }
    }
}

fn decode_g4(
    reader: &mut BitReader,
    tables: &Tables,
    columns: u32,
    rows: u32,
    reversed: bool,
) -> Result<Vec<u8>, String> {
    let bytes_per_row = (columns as usize + 7) / 8;
    let capacity = if rows > 0 {
        rows as usize * bytes_per_row
    } else {
        0
    };
    let mut output: Vec<u8> = Vec::with_capacity(capacity);
    let mut ref_line: Vec<u8> = vec![1u8; columns as usize];
    let mut row_count: u32 = 0;
    let mut eofb_seen = false;

    loop {
        if (rows > 0 && row_count >= rows) || reader.is_exhausted() || eofb_seen {
            break;
        }
        match decode_g4_row(reader, tables, &ref_line, columns, &mut eofb_seen) {
            Some(cur_line) => {
                encode_row_into(&cur_line, reversed, &mut output);
                ref_line = cur_line;
                row_count += 1;
            }
            None => break,
        }
    }

    Ok(output)
}

// ────────────────────────────────────────────────────────────────────────────
// T.4 Group 3 1D decoder (K=0)
// ────────────────────────────────────────────────────────────────────────────

fn decode_g3_1d(
    reader: &mut BitReader,
    tables: &Tables,
    columns: u32,
    rows: u32,
    end_of_line: bool,
    reversed: bool,
) -> Result<Vec<u8>, String> {
    let w = columns as usize;
    let mut output: Vec<u8> = Vec::new();
    let mut row_count: u32 = 0;

    loop {
        if rows > 0 && row_count >= rows {
            break;
        }
        if reader.is_exhausted() {
            break;
        }

        let mut cur_line: Vec<u8> = vec![1u8; w];
        let mut x: usize = 0;
        let mut color: u8 = 1;

        while x < w {
            let run_tree = if color == 1 {
                &tables.white_run
            } else {
                &tables.black_run
            };
            let run = match decode_full_run(reader, run_tree) {
                Some(v) => v,
                None => break,
            };
            let end = (x + run as usize).min(w);
            cur_line[x..end].fill(color);
            x += run as usize;
            if run > 0 || x < w {
                color = 1 - color;
            }
        }

        encode_row_into(&cur_line, reversed, &mut output);
        row_count += 1;

        if end_of_line {
            reader.align_to_byte();
        }
    }

    Ok(output)
}

// ────────────────────────────────────────────────────────────────────────────
// Output encoding
// ────────────────────────────────────────────────────────────────────────────

fn encode_row_into(line: &[u8], reversed: bool, out: &mut Vec<u8>) {
    let byte_count = (line.len() + 7) / 8;
    let base = out.len();
    out.resize(base + byte_count, 0u8);
    let dst = &mut out[base..];
    for (i, &b) in line.iter().enumerate() {
        let pixel = if reversed { 1 - b } else { b };
        if pixel != 0 {
            dst[i / 8] |= 0x80u8 >> (i % 8);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

pub fn ccitt_decode_impl(
    data: &[u8],
    k: i32,
    columns: u32,
    rows: u32,
    end_of_line: bool,
    black_is_1: bool,
    _damaged_rows_before_error: u32,
) -> Result<Vec<u8>, String> {
    if columns == 0 {
        return Err("Columns must be > 0".to_string());
    }

    let tables = get_tables();
    let mut reader = BitReader::new(data);

    // In standard T.4/T.6 white=1, black=0.  When BlackIs1=true the output
    // bits are inverted to match the Python CCITTFaxDecoder(reversed=True).
    let reversed = black_is_1;

    if k == -1 {
        decode_g4(&mut reader, tables, columns, rows, reversed)
    } else if k == 0 {
        decode_g3_1d(&mut reader, tables, columns, rows, end_of_line, reversed)
    } else {
        Err(format!(
            "CCITT K={k} not supported (only K=0 and K=-1)"
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (data, k=0, columns=1728, rows=0, end_of_line=false, black_is_1=false, damaged_rows_before_error=0))]
pub fn ccitt_decode(
    data: &[u8],
    k: i32,
    columns: u32,
    rows: u32,
    end_of_line: bool,
    black_is_1: bool,
    damaged_rows_before_error: u32,
) -> PyResult<Vec<u8>> {
    ccitt_decode_impl(
        data,
        k,
        columns,
        rows,
        end_of_line,
        black_is_1,
        damaged_rows_before_error,
    )
    .map_err(PyValueError::new_err)
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ccitt_decode, m)?)?;
    Ok(())
}
