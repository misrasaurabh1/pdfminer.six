"""Benchmark compression codecs: LZW, CCITT, RunLength, ASCII85."""

import timeit
from base64 import a85encode
from io import BytesIO

from pdfminer.ascii85 import ascii85decode
from pdfminer.ccitt import CCITTFaxDecoder
from pdfminer.lzw import lzwdecode
from pdfminer.runlength import rldecode

# ---------------------------------------------------------------------------
# Synthetic test-data generators
# ---------------------------------------------------------------------------


def _make_lzw_data() -> bytes:
    """Return valid LZW-encoded bytes.

    pdfminer ships only a decoder, so we hand-craft a valid PDF LZW bitstream
    (9-bit codes, clear=256, EOD=257) by encoding a repeating byte pattern.
    """
    source = bytes(range(256)) * 40  # 10240 bytes

    bits: list[int] = []

    def emit(code: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((code >> shift) & 1)

    emit(256, 9)  # clear code

    table: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    next_code = 258
    nbits = 9
    w: bytes = b""

    for byte in source:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
        else:
            emit(table[w], nbits)
            table[wc] = next_code
            next_code += 1
            # Widen code length at table size thresholds (PDF spec §3.3.3)
            if next_code == 512:
                nbits = 10
            elif next_code == 1024:
                nbits = 11
            elif next_code == 2048:
                nbits = 12
            w = bytes([byte])

    if w:
        emit(table[w], nbits)
    emit(257, nbits)  # EOD code

    while len(bits) % 8:
        bits.append(0)

    out = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        out.append(byte_val)

    return bytes(out)


def _make_ccitt_data() -> bytes:
    """Return a minimal CCITT G4 bitstream (64-pixel-wide all-white image).

    Each all-white row is 64 V(0) codes ("1" bits). Two EOFB sequences
    (000000000001000000000001) terminate the stream.
    """
    n_rows = 50
    stream_bits = "1" * 64 * n_rows + "000000000001000000000001" * 2
    # Pad to byte boundary
    padding = (-len(stream_bits)) % 8
    stream_bits += "0" * padding

    out = bytearray()
    for i in range(0, len(stream_bits), 8):
        out.append(int(stream_bits[i : i + 8], 2))
    return bytes(out)


def _make_runlength_data() -> bytes:
    """Return RunLength-encoded bytes that decode to ~50 KB of output."""
    out = bytearray()
    lit = bytes(range(128))
    for _ in range(200):
        out.append(127)   # 128 literal bytes follow
        out.extend(lit)
        out.append(129)   # repeat next byte 128 times (257-129)
        out.append(0x42)
    out.append(128)  # EOD
    return bytes(out)


def _make_ascii85_data() -> bytes:
    """Return ASCII85-encoded bytes that decode to ~50 KB of output."""
    return a85encode(bytes(range(256)) * 200, adobe=True)


# Pre-build test data at module load so it is excluded from timing.
LZW_DATA = _make_lzw_data()
CCITT_DATA = _make_ccitt_data()
CCITT_WIDTH = 64
RL_DATA = _make_runlength_data()
A85_DATA = _make_ascii85_data()


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_lzw_decode(data: bytes = LZW_DATA, iterations: int = 100) -> float:
    """Benchmark LZW decoding. Returns ms per decode."""
    total = timeit.timeit(lambda: lzwdecode(data), number=iterations)
    return total / iterations * 1000


def bench_ccitt_decode(
    data: bytes = CCITT_DATA,
    width: int = CCITT_WIDTH,
    iterations: int = 20,
) -> float:
    """Benchmark CCITT G4 fax decode. Returns ms per decode."""

    def run() -> None:
        parser = CCITTFaxDecoder(width, bytealign=False, reversed=False)
        parser.feedbytes(data)
        parser.close()

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


def bench_runlength_decode(data: bytes = RL_DATA, iterations: int = 1000) -> float:
    """Benchmark RunLength decoding. Returns ms per decode."""
    total = timeit.timeit(lambda: rldecode(data), number=iterations)
    return total / iterations * 1000


def bench_ascii85_decode(data: bytes = A85_DATA, iterations: int = 100) -> float:
    """Benchmark ASCII85 decoding. Returns ms per decode."""
    total = timeit.timeit(lambda: ascii85decode(data), number=iterations)
    return total / iterations * 1000


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Codec Benchmarks ===")
    print(f"LZW input size:       {len(LZW_DATA):>8} bytes")
    print(f"CCITT input size:     {len(CCITT_DATA):>8} bytes")
    print(f"RunLength input size: {len(RL_DATA):>8} bytes")
    print(f"ASCII85 input size:   {len(A85_DATA):>8} bytes")
    print()

    results = [
        ("LZW decode", bench_lzw_decode()),
        ("CCITT G4 decode", bench_ccitt_decode()),
        ("RunLength decode", bench_runlength_decode()),
        ("ASCII85 decode", bench_ascii85_decode()),
    ]

    print(f"{'Benchmark':<25} | {'ms/call':>10}")
    print("-" * 40)
    for name, ms in results:
        print(f"{name:<25} | {ms:>10.3f}")
