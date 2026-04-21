"""Benchmark CMap decoding with various code sequences and large input batches."""

import math
import struct
import timeit

from pdfminer.cmapdb import CMap, CMapDB, IdentityCMap, IdentityCMapByte

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# Large batch of 2-byte code sequences (UTF-16BE style, common in CJK PDFs)
_LARGE_BATCH_2BYTE = struct.pack(f">{10_000}H", *(i % 65536 for i in range(10_000)))

# Large batch of 1-byte code sequences
_LARGE_BATCH_1BYTE = bytes(range(256)) * 40  # 10240 bytes


def _make_cmap() -> CMap:
    cmap = CMap()
    cmap.code2cid = {i: i + 1 for i in range(256)}
    return cmap


_CMAP = _make_cmap()


def _make_2byte_cmap() -> CMap:
    """Two-level trie: first byte -> {second byte -> cid}, mimicking CJK CMaps."""
    cmap = CMap()
    d: dict[int, object] = {}
    for high in range(16):
        sub: dict[int, object] = {low: high * 256 + low for low in range(256)}
        d[high] = sub
    cmap.code2cid = d
    return cmap


_CMAP_2BYTE = _make_2byte_cmap()

_SINGLE_BYTE_INPUT = bytes(range(256)) * 40
_TWO_BYTE_INPUT = bytes(
    b for pair in [(h, l) for h in range(16) for l in range(256)] * 3
    for b in pair
)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_identity_cmap_decode(
    data: bytes = _LARGE_BATCH_2BYTE,
    iterations: int = 1_000,
) -> float:
    """Benchmark IdentityCMap.decode() on a large batch. Returns ms per call."""
    cmap = IdentityCMap()
    total = timeit.timeit(lambda: cmap.decode(data), number=iterations)
    return total / iterations * 1000


def bench_identity_cmap_byte_decode(
    data: bytes = _LARGE_BATCH_1BYTE,
    iterations: int = 1_000,
) -> float:
    """Benchmark IdentityCMapByte.decode() on a large batch. Returns ms per call."""
    cmap = IdentityCMapByte()
    total = timeit.timeit(lambda: cmap.decode(data), number=iterations)
    return total / iterations * 1000


def bench_cmap_decode_1byte(
    data: bytes = _SINGLE_BYTE_INPUT,
    iterations: int = 200,
) -> float:
    """Benchmark CMap.decode() with single-byte codes. Returns ms per call."""
    cmap = _CMAP
    total = timeit.timeit(lambda: list(cmap.decode(data)), number=iterations)
    return total / iterations * 1000


def bench_cmap_decode_2byte(
    data: bytes = _TWO_BYTE_INPUT,
    iterations: int = 200,
) -> float:
    """Benchmark CMap.decode() with two-byte codes (CJK-like trie). Returns ms per call."""
    cmap = _CMAP_2BYTE
    total = timeit.timeit(lambda: list(cmap.decode(data)), number=iterations)
    return total / iterations * 1000


def bench_cmap_db_load(cmap_name: str = "UniGB-UCS2-H", iterations: int = 5) -> float:
    """Benchmark CMapDB.get_cmap() for a named CMap (exercises disk I/O + parse).

    Returns ms per call, or NaN if the CMap is not installed.
    The cache is cleared before each timed iteration so disk I/O is included.
    """
    try:
        CMapDB._cmap_cache.pop(cmap_name, None)
        CMapDB._umap_cache.pop(cmap_name, None)
        CMapDB.get_cmap(cmap_name)  # existence check
    except CMapDB.CMapNotFound:
        return float("nan")

    def run() -> None:
        CMapDB._cmap_cache.pop(cmap_name, None)
        CMapDB._umap_cache.pop(cmap_name, None)
        CMapDB.get_cmap(cmap_name)

    total = timeit.timeit(run, number=iterations)
    return total / iterations * 1000


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CMapDB Benchmarks ===")
    print(f"IdentityCMap input:      {len(_LARGE_BATCH_2BYTE):>8} bytes ({len(_LARGE_BATCH_2BYTE)//2:,} code points)")
    print(f"IdentityCMapByte input:  {len(_LARGE_BATCH_1BYTE):>8} bytes")
    print(f"CMap 1-byte input:       {len(_SINGLE_BYTE_INPUT):>8} bytes")
    print(f"CMap 2-byte input:       {len(_TWO_BYTE_INPUT):>8} bytes")
    print()

    results = [
        ("IdentityCMap decode (2-byte)", bench_identity_cmap_decode()),
        ("IdentityCMapByte decode (1-byte)", bench_identity_cmap_byte_decode()),
        ("CMap decode (1-byte codes)", bench_cmap_decode_1byte()),
        ("CMap decode (2-byte trie)", bench_cmap_decode_2byte()),
        ("CMapDB load UniGB-UCS2-H", bench_cmap_db_load()),
    ]

    print(f"{'Benchmark':<35} | {'ms/call':>10}")
    print("-" * 50)
    for name, ms in results:
        if math.isnan(ms):
            print(f"{name:<35} | {'N/A (not found)':>10}")
        else:
            print(f"{name:<35} | {ms:>10.3f}")
