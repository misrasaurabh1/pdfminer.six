"""Tests for Rust-accelerated CMap decode.

These tests verify that:
1. The Rust cmap_decode function produces the same results as the Python fallback.
2. CMap.code2cid and CMap.attrs remain Python dicts accessible from Python.
3. CMapDB.get_cmap() and CMapDB.CMapNotFound work correctly.
"""

import pytest

from pdfminer.cmapdb import CMap, CMapDB


def test_cmap_decode_single_byte():
    """Basic single-byte CID mapping."""
    cmap = CMap()
    cmap.code2cid = {0x41: 65, 0x42: 66}
    result = list(cmap.decode(b"AB"))
    assert result == [65, 66]


def test_cmap_decode_multibyte():
    """Two-byte code sequences map to CIDs."""
    cmap = CMap()
    cmap.code2cid = {0x00: {0x41: 65, 0x42: 66}}
    result = list(cmap.decode(b"\x00\x41\x00\x42"))
    assert result == [65, 66]


def test_cmap_decode_missing_byte():
    """Bytes not in the mapping are silently skipped."""
    cmap = CMap()
    cmap.code2cid = {0x41: 65}
    # 0xFF is not in the map, should be ignored; 0x41 should still decode
    result = list(cmap.decode(b"\xff\x41"))
    assert result == [65]


def test_cmap_decode_empty():
    """Decoding empty bytes returns empty list."""
    cmap = CMap()
    cmap.code2cid = {0x41: 65}
    result = list(cmap.decode(b""))
    assert result == []


def test_cmap_decode_empty_code2cid():
    """Decoding with empty code2cid returns empty list."""
    cmap = CMap()
    result = list(cmap.decode(b"\x41\x42"))
    assert result == []


def test_cmap_code2cid_is_dict():
    """code2cid must be a Python dict and be mutable."""
    cmap = CMap()
    assert isinstance(cmap.code2cid, dict)
    cmap.code2cid[0x41] = 65
    assert cmap.code2cid[0x41] == 65


def test_cmap_attrs_is_dict():
    """attrs must be a Python dict."""
    cmap = CMap()
    assert isinstance(cmap.attrs, dict)


def test_cmap_attrs_wmode():
    """WMode attribute controls is_vertical()."""
    cmap = CMap(WMode=0)
    assert not cmap.is_vertical()
    cmap.attrs["WMode"] = 1
    assert cmap.is_vertical()


def test_cmapdv_get_cmap():
    """CMapDB.get_cmap returns a usable CMapBase object."""
    cmap = CMapDB.get_cmap("78-EUC-H")
    assert cmap is not None


def test_cmap_not_found():
    """CMapDB.CMapNotFound is raised for unknown CMap names."""
    with pytest.raises(CMapDB.CMapNotFound):
        CMapDB.get_cmap("NonExistentCMap_xyzzy")


def test_identity_cmaps():
    """Identity-H and Identity-V return IdentityCMap objects."""
    h = CMapDB.get_cmap("Identity-H")
    v = CMapDB.get_cmap("Identity-V")
    assert not h.is_vertical()
    assert v.is_vertical()


def test_python_rust_identical():
    """Python and Rust decode produce identical results on real CMap data."""
    cmap = CMapDB.get_cmap("78-EUC-H")
    # Compose a sample of 2-byte CID codes from code2cid
    if not isinstance(cmap, CMap) or not cmap.code2cid:
        pytest.skip("CMap has no code2cid")

    # Build a test code sequence from the first few entries
    test_bytes = bytearray()
    for outer_key, outer_val in list(cmap.code2cid.items())[:5]:
        if isinstance(outer_val, dict):
            for inner_key in list(outer_val.keys())[:3]:
                test_bytes.append(outer_key)
                test_bytes.append(inner_key)
        elif isinstance(outer_val, int):
            test_bytes.append(outer_key)

    code = bytes(test_bytes)

    # Get Python result via private fallback
    python_result = list(cmap._decode_python(code))

    # Get result via the public decode (uses Rust if available)
    rust_result = list(cmap.decode(code))

    assert python_result == rust_result
