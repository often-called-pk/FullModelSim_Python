"""ChassisSim v3.41 car-file codec of functions/carfile.py (ChassisSim phase 1).

The synthetic checks build records of every supported tag from arbitrary values and from raw float bits, round-trip
them, cut the file at every length and pin the ValueErrors of decode and encode. They write only into a temporary
directory and run no solve. The vendor section runs only when CHASSISSIM_DIR names the ChassisSim install folder. It
reads the 31 shipped car files, checks the encode(decode(f)) == f gate and pins the GP3 2013 Eastern Creek variants.
It writes nothing into the vendor folders.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_carfile.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_carfile.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import glob
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from functions.carfile import END_TAG, decode, decode_bytes, encode, write

PASS = 0
TYRE_DIMS = (2, 3, 4, 5, 6, 7, 8)
TYRE_SIZES = [8, 10, 10, 11, 71, 14, 10, 11, 8, 14, 12, 12, 12]  # table lengths of TYRE_DIMS, worked out by hand


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def err(fn):
    """Return the message of the ValueError that fn raises, or None when fn raises none."""
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


def same(a, b):
    """Exact equality. Arrays compare by shape and bytes (NaN and -0.0 included), other values by ==."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (bytes, int)):
        return a == b
    x, y = np.asarray(a, '<f8'), np.asarray(b, '<f8')
    return x.shape == y.shape and x.tobytes() == y.tobytes()


def recs_equal(A, B):
    return len(A) == len(B) and all(ta == tb and ca == cb and same(va, vb)
                                    for (ta, ca, va), (tb, cb, vb) in zip(A, B))


def close(a, b):
    """Equal up to 1e-6 relative on floats. Bytes and ints compare exactly."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if a is None or isinstance(a, (bytes, int)):
        return a == b
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    m = ~np.isnan(x)
    return bool(np.all(np.abs(x[m] - y[m]) <= 1e-6 * np.maximum(np.abs(x[m]), np.abs(y[m])) + 1e-12))


def rec(tag, cid, payload=b''):
    """One raw record: int32 tag, int32 cid, then the payload bytes."""
    return struct.pack('<ii', tag, cid) + payload


def read_bytes(path):
    with open(path, 'rb') as fh:
        return fh.read()


def arb(n, s):
    """n arbitrary finite numbers, different for each seed s."""
    return np.sin(np.arange(n) * 0.9 + s) * 7.3 - 0.61


def tyre_tables(s):
    tabs = [arb(k, s + i) for i, k in enumerate(TYRE_SIZES)]
    tabs[3][0] = np.nan
    tabs[4][1] = -0.0
    return tabs


def bits(*u):
    """float64 array from raw 64-bit patterns."""
    return np.array(u, dtype=np.uint64).view('<f8')


BIT_VALUES = {2: 0x8000000000000000, 3: 0x7FF80000DEADBEEF, 4: 0x7FF0000000000001}  # -0.0, NaN payload, sNaN
BIT_ARRAY = [0x7FF8DEAD00000001, 0x7FF0000000000001, 0x8000000000000000, 0x3FF0000000000001]


def sample_records():
    """One record of every supported tag, with arbitrary values, NaN, -0.0 and raw bit patterns."""
    v7x, v7y = arb(3, 0.3), arb(3, 1.1)
    v7x[1] = -0.0
    v7y[0] = np.nan
    z9z = arb(2, 0.7)
    z9z[1] = -0.0
    d9 = arb(6, 0.9)
    d9[0] = np.nan
    t10 = arb(6, 1.3)
    t10[3], t10[4] = np.nan, -0.0
    sy6 = arb(6, 2.3)
    sy6[1], sy6[2] = -0.0, np.nan
    dm21 = arb(21, 3.3)
    tyre_s = arb(18, 3.7)
    tyre_s[0], tyre_s[1] = np.nan, -0.0
    return [
        (1, 1, 2.718281828459045),
        (1, 2, float(bits(BIT_VALUES[2])[0])),
        (1, 3, float(bits(BIT_VALUES[3])[0])),
        (1, 4, float(bits(BIT_VALUES[4])[0])),
        (2, 10, bits(*BIT_ARRAY)),
        (2, 11, np.array([])),
        (6, 20, 37),
        (6, 21, 255),
        (7, 30, (v7x, v7y)),
        (8, 40, b'ChassisSim \x00\xff'),
        (8, 41, b''),
        (9, 50, (arb(3, 0.5), z9z, d9)),  # nx 3, nz 2
        (10, 60, (4, t10)),  # flag 4, 2 values per column
        (4, 70, arb(6, 1.7)),
        (4, 71, np.array([])),
        (11, 80, (0, 0.0173, 83113.7, arb(6, 2.1), sy6)),
        (11, 81, (1, -0.5137, 190517.3, arb(11, 2.5), arb(11, 2.9))),
        (12, 90, (1, arb(6, 3.0), dm21)),  # flag 1, n = 7
        (13, 100, (TYRE_DIMS, tyre_s, tyre_tables(4.1))),
    ]


print("[carfile] synthetic records: round trip")
RECS = sample_records()
FULL = RECS + [(END_TAG, END_TAG, None)]
DATA = encode(FULL)
DEC = decode_bytes(DATA)
ok("decode(encode(records)) gives equal values, in order", recs_equal(DEC, FULL))
ok("the end pair decodes as (4000, 4000, None)", DEC[-1] == (END_TAG, END_TAG, None))
ok("encode(decode(bytes)) is byte-equal", encode(DEC) == DATA)

DEC_BY_CID = {c: v for _, c, v in DEC}
ok("tag 1 keeps the bits of -0.0, a NaN with payload and an sNaN",
   all(struct.pack('<d', DEC_BY_CID[c]) == struct.pack('<Q', u) for c, u in BIT_VALUES.items()))
ok("tag 2 keeps the raw bits of its values", DEC_BY_CID[10].view('<u8').tolist() == BIT_ARRAY)
ok("tyre record keeps its dims and 13 table lengths",
   DEC_BY_CID[100][0] == TYRE_DIMS and [len(t) for t in DEC_BY_CID[100][2]] == TYRE_SIZES)

DATA_NO_END = encode(RECS)
DEC_NO_END = decode_bytes(DATA_NO_END)
ok("a file without the end pair decodes, with no end record",
   recs_equal(DEC_NO_END, RECS) and DEC_NO_END[-1][0] != END_TAG)
ok("a file without the end pair round-trips byte-equal", encode(DEC_NO_END) == DATA_NO_END)

print("[carfile] cuts")
bad_cuts, n_prefix = [], 0
for cut in range(len(DATA)):
    try:
        dec = decode_bytes(DATA[:cut])
    except ValueError:
        continue
    if len(dec) < len(FULL) and recs_equal(dec, FULL[:len(dec)]):
        n_prefix += 1
    else:
        bad_cuts.append(cut)
ok(f"every cut of the {len(DATA)} byte file: ValueError or a record prefix", not bad_cuts)
ok("the cuts that decode are the record boundaries (one per record)", n_prefix == len(RECS))

print("[carfile] decode errors")
ok("empty bytes raise ValueError", err(lambda: decode_bytes(b'')) is not None)
ok("3 bytes cut from the end raise ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(err(lambda: decode_bytes(DATA[:-3]))))
ok("unknown tag 99 raises at offset 0",
   (lambda m: m is not None and "at offset 0 " in m)(err(lambda: decode_bytes(rec(99, 1, bytes(8))))))
ok("tag 3 raises on decode", (lambda m: m is not None and "unknown" in m)(
    err(lambda: decode_bytes(rec(3, 1, bytes(16))))))
ok("tag 5 raises on decode", (lambda m: m is not None and "unknown" in m)(
    err(lambda: decode_bytes(rec(5, 1, bytes(16))))))
ok("negative count at offset 8 raises",
   (lambda m: m is not None and "offset 8" in m)(err(lambda: decode_bytes(rec(2, 1, struct.pack('<i', -1))))))
ok("spring flag 2 at offset 12 raises",
   (lambda m: m is not None and "offset 12" in m)(
       err(lambda: decode_bytes(rec(11, 1, struct.pack('<i', 0) + bytes([2]) + bytes(16))))))
ok(f"data after the end pair raises at offset {len(DATA)}",
   (lambda m: m is not None and f"offset {len(DATA)}." in m)(err(lambda: decode_bytes(DATA + bytes(4)))))

PREFIX = rec(1, 5, struct.pack('<d', 0.25))  # 16 bytes, so the bad record starts at offset 16
ok("bad count (10^6) at offset 24 raises",
   (lambda m: m is not None and "offset 24" in m)(
       err(lambda: decode_bytes(PREFIX + rec(2, 6, struct.pack('<i', 10**6) + bytes(8))))))
ok("bad count (-3) at offset 24 raises",
   (lambda m: m is not None and "offset 24" in m)(
       err(lambda: decode_bytes(PREFIX + rec(2, 6, struct.pack('<i', -3))))))
ok("oversized tag 9 table (nx, nz = 10^5) at offset 24 raises",
   (lambda m: m is not None and "offset 24" in m)(
       err(lambda: decode_bytes(PREFIX + rec(9, 7, struct.pack('<ii', 10**5, 10**5))))))
ok("oversized tag 13 tyre (dims 10^4) at offset 24 raises",
   (lambda m: m is not None and "offset 24" in m)(
       err(lambda: decode_bytes(PREFIX + rec(13, 8, struct.pack('<7i', 10**4, 2, 2, 2, 2, 2, 2))))))

print("[carfile] encode errors")
ok("encode([]) raises ValueError", err(lambda: encode([])) is not None)
ok("tag 3 and tag 5 raise on encode",
   err(lambda: encode([(3, 1, None)])) is not None and err(lambda: encode([(5, 1, None)])) is not None)
ok("tag 4 with values not a multiple of 3 raises", err(lambda: encode([(4, 1, np.arange(4.0))])) is not None)
REFUSED = {
    "end record with cid != 4000": [(END_TAG, 1, None)],
    "end record with a value": [(END_TAG, END_TAG, 1.0)],
    "end record not last": [(END_TAG, END_TAG, None), (1, 1, 2.0)],
    "spring flag 2": [(11, 1, (2, 0.1, 0.2, arb(3, 0.1), arb(3, 0.2)))],
    "tag 10 with 3 items": [(10, 1, (1, arb(3, 0.1), arb(3, 0.2)))],
    "tag 12 with 2 items": [(12, 1, (1, arb(6, 0.1)))],
    "tag 7 with 3 items": [(7, 1, (arb(2, 0.1), arb(2, 0.2), arb(2, 0.3)))],
    "tag 9 with 2 items": [(9, 1, (arb(2, 0.1), arb(2, 0.2)))],
    "tag 11 with 4 items": [(11, 1, (0, 0.1, 0.2, arb(3, 0.3)))],
    "tag 1 value None": [(1, 1, None)],
    "tag 1 value a string": [(1, 1, "abc")],
    "tag 2 values of strings": [(2, 1, ["a", "b"])],
    "tag 11 spring s1 None": [(11, 1, (0, None, 0.2, arb(3, 0.1), arb(3, 0.2)))],
    "tyre with 6 dims": [(13, 1, ((2, 3, 4, 5, 6, 7), arb(18, 0.1), tyre_tables(0.5)))],
    "tyre with 12 tables": [(13, 1, (TYRE_DIMS, arb(18, 0.1), tyre_tables(0.5)[:12]))],
}
for label, recs in REFUSED.items():
    m = err(lambda: encode(recs))
    ok(f"encode refuses: {label}", m is not None and m.startswith("The record 0 (tag ")
       and "is not valid." in m)

print("[carfile] write")
with tempfile.TemporaryDirectory() as tmp:
    new = os.path.join(tmp, "new.car")
    write(FULL, new)
    ok("write creates a new path that decodes equal", recs_equal(decode(new), FULL))
    ok("the written file holds encode(records)", read_bytes(new) == DATA)
    before = read_bytes(new)
    refused = False
    try:
        write(FULL, new)
    except FileExistsError:
        refused = True
    ok("write refuses an existing path (FileExistsError)", refused)
    ok("the refused write leaves the file unchanged", read_bytes(new) == before)
    no_end = os.path.join(tmp, "no_end.car")
    write(RECS, no_end)
    ok("write of a file without the end pair decodes equal", recs_equal(decode(no_end), RECS))


def vendor(cs):
    cs = Path(cs)
    parent = cs.parent
    models = parent / "Models"
    files = (sorted(glob.glob(os.path.join(cs, "*.car")))
             + sorted(glob.glob(os.path.join(cs, "help", "**", "*.car"), recursive=True))
             + sorted(glob.glob(os.path.join(models, "**", "*.car"), recursive=True))
             + [str(parent / "ChassisSim_v3_41def_carfile.car")])
    print("[carfile] vendor files")
    ok("31 shipped car files found", len(files) == 31)

    bad, n_end = [], 0
    for f in files:
        raw = read_bytes(f)
        recs = decode(f)
        if encode(recs) != raw:
            bad.append(os.path.basename(f))
        n_end += recs[-1][0] == END_TAG
    ok(f"gate 1: encode(decode(f)) == f for all {len(files)} files" + (f" (bad: {bad})" if bad else ""),
       not bad)
    ok("29 of the 31 files end with the end pair", n_end == 29)

    print("[carfile] vendor GP3 2013 Eastern Creek variants")
    gp3 = models / "GP3 2013" / "Eastern Creek, AU"
    DEF = "def_GP3 2013.car"
    R17 = "rideheight17mm_GP3 2013.car"
    R20 = "rideheight20mm_GP3 2013.car"
    T0 = "rideheight17mm_toe0_GP3 2013.car"
    T66 = "rideheight17mm_toe0.66_GP3 2013.car"
    FD33 = "rideheight17mm_toe0.33_frontdamper20000_GP3 2013.car"
    FD66 = "rideheight17mm_toe0.66_frontdamper20000_GP3 2013.car"

    def load(name):
        return decode(str(gp3 / name))

    def val(name, cid):
        return [x for _, c, x in load(name) if c == cid][0]

    def differing(a, b):
        """Cids whose values differ beyond 1e-6 relative (or in bytes, ints, shapes) between two files."""
        A, B = load(a), load(b)
        ok(f"{a} and {b} share one record layout", [(t, c) for t, c, _ in A] == [(t, c) for t, c, _ in B])
        return {c for (_, c, x), (_, _, y) in zip(A, B) if not close(x, y)}

    ok("ride height: 17 vs 20 mm differ beyond 1e-6 relative in cids {117, 118} only",
       differing(R17, R20) == {117, 118})
    # The stored value is within 1e-3 of the number in the name. The file stores mm.
    ok("ride height cids 117, 118 read 17 (17mm) and 20 (20mm)",
       all(abs(val(f, c) - n) < 1e-3 for c in (117, 118) for f, n in ((R17, 17.0), (R20, 20.0))))
    ok("base 17mm vs def_GP3 2013: differ beyond 1e-6 relative in cids {117, 118} only",
       differing(R17, DEF) == {117, 118})

    ok("toe 0 and toe 0.66 vs base: differ beyond 1e-6 relative in cids {80, 81} only",
       differing(T0, R17) == {80, 81} and differing(T66, R17) == {80, 81})
    a390, b390 = val(T0, 390), val(R17, 390)
    ok("cid 390 differs from base but by under 1e-6 relative (not counted)",
       a390 != b390 and abs(a390 - b390) / max(abs(a390), abs(b390)) < 1e-6)

    def toe(name, cid):
        return float(val(name, cid)[1][0])  # tyre scalar 0 is the toe

    ok("tyre scalar 0 is the toe: base 0.33, toe0 0.0, toe0.66 0.66 (cids 80, 81)",
       all(abs(toe(n, c) - x) < 1e-6 for c in (80, 81)
           for n, x in ((R17, 0.33), (T0, 0.0), (T66, 0.66))))

    ok("front damper 0.33 vs base: differ beyond 1e-6 relative in the exact cids",
       differing(FD33, R17) == {1, 2, 3, 4, 31, 32, 811, 812, 813, 814, 815, 816, 819, 820, 821, 823,
                                825, 826, 827, 828, 829, 830, 831})
    ok("front damper 0.66 vs def_GP3 2013: differ beyond 1e-6 relative in the exact cids",
       differing(FD66, DEF) == {1, 2, 3, 4, 811, 812, 813, 814, 815, 816, 819, 820, 821, 823, 827, 828,
                                829, 830, 831})


if not os.environ.get("CHASSISSIM_DIR") or not os.path.isdir(os.environ["CHASSISSIM_DIR"]):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(os.environ["CHASSISSIM_DIR"])

print(f"[carfile] {PASS} checks passed")
