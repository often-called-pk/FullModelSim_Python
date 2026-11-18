"""ChassisSim v3.41 car-file codec of functions/carfile.py (ChassisSim phase 1).

The synthetic checks build records of every supported tag, round-trip them and pin the ValueErrors. They write only
into a temporary directory and run no solve. The vendor section runs only when CHASSISSIM_DIR names the ChassisSim
install folder. It reads the 31 shipped car files, checks the encode(decode(f)) == f gate and pins the GP3 2013
Eastern Creek variants. It writes nothing into the vendor folders.

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
    """Equal up to float32 resave noise: 1e-6 relative on floats, exact on bytes and ints."""
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


def sample_records():
    """One record of every supported tag, with NaN and -0.0 inside float arrays. Tyre dims (2, 2, 3, 2, 2, 2, 2)."""
    f = np.array
    tyre_dims = (2, 2, 3, 2, 2, 2, 2)
    # table sizes of tyre_dims, worked out by hand from the tyre layout (2*d3, 2*d4, 2*d4, g(d1, d2), ...)
    tyre_sizes = [6, 4, 4, 8, 8, 4, 4, 8, 6, 4, 4, 4, 4]
    tables = [np.arange(k, dtype=float) + 0.5 for k in tyre_sizes]
    tables[3][0] = np.nan
    tables[4][1] = -0.0
    return [
        (1, 1, 3.5),
        (1, 2, -0.0),
        (1, 3, float('nan')),
        (2, 10, f([1.0, -0.0, np.nan, 2.5])),
        (2, 11, f([])),
        (6, 20, 0),
        (6, 21, 255),
        (7, 30, (f([0.0, 1.0, -0.0]), f([np.nan, 2.0, 3.0]))),
        (8, 40, b'ChassisSim \x00\xff'),
        (8, 41, b''),
        (9, 50, (f([0.0, 1.0, 2.0]), f([0.5, -0.0]), f([np.nan, -0.0, 1.0, 2.0, 3.0, 4.0]))),  # nx 3, nz 2
        (10, 60, (4, f([1.0, 2.0, 3.0, np.nan, -0.0, 6.0]))),  # flag 4, n = 2
        (4, 70, f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),  # n = 2, no flag
        (4, 71, f([])),
        (11, 80, (0, 0.002, 1.4e5, f(np.linspace(-1, 1, 6)), f([0.0, -0.0, np.nan, 1.0, 2.0, 3.0]))),
        (11, 81, (1, -0.5, 2.0e5, f(np.linspace(0, 1, 11)), f(np.arange(11.0) * 0.1))),
        (12, 90, (1, f([0.0, 0.0, 1000.0, 0.0, -0.0, 1000.0]), f(np.arange(21.0) * 1.5))),  # n = 7
        (13, 100, (tyre_dims, f(np.arange(18.0) * 0.25 - 1.0), tables)),
    ]


print("[carfile] synthetic records: round trip")
RECS = sample_records()
ok("sample holds the tags 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13",
   {t for t, _, _ in RECS} == {1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13})
FULL = RECS + [(END_TAG, END_TAG, None)]
DATA = encode(FULL)
DEC = decode_bytes(DATA)
ok("decode(encode(records)) gives equal values, in order", recs_equal(DEC, FULL))
ok("the end pair decodes as (4000, 4000, None)", DEC[-1] == (END_TAG, END_TAG, None))
ok("encode(decode(bytes)) is byte-equal", encode(DEC) == DATA)

DATA_NO_END = encode(RECS)
DEC_NO_END = decode_bytes(DATA_NO_END)
ok("a file without the end pair decodes, with no end record",
   recs_equal(DEC_NO_END, RECS) and DEC_NO_END[-1][0] != END_TAG)
ok("a file without the end pair round-trips byte-equal", encode(DEC_NO_END) == DATA_NO_END)

print("[carfile] errors")
ok("truncated file (3 bytes cut) raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(err(lambda: decode_bytes(DATA[:-3]))))
ok("unknown tag 99 raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(err(lambda: decode_bytes(rec(99, 1) + bytes(8)))))
ok("tag 3 raises on decode", (lambda m: m is not None and "unknown" in m)(
    err(lambda: decode_bytes(rec(3, 1, bytes(16))))))
ok("tag 5 raises on decode", (lambda m: m is not None and "unknown" in m)(
    err(lambda: decode_bytes(rec(5, 1, bytes(16))))))
ok("tag 3 and tag 5 raise on encode",
   err(lambda: encode([(3, 1, None)])) is not None and err(lambda: encode([(5, 1, None)])) is not None)
ok("negative count raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(err(lambda: decode_bytes(rec(2, 1, struct.pack('<i', -1))))))
ok("count larger than the file raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(
       err(lambda: decode_bytes(rec(2, 1, struct.pack('<i', 1000) + bytes(8))))))
ok("spring flag 2 raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(
       err(lambda: decode_bytes(rec(11, 1, struct.pack('<i', 0) + bytes([2]) + bytes(16))))))
ok("data after the end pair raises ValueError naming an offset",
   (lambda m: m is not None and "offset" in m)(err(lambda: decode_bytes(DATA + bytes(4)))))
ok("encode: tag 7 with X and Y of different length raises",
   (lambda m: m is not None and "needs" in m)(
       err(lambda: encode([(7, 1, (np.zeros(2), np.zeros(3)))]))))
ok("encode: tag 13 with 12 tables raises",
   (lambda m: m is not None and "13 tables" in m)(
       err(lambda: encode([(13, 1, ((2, 2, 3, 2, 2, 2, 2), np.zeros(18), [np.zeros(1)] * 12))]))))
ok("encode: tag 4 with values not a multiple of 3 raises",
   err(lambda: encode([(4, 1, np.arange(4.0))])) is not None)

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
    R17 = "rideheight17mm_GP3 2013.car"
    R20 = "rideheight20mm_GP3 2013.car"
    T0 = "rideheight17mm_toe0_GP3 2013.car"
    T66 = "rideheight17mm_toe0.66_GP3 2013.car"
    FD = "rideheight17mm_toe0.33_frontdamper20000_GP3 2013.car"

    def load(name):
        return decode(str(gp3 / name))

    def val(name, cid):
        return [x for _, c, x in load(name) if c == cid][0]

    def differing(a, b):
        A, B = load(a), load(b)
        ok(f"{a} and {b} share one record layout", [(t, c) for t, c, _ in A] == [(t, c) for t, c, _ in B])
        return {c for (_, c, x), (_, _, y) in zip(A, B) if not close(x, y)}

    ok("ride height 17 vs 20 mm differ in cids {117, 118} only",
       differing(R17, R20) == {117, 118})
    # the file stores millimetres, the number in the name; float32 resave noise is about 1e-5
    ok("ride height cids 117, 118 read 17 (17mm) and 20 (20mm), in mm",
       all(abs(val(f, c) - n) < 1e-3 for c in (117, 118) for f, n in ((R17, 17.0), (R20, 20.0))))

    ok("toe 0 and toe 0.66 vs base differ in cids {80, 81} only",
       differing(T0, R17) == {80, 81} and differing(T66, R17) == {80, 81})

    def toe(name, cid):
        return float(val(name, cid)[1][0])  # tyre scalar 0 is the toe

    ok("tyre scalar 0 is the toe: base 0.33, toe0 0.0, toe0.66 0.66 (cids 80, 81)",
       all(abs(toe(n, c) - x) < 1e-6 for c in (80, 81)
           for n, x in ((R17, 0.33), (T0, 0.0), (T66, 0.66))))

    fd_diff = differing(FD, R17)
    tag8 = {c for t, c, _ in load(FD) if t == 8}
    ok("front damper vs base: cids 31 and 32 differ", {31, 32} <= fd_diff)
    ok("front damper vs base: other differing cids are springs 1-4 or tag-8 labels",
       fd_diff - {31, 32} <= {1, 2, 3, 4} | tag8)


if not os.environ.get("CHASSISSIM_DIR") or not os.path.isdir(os.environ["CHASSISSIM_DIR"]):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    vendor(os.environ["CHASSISSIM_DIR"])

print(f"[carfile] {PASS} checks passed")
