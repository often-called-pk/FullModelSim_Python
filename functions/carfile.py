"""Decode and encode ChassisSim v3.41 car files.

Source: ChassisSim v3.41 car files, an MFC CArchive record stream. A file has
no header. It holds records `[int32 tag][int32 cid][payload]`, little-endian,
byte-packed, numbers in float64. Most files end with the pair `4000, 4000`.
Two shipped files end without it, so the pair is a record of its own
`(END_TAG, END_CID, None)` that `decode` returns only when the file has it.

Record values:
  1  float
  2  float64 array
  6  int (one byte)
  7  (X, Y) float64 arrays
  8  bytes
  9  (X, Z, D) float64 arrays, D flat with nx*nz points
  10 (flag, a) a = 3*n float64 values, n is the count stored on disk
  4  a = 3*n float64 values (n is the count stored on disk)
  11 (flag, s1, s2, X, Y) spring
  12 (flag, s, a) damper: 6 scalars s, a = 3*n values
  13 (dims, s, tables) tyre: 7 dims, 18 scalars, 13 flat float64 tables
  4000 None
Tags 3 and 5 are not in any shipped file. Their layout is unknown, so
`decode` and `encode` raise ValueError for them.

`write` never overwrites a file: a car file is vendor data.
"""
import os
import struct

import numpy as np

END_TAG = END_CID = 4000
_F = np.dtype('<f8')


def _tyre_sizes(d):
    d1, d2, d3, d4, d5, d6, d7 = d
    g = lambda nx, nz: nx + nz + nx * nz
    return [2 * d3, 2 * d4, 2 * d4, g(d1, d2), g(d6, d7), 2 * d6, 2 * d4,
            g(d1, d2), 2 * d3, 2 * d6, 2 * d5, 2 * d5, 2 * d5]


class _Reader:
    def __init__(self, b):
        self.b, self.p = b, 0

    def need(self, n):
        if n < 0 or self.p + n > len(self.b):
            raise ValueError(f"The car file ends early at offset {self.p}.")

    def int(self, n=1):
        self.need(4 * n)
        v = struct.unpack_from(f'<{n}i', self.b, self.p)
        self.p += 4 * n
        return v if n > 1 else v[0]

    def byte(self):
        self.need(1)
        self.p += 1
        return self.b[self.p - 1]

    def f64(self, n):
        self.need(8 * n)
        a = np.frombuffer(self.b, _F, n, self.p).copy()
        self.p += 8 * n
        return a

    def count(self, k=1):
        at = self.p
        n = self.int()
        if n < 0 or self.p + 8 * k * n > len(self.b):
            raise ValueError(f"The count {n} at offset {at} does not fit the file.")
        return n


def decode(path):
    """Read a car file. Return a list of (tag, cid, value) in file order."""
    with open(path, 'rb') as fh:
        b = fh.read()  # ponytail: car files are under 1 MB, stream if that changes
    return decode_bytes(b)


def decode_bytes(b):
    r, out = _Reader(b), []
    while r.p < len(b):
        at = r.p
        tag, cid = r.int(2)
        if tag == END_TAG and cid == END_CID:
            out.append((tag, cid, None))
            if r.p != len(b):
                raise ValueError(f"The file has data after the end marker at offset {r.p}.")
            break
        if tag == 1:
            v = float(r.f64(1)[0])
        elif tag == 2:
            v = r.f64(r.count())
        elif tag == 6:
            v = r.byte()
        elif tag == 7:
            n = r.count(2)
            v = (r.f64(n), r.f64(n))
        elif tag == 8:
            n = r.count(1 / 8)
            r.need(n)
            v = bytes(b[r.p:r.p + n])
            r.p += n
        elif tag == 9:
            nx, nz = r.int(2)
            if nx < 0 or nz < 0 or r.p + 8 * (nx + nz + nx * nz) > len(b):
                raise ValueError(f"The table size {nx} x {nz} at offset {at + 8} does not fit the file.")
            v = (r.f64(nx), r.f64(nz), r.f64(nx * nz))
        elif tag == 10:
            n = r.count(3)
            v = (r.byte(), r.f64(3 * n))
        elif tag == 4:
            v = r.f64(3 * r.count(3))
        elif tag == 11:
            n = r.count(2)
            fl = r.byte()
            if fl not in (0, 1):
                raise ValueError(f"The spring flag {fl} at offset {r.p - 1} is not 0 or 1.")
            s = r.f64(2)
            v = (fl, s[0], s[1], r.f64(n), r.f64(n))
        elif tag == 12:
            n = r.count(3)
            fl = r.byte()
            v = (fl, r.f64(6), r.f64(3 * n))
        elif tag == 13:
            d = r.int(7)
            if min(d) < 0 or sum(_tyre_sizes(d)) * 8 > len(b):
                raise ValueError(f"The tyre dimensions {d} at offset {at + 8} do not fit the file.")
            s = r.f64(18)
            v = (d, s, [r.f64(k) for k in _tyre_sizes(d)])
        else:
            raise ValueError(f"The record tag {tag} (cid {cid}) at offset {at} is unknown.")
        out.append((tag, cid, v))
    return out


def _f(a, n=None):
    a = np.ascontiguousarray(a, _F).ravel()
    if n is not None and len(a) != n:
        raise ValueError(f"The record needs {n} values. It has {len(a)}.")
    return a.tobytes()


def _i(*v):
    return struct.pack(f'<{len(v)}i', *v)


def encode(records):
    """Return the bytes of a car file. Inverse of `decode`."""
    out = []
    for tag, cid, v in records:
        out.append(_i(tag, cid))
        if tag == END_TAG:
            pass
        elif tag == 1:
            out.append(_f(v, 1))
        elif tag == 2:
            a = np.asarray(v, _F).ravel()
            out += [_i(len(a)), _f(a)]
        elif tag == 6:
            out.append(bytes([v]))
        elif tag == 7:
            x, y = (np.asarray(a, _F).ravel() for a in v)
            out += [_i(len(x)), _f(x), _f(y, len(x))]
        elif tag == 8:
            out += [_i(len(v)), bytes(v)]
        elif tag == 9:
            x, z, d = (np.asarray(a, _F).ravel() for a in v)
            out += [_i(len(x), len(z)), _f(x), _f(z), _f(d, len(x) * len(z))]
        elif tag in (10, 4, 12):
            fl, *rest = v if tag != 4 else (None, v)
            a = np.asarray(rest[-1], _F).ravel()
            if len(a) % 3:
                raise ValueError(f"Record tag {tag}, cid {cid}: the number of values must be a multiple of 3.")
            out.append(_i(len(a) // 3))
            if tag != 4:
                out.append(bytes([fl]))
            if tag == 12:
                out.append(_f(rest[0], 6))
            out.append(_f(a))
        elif tag == 11:
            fl, s1, s2, x, y = v
            x = np.asarray(x, _F).ravel()
            out += [_i(len(x)), bytes([fl]), _f([s1, s2]), _f(x), _f(y, len(x))]
        elif tag == 13:
            d, s, tabs = v
            if len(tabs) != 13:
                raise ValueError(f"Tyre cid {cid}: the record needs 13 tables. It has {len(tabs)}.")
            out += [_i(*d), _f(s, 18)]
            out += [_f(t, k) for t, k in zip(tabs, _tyre_sizes(d))]
        else:
            raise ValueError(f"The record tag {tag} (cid {cid}) is unknown.")
    return b''.join(out)


def write(records, path):
    """Write a car file. Raise FileExistsError when the path exists."""
    data = encode(records)
    with open(path, 'xb') as fh:
        fh.write(data)
