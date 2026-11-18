"""Decode and encode ChassisSim v3.41 car files.

Source: ChassisSim v3.41 car files, an MFC CArchive record stream. The layout below was measured on
the 31 car files that ChassisSim v3.41 ships: each one decodes and encodes back to the same bytes.

A file has no header. It holds records `[int32 tag][int32 cid][payload]`. All numbers are little-endian
and byte-packed: there is no alignment padding, so a float64 value can start at any offset. Most files
end with the pair `4000, 4000`. Two shipped files end without it, so the pair is a record of its own,
`(END_TAG, END_CID, None)`, and `decode` returns it only when the file has it. A file that is cut at a
record boundary therefore reads as a shorter car. A caller that needs a whole file checks for the end
record.

Payload of each tag, then the value that `decode` returns (n and the dims are int32, f64 is float64):
  1   f64                                               float
  2   n, n f64                                          float64 array
  4   n, 3n f64                                         float64 array with 3n values
  6   1 byte                                            int
  7   n, n f64 X, n f64 Y                               (X, Y)
  8   n, n bytes                                        bytes
  9   nx, nz, nx f64 X, nz f64 Z, nx*nz f64 data        (X, Z, data)
  10  n, 1 byte flag, 3n f64                            (flag, values)
  11  n, 1 byte flag (0 or 1), 2 f64, n f64 X, n f64 Y  (flag, s1, s2, X, Y), a spring
  12  n, 1 byte flag, 6 f64, 3n f64                     (flag, 6 scalars, values), a damper
  13  7 dims d1..d7, 18 f64, 13 tables                  (dims, 18 scalars, 13 tables), a tyre
The 13 tyre tables in file order: 2D(d3), 2D(d4), 2D(d4), 3D(d1, d2), 3D(d6, d7), 2D(d6), 2D(d4),
3D(d1, d2), 2D(d3), 2D(d6), 2D(d5), 2D(d5), 2D(d5). A 2D(k) table holds 2k f64 values and a 3D(a, b)
table holds a + b + a*b f64 values. The tyre header stores the dims once.
Tags 3 and 5 occur in no shipped file. Their layout is unknown, so `decode` and `encode` raise
ValueError for them.

`write` never overwrites a file: a car file is vendor data.
"""
import struct

import numpy as np

END_TAG = END_CID = 4000
_F = np.dtype('<f8')


def _tyre_sizes(d):
    """Return the f64 count of each of the 13 tyre tables for the dims d1..d7."""
    d1, d2, d3, d4, d5, d6, d7 = d

    def g(nx, nz):
        return nx + nz + nx * nz
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

    def count(self, size):
        """Read a count n. size is the byte size of one item, so that n items must fit the file."""
        at = self.p
        n = self.int()
        if n < 0 or self.p + size * n > len(self.b):
            raise ValueError(f"The count {n} at offset {at} does not fit the file.")
        return n


def decode(path):
    """Read a car file. Return a list of (tag, cid, value) in file order."""
    with open(path, 'rb') as fh:
        b = fh.read()  # ponytail: car files are under 1 MB, stream if that changes
    return decode_bytes(b)


def decode_bytes(b):
    """Decode the bytes of a car file. Return a list of (tag, cid, value) in file order."""
    if not b:
        raise ValueError("The car file is empty.")
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
            v = r.f64(r.count(8))
        elif tag == 4:
            v = r.f64(3 * r.count(24))
        elif tag == 6:
            v = r.byte()
        elif tag == 7:
            n = r.count(16)
            v = (r.f64(n), r.f64(n))
        elif tag == 8:
            n = r.count(1)
            v = bytes(b[r.p:r.p + n])
            r.p += n
        elif tag == 9:
            nx, nz = r.int(2)
            if nx < 0 or nz < 0 or r.p + 8 * (nx + nz + nx * nz) > len(b):
                raise ValueError(f"The table size {nx} x {nz} at offset {at + 8} does not fit the file.")
            v = (r.f64(nx), r.f64(nz), r.f64(nx * nz))
        elif tag == 10:
            n = r.count(24)
            v = (r.byte(), r.f64(3 * n))
        elif tag == 11:
            n = r.count(16)
            fl = r.byte()
            if fl not in (0, 1):
                raise ValueError(f"The spring flag {fl} at offset {r.p - 1} is not 0 or 1.")
            s = r.f64(2)
            v = (fl, s[0], s[1], r.f64(n), r.f64(n))
        elif tag == 12:
            n = r.count(24)
            v = (r.byte(), r.f64(6), r.f64(3 * n))
        elif tag == 13:
            d = r.int(7)
            if min(d) < 0 or r.p + 8 * (18 + sum(_tyre_sizes(d))) > len(b):
                raise ValueError(f"The tyre dimensions {d} at offset {at + 8} do not fit the file.")
            v = (d, r.f64(18), [r.f64(k) for k in _tyre_sizes(d)])
        else:
            raise ValueError(f"The record tag {tag} (cid {cid}) at offset {at} is unknown.")
        out.append((tag, cid, v))
    return out


def _f(a, n=None):
    """Return the float64 bytes of a. n is the value count that the record needs."""
    a = np.asarray(a)
    if a.dtype.kind not in 'biuf':
        raise ValueError(f"The values must be numbers. Their type is {a.dtype}.")
    a = np.ascontiguousarray(a, _F).ravel()
    if n is not None and len(a) != n:
        raise ValueError(f"The record needs {n} values. It has {len(a)}.")
    return a.tobytes()


def _i(*v):
    return struct.pack(f'<{len(v)}i', *v)


def _table3(a):
    """Return the count bytes and the value bytes of a table with 3 columns."""
    b = _f(a)
    if len(b) % 24:
        raise ValueError(f"The number of values must be a multiple of 3. It is {len(b) // 8}.")
    return _i(len(b) // 24), b


def _payload(tag, cid, v, last):
    """Return the payload bytes of one record. last is True for the last record of the car."""
    if tag == END_TAG:
        if cid != END_CID or v is not None or not last:
            raise ValueError("The end record must be (4000, 4000, None), and it must be the last record.")
        return b''
    if tag == 1:
        return _f(v, 1)
    if tag == 2:
        b = _f(v)
        return _i(len(b) // 8) + b
    if tag == 4:
        return b''.join(_table3(v))
    if tag == 6:
        return bytes([v])
    if tag == 7:
        x, y = v
        bx = _f(x)
        return _i(len(bx) // 8) + bx + _f(y, len(bx) // 8)
    if tag == 8:
        return _i(len(v)) + bytes(v)
    if tag == 9:
        x, z, d = v
        bx, bz = _f(x), _f(z)
        return _i(len(bx) // 8, len(bz) // 8) + bx + bz + _f(d, (len(bx) // 8) * (len(bz) // 8))
    if tag == 10:
        fl, a = v
        n, b = _table3(a)
        return n + bytes([fl]) + b
    if tag == 11:
        fl, s1, s2, x, y = v
        if fl not in (0, 1):
            raise ValueError(f"The spring flag {fl} is not 0 or 1.")
        bx = _f(x)
        return _i(len(bx) // 8) + bytes([fl]) + _f([s1, s2]) + bx + _f(y, len(bx) // 8)
    if tag == 12:
        fl, s, a = v
        n, b = _table3(a)
        return n + bytes([fl]) + _f(s, 6) + b
    if tag == 13:
        d, s, tabs = v
        if len(d) != 7 or len(tabs) != 13:
            raise ValueError(f"A tyre needs 7 dims and 13 tables. It has {len(d)} and {len(tabs)}.")
        return _i(*d) + _f(s, 18) + b''.join(_f(t, k) for t, k in zip(tabs, _tyre_sizes(d)))
    raise ValueError(f"The record tag {tag} is unknown.")


def encode(records):
    """Return the bytes of a car file. Inverse of `decode`."""
    if not records:
        raise ValueError("A car needs at least one record.")
    out = []
    for k, (tag, cid, v) in enumerate(records):
        try:
            out.append(_i(tag, cid) + _payload(tag, cid, v, k == len(records) - 1))
        except (ValueError, TypeError) as e:
            raise ValueError(f"The record {k} (tag {tag}, cid {cid}) is not valid. {e}") from None
    return b''.join(out)


def write(records, path):
    """Write a car file. Raise FileExistsError when the path exists."""
    data = encode(records)
    with open(path, 'xb') as fh:
        fh.write(data)
