"""Edit the scalar setup values of a ChassisSim .car file and save a new file.

The values are in the file units of ChassisSim (the units of the decoded records). No value is converted.
Only the records that the user edits change. All other records stay as read, so their bytes stay equal.
"""
import hashlib
import os
import tempfile
from collections import namedtuple

from functions import carfile, chassissim

Field = namedtuple("Field", "group label unit kind lo hi tip")

C4 = ("FR", "FL", "RR", "RL")
INF = float("inf")
NOT_READ_TIP = "The FullModelSim solve does not read this value. ChassisSim reads it."
CAMBER_TIP = "The file stores negative camber as a positive value."
LEFT_GAP_TIP = "ChassisSim reads a left gap of 0 as the right gap."


def _specs():
    """Return the field table: (cid, slot) -> Field."""
    t = {}

    def add(cids, slot, group, label, unit, kind="f", lo=-INF, hi=INF, names=None, tip=""):
        names = names or [""] * len(cids)
        for c, n in zip(cids, names):
            t[(c, slot)] = Field(group, (label + " " + n).strip(), unit, kind, lo, hi, tip)

    def one(cid, group, label, unit, kind="f", lo=-INF, hi=INF, tip=""):
        add([cid], None, group, label, unit, kind, lo, hi, tip=tip)

    def four(base, slot, group, label, unit, kind="f", lo=-INF, hi=INF, tip=""):
        add(range(base, base + 4), slot, group, label, unit, kind, lo, hi, C4, tip)

    M = "Mass and inertia"
    for cid, lab, unit in ((84, "Total mass", "kg"), (85, "Unsprung mass front", "kg"), (86, "Unsprung mass rear", "kg"),
                           (87, "Roll inertia Ix", "kg m^2"), (88, "Pitch inertia Iy", "kg m^2"),
                           (89, "Yaw inertia Iz", "kg m^2")):
        one(cid, M, lab, unit, "f", 0, 1e6)
    one(96, M, "Front mass share", "-", "f", 0, 1)
    G = "Geometry and ride height"
    for cid, lab in ((90, "Wheelbase"), (91, "Track front"), (92, "Track rear"), (93, "CG height")):
        one(cid, G, lab, "m", "f", 0, 10)
    four(117, None, G, "Ride height", "mm", "f", -100, 500)
    S = "Springs and bars"
    four(1, 2, S, "Spring rate", "N/m", "f", 0, 1e7)
    four(1, 1, S, "Spring preload", "m", "f", -1, 1)
    for cid, lab in ((5, "Third spring front"), (6, "Third spring rear")):
        add([cid], 2, S, lab + " rate", "N/m", "f", 0, 1e7)
        add([cid], 1, S, lab + " preload", "m", "f", -1, 1, tip=NOT_READ_TIP)
    add([19], 2, S, "Anti-roll bar front rate", "N/m", "f", 0, 1e8)
    add([21], 2, S, "Anti-roll bar rear rate", "N/m", "f", 0, 1e8)
    R = "Rubbers and gaps"
    four(13, 2, R, "Bump rubber rate", "N/m", "f", 0, 1e8)       # not read with the table flag at 1: CarEdit.tip
    four(13, 1, R, "Bump rubber preload", "m", "f", -1, 1, NOT_READ_TIP)
    add([17, 18], 2, R, "Third rubber rate", "N/m", "f", -1e8, 1e8, ("front", "rear"))
    add(range(7, 13), None, R, "Gap", "m", "f", 0, 1e3, (*C4, "third front", "third rear"))
    add(range(204, 210), None, R, "Gap flag", "0 or 1", "i", 0, 1, (*C4, "third front", "third rear"))
    add(range(657, 661), None, R, "Hysteresis flag", "0 or 1", "i", 0, 1, C4)
    B = "Brakes"
    one(107, B, "Brake force total", "N", "f", 0, 1e6)
    one(108, B, "Front brake bias", "-", "f", 0, 1)
    one(109, B, "Brake delay", "s", "f", 0, 10, NOT_READ_TIP)
    one(160, B, "Left brake share", "-", "f", 0, 1)
    E = "Engine"
    one(100, E, "Efficiency", "-", "f", 0, 1)
    one(102, E, "Idle speed", "rpm", "f", 0, 1e5)
    one(103, E, "Speed limit", "rpm", "f", 0, 1e5)
    one(104, E, "Shift cut delay", "s", "f", 0, 10)
    one(105, E, "Torque share during the shift cut", "-", "f", -1e-9, 1)
    one(106, E, "Shift cut ramp time", "s", "f", 0, 10)
    one(111, E, "Torque split", "-", "f", 0, 1)
    one(112, E, "Front differential type", "1 to 5", "i", 1, 5)
    one(113, E, "Rear differential type", "1 to 5", "i", 1, 5)
    A = "Aero"
    one(121, A, "Lift area CL x A", "m^2", "f", 0, 100)
    one(122, A, "Drag area CD x A", "m^2", "f", 0, 100)
    one(123, A, "Balance offset", "-", "f", -1, 1)
    one(125, A, "Air density", "kg/m^3", "f", 0.1, 5)
    one(336, A, "DRS on", "0 or 1", "i", 0, 1)
    one(337, A, "DRS lift area loss front", "m^2", "f", -100, 100)
    one(338, A, "DRS lift area loss rear", "m^2", "f", -100, 100)
    one(339, A, "DRS drag area loss", "m^2", "f", -100, 100)
    one(340, A, "DRS zone start", "m", "f", 0, 1e5)
    one(341, A, "DRS zone end", "m", "f", 0, 1e5)
    add(range(413, 416), None, A, "Speed correction on,", "0 or 1", "i", 0, 1, ("lift area", "drag area", "balance"))
    TY = "Tyres"
    for slot, lab, unit, lo, hi, tip in (
            (0, "Toe (out +)", "deg", -20, 20, ""), (1, "Camber (negative +)", "deg", -20, 20, CAMBER_TIP),
            (2, "Castor", "deg", -20, 30, ""), (4, "Radius", "m", 0.05, 2, ""), (5, "Pressure", "Pa", 1e3, 1e7, ""),
            (6, "Temperature", "K", 150, 600, ""), (7, "Rolling drag", "-", 0, 1, ""),
            (8, "Peak slip angle", "rad", 0.001, 1.5, ""), (9, "Peak slip ratio", "-", 0.001, 1.5, ""),
            (11, "Optimum camber", "deg", -20, 20, NOT_READ_TIP), (15, "Grip", "-", 0, 10, "")):
        four(80, slot, TY, lab, unit, "f", lo, hi, tip)
    H = "Hybrid"
    one(177, H, "Hybrid on", "0 or 1", "i", 0, 1)
    one(178, H, "Deploy mode", "0 or 1", "i", 0, 1)
    one(179, H, "Regen clamp", "W", "f", 0, 1e8)
    one(180, H, "Deploy scale", "W", "f", 0, 1e8)
    one(181, H, "Energy cap", "J", "f", 0, 1e10)
    one(182, H, "Regen scale", "-", "f", 0, 100)
    one(183, H, "Deploy scale after the map", "-", "f", 0, 100)
    one(836, H, "Free energy offset", "J", "f", -1e10, 1e10)
    one(464, H, "Deploy front share on", "0 or 1", "i", 0, 1)
    one(465, H, "Deploy front share", "-", "f", 0, 1)
    one(466, H, "Deploy map on", "0 or 1", "i", 0, 1)
    one(870, H, "Regen to front axle", "0 or 1", "i", 0, 1)
    one(871, H, "Regen front share", "-", "f", 0, 1)
    one(535, H, "Brake regen maps on", "0 or 1", "i", 0, 1)
    one(590, H, "Brake regen gate", "0 to 255", "i", 0, 255)
    one(973, H, "Brake regen blend", "0 or 1", "i", 0, 1)
    one(573, H, "Paddle regen on", "0 or 1", "i", 0, 1)
    one(589, H, "Paddle gate", "0 to 255", "i", 0, 255)
    N = "Environment"
    one(390, N, "Track temperature", "K", "f", 150, 450, NOT_READ_TIP)
    return t


FIELDS = _specs()
GROUPS = ("Mass and inertia", "Geometry and ride height", "Springs and bars", "Rubbers and gaps", "Brakes",
          "Engine", "Aero", "Tyres", "Hybrid", "Environment")
# Dampers have no field: the model reads their tables only, so a scalar of a damper record acts on nothing.


def _tag(key, f):
    """The record tag of a field: 6 for an int, 1 without a slot, 13 for a tyre, else 11 for a spring or rubber."""
    return 6 if f.kind == "i" else 1 if key[1] is None else 13 if 80 <= key[0] <= 83 else 11


def _slot_get(tag, v, slot):
    if tag == 11:
        return float(v[slot])
    if tag == 13:
        return float(v[1][slot])
    return v if tag == 6 else float(v)


def _slot_set(tag, v, slot, x):
    """Return a copy of the record value v with the slot set to x."""
    if tag == 11:
        return tuple(float(x) if i == slot else e for i, e in enumerate(v))
    if tag == 13:
        sc = v[1].copy()
        sc[slot] = x
        return (v[0], sc, v[2])
    return int(x) if tag == 6 else float(x)


def _read(by_cid, key):
    """Return the value that load_car reads for the field key from the records by_cid.
    A copied left record gives the value of its right sibling."""
    tag, v = by_cid[chassissim.left_copies(by_cid).get(key[0], key[0])]
    return _slot_get(tag, v, key[1])


def in_chassissim(path, cs):
    """Return True when path is inside the ChassisSim folder cs, or inside the Models folder beside it."""
    if not cs:
        return False
    norm = lambda p: os.path.normcase(os.path.realpath(p))
    path, cs = norm(path), norm(cs)
    for root in (cs, norm(os.path.join(os.path.dirname(cs), "Models"))):
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:      # another drive
            pass
    return False


class CarEdit:
    """The decoded records of a car file and the edits of the user. edits maps (cid, slot) to a value."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(self.path, "rb") as fh:
            b = fh.read()  # ponytail: car files are under 1 MB, stream if that changes
        self.records = carfile.decode_bytes(b)
        if carfile.encode(self.records) != b:
            raise ValueError("The file does not encode back to its own bytes, so a new file would differ from it "
                             "in places that we do not read. The GUI cannot edit the car.")
        self.base = os.path.splitext(os.path.basename(self.path))[0]
        self.by_cid = {c: (t, v) for t, c, v in self.records}
        self.copies = chassissim.left_copies(self.by_cid)       # left cid -> right cid that load_car reads
        self.edits = {}

    def _source(self, cid):
        """Return (tag, value) that load_car reads for cid, or None."""
        return self.by_cid.get(self.copies.get(cid, cid))

    def rows(self):
        """Return [(key, Field, file value)] of the fields that the file holds, in the order of FIELDS."""
        out = []
        for key, f in FIELDS.items():
            rec = self._source(key[0])
            if rec is not None and rec[0] == _tag(key, f):
                out.append((key, f, _slot_get(rec[0], rec[1], key[1])))
        return out

    def file_value(self, key):
        tag, v = self._source(key[0])
        return _slot_get(tag, v, key[1])

    def tip(self, key):
        """Return the note of a row for its tooltip, or ''.
        The solve does not read a rubber rate when the rubber uses its table (flag 1).
        The solve reads the castor of the front left tyre only. The front right one counts when the file copies it."""
        cid, slot = key
        if (13 <= cid <= 18 and slot == 2 and self._source(cid)[1][0] == 1) or (
                slot == 2 and cid in (82, 83)) or (key == (80, 2) and 81 not in self.copies):
            return NOT_READ_TIP
        return FIELDS[key].tip

    def set(self, key, x):
        """Set an edit. A value equal to the file value removes the edit."""
        fv = self.file_value(key)
        if x == fv or (x != x and fv != fv):     # a NaN of the file equals a NaN
            self.edits.pop(key, None)
        else:
            self.edits[key] = x

    def new_records(self):
        """Return the record list with the edits applied, slot by slot.
        A left record that the file copies from the right one keeps the value that its row shows.
        When that record, its right sibling or a gap flag of the pair has an edit, the left record comes first.
        It is a copy of the right record of the file, in place or after the right record. Then the edits apply."""
        recs = list(self.records)
        pos = {c: i for i, (_, c, _) in enumerate(recs)}
        edited = {c for c, _ in self.edits}
        for l, r in self.copies.items():
            d = 197 if 7 <= r <= 10 else -197 if 204 <= r <= 207 else 0     # a gap and its flag go together
            if edited & {l, r, l + d, r + d}:
                row = (self.by_cid[r][0], l, self.by_cid[r][1])
                if l in pos:
                    recs[pos[l]] = row
                else:
                    recs.insert(pos[r] + 1, row)
                    pos = {c: i for i, (_, c, _) in enumerate(recs)}
        for (cid, slot), x in self.edits.items():
            tag, _, v = recs[pos[cid]]
            recs[pos[cid]] = (tag, cid, _slot_set(tag, v, slot, x))
        return recs

    def data(self):
        return carfile.encode(self.new_records())

    def validate(self, chassissim_dir=None):
        """Return the error text of the edited car, or None when the car is valid.
        chassissim_dir is the ChassisSim folder of the run. load_car reads the atmospheric factor from it."""
        for key, x in self.edits.items():
            f = FIELDS[key]
            if not f.lo <= x <= f.hi:
                return f"{f.label}: {x:g} is outside the range {f.lo:g} to {f.hi:g}."
        data = self.data()
        new = {c: (t, v) for t, c, v in carfile.decode_bytes(data)}
        for key, f, val in self.rows():         # each row must show the value that the new file gives the model
            got, want = _read(new, key), self.edits.get(key, val)
            if got != want and (got == got or want == want):        # a NaN of the file equals itself here
                return f"{f.label}: " + (LEFT_GAP_TIP if key[0] in (8, 10) else
                                         "ChassisSim reads a left record that is all zero as the right record.")
        fd, tmp = tempfile.mkstemp(suffix=".car", prefix="fms_car_check_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            cs = chassissim.load_car(tmp, chassissim_dir)
            chassissim.check_car(cs)
            chassissim.power_refusals(cs)
            chassissim.aero_refusals(cs)
        except Exception as exc:
            return str(exc).replace(tmp, self.base + " (edited)")
        finally:
            os.remove(tmp)
        return None

    def save(self, path, chassissim_dir=None):
        """Write the edited car to a new file. Raise ValueError for an invalid car, FileExistsError for a file."""
        err = self.validate(chassissim_dir)
        if err:
            raise ValueError(err)
        carfile.write(self.new_records(), path)

    def materialize(self, folder):
        """Return the path to run. Without edits it is the source file.
        Else the path is <folder>/<base>_<sha8>.car. A file with equal bytes stays, and a new file gets the edited car."""
        if not self.edits:
            return self.path
        data = self.data()
        path = os.path.join(folder, f"{self.base}_{hashlib.sha256(data).hexdigest()[:8]}.car")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                if fh.read() != data:
                    raise ValueError(f"The file {path} exists with other content.")
            return path
        os.makedirs(folder, exist_ok=True)
        with open(path, "xb") as fh:        # mode x never overwrites a file
            fh.write(data)
        return path
