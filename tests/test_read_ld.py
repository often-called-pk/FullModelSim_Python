"""MoTeC .ld reader of functions/telemetry.py: read_ld, LD_MAP and the channel records (ChassisSim phase 1, task 2).

The synthetic checks pack a small .ld file (header, linked channel records, data blocks) into a temporary directory
and read it back. They need numpy only and run no solve. The vendor section runs only when CHASSISSIM_DIR names the
ChassisSim install folder. It reads every .ld log under the Models folder next to it, checks the GP3 lap against its
.ldx Fastest Time, and checks ay against the curvature of Eastern Creek. It writes nothing into the repo.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_read_ld.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_read_ld.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import os
import pathlib
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import functions.telemetry as P
from functions.chassissim import load_circuit

G = 9.80665   # expected standard gravity, pinned against telemetry.G below
HDR = 0x100   # header bytes; the channel records follow it
REC = 124     # bytes per channel record


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def chan(name, unit, raw, cls=(5, 4), dec=2, rate=50, shift=0, mul=1, scale=1):
    """A channel of the synthetic file. raw holds the stored integers; cls (class, size) sets the sample type."""
    dtype = {(5, 4): "<i4", (3, 2): "<i2"}.get(cls, "<i4")
    return dict(name=name, unit=unit, raw=np.asarray(raw), cls=cls, dtype=dtype, dec=dec, rate=rate,
                shift=shift, mul=mul, scale=scale)


def write_ld(path, chans, magic=0x40):
    """Pack channels into a .ld file with the layout of map/ld_reader.md sections 1-2."""
    n = len(chans)
    rec = [HDR + REC * i for i in range(n)]
    blob = bytearray(HDR + REC * n)
    struct.pack_into("<I", blob, 0, magic)
    struct.pack_into("<I", blob, 0x08, rec[0])
    struct.pack_into("<H", blob, 0x56, n)
    at = len(blob)
    for i, c in enumerate(chans):
        raw = c["raw"].astype(c["dtype"])
        struct.pack_into("<IIIIHHHHhhhh", blob, rec[i],
                         rec[i - 1] if i else 0, rec[i + 1] if i + 1 < n else 0, at, raw.size,
                         0x10 + i, c["cls"][0], c["cls"][1], c["rate"],
                         c["shift"], c["mul"], c["scale"], c["dec"])
        for off, text in ((32, c["name"]), (72, c["unit"])):
            blob[rec[i] + off:rec[i] + off + len(text)] = text.encode("latin1")
        blob += raw.tobytes()
        at += raw.nbytes
    pathlib.Path(path).write_bytes(bytes(blob))


# Synthetic run: 1 s of lead-in (tail of the previous lap, distance 950 to 999 m), one lap (0 to 1000 m, 40 s),
# a 4 s tail (0 to 30 m). The lead-in must end near the track length, or the wrap is not seen (the lap rule).
N_A, N_L, N_T, N_R = 50, 2000, 200, 1500
N = N_A + N_L + N_T
T = np.arange(N) / 50.0
LAP_EXPECT = np.r_[np.zeros(N_A, int), np.ones(N_L, int), np.full(N_T, 2)]
DIST_RAW = np.round(np.r_[950.0 + np.arange(N_A), np.linspace(0, 1000, N_L, endpoint=False),
                          np.linspace(0, 30, N_T, endpoint=False)] * 100).astype("<i4")
AY_RAW = np.round(100 * 0.8 * np.sin(2 * np.pi * T / 6)).astype("<i4")
STEER_RAW = np.round(100 * 5 * np.sin(2 * np.pi * T / 8)).astype("<i4")
YAW_RAW = np.round(100 * 20 * np.cos(2 * np.pi * T / 8)).astype("<i4")
RPM_RAW = (4000 + 5 * np.arange(N_R)).astype("<i2")


def base_chans():
    return [chan("Lap Distance", "m", DIST_RAW), chan("G Force - Lateral", "G", AY_RAW),
            chan("Steered Angle", "deg", STEER_RAW), chan("Yaw rate", "deg/s", YAW_RAW),
            chan("Engine RPM", "rpm", RPM_RAW, cls=(3, 2), dec=0)]


tmp_dir = tempfile.TemporaryDirectory()
tmp = tmp_dir.name
ld = os.path.join(tmp, "synth.ld")
write_ld(ld, base_chans())

print("[read_ld] synthetic .ld, default map")
ok("G is the standard gravity 9.80665", P.G == 9.80665)
with open(ld, "rb") as fh:
    recs = {c["name"]: c["unit"] for c in P._ld_channels(fh)}
ok("channel records read the names and units", recs == {"Lap Distance": "m", "G Force - Lateral": "G",
                                                       "Steered Angle": "deg", "Yaw rate": "deg/s",
                                                       "Engine RPM": "rpm"})
out = P.read_ld(ld)
ok("default map keeps only the channels the file holds", set(out) == {"s", "ay", "delta", "r", "om_gear",
                                                                      "t", "lap"})
ok("absent channels (vx, ax, xs_*, throttle) are not keys", all(k not in out for k in
                                                                ("vx", "ax", "xs_fl", "throttle")))
ok("t = arange(n) / 50 exactly", np.array_equal(out["t"], np.arange(N) / 50))
ok("s = raw / 100 (dec 2)", np.allclose(out["s"], DIST_RAW / 100, rtol=0, atol=1e-9))
ok("lap 0 lead-in, 1 the lap, 2 the tail", np.array_equal(out["lap"], LAP_EXPECT))
ok("ay = +G * raw / 100 (left positive, factor +G)",
   np.allclose(out["ay"], G * AY_RAW / 100, rtol=0, atol=1e-9))
ok("delta = -pi/180 * raw / 100 (sign flipped)",
   np.allclose(out["delta"], -np.pi / 180 * STEER_RAW / 100, rtol=0, atol=1e-12))
ok("r = -pi/180 * raw / 100 (sign flipped)",
   np.allclose(out["r"], -np.pi / 180 * YAW_RAW / 100, rtol=0, atol=1e-12))
ok("int16 Engine RPM (other count) is np.interp-ed onto t",
   out["om_gear"].size == N and np.allclose(out["om_gear"], np.interp(T, np.arange(N_R) / 50.0,
                                                                     np.pi / 30 * RPM_RAW), rtol=0, atol=1e-9))

print("[read_ld] map forms")
out_tuple = P.read_ld(ld, {"delta": (("Nope", "Steered Angle"), "deg", -np.pi / 180)})
ok("a tuple of names takes the first name present", np.allclose(out_tuple["delta"], out["delta"], rtol=0,
                                                              atol=1e-12))
ok("a tuple with no name present raises ValueError",
   raises(lambda: P.read_ld(ld, {"x": (("Nope", "Also nope"), "deg", 1.0)})))
ok("a user map naming an absent channel raises ValueError", raises(lambda: P.read_ld(ld, {"x": "Nope"})))
ok("a unit mismatch raises ValueError",
   raises(lambda: P.read_ld(ld, {"ay": ("G Force - Lateral", "m/s^2", G)})))
out_plain = P.read_ld(ld, {"ay": "G Force - Lateral"})
ok("plain {key: name} map gives factor 1 (ay = raw / 100)",
   np.allclose(out_plain["ay"], AY_RAW / 100, rtol=0, atol=1e-9) and set(out_plain) == {"ay", "t"})

print("[read_ld] refusals")
bad = os.path.join(tmp, "bad_magic.ld")
write_ld(bad, base_chans(), magic=0x41)
ok("bad magic raises ValueError", raises(lambda: P.read_ld(bad)))

shifted = base_chans()
shifted[2] = chan("Steered Angle", "deg", STEER_RAW, shift=1)
sh = os.path.join(tmp, "shift.ld")
write_ld(sh, shifted)
ok("shift != 0 raises ValueError", raises(lambda: P.read_ld(sh)))

unknown = base_chans()
unknown[3] = chan("Yaw rate", "deg/s", YAW_RAW, cls=(7, 4))
uk = os.path.join(tmp, "class.ld")
write_ld(uk, unknown)
ok("unknown class raises ValueError", raises(lambda: P.read_ld(uk)))

tmp_dir.cleanup()
print("[read_ld] all synthetic checks passed")

print("[read_ld] vendor section")
vendor = os.environ.get("CHASSISSIM_DIR")
if not vendor or not os.path.isdir(vendor):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    models = pathlib.Path(vendor).parent / "Models"
    logs = sorted(models.rglob("*.ld"))
    ok(f"found {len(logs)} .ld logs under Models", len(logs) > 0)
    for p in logs:
        v = P.read_ld(str(p))
        ok(f"default map reads {p.name}", all(k in v for k in ("s", "vx", "ay", "t", "lap")))

    gp3 = models / "GP3 2013" / "Eastern Creek, AU" / "gp3_baseline_easterncreek.ld"
    ldx_root = ET.parse(gp3.with_suffix(".ldx")).getroot()
    fastest = next(e.get("Value") for e in ldx_root.iter("String") if e.get("Id") == "Fastest Time")
    mins, secs = fastest.split(":")
    t_ldx = 60 * int(mins) + float(secs)
    gp = P.read_ld(str(gp3))
    lap1 = gp["lap"] == 1
    dur = lap1.sum() / 50.0
    ok(f"GP3 lap 1 = {dur:.3f} s within 0.05 s of Fastest Time {fastest}", abs(dur - t_ldx) <= 0.05)

    tr = load_circuit("Eastern Creek", vendor)
    k = np.interp(gp["s"][lap1], tr.s, tr.k)
    corr = np.corrcoef(gp["ay"][lap1], gp["vx"][lap1] ** 2 * k)[0, 1]
    ok(f"corr(ay, vx^2 k) on GP3 lap 1 = {corr:.4f} > 0.99", corr > 0.99)
