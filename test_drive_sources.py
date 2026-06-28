"""Casadi-free unit tests for the drive-source abstraction."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from types import SimpleNamespace
import numpy as np

from functions.drive_sources import (
    DriveSource, SplitPolicy, node_wheels, aero_keys, control_keys,
    _topology, build_topology)

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

# fake pt/vp with the fields build_topology reads
pt = SimpleNamespace(Pmax=450000.0, Tmax=429.0, OMmax=25000 * np.pi / 30,
                     Vmax=380 / 3.6, eff=0.9)
vp = SimpleNamespace(gear=(25000 * np.pi / 30) * 0.35 / (380 / 3.6),
                     Tdist=0.33, brkB=0.65)

print("node_wheels / aero_keys")
ok("node all -> 4 wheels", node_wheels("all") == ["fl", "fr", "rl", "rr"])
ok("node rear -> rl,rr", node_wheels("rear") == ["rl", "rr"])
ok("node fl -> fl", node_wheels("fl") == ["fl"])
ok("aero 0 -> []", aero_keys(0) == [])
ok("aero 2 -> FW,RW", aero_keys(2) == ["FW", "RW"])
ok("aero 3 -> FW,FW,RW,TW", aero_keys(3) == ["FW", "FW", "RW", "TW"])

print("_topology")
ok("single", _topology(0, 0, 0) == "single")
ok("single+atd", _topology(0, 1, 0) == "single_atd")
ok("four_motor", _topology(1, 0, 0) == "four_motor")
ok("hybrid overrides", _topology(1, 1, 1) == "hybrid")

print("build_topology + control_keys back-compat")
s, sp = build_topology("single", pt, vp)
ok("single one source node all", len(s) == 1 and s[0].node == "all" and s[0].ctrl_key == "T_motor")
ok("single fixed split", len(sp) == 1 and sp[0].kind == "fixed" and sp[0].keys == [])
ok("single Static keys", control_keys(s, sp, 0) == ["T_motor", "T_brake", "delta"])
ok("single AALB keys", control_keys(s, sp, 3) ==
   ["T_motor", "T_brake", "FW", "FW", "RW", "TW", "delta"])

s_atd, sp_atd = build_topology("single_atd", pt, vp)
ok("single+ATD keys", control_keys(s_atd, sp_atd, 0) ==
   ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"])

s4, sp4 = build_topology("four_motor", pt, vp)
ok("four_motor 4 sources", [x.ctrl_key for x in s4] ==
   ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr"])
ok("four_motor no splits", sp4 == [])
ok("four_motor Static keys", control_keys(s4, sp4, 0) ==
   ["T_motor_fl", "T_motor_fr", "T_motor_rl", "T_motor_rr", "T_brake", "delta"])

sh, sph = build_topology("hybrid", pt, vp, ice_gear=1.0)
ok("hybrid source ctrl_keys", [x.ctrl_key for x in sh] ==
   ["T_motor_fl", "T_motor_fr", "T_motor_r", "T_ice_r"])
ok("hybrid has one ice", sum(1 for x in sh if x.type == "ice") == 1)
ok("hybrid tv split on rear", len(sph) == 1 and sph[0].kind == "tv" and sph[0].keys == ["split_R"])
ok("hybrid Static keys", control_keys(sh, sph, 0) ==
   ["T_motor_fl", "T_motor_fr", "T_motor_r", "T_ice_r", "T_brake", "split_R", "delta"])
ok("hybrid front emotor gear 6.0", sh[0].gear == 6.0)
ok("hybrid ice gear = ice_gear*3.5", sh[3].gear == 1.0 * 3.5)
ok("hybrid rear emotor gear = 1.5*ice_gear*3.5", sh[2].gear == 1.5 * 1.0 * 3.5)
ok("hybrid front emotor 150kW/143Nm", sh[0].Pmax == 150e3 and sh[0].Tmax == 143.0)

print("\nALL DRIVE-SOURCE TESTS PASSED")
