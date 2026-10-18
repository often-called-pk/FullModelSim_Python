# Physics Fixes A1-A6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** opt-in `physics='corrected'` flag fixing inherited MATLAB physics A1-A6. Default `'matlab'` stays bit-identical, parity gates green.
**Architecture:** one string flag, `userOpts(physics=...)` -> `ctx.physics`. vehParams / vehModel / ggv / ladder branch with python `if phys == "corrected":` (SX graph for 'matlab' untouched). No new vp primaries (GUI registry untouched). Corrected-only derived vp fields OK (never added under 'matlab', gate 1 compares the matlab vp).
**Tech:** Python 3.14, casadi 3.7.2 SX, numpy, IPOPT + MUMPS (Linux box, no Coin-HSL).
**Spec:** `docs/laptime_sim_improvements.md` section A (A1-A6).
**Roles:** Opus orchestrates + verifies + applies fixes from test results. Sonnet builds. Haiku writes tests. Fable only when Opus fix fails first pass.

## Global constraints
- Python: `/home/pk/.venvs/fullmodelsim/bin/python`, cwd = worktree root `.claude/worktrees/physics-fixes`.
- physics='matlab' (default): every existing test passes unchanged. Parity guards: `tests/test_vehmodel_matlab.py`, `tests/test_matlab_parity.py`. Do not edit their asserts (only exception: CG_p hit, Task 5).
- Values: `PHYSICS = ("matlab", "corrected")`, case-sensitive, anything else -> ValueError.
- Tests: plain scripts `tests/test_*.py`, `import _bootstrap` first line, top-level asserts, no pytest.
- No em dash anywhere (hooks block). ASCII comments. Ponytail + karpathy surgical diffs.
- No NLP solves in build/test tasks. One solve at a time (15 GB box). Validation solves = Task 7 only.
- Commit per task on branch `physics-fixes`. No push.

## Review focus
1. matlab default must stay bit-identical -> parity guards above, run after every task.
2. Seeds (`ladder.quasi_static_states`) under corrected must sit at corrected equilibrium -> Task 3 test 7.
3. Corrected results must not overwrite default files -> Task 2 stem test.
4. QSS screener must follow physics -> Task 3 test 6, Task 4 test 7.
5. Smooth stops: no NaN/inf for |xs| up to 0.3 m -> Task 4 test 4.

## Shared test recipe (all new tests)
```python
import _bootstrap
import warnings
import numpy as np
import casadi as ca
from functions.context import Ctx
from userOpts import userOpts
from vehModel import vehModel

def model(physics="matlab", **kw):          # kw -> userOpts (circuit, ATD, Electric_4Motors, vp_overrides, Steering, CamberGain)
    c = Ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        userOpts(c, **dict(dict(circuit="Sturn", physics=physics), **kw))
    vehModel(c)
    return c

def sym_time(c, names=()):
    """Physical symbols Xp (nx), Up (nu), K; dxdt = physical time derivative (nx); sigs = m attributes."""
    m = c.m23
    xs, us = ca.DM(np.asarray(m.x_s, float)), ca.DM(np.asarray(m.u_s, float))
    F = ca.Function("F", [m.x, m.u, m.pv], [m.dx, m.sf] + [getattr(m, n) for n in names])
    Xp, Up, K = ca.SX.sym("Xp", m.nx), ca.SX.sym("Up", m.nu), ca.SX.sym("K")
    o = F(Xp / xs, Up / us, K)
    return Xp, Up, K, o[0] * xs / o[1], list(o[2:])

def static_state(c, vx):
    vp = c.vp
    X = np.zeros(c.m23.nx)
    X[0] = vx
    X[5:9] = vx / vp.Rw
    X[19:23] = [vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr]
    return X
# state order nx=23: 0 vx, 1 vy, 2 r, 3 n, 4 eps, 5-8 Om_fl/fr/rl/rr, 9 zs, 10 zsdot, 11 theta,
# 12 thetadot, 13 phi, 14 phidot, 15-18 wu_fl..rr, 19-22 zt_fl..rr. theta > 0 = nose down,
# phi > 0 lifts left side (fl, rl). xs > 0 = compression. f_lift_i > 0 = downforce.
# inputs (Static, ATD On, EM4 Off, nu=7): T_motor, T_brake, ATD_FL, ATD_FR, ATD_RL, ATD_RR, delta.
# coast input: zeros except ATD_* = 0.25.
```

---

### Task 1: Setup + baseline (Opus)
- [x] venv `/home/pk/.venvs/fullmodelsim` (casadi 3.7.2, numpy, scipy, plotly, PySide6).
- [x] worktree `.claude/worktrees/physics-fixes`, branch `physics-fixes` from `worktree-speed-phase1` (d54864d).
- [x] `git mv circuits Circuits` (code + MATLAB use `Circuits/`, Linux is case-sensitive).
- [ ] baseline: `python /tmp/physfix/run_tests.py <worktree> /tmp/physfix/baseline` -> record pass/fail, fix Linux-only failures.
- [ ] commit: rename + plan + spec copy.

### Task 2: physics flag plumbing (Sonnet build, Haiku test)
**Files:** `userOpts.py`, `functions/importfile.py` (`result_stem`), `MLTP.py`, `MLTP_screen.py`, `MLTP_paramOptim.py`, `app/results.py`. Test: `tests/test_physics_flag.py`.
**Produces:**
- `userOpts(ctx, ..., physics="matlab")`; module const `PHYSICS`; sets `ctx.physics` BEFORE Powertrain/vehParams; other modules read `getattr(ctx, "physics", "matlab")`.
- `result_stem(circuit, cfg, tyre_set="MF205", mesh_requested="auto", physics="matlab")` -> `<circuit>_<cfg>[_<tyre_set>][_physCorrected][_mesh<Mesh>]`. Every caller passes physics.
- `data["physics"]` in MLTP, MLTP_screen, optimise_design results.
- MLTP/MLTP_initial/MLTP_screen/optimise_design forward `physics` via existing `**useropts_kwargs` (check MLTP's inline MLTP_initial call forwards it).
- [ ] tests: default `c.physics == "matlab"`; `"corrected"` kept; `"Corrected"`, `"foo"` -> ValueError. `result_stem("BCN","Static_ATDOn_EM4Off") == "BCN_Static_ATDOn_EM4Off"`; `physics="corrected"` -> `"BCN_Static_ATDOn_EM4Off_physCorrected"`; `result_stem("BCN","Static_ATDOn_EM4Off","CopyB","curvature","corrected") == "BCN_Static_ATDOn_EM4Off_CopyB_physCorrected_meshCurvature"`. MLTP_screen Sturn, save off, physics corrected -> result data physics == "corrected".
- [ ] build, run new test + parity guards, commit `feat: physics flag (matlab default, corrected opt-in)`.

### Task 3: A1 mass bookkeeping + A3 frame terms (Sonnet build, Haiku test)
**Files:** `vehParams.py`, `vehModel.py`, `functions/ggv.py`, `functions/ladder.py`. Test: `tests/test_physics_mass_frame.py`.
**Corrected spec:**
- vehParams: per-corner unsprung `m_u_fl = m_u_fr = muf/2`, `m_u_rl = m_u_rr = mur/2` (new corrected-only fields). Lever rule front share `l_r/l`: `Wfl0 = Wfr0 = g*(muf/2 + ms*l_r/(2*l))`, `Wrl0 = Wrr0 = g*(mur/2 + ms*l_f/(2*l))`, `xsi_fl/fr = g*ms*l_r/(2*l*k_fl/fr)`, `xsi_rl/rr = g*ms*l_f/(2*l*k_rl/rr)`, `xti_i = W_i0/kt`.
- vehModel: `dwu_i = (zt_i*kt - m_u_i*g - fj_i - fzs_i)*sf/m_u_i`. `dvx = (sum fxg - f_drag - f_roll + m*vy*r)*sf/m`. `dvy = (sum fyg + f_side - m*vx*r)*sf/m`. `dzsdot = (-f_lift + sum fzs + sum fj - ms*g)*sf/ms` (A3: no cos(phi)cos(theta), no thetadot*vx, no phidot*vy, no g*sin(theta), no thetadot*zsdot, no g*sin(phi)cos(theta)).
- vehModel exposes on m (both physics): `f_roll`, `f_side`, `f_drag`, `fzs_fl..rr`, `xs_fl..rr`, `xsdot_fl..rr`.
- ggv: `_load_basis(vp, aero_c, load_model, physics="matlab")`; 'vehModel' + corrected -> `M = vp.m`, `Fz_f0 = g*(muf + ms*l_r/l)`, `Fz_r0 = g*(mur + ms*l_f/l)`, rolling basis on m*g; `build_envelope` passes ctx physics. Same in `ladder.qss_profile`.
- ladder: `quasi_static_states(vp, vx, physics="matlab")` corrected -> corrected equilibrium (zt, zs etc. with per-corner unsprung masses); callers pass ctx physics.
- [ ] tests:
  Tests 1, 2, 7 use `vp_overrides={"pHy1": 0, "pHy2": 0, "pVy1": 0, "pVy2": 0}` (MF205 makes about 390 N side force per corner at zero slip, so a static state is not in equilibrium otherwise).
  1. corrected, static_state vx=0.5, coast: |dxdt[10]|, |dxdt[12]|, |dxdt[14]|, |dxdt[15:19]| < 1e-3; sum fz_i == m*g (rel 1e-3); (fz_fl+fz_fr)/sum == (muf + ms*l_r/l)/m (abs 1e-3).
  2. matlab same state: dxdt[15] == (kt*xti_fl - mus*g - k_fl*xsi_fl)/mus (abs 1e-3) and that value < -5 (quirk pinned).
  3. vx=30 coast: `(sum fx_i - f_drag - f_roll)/dxdt[0]` == m (corrected) / ms (matlab), rel 1e-9. vy=0.5, r=0: `(sum fy_i + f_side)/dxdt[1]` same.
  4. state vx=30, vy=0.3, r=0.1, theta=0.01, phi=0.02, zt=xti, Om=vx/Rw: corrected d dxdt[0]/d theta == 0 and d dxdt[1]/d phi == 0 (abs 1e-12); matlab d dxdt[0]/d theta > 9, d dxdt[1]/d phi < -9.
  5. same state: d2 dxdt[10]/(d vx d thetadot) == 0, d2 dxdt[10]/(d vy d phidot) == 0 (corrected); matlab +1 and -1 (abs 1e-9).
  6. QSS Sturn lap corrected / lap matlab in [1.07, 1.13].
  7. `quasi_static_states(vp, 20, physics="corrected")` state -> |dxdt[15:19]| < 0.05 (corrected model).
- [ ] build, run new test + parity guards, commit `feat(physics): per-corner unsprung mass, planar m, frame-consistent body (corrected only)`.

### Task 4: A2 suspension + A4 aero load path (Sonnet build, Haiku test)
**Files:** `vehParams.py`, `vehModel.py`, `functions/ggv.py`. Test: `tests/test_physics_suspension_aero.py`.
**Corrected spec:**
- vehParams: sprung corner mass `m_c_fl/fr = ms*l_r/(2*l)`, `m_c_rl/rr = ms*l_f/(2*l)` (corrected-only fields); `c_i = 2*zeta_i*sqrt(k_i*m_c_i)`.
- vehModel kinematics: small-angle body + wheel, so xsdot is the exact derivative: `xs_fl = -(zs - l_f*theta + (t/2)*phi) - (zt_fl - xti_fl)`, fr `-(t/2)*phi`, rl `+l_r*theta`, rr both; xsdot unchanged formula.
- no `_cap` clamp, no fzs zeroing, no wheel-torque cutoff at xs >= 0.075. Stops: module consts `BUMP_GAP = 0.075`, `REBOUND_GAP = 0.05`, `K_STOP = 1e6`, `W_STOP = 0.002` (ponytail comment: promote to vp primaries when stop tuning matters); `fzs_i += K_STOP*W_STOP*(softplus((xs_i - BUMP_GAP)/W_STOP) - softplus((-xs_i - REBOUND_GAP)/W_STOP))`, `softplus(z) = if_else(z > 30, z, log1p(exp(z)))`.
- A4: corrected `fzs_i` has no `f_lift_i`; heave keeps `-f_lift`; `dthetadot` numerator `+ (f_lift_fl + f_lift_fr)*l_f - (f_lift_rl + f_lift_rr)*l_r`; `dphidot` numerator `- (f_lift_fl + f_lift_rl - f_lift_fr - f_lift_rr)*0.5*t`. Skipped: drag pitch moment, add when drag centre-of-pressure height is known.
- ggv: 'vehModel' + corrected -> `aero_front` as the 'nominal' basis (Cl split).
- [ ] tests:
  1. corrected `c_fl == 2*0.7*sqrt(75000*ms*l_r/(2*l))`, `c_rl == 2*0.7*sqrt(80000*ms*l_f/(2*l))` (rel 1e-12); matlab `c_fl == 0.7*2*sqrt(m*(1-wB)/2)*75000`.
  2. corrected, 5 random states (|xs| < 0.04): `jacobian(xs_i, Xp) @ dxdt == xsdot_i` (abs 1e-9). matlab, wu_fl = 0.5: mismatch > 0.1.
  3. corrected: `A = jacobian(dxdt, Xp)` at static_state vx=20, block [9:23, 9:23]: max|eig| < 2000 and a complex eig with |Im|/(2 pi) in [1.0, 4.0] Hz. matlab: max|eig| > 5000.
  4. corrected: zs swept -0.3..0.3 (step 1e-4), others static vx=20: fzs_fl finite, max adjacent |dfzs| < 500 N. State zs=-0.08 (xs_fl about +0.08), T_motor=100, ATD_FL=1: corrected `T_fl == 100*gear` (rel 1e-9), matlab `T_fl == 0`.
  5. corrected: d fzs_fl / d vx == 0 at static_state vx=40; matlab |d fzs_fl / d vx| > 1.
  6. static_state, coast: corrected dxdt[12](vx=60) - dxdt[12](vx=10) > 0; matlab < 0.
  7. `_load_basis(vp, aero_coefficients(...), "vehModel", physics="corrected")` aero_front == `_load_basis(..., "nominal")` aero_front.
- [ ] build, run new test + parity guards, commit `feat(physics): damper formula, spring to wheel, smooth stops, aero through springs (corrected only)`.

### Task 5: A5 EM4 motor speed + A6 kinematics (Sonnet build, Haiku test)
**Files:** `vehModel.py`, `userOpts.py`, `vehParams.py`, `tests/test_matlab_parity.py` (CG_p expected hit only). Test: `tests/test_physics_em4_kinematics.py`.
**Spec:**
- corrected, EM4: `Om_motor_i = Om_i * gear` (vehModel.py:514-515). Skipped: per-motor ratings, gear primary, pack power, regen: add when owner picks ratings.
- corrected: `sa_rr` uses `-toe_rear_rad`; Ackermann `sa_fl/sa_fr` use `+/-toe_front_rad`.
- both physics: `userOpts(..., Steering="NA", CamberGain="Off")` -> `ctx.Steering`, `ctx.CamberGain` (validate: Steering in ('NA','Ackermann'), CamberGain in ('Off','Linear','Table'), else ValueError); `vehModel(ctx, Steering=None, CamberGain=None, TyreModel=...)`, None -> ctx attribute.
- both physics: port MATLAB `vehParams.m` CG_p table (14 values) into `vp.CG_p_deg_per_deg_table` (dormant at CamberGain Off). test_matlab_parity: CG_p no longer expected as a hit.
- [ ] tests:
  1. EM4 (Electric_4Motors='On', ATD='Off'), static_state vx=20: corrected `Om_motor_fl == gear*Om_fl` (rel 1e-12); matlab `Om_motor_fl == Om_fl/gear`.
  2. vp_overrides toe_front=0.2, toe_rear=0.1, static_state vx=20, delta=0: corrected `sa_rl == +toe_rear_rad`, `sa_rr == -toe_rear_rad`; matlab `sa_rr == -toe_front_rad` (abs 1e-12).
  3. Steering='Ackermann', same: corrected `sa_fl == +toe_front_rad`, `sa_fr == -toe_front_rad`; matlab `sa_fl == +toe_rear_rad`.
  4. CamberGain='Table' builds (both physics); table has 14 finite values.
  5. default ctx.Steering == 'NA', ctx.CamberGain == 'Off'; Steering='x' -> ValueError.
- [ ] build, run new test + parity guards, commit `feat(physics): EM4 motor speed, toe indices (corrected); Steering/CamberGain options; CG_p table`.

### Task 6: full suite + parity (Opus)
- [ ] run all `tests/test_*.py` one at a time, compare with baseline; fix (Opus first pass, Fable on a second failure); commit fixes.

### Task 7: validation solves (Opus, one at a time, MUMPS)
- [ ] Sturn default: Solve_Succeeded, lap within 0.05% of 18.01633 s.
- [ ] Sturn corrected: Solve_Succeeded; lap, iterations; compare with QSS ratio. Rescue if needed: `ladder='qss23'` or `homotopy=True`.
- [ ] BCN corrected (check free memory first). Record results in `docs/laptime_sim_improvements.md`.

### Task 8: docs + commit (Opus)
- [ ] CLAUDE.md conventions: physics flag bullet. Improvement list: A1-A6 status. Commit `docs: corrected physics flag and status`.
