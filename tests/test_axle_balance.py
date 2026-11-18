"""Axle balance channel (ChassisSim phase 2, task 9): tests/test_axle_balance.py.

MLTP appends the channel "balance" to data.vehicle and data.vehicle_full: the rear axle's share of
its grip limit minus the front axle's share, oversteer positive. Per corner L_c = sqrt((fx/mu_x)^2 +
(fy/mu_y)^2) [N], and balance = (L_rl + L_rr)/(fz_rl + fz_rr) - (L_fl + L_fr)/(fz_fl + fz_fr).
plotSDI.plot_friction adds a trace "axle balance (rear - front)" when the channel is there.

Sections 1, 2, 3, 4 and 5 use a fake IPOPT that returns the stored Sturn optimum (warm start from
Results/Sturn_Static_ATDOn_EM4Off.mat, as tests/test_telemetry.py section 4), so forces and slips are
real. Section 5 seeds the corrected physics from the same stored point (interpolated, no duals).
Each 4EM build asserts preconditions first: the balance spans more than 1e-2, and some corner has
|fx| and |fy| above 100 N. Section 6 is the vendor ChassisSim GP3 2013 car on Eastern Creek. It runs
only when CHASSISSIM_DIR names the install folder and prints SKIP and exits 0 otherwise.
Sections 1-5 need casadi and Data/DATA_AA.mat (SKIP without it). Scratch files: none.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_axle_balance.py
    CHASSISSIM_DIR="<ChassisSim_v3_41 folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_axle_balance.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import copy
import io
import os
import warnings

import numpy as np

import plotSDI

DATA_AA = os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")
STURN = os.path.join(_bootstrap.ROOT, "Results", "Sturn_Static_ATDOn_EM4Off.mat")
CORNERS = ("fl", "fr", "rl", "rr")
PASS = 0


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout and warnings swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def channels(veh):
    """data.vehicle as a dict: an in-memory result holds a dict, a loaded .mat a namespace."""
    return veh if isinstance(veh, dict) else vars(veh)


def hand_balance(veh):
    """Balance from the stored channels, written out per corner (the definition of the brief)."""
    L = {c: np.sqrt((np.asarray(veh[f"fx_{c}"], float) / np.asarray(veh[f"mu_{c}_x"], float)) ** 2
                    + (np.asarray(veh[f"fy_{c}"], float) / np.asarray(veh[f"mu_{c}_y"], float)) ** 2)
         for c in CORNERS}
    fz = {c: np.asarray(veh[f"fz_{c}"], float) for c in CORNERS}
    return (L["rl"] + L["rr"]) / (fz["rl"] + fz["rr"]) - (L["fl"] + L["fr"]) / (fz["fl"] + fz["fr"])


def is_real(veh):
    """True when the state is not trivial: balance spans > 1e-2 and some corner has |fx|, |fy| > 100 N."""
    bal = np.asarray(veh["balance"], float)
    fx = max(np.abs(np.asarray(veh[f"fx_{c}"], float)).max() for c in CORNERS)
    fy = max(np.abs(np.asarray(veh[f"fy_{c}"], float)).max() for c in CORNERS)
    return np.ptp(bal) > 1e-2 and fx > 100.0 and fy > 100.0


def stored_optimum_solver():
    """Context manager: transcription._make_solver -> a solver that returns x = x0 and zero duals.
    Used with warm_start=STURN, so x0 is the stored optimum (no IPOPT)."""
    import functions.transcription as T

    def fake(ca_, nlp, opts, info=None):
        if info is not None:
            info["linear_solver"] = "fake"

        def solver(**k):
            return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()),
                    "lam_x": np.zeros(nlp["x"].numel())}

        solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
        return solver

    @contextlib.contextmanager
    def cm():
        real, T._make_solver = T._make_solver, fake
        try:
            yield
        finally:
            T._make_solver = real

    return cm()


def check_formula(tag, c):
    """Channel present, finite, full-grid length, bitwise knot slice, hand formula, preconditions, signs."""
    d = c.data
    D = int(d.OPT_d)
    full, veh = channels(d.vehicle_full), channels(d.vehicle)
    n_pts = np.size(d.s_full)
    ok(f"{tag}: 'balance' in data.vehicle and data.vehicle_full",
       "balance" in veh and "balance" in full)
    ok(f"{tag}: balance finite, {n_pts} full-grid points like the other channels",
       np.all(np.isfinite(full["balance"])) and np.size(full["balance"]) == n_pts
       and all(np.size(full[k]) == n_pts for k in full))
    ok(f"{tag}: vehicle['balance'] == vehicle_full['balance'][::OPT_d+1] bitwise",
       np.asarray(veh["balance"]).tobytes() == np.asarray(full["balance"])[::D + 1].tobytes())
    # preconditions: the state is real, so the formula checks below mean something
    bal = np.asarray(full["balance"], float)
    fx_max = max(np.abs(np.asarray(full[f"fx_{k}"], float)).max() for k in CORNERS)
    fy_max = max(np.abs(np.asarray(full[f"fy_{k}"], float)).max() for k in CORNERS)
    ok(f"{tag}: precondition balance ptp {np.ptp(bal):.3g} > 1e-2", np.ptp(bal) > 1e-2)
    ok(f"{tag}: precondition max |fx| {fx_max:.0f} N > 100 N", fx_max > 100.0)
    ok(f"{tag}: precondition max |fy| {fy_max:.0f} N > 100 N", fy_max > 100.0)
    hand = hand_balance(full)
    ok(f"{tag}: balance equals the hand formula at every point (rtol 1e-12)",
       np.allclose(bal, hand, rtol=1e-12, atol=0.0))
    # sign: the most rear-limited and most front-limited points
    i_hi, i_lo = int(np.argmax(hand)), int(np.argmin(hand))
    signs = []
    if hand[i_hi] > 0:
        ok(f"{tag}: channel > 0 at the largest rear-minus-front point (s = {d.s_full[i_hi]:.1f} m)",
           bal[i_hi] > 0)
        signs.append("+")
    if hand[i_lo] < 0:
        ok(f"{tag}: channel < 0 at the smallest point (s = {d.s_full[i_lo]:.1f} m)", bal[i_lo] < 0)
        signs.append("-")
    ok(f"{tag}: at least one sign of the balance occurs (seen {''.join(signs)})", bool(signs))
    return d, full


# =============================================================================================
if not os.path.exists(DATA_AA):
    print("SKIP: Data/DATA_AA.mat not found (sections 1-5 need it)")
else:
    import MLTP as mltp_mod

    print("1. stored-optimum Sturn build, default 4EM car (matlab physics): channel, formula, preconditions, signs")
    with stored_optimum_solver():
        c = quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False)
    d, full = check_formula("default", c)

    print("3. independent path: fz-weighted mean of the friction_usage rho per axle")
    # per-corner rho = L_c / fz_c, so the fz-weighted mean is sum(L_c) / sum(fz_c) (rear minus front)
    usage = plotSDI.friction_usage(d)
    ok("friction_usage has the per-corner rho_fl..rr (no PureSlip rows in this build)",
       all(f"rho_{k}" in usage for k in CORNERS))
    fz = {k: np.asarray(channels(d.vehicle)[f"fz_{k}"], float) for k in CORNERS}     # knots, like rho
    ax_mean = lambda axle: sum(usage[f"rho_{k}"] * fz[k] for k in axle) / sum(fz[k] for k in axle)
    via_rho = ax_mean(("rl", "rr")) - ax_mean(("fl", "fr"))
    bal_knot = np.asarray(d.vehicle["balance"], float)
    ok("balance equals the fz-weighted rho mean, rear minus front (rtol 1e-9)",
       np.allclose(bal_knot, via_rho, rtol=1e-9, atol=0.0))

    print("4. plot_friction: one balance trace, and a result without the channel plots as before")
    fig = plotSDI.plot_friction(d)
    named = [t for t in fig.data if t.name == "axle balance (rear - front)"]
    ok("exactly one trace named 'axle balance (rear - front)'", len(named) == 1)
    ok("its y equals data.vehicle['balance']",
       np.array_equal(np.asarray(named[0].y, float), bal_knot))
    d_old = copy.copy(d)
    d_old.vehicle = {k: v for k, v in channels(d.vehicle).items() if k != "balance"}
    fig_old = quiet(plotSDI.plot_friction, d_old)
    ok("copy without 'balance': no balance trace, no error",
       not any(t.name == "axle balance (rear - front)" for t in fig_old.data)
       and len(fig_old.data) == len(fig.data) - 1)

    print("5. physics='corrected' (stored Sturn point seeds it, interpolated; fake IPOPT)")
    with stored_optimum_solver():
        cc = quiet(mltp_mod.MLTP, circuit="Sturn", warm_start=STURN, save=False, plot=False,
                   physics="corrected")
    check_formula("corrected", cc)

    print("6. vendor ChassisSim GP3 2013 car on Eastern Creek")
    cs_dir = os.environ.get("CHASSISSIM_DIR")
    if not cs_dir or not os.path.isdir(cs_dir):
        print("SKIP: CHASSISSIM_DIR not set or not a folder")
    else:
        car_path = os.path.join(os.path.dirname(cs_dir.rstrip("/\\")), "Models", "GP3 2013",
                                "Eastern Creek, AU", "def_GP3 2013.car")
        ok("GP3 2013 Eastern Creek car file found", os.path.isfile(car_path))
        # The car build is cold (ladder qss23): the seed is the QSS point, not the stored optimum.
        # Presence and finite checks always run; the formula and the preconditions run only when the
        # QSS seed state is real (the print says which).
        with stored_optimum_solver():
            v = quiet(mltp_mod.MLTP, car=f"chassissim:{car_path}", circuit="chassissim:Eastern Creek",
                      chassissim_dir=cs_dir, physics="corrected", save=False, plot=False,
                      linear_solver="mumps")
        full_v = channels(v.data.vehicle_full)
        ok("GP3 car: 'balance' in data.vehicle and data.vehicle_full",
           "balance" in full_v and "balance" in channels(v.data.vehicle))
        ok("GP3 car: balance finite on every point",
           np.all(np.isfinite(np.asarray(full_v["balance"], float))))
        if is_real(full_v):
            print("  GP3 car: QSS seed state is real, running the full formula and sign checks")
            check_formula("GP3 car", v)
        else:
            print("  GP3 car: seed state is trivial (preconditions do not hold); presence and finite checks only")

print(f"{PASS} checks passed")
