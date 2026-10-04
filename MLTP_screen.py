"""MLTP_screen.py - fast quasi-steady-state (QSS) lap-time SCREEN (estimate tier).

Call shape follows MLTP(). Accepted: circuit, vi, AeroConfig, ATD,
Electric_4Motors, save, results_dir and **useropts_kwargs (forwarded to
userOpts; ni is accepted but unused). TyreModel and plot are accepted and
ignored. warm_start, plots_dir and warm_start_duals are not accepted
(TypeError). The 23-state optimal-control problem is replaced by the g-g-v
envelope + forward/backward speed march of functions/ggv.py:

    userOpts -> build_envelope -> march (centreline, n = 0) -> ctx.data (+ .mat)

The lap time is an ESTIMATE for screening / ranking setups (tens of ms per
setup once ctx exists), never an optimum: the line is fixed to the centreline
(the track width is not used) and the transient states of the 23-state model
(suspension, wheel spin, yaw dynamics) are absent. Use MLTP() for the optimum.

Output schema follows MLTP where it can (s_full/k_full from discretise(),
lap_time, track, N, circuit/AeroConfig/ATD/EM4, mesh/mesh_opts/tyre_set), plus the
QSS profile and envelope; data.fidelity = 'qss'. Note t_opt is sampled at s_full (the NLP's
t_opt is at the N+1 knots). Saved to Results/<circuit>_<cfg>_qss.mat.
"""

import os
import time
import warnings
import numpy as np
import scipy.io as sio
from types import SimpleNamespace

from functions.context import Ctx
from functions.ggv import build_envelope, march
from functions.importfile import result_stem
from functions.mesh import mesh_opts_record
from functions.transcription import discretise, reconstruct_track
from userOpts import userOpts

# vehModel hard-codes the lateral-offset bound n in [-4, 4] m (x_lim row 4) and
# MLTP plots the track limits with n_width = 2*x_s[3]*x_max[3] = 8 m; mirrored
# here as a constant so the screen never has to build the CasADi model.
N_HALF_WIDTH = 4.0


def MLTP_screen(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On",
                Electric_4Motors="Off", save=True, results_dir="Results",
                load_model="vehModel", ds_fine=1.0, verbose=True,
                TyreModel=None, plot=False, **useropts_kwargs):
    """QSS lap-time estimate. Extra userOpts arguments (vp_overrides, OPT_ds,
    circuits_dir, ...) are forwarded via **useropts_kwargs. load_model selects
    the vertical-load basis of the envelope ('vehModel' = the 23-state model's
    quasi-steady equilibrium, 'nominal' = textbook m*g / Wfl0 split); see
    functions/ggv.py. Returns ctx with ctx.data, ctx.envelope, ctx.elapsed.
    TyreModel and plot are accepted for MLTP() call parity and ignored."""
    t0 = time.perf_counter()
    ctx = Ctx()
    userOpts(ctx, circuit=circuit, vi=vi, AeroConfig=AeroConfig, ATD=ATD,
             Electric_4Motors=Electric_4Motors, **useropts_kwargs)
    t1 = time.perf_counter()

    env = build_envelope(ctx, load_model=load_model)
    t2 = time.perf_counter()
    disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d,         # NLP grid, for schema parity
                      mesh=getattr(ctx, "mesh", "uniform"),
                      mesh_opts=getattr(ctx, "mesh_opts", None))
    t3 = time.perf_counter()
    prof = march(env, ctx.track.s, ctx.track.k, vi, ds_fine=ds_fine, s_out=disc["s_full"])
    t4 = time.perf_counter()

    if not prof["vi_feasible"]:
        warnings.warn(f"[MLTP_screen] vi={vi} m/s cannot be held at s=0 (the first corner "
                      f"needs braking before the start); the QSS profile starts at "
                      f"{prof['v'][0]:.2f} m/s.")

    track = reconstruct_track(ctx.track, disc["s_full"], disc["k_full"],
                              np.zeros_like(disc["s_full"]), 2.0 * N_HALF_WIDTH)
    lap = prof["lap_time"]
    data = {
        "fidelity": "qss",
        "s_full": disc["s_full"], "k_full": disc["k_full"],
        "v": prof["v_out"], "t_opt": prof["t_out"], "ax": prof["ax_out"], "ay": prof["ay_out"],
        "v_apex": prof["v_apex_out"],
        "s_fine": prof["s"], "k_fine": prof["k"], "v_fine": prof["v"], "t_fine": prof["t"],
        "ax_fine": prof["ax"], "ay_fine": prof["ay"],
        "lap_time": lap,
        "track": {k: v for k, v in track.items()},
        "envelope": env.to_dict(),
        "N": disc["N"], "OPT_ds": ctx.OPT_ds, "OPT_d": ctx.OPT_d, "ds_fine": prof["ds"],
        "circuit": circuit, "AeroConfig": AeroConfig, "ATD": ctx.ATD,
        "EM4": ctx.Electric_4Motors, "vi": float(vi), "load_model": load_model,
        "mesh": getattr(ctx, "mesh", "uniform"),
        "mesh_requested": getattr(ctx, "mesh_requested", "auto"),
        "mesh_opts": mesh_opts_record(getattr(ctx, "mesh_opts", None)),
        "tyre_set": getattr(ctx, "tyre_set", "MF205"),
        "mf_overrides": list(getattr(ctx, "mf_overrides", [])),
        "note": "ESTIMATE (QSS screen): centreline n=0, quasi-steady point mass, "
                "no transients - not an optimum; see MLTP() for the NLP solution",
    }
    ctx.data = SimpleNamespace(**data)
    ctx.envelope = env
    ctx.elapsed = {"setup": t1 - t0, "envelope": t2 - t1, "grid": t3 - t2,
                   "march": t4 - t3, "total": time.perf_counter() - t0}

    if save:
        os.makedirs(results_dir, exist_ok=True)
        cfg = f"{AeroConfig}_ATD{ctx.ATD}_EM4{ctx.Electric_4Motors}"
        out_path = os.path.join(results_dir, result_stem(
            circuit, cfg, getattr(ctx, "tyre_set", "MF205"),
            getattr(ctx, "mesh_requested", "auto")) + "_qss.mat")
        sio.savemat(out_path, {"data": data}, do_compression=True)
        ctx.out_path = out_path
        if verbose:
            print(f"Saved QSS screen estimate -> {out_path}")

    if verbose:
        print(f"[MLTP_screen] ESTIMATE (QSS screen: centreline, quasi-steady, no transients)  "
              f"circuit={circuit}  config={AeroConfig}/ATD={ctx.ATD}/EM4={ctx.Electric_4Motors}  "
              f"N={disc['N']}  lap time ~ {lap:.3f} s  "
              f"(envelope {1e3 * ctx.elapsed['envelope']:.1f} ms, march {1e3 * ctx.elapsed['march']:.1f} ms)")
    return ctx


def screen_sweep(circuit, overrides_list, vi=60.0, AeroConfig="Static", ATD="On",
                 Electric_4Motors="Off", load_model="vehModel", ds_fine=1.0,
                 **useropts_kwargs):
    """QSS lap time for each vp_overrides dict in overrides_list (the hook for a
    DoE orchestrator). A base vp_overrides passed in **useropts_kwargs is merged
    under each entry. Rebuilds ctx (userOpts) per entry, then envelope + march;
    nothing is saved. Returns [{'lap_time': float, 'overrides': dict}, ...]."""
    base = dict(useropts_kwargs.pop("vp_overrides", None) or {})
    results = []
    for ov in overrides_list:
        merged = {**base, **dict(ov or {})}
        ctx = Ctx()
        userOpts(ctx, circuit=circuit, vi=vi, AeroConfig=AeroConfig, ATD=ATD,
                 Electric_4Motors=Electric_4Motors, vp_overrides=merged, **useropts_kwargs)
        env = build_envelope(ctx, load_model=load_model)
        prof = march(env, ctx.track.s, ctx.track.k, vi, ds_fine=ds_fine)
        results.append({"lap_time": prof["lap_time"], "overrides": dict(ov or {})})
    return results


if __name__ == "__main__":
    MLTP_screen(circuit="Sturn", vi=60.0)
