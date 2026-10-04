"""transcription.py - shared direct-collocation transcription utilities.

The transcription (discretisation, NLP assembly, solve, reconstruction) is
byte-for-byte identical between MLTP_initial.m and MLTP.m, so it lives here once
and is called by both ports. This keeps the per-script files thin while exactly
reproducing the MATLAB numerics.

Conventions reproduced from MATLAB:
  * decision vector  w = [Xk(:); Uk(:); Yk(:); Xkj(:)]  (Yk omitted when ny == 0)
  * all reshapes are column-major (order='F') to match MATLAB
  * boundary bounds use NaN-ignoring max/min (np.fmax/np.fmin) so that NaN entries
    in Xi/Xf leave the corresponding state free, exactly like MATLAB max/min.

NLP symbol type (build_and_solve_nlp `sym_type`, default from env MLTP_SYM_TYPE,
else "SX"). Both build the same NLP: same packing, constraint order, bounds and
objective accumulation order; the model Functions are evaluated over all points
at once via f_dyn.map(N*d) / h_eq.map(N+1) / f_sf.map(N+1).
  * "SX" (default): the mapped calls are inlined symbolically -- bit-identical to
    the former per-interval loop (same IPOPT iterates).
  * "MX": the decision variables are MX, so each map stays ONE call node. nlpsol
    builds ~15x faster, but CasADi's AD through a mapped call evaluates every
    derivative direction at every point, so jac_g / hess_l evaluations are
    ~8x slower, and the Jacobian/Hessian keep extra structural zeros (e.g. at
    zero-curvature points) that change IPOPT's path. Opt-in only.
"""

import os
import time

import numpy as np

from .collocation import collocation_points, collocation_coeff
from .curv2cart import curv2cart
from .cartPath import cartPath
from .trackLimits import trackLimits


# ============================================================================
# 1. Discretisation (pure numpy - testable without casadi)
# ============================================================================
def discretise(track, OPT_ds, OPT_d, mesh="uniform", mesh_opts=None, s_knot=None):
    """Build the collocation grid and curvature samples. Returns a namespace-like
    dict with N, mesh, s_knot, dsk, s_col, s_full, k_knot, k_col, k_full, pv_*,
    tau, C, D, B.

    mesh      : 'uniform' (default) -> N = round(L/OPT_ds) equal intervals;
                'curvature' -> functions.mesh.curvature_mesh: same N by default,
                knots redistributed towards corners and corner entry / exit.
    mesh_opts : dict of keyword arguments for curvature_mesh (a, b, ds_min,
                ds_max, smooth_window, N, pct, grid_ds); ignored when uniform.
    s_knot    : explicit knot vector (overrides ``mesh``; reported as 'custom'),
                strictly increasing, normally spanning the track [s[0], s[-1]].
    Every output below is computed from s_knot alone, so it holds for any mesh.
    """
    s = np.asarray(track.s, dtype=float).reshape(-1)
    k = np.asarray(track.k, dtype=float).reshape(-1)

    tau = np.asarray(collocation_points(OPT_d, "legendre"), dtype=float)
    C, D, B = collocation_coeff(tau)

    if s_knot is not None:
        s_knot = np.asarray(s_knot, dtype=float).reshape(-1)
        if s_knot.size < 2 or not np.all(np.diff(s_knot) > 0):
            raise ValueError("s_knot must be strictly increasing with >= 2 entries.")
        mesh = "custom"
        N = s_knot.size - 1
    elif mesh == "uniform":
        N = max(1, int(round((s[-1] - s[0]) / OPT_ds)))
        s_knot = np.linspace(s.min(), s.max(), N + 1)
    elif mesh == "curvature":
        from .mesh import curvature_mesh
        s_knot = curvature_mesh(s, k, OPT_ds, **dict(mesh_opts or {}))
        N = s_knot.size - 1
    else:
        raise ValueError(f"mesh must be 'uniform' or 'curvature', got {mesh!r}")
    dsk = np.diff(s_knot)                                   # length N

    # s at collocation points (absolute). start[i] = s_knot[i]: the cumulative
    # sum is kept (offset by s_knot[0]) so the uniform grid stays bit-identical.
    start = s_knot[0] + np.concatenate(([0.0], np.cumsum(dsk[:-1])))
    s_col = np.kron(dsk, tau) + np.kron(start, np.ones(OPT_d))

    # full ordered array of knot + collocation points
    s_col_mat = s_col.reshape(OPT_d, N, order="F")
    block = np.vstack([np.zeros((1, N)), s_col_mat])        # (OPT_d+1) x N
    unit = np.concatenate(([1.0], np.zeros(OPT_d)))
    s_full = np.kron(s_knot[:-1], unit) + block.reshape(-1, order="F")
    s_full = np.append(s_full, s_knot[-1])

    # curvature at the various point sets (linear interpolation)
    k_knot = np.interp(s_knot, s, k)
    k_col = np.interp(s_col, s, k)
    k_full = np.interp(s_full, s, k)

    return dict(N=N, mesh=mesh, s_knot=s_knot, dsk=dsk, s_col=s_col, s_full=s_full,
                k_knot=k_knot, k_col=k_col, k_full=k_full,
                pv_knot=k_knot.reshape(1, -1), pv_col=k_col.reshape(1, -1),
                pv_full=k_full.reshape(1, -1), tau=tau, C=C, D=D, B=B)


# ============================================================================
# 2. Solution unpacking + reconstruction (pure numpy - testable)
# ============================================================================
def unpack_solution(w_opt, nx, nu, ny, N, OPT_d, x_s, u_s, y_s):
    """Undo the column-major packing and scaling of the solution vector."""
    w = np.asarray(w_opt, dtype=float).reshape(-1)
    x_s = np.asarray(x_s).reshape(-1, 1)
    u_s = np.asarray(u_s).reshape(-1, 1)

    off = 0
    x_opt = w[off:off + nx * (N + 1)].reshape(nx, N + 1, order="F") * x_s
    off += nx * (N + 1)
    u_opt = w[off:off + nu * (N + 1)].reshape(nu, N + 1, order="F") * u_s
    off += nu * (N + 1)
    if ny > 0:
        y_s = np.asarray(y_s).reshape(-1, 1)
        y_opt = w[off:off + ny * (N + 1)].reshape(ny, N + 1, order="F") * y_s
        off += ny * (N + 1)
    else:
        y_opt = None
    xc_opt = w[off:off + nx * N * OPT_d].reshape(nx, N * OPT_d, order="F") * x_s
    return x_opt, u_opt, y_opt, xc_opt


def reconstruct_x_full(x_opt, xc_opt, nx, N, OPT_d):
    """Interleave knot states and collocation states into the full trajectory,
    x_full = [X1 X11..X1d, X2 X21..X2d, ..., XN ..., XN+1]."""
    unit = np.concatenate(([1.0], np.zeros(OPT_d))).reshape(1, -1)
    kron_part = np.kron(x_opt[:, :-1], unit)               # nx x (N*(d+1))
    xc_block = xc_opt.reshape(nx * OPT_d, N, order="F")
    tmp = np.vstack([np.zeros((nx, N)), xc_block])         # (nx*(d+1)) x N
    addend = tmp.reshape(nx, N * (OPT_d + 1), order="F")
    x_full = kron_part + addend
    x_full = np.hstack([x_full, x_opt[:, -1:]])
    return x_full


def compute_time(ca, f_sf, disc, xc_opt, x_s):
    """Reconstruct t_opt at the knot points. Because the objective integrand
    L = sf, each interval's time is Qk·B·dsk with Qk = sf evaluated at the d
    collocation points. Equivalent to the MATLAB f_t_opt postprocessing loop."""
    N = disc["N"]
    B = np.asarray(disc["B"]).reshape(-1)              # (d,)
    d = B.size
    k_col = np.asarray(disc["k_col"]).reshape(1, -1)   # (1, N*d)
    x_s = np.asarray(x_s).reshape(-1, 1)
    xc_scaled = xc_opt / x_s                            # scaled states at collocation pts
    sf_vals = np.array(f_sf(ca.DM(xc_scaled), ca.DM(k_col)).full()).reshape(-1)  # (N*d,)
    sf_mat = sf_vals.reshape(d, N, order="F")           # column k = sf at interval k's pts
    dt = (B @ sf_mat) * np.asarray(disc["dsk"]).reshape(-1)   # (N,)
    return np.concatenate([[0.0], np.cumsum(dt)])


def interp_inputs(u_opt, s_knot, s_full, OPT_uinter):
    """u_full from u_opt over the full point set (linear or previous-hold)."""
    nu = u_opt.shape[0]
    u_full = np.empty((nu, s_full.size))
    if OPT_uinter == "linear":
        for i in range(nu):
            u_full[i, :] = np.interp(s_full, s_knot, u_opt[i, :])
    elif OPT_uinter == "constant":
        idx = np.searchsorted(s_knot, s_full, side="right") - 1
        idx = np.clip(idx, 0, u_opt.shape[1] - 1)
        u_full = u_opt[:, idx]
    else:
        raise ValueError("OPT_uinter must be 'linear' or 'constant'")
    return u_full


def reconstruct_track(track0, s_full, k_full, n_full, n_halfwidth_bound):
    """Rebuild cartesian track + racing line from the curvilinear solution.

    track0 : original track namespace (may carry x,y centreline)
    n_full : lateral offset (x_full[3, :])
    n_halfwidth_bound : value used for trackLimits width = x_s(4)*x_max(4)*2 + 2
    Returns a dict with s, k, x, y, xopt, yopt, Xl, Xr.
    """
    if not hasattr(track0, "x") or not hasattr(track0, "y") \
            or getattr(track0, "x", None) is None or getattr(track0, "y", None) is None:
        x, y = curv2cart(s_full, k_full)
    else:
        x = np.interp(s_full, np.asarray(track0.s, float).reshape(-1),
                      np.asarray(track0.x, float).reshape(-1))
        y = np.interp(s_full, np.asarray(track0.s, float).reshape(-1),
                      np.asarray(track0.y, float).reshape(-1))
    xopt, yopt = cartPath(x, y, n_full)
    Xl, Xr = trackLimits(x, y, n_halfwidth_bound)
    return dict(s=s_full, k=k_full, x=x, y=y, xopt=xopt, yopt=yopt, Xl=Xl, Xr=Xr)


# ============================================================================
# 3. NLP assembly + solve (requires casadi)
# ============================================================================
def _col_diff(M, dsk, ca):
    """(M[:,1:]-M[:,:-1]) / dsk  -> per-interval derivative (cols = N)."""
    n = M.shape[0]
    dM = M[:, 1:] - M[:, :-1]
    return dM / ca.repmat(ca.DM(np.asarray(dsk).reshape(1, -1)), n, 1)


def _pad_last(M, ca):
    """diff along columns then pad the last column (MATLAB [d d(:,end)])."""
    d = M[:, 1:] - M[:, :-1]
    return ca.horzcat(d, d[:, -1])


def build_and_solve_nlp(ca, m, f_dyn, f_sf, h_eq, h_lb, h_ub,
                        disc, guesses, reg, duk_lb, duk_ub,
                        Xi, Xf, OPT_d, OPT_uinter, OPT_e, opts, param=None,
                        sym_type=None, warm=None):
    """Assemble and solve the collocation NLP. Mirrors the NLP sections of
    MLTP_initial.m / MLTP.m exactly.

    m       : model namespace (nx,nu,ny, x_min/max,u_min/max,y_min/max, x_s,u_s,y_s)
    f_dyn   : casadi Function (x,u[,y],pv[,P]) -> (dx, L)
    f_sf    : casadi Function (x,kappa) -> sf
    h_eq    : casadi Function (x,u[,y],pv[,P]) -> h
    disc    : output of discretise()
    guesses : dict with x0,u0,xc0[,y0]  (already scaled, shapes match)
    reg     : dict with ru,rdu,rdu2[,rdy,rdy2]  (column vectors)
    param   : optional static design parameters dict(sym, lb, ub, x0); appended
              last to w. Only the shape of ``sym`` is used on the MX route.
    sym_type: "SX" | "MX" | None (None -> env MLTP_SYM_TYPE, default "SX").
              SX: the mapped model calls are inlined (bit-identical to the
              legacy per-interval build). MX: decision variables are MX and
              f_dyn / h_eq / f_sf are each ONE mapped call node -- much faster
              to build, slower per IPOPT iteration (see module docstring).
    warm    : optional warm start from a previous solve of an NLP with the SAME
              structure: dict with any of
                x0     full decision vector (length n_w) used instead of the
                       guesses-built w0 (guesses are still required: they are
                       the fallback when x0 is absent / the wrong length);
                lam_g0 constraint multipliers (length n_g);
                lam_x0 bound multipliers (length n_w);
                ipopt  IPOPT options merged into opts["ipopt"] ONLY when the
                       duals are injected (functions.warmstart.
                       warm_start_ipopt_opts(): IPOPT ignores lam_*0 unless
                       warm_start_init_point='yes').
              Entries of the wrong length / non-finite are skipped with a
              warning; duals are used only together with an accepted x0.
    Returns : dict(sol, solver, N, nx, nu, ny, Xk, Uk, Yk, Xkj, dt_opt,
              sym_type, t_build, w_opt, lam_g, lam_x, n_w, n_g, structure,
              warm_info, linear_solver) -- t_build = NLP construction wall
              time [s] (entry -> just before the IPOPT call, incl. nlpsol
              creation); w_opt/lam_g/lam_x = numpy primal/dual solution;
              structure = {nx, nu, ny, N, OPT_d, n_w, n_g, n_param} (what a
              later solve must match to re-inject them); warm_info = which
              warm-start parts were used (x0, lam_g0, lam_x0, ipopt, duals);
              linear_solver = the one actually used (after any HSL fallback).
    """
    t_enter = time.perf_counter()
    if sym_type is None:
        sym_type = os.environ.get("MLTP_SYM_TYPE", "SX")
    sym_type = str(sym_type).strip().upper()
    if sym_type not in ("MX", "SX"):
        raise ValueError(f"sym_type must be 'MX' or 'SX', got {sym_type!r}")
    Sym = ca.MX if sym_type == "MX" else ca.SX

    nx, nu, ny = m.nx, m.nu, m.ny
    has_aux = ny > 0
    has_param = param is not None
    P = None
    if has_param:
        P = param["sym"]
        if not isinstance(P, Sym):
            # the caller's symbol (SX) cannot enter an MX graph: use a fresh
            # decision symbol of the same shape as the Function input instead
            P = Sym.sym("P", P.size1(), P.size2())
    N = disc["N"]
    dsk = disc["dsk"]
    tau = disc["tau"]
    C, D, B = ca.DM(disc["C"]), ca.DM(disc["D"]), ca.DM(disc["B"])
    pv_col = ca.DM(disc["pv_col"])
    k_knot = ca.DM(disc["k_knot"].reshape(1, -1))

    # decision variables
    Xk = Sym.sym("Xk", nx, N + 1)
    Uk = Sym.sym("Uk", nu, N + 1)
    Yk = Sym.sym("Yk", ny, N + 1) if has_aux else Sym.zeros(0, N + 1)
    Xkj = Sym.sym("Xkj", nx, N * OPT_d)

    # input (and aux) derivatives for regularisation
    duk = _col_diff(Uk, dsk, ca)
    duk2 = _pad_last(duk, ca)
    if has_aux:
        dyk = _col_diff(Yk, dsk, ca)
        dyk2 = _pad_last(dyk, ca)

    ru = ca.DM(np.asarray(reg["ru"]).reshape(-1, 1))
    rdu = ca.DM(np.asarray(reg["rdu"]).reshape(-1, 1))
    rdu2 = ca.DM(np.asarray(reg["rdu2"]).reshape(-1, 1))
    if has_aux:
        rdy = ca.DM(np.asarray(reg["rdy"]).reshape(-1, 1))
        rdy2 = ca.DM(np.asarray(reg["rdy2"]).reshape(-1, 1))

    x_s = np.asarray(m.x_s).reshape(-1)
    J_s = 1.0

    # boundary bounds (NaN-ignoring, as MATLAB max/min)
    x_min, x_max = m.x_min, m.x_max
    x0_min = np.fmax(x_min, Xi / x_s - OPT_e)
    x0_max = np.fmin(x_max, Xi / x_s + OPT_e)
    xf_min = np.fmax(x_min, Xf / x_s - OPT_e)
    xf_max = np.fmin(x_max, Xf / x_s + OPT_e)

    gb = [Xk[:, 0], Xk[:, -1]]
    lbg = [x0_min.reshape(-1, 1), xf_min.reshape(-1, 1)]
    ubg = [x0_max.reshape(-1, 1), xf_max.reshape(-1, 1)]

    # dynamics + objective integrand at ALL N*d collocation points in one call.
    # Column OPT_d*k + j is point j of interval k (the Xkj column order). Inputs
    # use the same arithmetic as the per-interval form Uk[:,k] + kron(duk[:,k], tau):
    # element (i, OPT_d*k+j) = Uk[i,k] + duk[i,k]*tau[j]; 'constant' holds Uk[:,k].
    nd = N * OPT_d
    rep = np.repeat(np.arange(N), OPT_d).tolist()       # interval index per point

    def _at_points(V, dV):
        if OPT_uinter == "linear":
            tau_rep = ca.DM(np.tile(np.asarray(tau).reshape(1, -1), (V.size1(), N)))
            return V[:, rep] + dV[:, rep] * tau_rep
        return V[:, rep]

    dyn_args = [Xkj, _at_points(Uk, duk)]
    if has_aux:
        dyn_args.append(_at_points(Yk, dyk))
    dyn_args.append(pv_col)
    if has_param:
        dyn_args.append(ca.repmat(P, 1, nd))            # shared by every point
    # MX: a single call node; SX: inlined per point (same as the legacy call)
    dX_all, L_all = f_dyn.map(nd)(*dyn_args)

    # collocation constraints + objective (per-interval linear assembly)
    gck = []
    J = 0
    dt_opt = []
    for k in range(N):
        cols = slice(OPT_d * k, OPT_d * k + OPT_d)
        Z = ca.horzcat(Xk[:, k], Xkj[:, cols])
        dPi = ca.mtimes(Z, C)

        dXkj = dX_all[:, cols]
        Qk = L_all[:, cols]

        Xk_end = ca.mtimes(Z, D)
        gck.append(dsk[k] * ca.reshape(dXkj, -1, 1) - ca.reshape(dPi, -1, 1))
        gck.append(Xk_end - Xk[:, k + 1])

        J = J + ca.mtimes(Qk, B) * dsk[k] / J_s \
            + ca.sumsqr(ru * Uk[:, k]) + ca.sumsqr(rdu * duk[:, k]) \
            + ca.sumsqr(rdu2 * duk2[:, k])
        if has_aux:
            J = J + ca.sumsqr(rdy * dyk[:, k]) + ca.sumsqr(rdy2 * dyk2[:, k])

        dt_opt.append(ca.mtimes(Qk, B) * dsk[k])

    # path constraints (one mapped call over the N+1 knots; a plain call with
    # N+1 columns would unroll into N+1 call nodes on MX)
    hargs = [Xk, Uk]
    if has_aux:
        hargs.append(Yk)
    hargs.append(ca.DM(disc["pv_knot"]))
    if has_param:
        hargs.append(ca.repmat(P, 1, N + 1))
    ghk = h_eq.map(N + 1)(*hargs)
    ghk = ca.reshape(ghk, -1, 1)

    # rate-of-input constraints (in time): du/dt = du/ds * 1/sf
    Sfk = f_sf.map(N + 1)(Xk, k_knot)
    Sfk = ca.reshape(Sfk, 1, -1)
    duk_t = duk / ca.repmat(Sfk[:, :N], nu, 1)
    gduk = ca.reshape(duk_t, -1, 1)

    # assemble w, bounds
    w_parts = [ca.reshape(Xk, -1, 1), ca.reshape(Uk, -1, 1)]
    lbw = [np.tile(m.x_min.reshape(-1, 1), (N + 1, 1)),
           np.tile(m.u_min.reshape(-1, 1), (N + 1, 1))]
    ubw = [np.tile(m.x_max.reshape(-1, 1), (N + 1, 1)),
           np.tile(m.u_max.reshape(-1, 1), (N + 1, 1))]
    w0 = [guesses["x0"].reshape(-1, 1, order="F"), guesses["u0"].reshape(-1, 1, order="F")]
    if has_aux:
        w_parts.append(ca.reshape(Yk, -1, 1))
        lbw.append(np.tile(m.y_min.reshape(-1, 1), (N + 1, 1)))
        ubw.append(np.tile(m.y_max.reshape(-1, 1), (N + 1, 1)))
        w0.append(guesses["y0"].reshape(-1, 1, order="F"))
    w_parts.append(ca.reshape(Xkj, -1, 1))
    lbw.append(np.tile(m.x_min.reshape(-1, 1), (N * OPT_d, 1)))
    ubw.append(np.tile(m.x_max.reshape(-1, 1), (N * OPT_d, 1)))
    w0.append(np.asarray(guesses["xc0"]).reshape(-1, 1, order="F"))
    if has_param:                                       # static design parameters
        w_parts.append(ca.reshape(P, -1, 1))
        lbw.append(np.asarray(param["lb"], dtype=float).reshape(-1, 1))
        ubw.append(np.asarray(param["ub"], dtype=float).reshape(-1, 1))
        w0.append(np.asarray(param["x0"], dtype=float).reshape(-1, 1))

    w = ca.vertcat(*w_parts)
    lbw = np.vstack(lbw)
    ubw = np.vstack(ubw)
    w0 = np.vstack(w0)

    # assemble g and bounds
    g = ca.vertcat(*gb, *gck, ghk, gduk)
    lbg = np.vstack(lbg
                    + [np.zeros(((OPT_d + 1) * N * nx, 1))]
                    + [np.tile(np.asarray(h_lb).reshape(-1, 1), (N + 1, 1))]
                    + [np.tile(np.asarray(duk_lb).reshape(-1, 1), (N, 1))])
    ubg = np.vstack(ubg
                    + [np.zeros(((OPT_d + 1) * N * nx, 1))]
                    + [np.tile(np.asarray(h_ub).reshape(-1, 1), (N + 1, 1))]
                    + [np.tile(np.asarray(duk_ub).reshape(-1, 1), (N, 1))])

    nlp = {"f": J, "x": w, "g": g}

    # ---- structure record + optional primal/dual warm start ---------------
    n_w, n_g = int(w.size1()), int(g.size1())
    structure = dict(nx=int(nx), nu=int(nu), ny=int(ny), N=int(N), OPT_d=int(OPT_d),
                     n_w=n_w, n_g=n_g, n_param=int(P.numel()) if has_param else 0)
    solve_kw = {}
    warm_info = dict(x0=False, lam_g0=False, lam_x0=False, ipopt=False, duals=False)
    if warm:
        import warnings

        def _seed(key, n):
            v = warm.get(key)
            if v is None:
                return None
            v = np.asarray(v, dtype=float).reshape(-1, 1)
            if v.shape[0] != n or not np.all(np.isfinite(v)):
                warnings.warn(f"warm start: {key} ignored ({v.shape[0]} entries, NLP needs {n}"
                              f"{'' if np.all(np.isfinite(v)) else ', non-finite values'})",
                              RuntimeWarning, stacklevel=3)
                return None
            return v

        x0_warm = _seed("x0", n_w)
        if x0_warm is not None:
            w0 = x0_warm
            warm_info["x0"] = True
            for key, n in (("lam_g0", n_g), ("lam_x0", n_w)):
                lam = _seed(key, n)
                if lam is not None:
                    solve_kw[key] = lam
                    warm_info[key] = True
        elif warm.get("lam_g0") is not None or warm.get("lam_x0") is not None:
            warnings.warn("warm start: dual seeds ignored without an accepted primal x0",
                          RuntimeWarning, stacklevel=2)
        if warm_info["lam_g0"] and warm.get("ipopt"):
            opts = dict(opts)
            opts["ipopt"] = {**opts.get("ipopt", {}), **dict(warm["ipopt"])}
            warm_info["ipopt"] = True
        ws_point = str(opts.get("ipopt", {}).get("warm_start_init_point", "no")).lower()
        warm_info["duals"] = bool(warm_info["lam_g0"] and ws_point == "yes")
        if warm_info["lam_g0"] and not warm_info["duals"]:
            warnings.warn("warm start: duals passed but warm_start_init_point != 'yes', "
                          "so IPOPT ignores them", RuntimeWarning, stacklevel=2)
        print(f"[transcription] warm start: primal x0 {'yes' if warm_info['x0'] else 'no'}, "
              f"duals {'yes' if warm_info['duals'] else 'no'}"
              + (" (IPOPT warm-start options applied)" if warm_info["ipopt"] else ""))

    t_asm = time.perf_counter() - t_enter
    solver_info = {}
    solver = _make_solver(ca, nlp, opts, info=solver_info)
    t_build = time.perf_counter() - t_enter
    print(f"[transcription] NLP build ({sym_type}, N={N}): {t_build:.2f} s "
          f"(assembly {t_asm:.2f} s, nlpsol {t_build - t_asm:.2f} s)")

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, **solve_kw)

    return dict(sol=sol, solver=solver, N=N, nx=nx, nu=nu, ny=ny,
                Xk=Xk, Uk=Uk, Yk=Yk, Xkj=Xkj, dt_opt=dt_opt,
                sym_type=sym_type, t_build=t_build,
                w_opt=np.array(sol["x"]).reshape(-1),
                lam_g=np.array(sol["lam_g"]).reshape(-1),
                lam_x=np.array(sol["lam_x"]).reshape(-1),
                n_w=n_w, n_g=n_g, structure=structure, warm_info=warm_info,
                linear_solver=solver_info.get("linear_solver"))


def _make_solver(ca, nlp, opts, info=None):
    """Create the IPOPT solver with a configurable linear solver.

    If an HSL solver (ma*) is requested, register the Coin-HSL DLL directory,
    probe once that IPOPT can load it, and use it via the `hsllib` option; if
    HSL is unavailable or fails to load, transparently fall back to MUMPS
    (bundled in the casadi wheel) so a solve never crashes on a missing or
    incompatible HSL DLL. The HSL directory is taken from a private top-level
    `opts["_hsl_dir"]` hint (set by userOpts) resolved against COINHSL_DIR and
    a seeded default; the hint is always stripped before reaching CasADi.
    `info` (optional dict) receives the effective "linear_solver".
    """
    import copy
    from functions.hsl import apply_linear_solver, resolve_hsl_dir

    opts = copy.deepcopy(opts)
    hsl_dir = resolve_hsl_dir(explicit=opts.pop("_hsl_dir", None))
    ip = opts.setdefault("ipopt", {})
    linear_solver = ip.get("linear_solver", "mumps")
    if str(linear_solver).startswith("ma"):
        opts = apply_linear_solver(opts, linear_solver=linear_solver,
                                   hsl_dir=hsl_dir)
    if info is not None:
        info["linear_solver"] = opts.get("ipopt", {}).get("linear_solver", "mumps")
    return ca.nlpsol("solver", "ipopt", nlp, opts)
