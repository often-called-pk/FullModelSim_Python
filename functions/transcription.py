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
"""

import numpy as np

from .collocation import collocation_points, collocation_coeff
from .curv2cart import curv2cart
from .cartPath import cartPath
from .trackLimits import trackLimits


# ============================================================================
# 1. Discretisation (pure numpy - testable without casadi)
# ============================================================================
def discretise(track, OPT_ds, OPT_d):
    """Build the collocation grid and curvature samples. Returns a namespace-like
    dict with N, s_knot, dsk, s_col, s_full, k_knot, k_col, k_full, pv_*, tau, C, D, B."""
    s = np.asarray(track.s, dtype=float).reshape(-1)
    k = np.asarray(track.k, dtype=float).reshape(-1)

    tau = np.asarray(collocation_points(OPT_d, "legendre"), dtype=float)
    C, D, B = collocation_coeff(tau)

    N = int(round(s[-1] / OPT_ds))
    s_knot = np.linspace(s.min(), s.max(), N + 1)
    dsk = np.diff(s_knot)                                   # length N

    # s at collocation points (absolute)
    start = np.concatenate(([0.0], np.cumsum(dsk[:-1])))    # start of each interval
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

    return dict(N=N, s_knot=s_knot, dsk=dsk, s_col=s_col, s_full=s_full,
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
                        Xi, Xf, OPT_d, OPT_uinter, OPT_e, opts, param=None):
    """Assemble and solve the collocation NLP. Mirrors the NLP sections of
    MLTP_initial.m / MLTP.m exactly.

    m       : model namespace (nx,nu,ny, x_min/max,u_min/max,y_min/max, x_s,u_s,y_s)
    f_dyn   : casadi Function (x,u[,y],pv) -> (dx, L)
    f_sf    : casadi Function (x,kappa) -> sf
    h_eq    : casadi Function (x,u[,y],pv) -> h
    disc    : output of discretise()
    guesses : dict with x0,u0,xc0[,y0]  (already scaled, shapes match)
    reg     : dict with ru,rdu,rdu2[,rdy,rdy2]  (column vectors)
    Returns : dict(sol, N, nx, nu, ny, dt_opt_funcs, Xkj, Uk, Yk)
    """
    SX = ca.SX
    nx, nu, ny = m.nx, m.nu, m.ny
    has_aux = ny > 0
    has_param = param is not None
    P = param["sym"] if has_param else None
    N = disc["N"]
    dsk = disc["dsk"]
    tau = disc["tau"]
    C, D, B = ca.DM(disc["C"]), ca.DM(disc["D"]), ca.DM(disc["B"])
    pv_col = ca.DM(disc["pv_col"])
    k_knot = ca.DM(disc["k_knot"].reshape(1, -1))

    # decision variables
    Xk = SX.sym("Xk", nx, N + 1)
    Uk = SX.sym("Uk", nu, N + 1)
    Yk = SX.sym("Yk", ny, N + 1) if has_aux else SX.zeros(0, N + 1)
    Xkj = SX.sym("Xkj", nx, N * OPT_d)

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

    # collocation constraints + objective
    gck = []
    J = 0
    dt_opt = []
    for k in range(N):
        cols = slice(OPT_d * k, OPT_d * k + OPT_d)
        Z = ca.horzcat(Xk[:, k], Xkj[:, cols])
        dPi = ca.mtimes(Z, C)

        Ucol = Uk[:, k]
        if OPT_uinter == "linear":
            Ucol = Uk[:, k] + ca.kron(duk[:, k], ca.DM(tau.reshape(1, -1)))
        args = [Xkj[:, cols], Ucol]
        if has_aux:
            Ycol = Yk[:, k]
            if OPT_uinter == "linear":
                Ycol = Yk[:, k] + ca.kron(dyk[:, k], ca.DM(tau.reshape(1, -1)))
            args.append(Ycol)
        args.append(pv_col[:, cols])
        if has_param:
            args.append(P)
        dXkj, Qk = f_dyn(*args)

        Xk_end = ca.mtimes(Z, D)
        gck.append(dsk[k] * ca.reshape(dXkj, -1, 1) - ca.reshape(dPi, -1, 1))
        gck.append(Xk_end - Xk[:, k + 1])

        J = J + ca.mtimes(Qk, B) * dsk[k] / J_s \
            + ca.sumsqr(ru * Uk[:, k]) + ca.sumsqr(rdu * duk[:, k]) \
            + ca.sumsqr(rdu2 * duk2[:, k])
        if has_aux:
            J = J + ca.sumsqr(rdy * dyk[:, k]) + ca.sumsqr(rdy2 * dyk2[:, k])

        dt_opt.append(ca.mtimes(Qk, B) * dsk[k])

    # path constraints
    hargs = [Xk, Uk]
    if has_aux:
        hargs.append(Yk)
    hargs.append(ca.DM(disc["pv_knot"]))
    if has_param:
        hargs.append(P)
    ghk = h_eq(*hargs)
    ghk = ca.reshape(ghk, -1, 1)

    # rate-of-input constraints (in time): du/dt = du/ds * 1/sf
    Sfk = f_sf(Xk, k_knot)
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
    solver = _make_solver(ca, nlp, opts)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    return dict(sol=sol, solver=solver, N=N, nx=nx, nu=nu, ny=ny,
                Xk=Xk, Uk=Uk, Yk=Yk, Xkj=Xkj, dt_opt=dt_opt)


def _make_solver(ca, nlp, opts):
    """Create the IPOPT solver with a configurable linear solver.

    If an HSL solver (ma*) is requested, register the Coin-HSL DLL directory,
    probe once that IPOPT can load it, and use it via the `hsllib` option; if
    HSL is unavailable or fails to load, transparently fall back to MUMPS
    (bundled in the casadi wheel) so a solve never crashes on a missing or
    incompatible HSL DLL. The HSL directory is taken from a private top-level
    `opts["_hsl_dir"]` hint (set by userOpts) resolved against COINHSL_DIR and
    a seeded default; the hint is always stripped before reaching CasADi.
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
    return ca.nlpsol("solver", "ipopt", nlp, opts)
