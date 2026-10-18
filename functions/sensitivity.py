"""sensitivity.py - limit prices from the multipliers that a full MLTP result stores.

Numpy only. No MLTP module imports it.

At an optimum, the multiplier of an active bound gives the first-order change of the
objective J per unit of that bound. CasADi marks an active upper bound with lam > 0.
Then dJ*/d(bound) = -lam. So one solve prices every limit, with no second solve.

- nlp_index() gives the index arrays of the NLP packing (functions/transcription.py).
- limit_prices() gives scalar prices: track bounds, torque and power limits, start speed.
- lateral_prices() gives the price of the track bounds at every grid point of the lap.
- costates() gives the continuity multipliers per physical unit of each state.

Units are SI. A price below zero means that a looser limit makes the lap faster.
A price is correct to first order only. It is wrong across an active-set change, where
another constraint becomes active or inactive. A price is dJ/dp. J includes the input
regularisation R (0.27% of the Sturn lap, 0.12% of the BCN lap). So a price is not exactly
the change of the lap time.

Skipped: the price of the lap time alone. It needs the primal sensitivity (one KKT solve).
Add it when the share of R matters. Skipped: gradients with respect to the setup
parameters. Add them when a setup optimiser needs them.
"""
from types import SimpleNamespace

import numpy as np

from functions.warmstart import as_dict, get_field, knots_from_full, norm_keys, vec


def nlp_index(structure):
    """Index arrays (0-based) of the rows of w, lam_x, g and lam_g of one NLP.

    ``structure`` is data["nlp"]["structure"] or the output of nlp_structure().
    The layout copies the packing in build_and_solve_nlp. Change the two together.
    The rows of w are X (nx, N+1), U (nu, N+1), Y (ny, N+1), XC (nx, N*d) and P (n_param,).
    The rows of g are start (nx,), end (nx,), defect (nx, N*d), cont (nx, N), h (nh, N+1),
    rate (nu, N) and periodic (n_periodic,).
    Column c = d*k + j of XC and defect is collocation point j of interval k.
    The dict also holds the ints n_w, n_g and nh.
    A missing n_param or n_periodic counts as 0.
    Raise ValueError if the sizes do not fit the packing.
    """
    nx, nu, ny, N, d, n_w, n_g = (int(get_field(structure, k)) for k in
                                  ("nx", "nu", "ny", "N", "OPT_d", "n_w", "n_g"))
    n_param, n_per = (int(get_field(structure, k, 0)) for k in ("n_param", "n_periodic"))
    nh, rest = divmod(n_g - 2 * nx - (d + 1) * N * nx - nu * N - n_per, N + 1)
    if rest or nh < 0 or n_w != (nx + nu + ny) * (N + 1) + nx * N * d + n_param:
        raise ValueError(f"The NLP sizes do not fit the packing: n_w {n_w}, n_g {n_g}, nx {nx}, "
                         f"nu {nu}, ny {ny}, N {N}, d {d}, n_param {n_param}, n_periodic {n_per}.")
    i, j = np.arange(nx)[:, None], np.arange(nu)[:, None]
    k, c = np.arange(N + 1), np.arange(N * d)
    off_h = 2 * nx + (d + 1) * nx * N
    off_r = off_h + nh * (N + 1)
    return dict(
        X=i + nx * k,
        U=nx * (N + 1) + j + nu * k,
        Y=(nx + nu) * (N + 1) + np.arange(ny)[:, None] + ny * k,
        XC=(nx + nu + ny) * (N + 1) + i + nx * c,
        P=n_w - n_param + np.arange(n_param),
        start=np.arange(nx), end=nx + np.arange(nx),
        defect=2 * nx + (d + 1) * nx * (c // d) + nx * (c % d) + i,
        cont=2 * nx + (d + 1) * nx * k[:N] + d * nx + i,
        h=off_h + np.arange(nh)[:, None] + nh * k,
        rate=off_r + j + nu * k[:N],
        periodic=off_r + nu * N + np.arange(n_per),
        n_w=n_w, n_g=n_g, nh=nh)


def _record(data):
    """The checked pieces of a full result: index, multipliers, scales, grid sizes and Xi."""
    nlp = get_field(data, "nlp")
    if nlp is None or get_field(nlp, "x_s") is None or get_field(nlp, "u_s") is None:
        raise ValueError("The result has no nlp record with x_s and u_s. Use a full MLTP result.")
    st = get_field(nlp, "structure")
    ix = nlp_index(st)
    lg, lx = vec(get_field(nlp, "lam_g")), vec(get_field(nlp, "lam_x"))
    if lg.size != ix["n_g"] or lx.size != ix["n_w"]:
        raise ValueError(f"lam_g has {lg.size} entries and lam_x has {lx.size}. "
                         f"The NLP structure needs {ix['n_g']} and {ix['n_w']}.")
    xi = get_field(data, "Xi")
    return SimpleNamespace(ix=ix, lg=lg, lx=lx, x_s=vec(get_field(nlp, "x_s")),
                           u_s=vec(get_field(nlp, "u_s")), N=int(get_field(st, "N")),
                           d=int(get_field(st, "OPT_d")), xi=None if xi is None else vec(xi))


def _n_rows(r):
    """Multipliers on the bounds of n and the s_full index that each one prices.

    The rows are the knots, the collocation points, the end row and the start row.
    A start row counts only if Xi leaves n(0) free. Otherwise it prices ni, not the track.
    """
    c = np.arange(r.N * r.d)
    lam = np.r_[r.lx[r.ix["X"][3]], r.lx[r.ix["XC"][3]], r.lg[r.ix["end"][3]]]
    at = np.r_[(r.d + 1) * np.arange(r.N + 1), (r.d + 1) * (c // r.d) + 1 + c % r.d, (r.d + 1) * r.N]
    if r.xi is None or np.isnan(r.xi[3]):
        lam, at = np.r_[lam, r.lg[r.ix["start"][3]]], np.r_[at, 0]
    return lam, at


def limit_prices(data, Pmax=None):
    """Prices dJ/d(limit) of one full MLTP result, as a dict of floats.

    ``data`` is importfile(p)["data"], load_solution(p), ctx.data or a dict.
    It needs the nlp record and input_keys. It needs constraints for the Pmax price.
    Xi is optional. A missing Xi means the matlab start. The keys are:
    - n_hi, n_lo [s/m]: the price of the upper and of the lower bound of n.
    - W [s/m]: n_hi - n_lo. Both sides of the track grow by 1 m.
    - Tmax [s/Nm]: pt.Tmax. With 4 motors, the sum over the 4 motors.
    - Tbrake_max [s/Nm]: vp.Tbrake_max.
    - Pmax [s/W]: pt.Pmax. NaN if the result has no constraints.
    - vi [s per m/s]: the centre of the start speed box. NaN if the start has no speed box.
    ``Pmax`` [W] is the rating that the solve used. None takes the default of Powertrain.
    Raise ValueError if the result has no nlp record or the multipliers do not fit it.
    """
    r = _record(data)
    lam, _ = _n_rows(r)
    n_hi, n_lo = -np.maximum(lam, 0).sum() / r.x_s[3], np.maximum(-lam, 0).sum() / r.x_s[3]
    keys = norm_keys(get_field(data, "input_keys"))
    Tmax = -sum(np.maximum(r.lx[r.ix["U"][j]], 0).sum() / r.u_s[j]
                for j, key in enumerate(keys) if key.startswith("T_motor"))
    jb = keys.index("T_brake")
    Tbrake_max = np.minimum(r.lx[r.ix["U"][jb]], 0).sum() / r.u_s[jb]
    if Pmax is None:
        from Powertrain import Powertrain
        Pmax = Powertrain(SimpleNamespace()).pt.Pmax
    cons = get_field(data, "constraints")
    # motor_power = (Pmax - P) / Pmax >= 0, so d(row)/dPmax = (1 - row) / Pmax
    power = np.nan if cons is None else sum(
        (r.lg[r.ix["h"][i]] * (1 - vec(h))).sum()
        for i, (name, h) in enumerate(as_dict(cons).items()) if name.startswith("motor_power")) / Pmax
    vi = np.nan if r.xi is not None and np.isnan(r.xi[0]) else -r.lg[r.ix["start"][0]] / r.x_s[0]
    prices = dict(W=n_hi - n_lo, n_hi=n_hi, n_lo=n_lo, Tmax=Tmax, Tbrake_max=Tbrake_max,
                  Pmax=power, vi=vi)
    return {k: float(v) for k, v in prices.items()}


def lateral_prices(data):
    """The price of the bounds of n at every point of s_full, as dict(s, hi, lo) in s/m.

    hi <= 0 is the price of the upper bound. lo >= 0 is the price of the lower bound.
    The start row sits at index 0 and the end row at index -1. Each follows the rule of W.
    So sum(hi - lo) equals limit_prices(data)["W"].
    Raise ValueError if s_full does not fit the NLP grid.
    """
    r = _record(data)
    s = vec(get_field(data, "s_full"))
    if s.size != (r.d + 1) * r.N + 1:
        raise ValueError(f"s_full has {s.size} points. The NLP grid has {(r.d + 1) * r.N + 1}.")
    lam, at = _n_rows(r)
    hi, lo = np.zeros(s.size), np.zeros(s.size)
    np.add.at(hi, at, -np.maximum(lam, 0) / r.x_s[3])
    np.add.at(lo, at, np.maximum(-lam, 0) / r.x_s[3])
    return dict(s=s, hi=hi, lo=lo)


def costates(data):
    """The multipliers of the continuity rows, as dict(s, P). P has the shape (nx, N).

    P[i, k] is the first-order change of J [s] per physical unit of a kick to state i
    at knot k+1. That knot is the end of interval k, at s[k].
    Raise ValueError if s_full does not fit the NLP grid.
    """
    r = _record(data)
    s = knots_from_full(get_field(data, "s_full"), r.d, r.N + 1)[1:]
    return dict(s=s, P=r.lg[r.ix["cont"]] / r.x_s[:, None])
