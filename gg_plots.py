#!/usr/bin/env python3
"""Interactive Plotly friction-circle / g-g plots for the MLTP pipeline.

Adds two standalone HTML plots; the existing friction_circle.html is untouched:

  * <outdir>/friction_circle_gg.html  per-tyre friction circle (4 tyres);
        axes Fx/(mu*Fz) and Fy/(mu*Fz); the unit circle is the grip limit.
  * <outdir>/gg_diagram.html          vehicle g-g diagram; lateral vs
        longitudinal acceleration in g, points coloured by speed.

Integration - one import + one call, placed wherever your other plots are made:

    from gg_plots import generate_gg_plots
    generate_gg_plots(data, fullFolderPath)      # writes both HTML files

Uses only quantities the solution already carries (per-tyre fx/fy/fz, mu_x/mu_y,
vehicle ax_g/ay_g). The two extraction helpers at the bottom map field names to
your `data` struct; or bypass them by passing arrays directly:

    generate_gg_plots(
        outdir=fullFolderPath,
        tyre={"FL": (gx_fl, gy_fl), "FR": (gx_fr, gy_fr),
              "RL": (gx_rl, gy_rl), "RR": (gx_rr, gy_rr)},
        vehicle={"ax_g": ax_g, "ay_g": ay_g, "speed": vx, "s": s_full},
    )
"""
import os
import numpy as np

try:
    from scipy.spatial import ConvexHull
    _HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    _HAVE_SCIPY = False

_TYRE_COLORS = {"FL": "#FFD166", "FR": "#FF8C42", "RL": "#EF476F", "RR": "#4CC9F0"}
_FALLBACK = ["#FFD166", "#FF8C42", "#EF476F", "#4CC9F0", "#C77DFF", "#06D6A0"]
_RHO = "\u03c1"
_MU = "\u03bc"


# ---------------------------------------------------------------------------
# geometry / styling helpers (no plotly needed)
# ---------------------------------------------------------------------------
def _hull_loop(gx, gy):
    pts = np.column_stack([np.asarray(gx, float), np.asarray(gy, float)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(np.unique(pts, axis=0)) < 3:
        return None
    if _HAVE_SCIPY:
        try:
            loop = pts[ConvexHull(pts).vertices]
        except Exception:
            return None
    else:                                               # monotone-chain fallback
        p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        def half(P):
            out = []
            for q in P:
                while len(out) >= 2 and np.cross(out[-1] - out[-2], q - out[-2]) <= 0:
                    out.pop()
                out.append(q)
            return out
        loop = np.array(half(p)[:-1] + half(p[::-1])[:-1])
        if len(loop) < 3:
            return None
    return np.vstack([loop, loop[0]])


def _rings_and_lim(arrays, units, max_ring, ring_step):
    is_ratio = units == "ratio"
    if ring_step is None:
        ring_step = 0.25 if is_ratio else 1.0
    if max_ring is None:
        if is_ratio:
            max_ring = 1.0
        else:
            rmax = max(float(np.nanmax(np.hypot(np.asarray(gx, float),
                                                np.asarray(gy, float))))
                       for gx, gy in arrays)
            max_ring = float(np.ceil(rmax / ring_step) * ring_step)
    rings = np.arange(ring_step, max_ring + 1e-9, ring_step)
    return is_ratio, ring_step, max_ring, rings


def _add_rings_and_guides(fig, lim, rings, max_ring, is_ratio):
    for r in rings:
        is_limit = is_ratio and abs(r - max_ring) < 1e-9
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r,
                      layer="below",
                      line=dict(color="white", width=1.6 if is_limit else 0.8,
                                dash="dash" if is_limit else "dot"),
                      opacity=0.85 if is_limit else 0.30)
    aw = "rgba(255,255,255,0.85)"
    for sgn, label in ((1, "Traction"), (-1, "Braking")):
        fig.add_annotation(x=0, y=sgn * lim * 0.90, ax=0, ay=sgn * lim * 0.55,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.4, arrowcolor=aw, text="")
        fig.add_annotation(x=-lim * 0.045, y=sgn * lim * 0.72, text=label, showarrow=False,
                           textangle=-90, xanchor="right", font=dict(color="white", size=13))
    for sgn in (1, -1):
        fig.add_annotation(x=sgn * lim * 0.99, y=0, ax=sgn * lim * 0.70, ay=0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.4, arrowcolor=aw, text="")
    fig.add_annotation(x=-lim * 0.92, y=lim * 0.02, text="right corners", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       font=dict(color="rgba(255,255,255,0.65)", size=11))
    fig.add_annotation(x=lim * 0.92, y=lim * 0.02, text="left corners", showarrow=False,
                       xanchor="right", yanchor="bottom",
                       font=dict(color="rgba(255,255,255,0.65)", size=11))
    if is_ratio:
        ang = np.deg2rad(57)
        fig.add_annotation(x=max_ring * np.cos(ang), y=max_ring * np.sin(ang),
                           text="grip limit  (" + _RHO + " = 1)", showarrow=True,
                           arrowhead=0, arrowwidth=0.8, arrowcolor="rgba(255,255,255,0.5)",
                           ax=42, ay=-34, xanchor="left", font=dict(color="white", size=11))
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.085, showarrow=False,
                       text=("Points near the outer circle are grip-limited" if is_ratio
                             else "Points far from the centre indicate high combined g"),
                       font=dict(color="rgba(255,255,255,0.7)", size=10))


def _apply_theme(fig, lim, rings, is_ratio, title, size, show_legend, right_margin=40):
    if is_ratio:
        xlabel = "Lateral grip usage  F<sub>y</sub> / " + _MU + "F<sub>z</sub>"
        ylabel = "Longitudinal grip usage  F<sub>x</sub> / " + _MU + "F<sub>z</sub>"
    else:
        xlabel = "Lateral acceleration (g)"
        ylabel = "Longitudinal acceleration (g)"
    ticks = np.concatenate([-rings[::-1], [0], rings])
    common = dict(range=[-lim, lim], zeroline=True, zerolinecolor="rgba(255,255,255,0.5)",
                  zerolinewidth=1, showgrid=False, tickvals=ticks, color="white",
                  tickfont=dict(color="white"), constrain="domain")
    fig.update_xaxes(title=dict(text=xlabel, font=dict(color="white", size=14)), **common)
    fig.update_yaxes(title=dict(text=ylabel, font=dict(color="white", size=14)),
                     scaleanchor="x", scaleratio=1, **common)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="white", size=20)),
        paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"),
        width=size, height=size, margin=dict(l=80, r=right_margin, t=70, b=90),
        showlegend=show_legend,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(17,17,17,0.85)", bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1, font=dict(color="white", size=11)),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.85)", font=dict(color="white")))


# ---------------------------------------------------------------------------
# figure builders
# ---------------------------------------------------------------------------
def plot_friction_circle_plotly(series, units="ratio",
                                title="Friction Circle  \u2014  Tyre Grip Utilisation",
                                max_ring=None, ring_step=None, point_size=5,
                                opacity=0.5, show_hull=True, colors=None,
                                s=None, size=820):
    """Per-tyre friction circle. series: dict name -> (gx, gy)."""
    import plotly.graph_objects as go
    is_ratio, ring_step, max_ring, rings = _rings_and_lim(
        series.values(), units, max_ring, ring_step)
    colors = {**_TYRE_COLORS, **(colors or {})}
    lim = max_ring * 1.16
    s_arr = None if s is None else np.asarray(s, float).reshape(-1)

    fig = go.Figure()
    _add_rings_and_guides(fig, lim, rings, max_ring, is_ratio)
    for i, (name, (gx, gy)) in enumerate(series.items()):
        c = colors.get(name, _FALLBACK[i % len(_FALLBACK)])
        gx = np.asarray(gx, float); gy = np.asarray(gy, float)
        rho = np.hypot(gx, gy)
        if s_arr is not None and s_arr.size == rho.size:
            cd = np.column_stack([rho, s_arr])
            ht = (name + "<br>" + _RHO + " = %{customdata[0]:.3f}"
                       + "<br>s = %{customdata[1]:.0f} m"
                       + "<br>long = %{y:.2f}   lat = %{x:.2f}<extra></extra>")
        else:
            cd = rho.reshape(-1, 1)
            ht = (name + "<br>" + _RHO + " = %{customdata[0]:.3f}"
                       + "<br>long = %{y:.2f}   lat = %{x:.2f}<extra></extra>")
        fig.add_trace(go.Scattergl(
            x=gy, y=gx, mode="markers", legendgroup=name, customdata=cd, hovertemplate=ht,
            name=name + "   peak " + _RHO + " = %.2f" % float(np.nanmax(rho)),
            marker=dict(size=point_size, color=c, opacity=opacity, line=dict(width=0))))
        if show_hull:
            loop = _hull_loop(gx, gy)
            if loop is not None:
                fig.add_trace(go.Scatter(
                    x=loop[:, 1], y=loop[:, 0], mode="lines", legendgroup=name,
                    showlegend=False, hoverinfo="skip",
                    line=dict(color=c, width=1.8, dash="dash")))
    _apply_theme(fig, lim, rings, is_ratio, title, size, show_legend=True)
    return fig


def plot_gg_vehicle_plotly(ax_g, ay_g, speed=None, s=None,
                           title="G-G Diagram  \u2014  Vehicle Accelerations",
                           max_ring=None, ring_step=1.0, point_size=5,
                           opacity=0.8, show_hull=True, size=820):
    """Vehicle g-g: lateral vs longitudinal acceleration in g, coloured by speed."""
    import plotly.graph_objects as go
    ax_g = np.asarray(ax_g, float).reshape(-1)
    ay_g = np.asarray(ay_g, float).reshape(-1)
    is_ratio, ring_step, max_ring, rings = _rings_and_lim(
        [(ax_g, ay_g)], "g", max_ring, ring_step)
    lim = max_ring * 1.16

    fig = go.Figure()
    _add_rings_and_guides(fig, lim, rings, max_ring, is_ratio)

    marker = dict(size=point_size, opacity=opacity, line=dict(width=0))
    cols, extra = [], ["a<sub>x</sub> = %{y:.2f} g", "a<sub>y</sub> = %{x:.2f} g"]
    has_cbar = False
    if speed is not None:
        sp = np.asarray(speed, float).reshape(-1)
        marker.update(color=sp, colorscale="Turbo", showscale=True,
                      colorbar=dict(title=dict(text="speed [m/s]", font=dict(color="white")),
                                    tickfont=dict(color="white"), outlinewidth=0,
                                    thickness=14, len=0.7))
        extra.append("speed = %{customdata[" + str(len(cols)) + "]:.1f} m/s")
        cols.append(sp); has_cbar = True
    else:
        marker.update(color="#FF8C42")
    if s is not None:
        extra.append("s = %{customdata[" + str(len(cols)) + "]:.0f} m")
        cols.append(np.asarray(s, float).reshape(-1))
    cd = np.column_stack(cols) if cols else None
    ht = "vehicle<br>" + "<br>".join(extra) + "<extra></extra>"

    fig.add_trace(go.Scattergl(x=ay_g, y=ax_g, mode="markers", name="vehicle",
                               marker=marker, customdata=cd, hovertemplate=ht))
    if show_hull:
        loop = _hull_loop(ax_g, ay_g)
        if loop is not None:
            fig.add_trace(go.Scatter(x=loop[:, 1], y=loop[:, 0], mode="lines",
                                     showlegend=False, hoverinfo="skip",
                                     line=dict(color="rgba(255,255,255,0.85)", width=1.6,
                                               dash="dash")))
    _apply_theme(fig, lim, rings, is_ratio, title, size, show_legend=False,
                 right_margin=110 if has_cbar else 40)
    return fig


# ---------------------------------------------------------------------------
# pipeline entry point
# ---------------------------------------------------------------------------
def generate_gg_plots(data=None, outdir=".", *, tyre=None, vehicle=None,
                      embed=True, s=None, size=820,
                      friction_filename="friction_circle_gg.html",
                      gg_filename="gg_diagram.html"):
    """Build and write BOTH plots into `outdir`. Returns {key: path}.

    data    : your solution object (used by the extraction helpers below).
    tyre    : optional {"FL": (gx, gy), ...} to bypass per-tyre extraction.
    vehicle : optional {"ax_g":.., "ay_g":.., "speed":.., "s":..} to bypass
              vehicle extraction.
    embed   : True -> standalone HTML (embeds plotly.js, like your current file);
              False -> "cdn" (tiny HTML that loads plotly from the web).
    """
    os.makedirs(outdir, exist_ok=True)
    if tyre is None:
        tyre = build_series_from_solution(data)
    if vehicle is None:
        vehicle = vehicle_gg_from_solution(data)
    incl = True if embed else "cdn"
    out = {}

    fig1 = plot_friction_circle_plotly(tyre, units="ratio",
                                       s=vehicle.get("s", s), size=size)
    p1 = os.path.join(outdir, friction_filename)
    fig1.write_html(p1, include_plotlyjs=incl)
    out["friction_circle_gg"] = p1

    fig2 = plot_gg_vehicle_plotly(vehicle["ax_g"], vehicle["ay_g"],
                                  speed=vehicle.get("speed"),
                                  s=vehicle.get("s", s), size=size)
    p2 = os.path.join(outdir, gg_filename)
    fig2.write_html(p2, include_plotlyjs=incl)
    out["gg_diagram"] = p2

    print("gg_plots: wrote\n  %s\n  %s" % (p1, p2))
    return out


# ---------------------------------------------------------------------------
# data extraction  --  field-name map for the Python port (MLTP.py / plotSDI.py):
#   `data` is a SimpleNamespace; `data.vehicle` is a dict of flat (N+1,) arrays.
#     per-tyre forces : fx_<w> / fy_<w> / fz_<w>   (w in fl,fr,rl,rr)  [present]
#     per-tyre mu     : mu_<w>_x / mu_<w>_y        [ADD to veh_syms in MLTP.py]
#     vehicle accel   : Lon_acc / Lat_acc in m/s^2                     [present]
#     vx at knots     : data.x_opt[0, :]   |   s at knots: from data.s_full
# ---------------------------------------------------------------------------
_G = 9.81


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):                     # scipy mat_struct (from .mat)
        return {f: getattr(obj, f) for f in obj._fieldnames}
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def _field(data, name):
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def _vehicle(data):
    v = _field(data, "vehicle")
    if v is None:
        raise AttributeError("gg_plots: no 'vehicle' on `data`; pass tyre=/vehicle= directly.")
    return _as_dict(v)


def _get(veh, name):
    obj = getattr(veh[name], "Data", veh[name])         # tolerate timeseries-like
    return np.asarray(obj, float).reshape(-1)


def build_series_from_solution(data):
    """Per-tyre (gx, gy). Uses mu_<w>_x / mu_<w>_y if present (unit circle = grip
    limit); otherwise falls back to Fx/Fz, Fy/Fz (force coefficients) and says so."""
    v = _vehicle(data)
    out = {}
    for t in ("fl", "fr", "rl", "rr"):
        fx, fy, fz = _get(v, "fx_" + t), _get(v, "fy_" + t), _get(v, "fz_" + t)
        if ("mu_" + t + "_x") in v and ("mu_" + t + "_y") in v:
            mux, muy = _get(v, "mu_" + t + "_x"), _get(v, "mu_" + t + "_y")
            out[t.upper()] = (fx / (mux * fz), fy / (muy * fz))
        else:
            print("gg_plots: mu_%s_x/_y missing from data.vehicle - plotting Fx/Fz, "
                  "Fy/Fz (force coefficients). Add mu_*_x/_y to veh_syms in MLTP.py "
                  "for a true unit-circle friction plot." % t)
            out[t.upper()] = (fx / fz, fy / fz)
    return out


def vehicle_gg_from_solution(data):
    """Vehicle g-g in g from Lon_acc / Lat_acc (m/s^2). Speed (vx at knots) and s
    are optional, used for colour and hover."""
    v = _vehicle(data)
    out = {"ax_g": _get(v, "Lon_acc") / _G, "ay_g": _get(v, "Lat_acc") / _G}
    n = out["ax_g"].size
    x_opt = _field(data, "x_opt")
    if x_opt is not None:
        try:
            out["speed"] = np.asarray(x_opt, float)[0, :]
        except Exception:
            pass
    s_full = _field(data, "s_full")
    if s_full is not None:
        sf = np.asarray(s_full, float).reshape(-1)
        out["s"] = np.linspace(sf[0], sf[-1], n)
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic demo so the module runs standalone
    def synth(n=1400, lat=1.0, lon=1.0, seed=0):
        rng = np.random.default_rng(seed)
        th = np.concatenate([
            rng.normal(0, 0.55, int(n * 0.34)), rng.normal(np.pi, 0.55, int(n * 0.30)),
            rng.normal(np.pi / 2, 0.65, int(n * 0.18)) * rng.choice([-1, 1], int(n * 0.18)),
            rng.uniform(-np.pi, np.pi, int(n * 0.18))])
        rho = np.clip(rng.beta(2.4, 1.7, th.size) * (0.55 + 0.45 * np.abs(np.cos(th))), 0, 1)
        return rho * np.cos(th) * lon, rho * np.sin(th) * lat

    tyre = {"FL": synth(1.0, 0.8, 1), "FR": synth(1.0, 0.78, 2),
            "RL": synth(0.85, 1.0, 3), "RR": synth(0.85, 1.0, 4)}
    axg, ayg = synth(1400, 4.2, 3.4, 7)
    speed = 60 + 25 * np.cos(np.linspace(0, 6 * np.pi, axg.size))
    s_demo = np.linspace(0, 4655, axg.size)

    generate_gg_plots(outdir=".", tyre=tyre,
                      vehicle={"ax_g": axg, "ay_g": ayg, "speed": speed, "s": s_demo})
