"""plotSDI.py - Plotly replacement for plotSDI.m (Simulink Data Inspector).

The MATLAB original streams every channel into Simulink's Simulation Data
Inspector. Plotly has no SDI equivalent, so this module reproduces the same
*information* as standalone interactive HTML figures:

  * racing line coloured by velocity (with track boundaries + centreline)
  * speed & longitudinal/lateral acceleration vs distance
  * per-wheel tyre forces (fx, fy, fz)
  * friction-circle usage (rho_lim path constraints)
  * suspension: heave / pitch / roll
  * powertrain: wheel torques, motor power & speed, cumulative energy
  * control inputs

Usage:
    from plotSDI import plotSDI, plot_racing_line
    plotSDI(ctx)                          # after a MLTP solve (uses ctx.data)
    plotSDI("Results/Sturn_Static_ATDOff_EM4Off.mat")   # from a saved file

    from functions.importfile import load_solution
    plot_racing_line(load_solution(path)).show()

Figures are written to Plots/<circuit>/<config>/<name>.html and also returned in
a dict so a caller can display them inline.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from types import SimpleNamespace


# --------------------------------------------------------------------------- 
# Accessors that work with dict / SimpleNamespace / scipy mat_struct
# --------------------------------------------------------------------------- 
def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, SimpleNamespace):
        return vars(obj)
    if hasattr(obj, "_fieldnames"):
        return {f: getattr(obj, f) for f in obj._fieldnames}
    return obj


def _get(obj, name, default=None):
    d = _as_dict(obj)
    return d.get(name, default) if isinstance(d, dict) else default


def _arr(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _resolve_data(source):
    """Return the solution data object from a ctx, a path string, or a data struct."""
    if isinstance(source, str):
        from functions.importfile import load_solution
        return load_solution(source)
    if hasattr(source, "data"):          # a Ctx
        return source.data
    return source                        # already a data namespace/dict


def _knot_grid(data):
    s_full = _arr(_get(data, "s_full"))
    N = int(_get(data, "N", (s_full.size - 1) // (int(_get(data, "OPT_d", 3)) + 1)))
    return np.linspace(s_full[0], s_full[-1], N + 1), s_full


# --------------------------------------------------------------------------- 
# Individual figures
# --------------------------------------------------------------------------- 
def plot_racing_line(data):
    """Racing line coloured by velocity, with track boundaries and centreline."""
    track = _as_dict(_get(data, "track"))
    xopt = _arr(track.get("xopt")); yopt = _arr(track.get("yopt"))

    # velocity along the full grid (for colour) if available
    x_full = _get(data, "x_full")
    if x_full is not None:
        vx_full = np.asarray(x_full, dtype=float)[0, :]
        colour = vx_full[:xopt.size] if vx_full.size >= xopt.size else None
    else:
        colour = None

    fig = go.Figure()
    # track boundaries
    for key, nm in (("Xl", "left limit"), ("Xr", "right limit")):
        XY = track.get(key)
        if XY is not None:
            XY = np.asarray(XY, dtype=float)
            fig.add_trace(go.Scatter(x=XY[:, 0], y=XY[:, 1], mode="lines",
                                     line=dict(color="#888", width=1), name=nm))
    # centreline
    if track.get("x") is not None and track.get("y") is not None:
        fig.add_trace(go.Scatter(x=_arr(track["x"]), y=_arr(track["y"]), mode="lines",
                                 line=dict(color="#ccc", width=1, dash="dash"),
                                 name="centreline"))
    # racing line
    if colour is not None:
        fig.add_trace(go.Scatter(
            x=xopt, y=yopt, mode="markers",
            marker=dict(size=4, color=colour, colorscale="Turbo",
                        colorbar=dict(title="v [m/s]")),
            name="racing line"))
    else:
        fig.add_trace(go.Scatter(x=xopt, y=yopt, mode="lines",
                                 line=dict(color="#e24b4a", width=2), name="racing line"))
    lap = _get(data, "lap_time")
    title = "Racing line" + (f"  —  lap time {float(lap):.3f} s" if lap is not None else "")
    fig.update_layout(title=title, xaxis_title="x [m]", yaxis_title="y [m]",
                      template="plotly_white")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_speed(data):
    s_knot, s_full = _knot_grid(data)
    veh = _as_dict(_get(data, "vehicle"))
    x_full = _get(data, "x_full")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Speed", "Acceleration"))
    if x_full is not None:
        fig.add_trace(go.Scatter(x=s_full, y=np.asarray(x_full, float)[0, :],
                                 name="vx", line=dict(color="#378ADD")), row=1, col=1)
    if "Lon_acc" in veh:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh["Lon_acc"]), name="long. acc"),
                      row=2, col=1)
    if "Lat_acc" in veh:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh["Lat_acc"]), name="lat. acc"),
                      row=2, col=1)
    fig.update_xaxes(title_text="s [m]", row=2, col=1)
    fig.update_yaxes(title_text="v [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="a [m/s²]", row=2, col=1)
    fig.update_layout(template="plotly_white", title="Speed & acceleration")
    return fig


def plot_tyre_forces(data):
    s_knot, _ = _knot_grid(data)
    veh = _as_dict(_get(data, "vehicle"))
    wheels = ["fl", "fr", "rl", "rr"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Longitudinal fx", "Lateral fy", "Vertical fz"))
    for comp, row in (("fx", 1), ("fy", 2), ("fz", 3)):
        for w in wheels:
            key = f"{comp}_{w}"
            if key in veh:
                fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]), name=key),
                              row=row, col=1)
    fig.update_xaxes(title_text="s [m]", row=3, col=1)
    fig.update_layout(template="plotly_white", title="Tyre forces [N]")
    return fig


def plot_friction(data):
    s_knot, _ = _knot_grid(data)
    con = _as_dict(_get(data, "constraints"))
    fig = go.Figure()
    for key in con:
        if key.startswith("rho_lim"):
            fig.add_trace(go.Scatter(x=s_knot, y=_arr(con[key]), name=key))
    fig.add_hline(y=1.0, line=dict(color="#e24b4a", dash="dash"),
                  annotation_text="grip limit")
    fig.update_layout(template="plotly_white", title="Friction-circle usage",
                      xaxis_title="s [m]", yaxis_title="rho (≤ 1)")
    return fig


def plot_suspension(data):
    s_knot, _ = _knot_grid(data)
    veh = _as_dict(_get(data, "vehicle"))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Heave zs", "Pitch θ", "Roll φ"))
    for key, row, unit in (("zs", 1, "m"), ("theta", 2, "rad"), ("phi", 3, "rad")):
        if key in veh:
            fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]), name=key), row=row, col=1)
            fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="s [m]", row=3, col=1)
    fig.update_layout(template="plotly_white", title="Suspension")
    return fig


def plot_powertrain(data):
    s_knot, _ = _knot_grid(data)
    veh = _as_dict(_get(data, "vehicle"))
    t_opt = _arr(_get(data, "t_opt"))
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Wheel torque [Nm]", "Motor power [kW]", "Motor speed [rad/s]", "Energy [kWh]"))
    for w in ["fl", "fr", "rl", "rr"]:
        if f"T_{w}" in veh:
            fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[f"T_{w}"]), name=f"T_{w}"),
                          row=1, col=1)
    # power
    for key in [k for k in veh if k.startswith("P_motor")]:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]) * 1e-3, name=key), row=1, col=2)
    # speed
    for key in [k for k in veh if k.startswith("Om_motor")]:
        fig.add_trace(go.Scatter(x=s_knot, y=_arr(veh[key]), name=key), row=2, col=1)
    # energy
    if "E_motor" in veh:
        fig.add_trace(go.Scatter(x=t_opt, y=_arr(veh["E_motor"]), name="E_motor"),
                      row=2, col=2)
        fig.update_xaxes(title_text="t [s]", row=2, col=2)
    fig.update_layout(template="plotly_white", title="Powertrain")
    return fig


def plot_inputs(data):
    s_knot, _ = _knot_grid(data)
    u_opt = _get(data, "u_opt")
    keys = _get(data, "input_keys")
    fig = go.Figure()
    if u_opt is not None:
        u_opt = np.asarray(u_opt, dtype=float)
        if keys is None:
            keys = [f"u{i}" for i in range(u_opt.shape[0])]
        keys = [str(k) for k in np.asarray(keys).reshape(-1)]
        for i in range(u_opt.shape[0]):
            fig.add_trace(go.Scatter(x=s_knot, y=u_opt[i, :], name=keys[i]))
    fig.update_layout(template="plotly_white", title="Control inputs",
                      xaxis_title="s [m]")
    return fig


# --------------------------------------------------------------------------- 
# Top-level: build, save, return all figures
# --------------------------------------------------------------------------- 
def plotSDI(source, save_dir="Plots", show=False):
    data = _resolve_data(source)

    circuit = str(_get(data, "circuit", "track"))
    cfg = f"{_get(data, 'AeroConfig', 'cfg')}_ATD{_get(data, 'ATD', '')}_EM4{_get(data, 'EM4', '')}"
    out_dir = os.path.join(save_dir, circuit, cfg)
    os.makedirs(out_dir, exist_ok=True)

    figs = {
        "racing_line": plot_racing_line(data),
        "speed": plot_speed(data),
        "tyre_forces": plot_tyre_forces(data),
        "friction_circle": plot_friction(data),
        "suspension": plot_suspension(data),
        "powertrain": plot_powertrain(data),
        "inputs": plot_inputs(data),
    }
    for name, fig in figs.items():
        path = os.path.join(out_dir, f"{name}.html")
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        if show:
            fig.show()
    # --- g-g / friction-circle plots (added) -------------------------------
    try:
        from gg_plots import generate_gg_plots
        generate_gg_plots(data, out_dir)          # writes friction_circle_gg.html + gg_diagram.html
    except Exception as exc:
        print(f"[plotSDI] gg plots skipped: {exc}")
    print(f"[plotSDI] wrote {len(figs)} figures -> {out_dir}")
    return figs


if __name__ == "__main__":
    import sys
    plotSDI(sys.argv[1] if len(sys.argv) > 1 else "Results")
