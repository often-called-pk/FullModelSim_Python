"""Parse a saved solve .mat into summary numbers and list its Plotly plots.
Path scheme mirrors MLTP.save (Results/<result_stem>.mat) and plotSDI
(Plots/<circuit>/<cfg>/<name>.html), with cfg = <AeroConfig>_ATD<ATD>_EM4<EM4>.
``setup_tag`` is the 8 hex digits of the setup knobs that differ from their defaults
(RunConfig.setup_tag()). It is "" for the default setup, whose stem has no tag.
``result_paths(rc)`` maps a RunConfig to the two paths that its solve writes.
"""
import glob
import os
import numpy as np
import scipy.io as sio

from functions.importfile import result_stem, n_tag


def _cfg(AeroConfig, ATD, EM4):
    return f"{AeroConfig}_ATD{ATD}_EM4{EM4}"


def result_mat_path(output_dir, circuit, AeroConfig, ATD, EM4,
                    tyre_set="MF205", mesh_requested="auto", setup_tag="", qss=False, **stem_kw):
    """stem_kw = the physics, start_state and n_tag arguments of result_stem.
    qss: the MLTP_screen file, <stem>_qss.mat."""
    stem = result_stem(circuit, _cfg(AeroConfig, ATD, EM4), tyre_set, mesh_requested, setup_tag=setup_tag, **stem_kw)
    return os.path.join(output_dir, "Results", f"{stem}{'_qss' if qss else ''}.mat")


def plot_dir(output_dir, circuit, AeroConfig, ATD, EM4,
             tyre_set="MF205", mesh_requested="auto", setup_tag="", **stem_kw):
    # plotSDI names the folder after the result stem minus the "<circuit>_" prefix
    stem = result_stem(circuit, _cfg(AeroConfig, ATD, EM4), tyre_set, mesh_requested, setup_tag=setup_tag, **stem_kw)
    return os.path.join(output_dir, "Plots", circuit, stem[len(circuit) + 1:])


def result_paths(rc):
    """(.mat path, plot folder) that the solve of the RunConfig `rc` writes. The stem follows the tyre set, mesh,
    physics, start state, track band and setup tag. Mode 'qss' writes <stem>_qss.mat and no plots. Its plot folder
    is "", so the Output tab does not show the plots of an NLP run of the same setup."""
    a = (rc.output_dir, rc.circuit, rc.AeroConfig, rc.ATD, rc.Electric_4Motors, rc.tyre_set, rc.mesh, rc.setup_tag())
    kw = dict(physics=rc.physics, start_state=rc.start_state, n_tag=n_tag(rc.n_bounds))
    qss = rc.mode == "qss"
    return result_mat_path(*a, qss=qss, **kw), "" if qss else plot_dir(*a, **kw)


def parse_summary(mat_path):
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = raw["data"]
    lap_time = float(np.asarray(data.lap_time).reshape(-1)[-1])
    energy = None
    veh = getattr(data, "vehicle", None)
    if veh is not None and hasattr(veh, "E_motor"):
        energy = float(np.asarray(veh.E_motor).reshape(-1)[-1]) * 1e-3     # stored in Wh
    return {"lap_time_s": lap_time, "energy_kWh": energy}


def list_plots(plot_directory):
    if not os.path.isdir(plot_directory):
        return []
    return sorted(glob.glob(os.path.join(plot_directory, "*.html")))
