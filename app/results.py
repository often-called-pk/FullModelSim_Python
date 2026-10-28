"""Parse a saved solve .mat into summary numbers and list its Plotly plots.
Path scheme mirrors MLTP.save (Results/<result_stem>.mat) and plotSDI
(Plots/<circuit>/<cfg>/<name>.html), with cfg = <AeroConfig>_ATD<ATD>_EM4<EM4>.
``setup_tag`` is the 8 hex digits of the setup knobs that differ from their defaults
(RunConfig.setup_tag()). It is "" for the default setup, whose stem has no tag.
"""
import glob
import os
import numpy as np
import scipy.io as sio

from functions.importfile import result_stem


def _cfg(AeroConfig, ATD, EM4):
    return f"{AeroConfig}_ATD{ATD}_EM4{EM4}"


def result_mat_path(output_dir, circuit, AeroConfig, ATD, EM4,
                    tyre_set="MF205", mesh_requested="auto", setup_tag=""):
    stem = result_stem(circuit, _cfg(AeroConfig, ATD, EM4), tyre_set, mesh_requested, setup_tag=setup_tag)
    return os.path.join(output_dir, "Results", f"{stem}.mat")


def plot_dir(output_dir, circuit, AeroConfig, ATD, EM4,
             tyre_set="MF205", mesh_requested="auto", setup_tag=""):
    # plotSDI names the folder after the result stem minus the "<circuit>_" prefix
    stem = result_stem(circuit, _cfg(AeroConfig, ATD, EM4), tyre_set, mesh_requested, setup_tag=setup_tag)
    return os.path.join(output_dir, "Plots", circuit, stem[len(circuit) + 1:])


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
