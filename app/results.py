"""Parse a saved solve .mat into summary numbers and list its Plotly plots.
Path scheme mirrors MLTP.save (Results/<circuit>_<cfg>.mat) and plotSDI
(Plots/<circuit>/<cfg>/<name>.html), with cfg = <AeroConfig>_ATD<ATD>_EM4<EM4>.
"""
import glob
import os
import numpy as np
import scipy.io as sio


def _cfg(AeroConfig, ATD, EM4):
    return f"{AeroConfig}_ATD{ATD}_EM4{EM4}"


def result_mat_path(output_dir, circuit, AeroConfig, ATD, EM4):
    return os.path.join(output_dir, "Results",
                        f"{circuit}_{_cfg(AeroConfig, ATD, EM4)}.mat")


def plot_dir(output_dir, circuit, AeroConfig, ATD, EM4):
    return os.path.join(output_dir, "Plots", circuit, _cfg(AeroConfig, ATD, EM4))


def parse_summary(mat_path):
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = raw["data"]
    lap_time = float(np.asarray(data.lap_time).reshape(-1)[-1])
    energy = None
    veh = getattr(data, "vehicle", None)
    if veh is not None and hasattr(veh, "E_motor"):
        energy = float(np.asarray(veh.E_motor).reshape(-1)[-1])
    return {"lap_time_s": lap_time, "energy_kWh": energy}


def list_plots(plot_directory):
    if not os.path.isdir(plot_directory):
        return []
    return sorted(glob.glob(os.path.join(plot_directory, "*.html")))
