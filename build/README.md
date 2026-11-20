# Building the FullModelSim app

One spec, `build/app.spec`, builds a onedir app on Windows and on Linux.

## Build locally

Run all commands from the repository root.

```
uv sync
uv run pyinstaller build/app.spec
```

Output on Windows: `dist\FullModelSim\FullModelSim.exe`.
Output on Linux: `dist/FullModelSim/FullModelSim`.
Resources are in the `_internal` folder next to the program.

On Linux, install these libraries before the build. PyInstaller bundles them, but not libGL, libEGL and libxcb:
`libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1
libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libegl1 libgl1 libfontconfig1 libdbus-1-3`.

The Linux build took about 40 s on the dev machine. The folder is 513 MB. The spec leaves out the Qt GTK3 platform theme and the libraries that only it needs. These are GTK,
pango, cairo, atk and some X11 libraries. So the file dialogs and the style are those of Qt. Before that cut the folder was 533 MB and the `tar.gz` 192 MiB.
The spec collects only what the app reads. From Qt, these are QtCore, QtGui, QtWidgets, their plugins and the Qt libraries of the plugins.
The data are the circuit files of the app, `Data/DATA_AA.mat` and the presets. The spec leaves out the casadi
C headers (the frozen app runs without JIT) and the scipy tests.
Delete `dist/` and `build/app/` after a test. Git ignores them.

### Build options

- **Windows window mode.** The exe has no console window. The IPOPT log goes to the GUI log pane.
  Set `FMS_CONSOLE=1` at build time for a debug build with a console. Linux always has a console.
- **Windows version data.** `build/version_info.txt` goes into the exe. Linux builds have no version data.
- **Icon.** Put `build/app.ico` in place and the Windows build uses it. No icon ships now.
- **Coin-HSL.** Set `COINHSL_DIR` to the folder of the Coin-HSL libraries before the build.
  The spec then bundles the `.dll` files (Windows) or the `.so` files (Linux) into the `hsl` folder
  of the bundle. Without it, the app uses MUMPS. The CI release never sets it.
- **Unhandled errors.** `headless_solve.main` catches solve failures and prints the traceback.
  It exits with a non-zero code. The GUI shows no modal dialog from the solve process.

## Release with CI

Push a tag that starts with `v`:

```
git tag v1.1.0
git push origin v1.1.0
```

The workflow `.github/workflows/release.yml` then does these steps on Windows and on Ubuntu 22.04:

1. It runs `uv sync --locked`, the quick tests and `uv run pyinstaller build/app.spec`.
2. It runs three smoke checks on the frozen app. A non-zero exit fails the job.
   - `--headless` on a QSS config (circuit Sturn). The result file must exist.
   - `--headless` on an NLP config with `max_iter` 1. This loads casadi, IPOPT and MUMPS.
   - `--smoke` with the offscreen Qt platform. The window opens and closes after 1 s.
3. It packs `FullModelSim-<tag>-windows.zip` and `FullModelSim-<tag>-linux.tar.gz`.
4. On Linux, it extracts the archive in an `ubuntu:22.04` container.
   The container gets only `libgl1`, `libegl1` and `libxcb1` from apt.
   It runs the three smoke checks again: the headless QSS, the headless NLP and the offscreen GUI start.
   It also runs `ldd` on the bundled `libqxcb.so`. Any line with `not found` fails the job.
   The container has no X server, so no check opens a real window.
5. The release job creates the GitHub Release with both archives.
   A tag with a `-` (for example `v1.1.0-rc1`) gives a prerelease.

The release has MUMPS only and no vendor data. The workflow `.github/workflows/tests.yml`
runs `tests/run_tests.py --quick` on every push and pull request, on both systems.
It also runs the setup script (`start.bat` or `start.sh`) with `--yes --no-gui --no-tests`.

## Requirements of the Linux download

- glibc 2.35 or later. The CI builds on Ubuntu 22.04.
- libGL, libEGL and libxcb from the system. A Linux desktop has them.
  On a minimal Ubuntu: `sudo apt-get install libgl1 libegl1 libxcb1`.

The bundle carries the other xcb libraries, xkbcommon, fontconfig and dbus.

## Packaging notes

- **Absolute entry point.** The spec uses `Analysis([os.path.join(ROOT, "app", "main.py")])`.
  PyInstaller resolves a relative script path against the folder of the spec (`build/`).
  A bare `app/main.py` fails with "script not found".
- **casadi DLL hook (`build/rthook_casadi.py`).** On Windows, the loader does not search
  `_internal\casadi\` for `_casadi.pyd`. The hook adds that folder to the DLL search path
  before casadi loads. On Linux the hook only adds that folder to `PATH`. The Linux loader
  does not read `PATH`, and casadi loads without the hook.
- **HSL in a frozen app.** `resolve_hsl_dir()` in `functions/hsl.py` searches in this order:
  `COINHSL_DIR` env, explicit argument (the Coin-HSL field of the GUI), the `hsl` folder of
  the frozen bundle (`sys._MEIPASS/hsl`). If HSL does not load, the solve uses MUMPS.

## Acceptance floor

Run the checks on a machine without Python. Use a clean Windows VM and a clean Ubuntu 22.04 install.

1. Copy or extract the `FullModelSim` folder and start the program.
   On Linux, start `./FullModelSim`.
2. Main tab: circuit `Sturn`, Aero `Static`, ATD on. Advanced tab: linear solver `mumps`.
3. Click Run. The IPOPT log shows in the log pane.
4. When the solve ends, the summary shows a lap time. The Output tab lists the plots.
   A double-click opens a plot in the browser.
   Results and plots go to the `FullModelSim` folder of the user's documents folder.
5. Advanced tab: set Run to `qss` and click Run. After about 1 s, the summary shows
   "QSS estimate (not an optimum)" and a lap time. The Output tab lists no plots.

On Linux, run the QSS check headless in a terminal:
`QT_QPA_PLATFORM=offscreen ./FullModelSim --headless cfg.json`.
The CI container run covers the headless QSS, one NLP iteration and the offscreen GUI start.
It does not cover a window on a real X server. Do steps 1 to 5 by hand for that.
The Windows frozen app has no run on the dev machine. Its first run is the CI smoke test.

## Earlier Windows check of the HSL bundle (old record)

This record is from the Windows-only spec, with the legacy tyre set and the libraries at the bundle root.
It is not a check of the current spec. Its lap time (25.574 s) is not the current Sturn lap.

`FullModelSim.exe --headless cfg.json` with `circuit=Sturn` and `linear_solver=ma57`:
casadi loaded from the bundle, IPOPT ran `ma57` from the bundled Coin-HSL,
and the solve ended with `Optimal Solution Found`.
A run with a stripped `PATH` gave the same result, so no DLL came from the dev machine.
The `hsl` folder of the current spec has no run with a real Coin-HSL library yet.
