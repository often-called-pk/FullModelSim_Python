# Building the FullModelSim Windows app

## Build (onedir, for debugging)

```powershell
venv\Scripts\Activate.ps1
# optional: bundle Coin-HSL DLLs
$env:COINHSL_DIR = "C:\path\to\coinhsl\bin"
pyinstaller build\windows-app.spec
```

Output: `dist\FullModelSim\FullModelSim.exe`.

## Patch needed for HSL in a frozen app

`functions/hsl.py` must also look for the HSL DLL directory at the frozen
resource root. Add `sys._MEIPASS` to its search order (alongside `COINHSL_DIR`
and the seeded default). If HSL still fails to load, the solve falls back to
MUMPS automatically — the app always solves.

## Acceptance floor (manual, on a clean Windows VM with no Python)

1. Copy `dist\FullModelSim\` to the VM and launch `FullModelSim.exe`.
2. Main tab: circuit `Sturn`, Aero `Static`, ATD on, linear solver `mumps`.
3. Run. The IPOPT iteration log streams into the log pane.
4. On finish: summary shows a lap time; the Output tab lists plots; double-click
   opens one in the browser. Results/Plots are written under
   `%USERPROFILE%\Documents\FullModelSim`.

HSL working in the frozen app is a stretch goal beyond this floor.
