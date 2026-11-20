"""Tests for start.py, the one-command setup (plain script, no pytest).

Imports start.py as a module (it must not run main on import) and checks its helpers:
find_uv, install_uv, ask, xcb_missing, the HSL probe texts, the argparse flags and the
--yes prompt skip, and the --quick list of tests/run_tests.py. run_tests.py is read with
ast, not imported (it runs its tests on import). Every outside call is patched: no
network, no real uv, no subprocess runs the real uv.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import argparse
import ast
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import functions.hsl as hsl

ROOT = _bootstrap.ROOT
TESTS = Path(_bootstrap.TESTS_DIR)

_spec = importlib.util.spec_from_file_location("start", os.path.join(ROOT, "start.py"))
start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start)

PASSED = 0


def ok(name, cond):
    global PASSED
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASSED += 1


class Stop(Exception):
    """Raised by a patched install step to end main() early."""


def exits(fn):
    """Run fn. Return (exited, exit code, stdout text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn()
        except SystemExit as exc:
            return True, exc.code, out.getvalue()
    return False, None, out.getvalue()


def fake_uv(folder):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / ("uv.exe" if start.WIN else "uv")
    p.write_text("#!/bin/sh\necho uv 0.0.0\n", encoding="ascii")
    p.chmod(0o755)
    return p


def find_with(extra, base):
    """find_uv with a clean environment: empty PATH, HOME without uv, plus extra."""
    env = {"HOME": str(base / "nohome"), "PATH": str(base / "nopath")}
    env.update({k: str(v) for k, v in extra.items()})
    env["USERPROFILE"] = env["HOME"]  # Path.home() reads USERPROFILE on Windows
    with mock.patch.dict(os.environ, env, clear=True):
        return start.find_uv()


def section_find_uv(base):
    print("1. find_uv")
    (base / "nopath").mkdir(exist_ok=True)
    (base / "nohome").mkdir(exist_ok=True)

    p1 = fake_uv(base / "p1")
    ok("found through PATH", find_with({"PATH": base / "p1"}, base) == str(p1))
    i1 = fake_uv(base / "i1")
    ok("found in UV_INSTALL_DIR", find_with({"UV_INSTALL_DIR": base / "i1"}, base) == str(i1))
    i2 = fake_uv(base / "i2" / "bin")
    ok("found in UV_INSTALL_DIR/bin", find_with({"UV_INSTALL_DIR": base / "i2"}, base) == str(i2))
    xb = fake_uv(base / "xbin")
    ok("found in XDG_BIN_HOME", find_with({"XDG_BIN_HOME": base / "xbin"}, base) == str(xb))
    xd = fake_uv(base / "d" / "bin")
    ok("found in XDG_DATA_HOME/../bin",
       find_with({"XDG_DATA_HOME": base / "d" / "share"}, base) == str(xd))
    hm = fake_uv(base / "home2" / ".local" / "bin")
    ok("found in ~/.local/bin", find_with({"HOME": base / "home2"}, base) == str(hm))
    ok("none -> None", find_with({}, base) is None)


def section_install(base):
    print("2. install_uv")

    def no_run(*a, **k):
        raise AssertionError("subprocess.run must not run in this case")

    def no_download(*a, **k):
        raise AssertionError("urlretrieve must not run in this case")

    # POSIX: the download fails
    calls = []

    def broken(url, dest, *a, **k):
        calls.append((url, dest))
        raise OSError("network down")

    with mock.patch.object(start, "WIN", False), \
            mock.patch.object(urllib.request, "urlretrieve", broken), \
            mock.patch.object(start.subprocess, "run", no_run):
        exited, code, out = exits(start.install_uv)
    ok("POSIX download error exits", exited and code == 1)
    ok("POSIX download error message", "could not download the uv installer" in out)
    ok("POSIX download error prints the docs hint", start.UV_DOCS in out)
    ok("POSIX download fetches the official script", bool(calls) and calls[0][0] == start.UV_SH)
    ok("POSIX script goes to a temp folder, not the repo",
       bool(calls) and not str(calls[0][1]).startswith(ROOT))

    # POSIX: download works, sh runs the script, the installer exits 0
    seen = []

    def good(url, dest, *a, **k):
        Path(dest).write_text("exit 0\n", encoding="ascii")

    def ran_ok(cmd, *a, **k):
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    with mock.patch.object(start, "WIN", False), \
            mock.patch.object(urllib.request, "urlretrieve", good), \
            mock.patch.object(start.subprocess, "run", ran_ok):
        exited, code, out = exits(start.install_uv)
    ok("POSIX install runs sh on the downloaded script",
       not exited and bool(seen) and seen[0][0] == "sh" and seen[0][1].endswith("install.sh"))

    # POSIX: the installer exits 2
    def ran_bad(cmd, *a, **k):
        return SimpleNamespace(returncode=2)

    with mock.patch.object(start, "WIN", False), \
            mock.patch.object(urllib.request, "urlretrieve", good), \
            mock.patch.object(start.subprocess, "run", ran_bad):
        exited, code, out = exits(start.install_uv)
    ok("POSIX installer failure exits 1 with its message",
       exited and code == 1 and "the uv installer failed" in out)

    # Windows: the command list only, nothing runs
    wcalls = []

    def wrun(cmd, *a, **k):
        wcalls.append((cmd, k))
        return SimpleNamespace(returncode=0)

    with mock.patch.object(start, "WIN", True), \
            mock.patch.object(urllib.request, "urlretrieve", no_download), \
            mock.patch.object(start.subprocess, "run", wrun):
        exited, code, out = exits(start.install_uv)
    cmd, kw = wcalls[0]
    ok("Windows command is Astral's PowerShell line", cmd == start.UV_PS)
    ok("Windows command names install.ps1 and ExecutionPolicy ByPass",
       cmd.startswith("powershell") and "https://astral.sh/uv/install.ps1" in cmd
       and "ExecutionPolicy ByPass" in cmd)
    ok("Windows command runs through the shell", kw.get("shell") is True)
    ok("Windows success does not exit", not exited)

    with mock.patch.object(start, "WIN", True), \
            mock.patch.object(urllib.request, "urlretrieve", no_download), \
            mock.patch.object(start.subprocess, "run",
                              lambda *a, **k: SimpleNamespace(returncode=1)):
        exited, code, out = exits(start.install_uv)
    ok("Windows installer failure exits 1 with its message",
       exited and code == 1 and "the uv installer failed" in out)


def section_ask():
    print("3. ask")

    def ask_with(side_effect):
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=side_effect), \
                contextlib.redirect_stdout(out):
            ans = start.ask("Q? ")
        return ans, out.getvalue()

    ans, out = ask_with(EOFError)
    ok("EOFError gives n", ans == "n")
    ok("EOFError prints the --yes hint", "--yes" in out)

    ans, out = ask_with(RuntimeError("input(): lost sys.stdin"))
    ok("closed input (RuntimeError) gives n", ans == "n")
    ok("closed input prints the --yes hint", "--yes" in out)

    out = io.StringIO()
    with mock.patch.object(sys, "stdin", None), contextlib.redirect_stdout(out):
        ans = start.ask("Q? ")
    ok("real closed stdin (sys.stdin None) gives n", ans == "n")
    ok("real closed stdin prints the --yes hint", "--yes" in out.getvalue())

    ans, _ = ask_with(lambda prompt: "  Y \n")
    ok("answer is stripped and lower-cased", ans == "y")


def section_xcb(base):
    print("4. xcb_missing")
    lib = base / "pyside"
    (lib / "Qt" / "plugins" / "platforms").mkdir(parents=True, exist_ok=True)
    (lib / "Qt" / "plugins" / "platforms" / "libqxcb.so").write_bytes(b"")
    nolib = base / "pyside_noplugin"
    nolib.mkdir(exist_ok=True)

    def run_for(ldd_text, folder):
        def fake(cmd, *a, **k):
            if cmd[0] == "ldd":
                return SimpleNamespace(stdout=ldd_text)
            if "PySide6" in cmd[-1]:
                return SimpleNamespace(stdout=str(folder) + "\n")
            raise AssertionError(f"unexpected command {cmd}")
        return fake

    def check(ldd_text, folder=lib, system="Linux", ldd_found=True):
        def which(name):
            return "/usr/bin/ldd" if (name == "ldd" and ldd_found) else None
        with mock.patch.object(start.platform, "system", return_value=system), \
                mock.patch.object(start.shutil, "which", side_effect=which), \
                mock.patch.object(start.subprocess, "run", side_effect=run_for(ldd_text, folder)):
            return start.xcb_missing("uv")

    one = "\tlibxcb-cursor.so.0 => not found\n\tlibc.so.6 => /lib/libc.so.6 (0x7f)\n"
    ok("one 'not found' line gives that name", check(one) == ["libxcb-cursor.so.0"])
    two = one + "\tlibEGL.so.1 => not found\n"
    ok("two 'not found' lines give both names in order",
       check(two) == ["libxcb-cursor.so.0", "libEGL.so.1"])
    ok("no missing lib gives []",
       check("\tlibc.so.6 => /lib/libc.so.6 (0x7f)\n") == [])

    def boom(*a, **k):
        raise AssertionError("no subprocess in this case")

    with mock.patch.object(start.platform, "system", return_value="Windows"), \
            mock.patch.object(start.subprocess, "run", side_effect=boom):
        ok("not Linux gives None (not checked)", start.xcb_missing("uv") is None)
    with mock.patch.object(start.platform, "system", return_value="Linux"), \
            mock.patch.object(start.shutil, "which", return_value=None), \
            mock.patch.object(start.subprocess, "run", side_effect=boom):
        ok("no ldd gives None (not checked)", start.xcb_missing("uv") is None)
    ok("no Qt xcb plugin gives None (not checked)", check(one, folder=nolib) is None)


def section_hsl():
    print("5. HSL probe texts")

    def probe_text(resolve, lib, probe):
        out = io.StringIO()
        with mock.patch.object(hsl, "resolve_hsl_dir", return_value=resolve), \
                mock.patch.object(hsl, "hsllib_path", return_value=lib), \
                mock.patch.object(hsl, "probe_linear_solver", return_value=probe) as pm, \
                contextlib.redirect_stdout(out):
            exec(start.HSL_PROBE, {"__name__": "hsl_probe"})
        return out.getvalue().strip(), pm

    lib_path = os.path.join("/hsl", "libhsl.so")
    text, pm = probe_text("/hsl", lib_path, True)
    ok("found and loads", text == "Coin-HSL: found, " + lib_path)
    ok("probe runs ma57 on that library", pm.call_args[0] == ("ma57", lib_path))
    text, _ = probe_text("/hsl", lib_path, False)
    ok("found but does not load",
       text == "Coin-HSL: found, does not load: solves use MUMPS")
    text, _ = probe_text(None, None, False)
    ok("not found", text == "Coin-HSL: not found, solves use MUMPS (slower, no licence)")


def section_main():
    print("6. argparse and the --yes prompt")
    seen = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        ns = real_parse(self, *a, **k)
        seen["ns"] = ns
        return ns

    def run_main(argv, answer="n"):
        seen.clear()
        ask = mock.Mock(return_value=answer)
        install = mock.Mock(side_effect=Stop)
        with mock.patch.object(sys, "argv", ["start.py"] + argv), \
                mock.patch.object(argparse.ArgumentParser, "parse_args", spy), \
                mock.patch.object(start, "find_uv", return_value=None), \
                mock.patch.object(start, "ask", ask), \
                mock.patch.object(start, "install_uv", install):
            try:
                exited, code, out = exits(start.main)
            except Stop:
                exited, code, out = "stopped", None, ""
        return exited, code, out, ask, install

    exited, code, out, ask, install = run_main(["--yes", "--no-gui", "--no-tests"])
    ns = seen["ns"]
    ok("--yes --no-gui --no-tests parse", ns.yes and ns.no_gui and ns.no_tests)
    ok("--yes skips the prompt", ask.call_count == 0)
    ok("--yes goes straight to the install", install.call_count == 1)

    exited, code, out, ask, install = run_main([])
    ns = seen["ns"]
    ok("defaults are all off", not (ns.yes or ns.no_gui or ns.no_tests))
    ok("no flag asks once", ask.call_count == 1)
    ok("answer n exits 1 with 'uv is required'",
       exited and code == 1 and "uv is required" in out)
    ok("answer n does not install", install.call_count == 0)

    exited, code, out, ask, install = run_main([], answer="y")
    ok("answer y goes to the install", install.call_count == 1)


def section_run_tests():
    print("7. tests/run_tests.py --quick list (read, not run)")
    src = (TESTS / "run_tests.py").read_text(encoding="utf-8")
    real_ipopt = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "REAL_IPOPT"
                                                for t in node.targets):
            real_ipopt = set(ast.literal_eval(node.value))
    ok("REAL_IPOPT is a literal set", real_ipopt is not None)
    slow = {"test_setup_sweep.py", "test_paramoptim_warmstart.py", "test_ladder.py",
            "test_refine.py"}
    ok("REAL_IPOPT names the four slow files", real_ipopt == slow)
    ok("the quick filter uses REAL_IPOPT", "f.name not in REAL_IPOPT" in src)
    on_disk = {f.name for f in TESTS.glob("test_*.py")}
    ok("the four slow files exist (exclusion is not vacuous)", slow <= on_disk)
    quick = {n for n in on_disk if n not in real_ipopt}
    ok("quick list includes test_matlab_parity.py", "test_matlab_parity.py" in quick)
    ok("quick list includes test_hsl.py", "test_hsl.py" in quick)
    ok("quick list excludes the four slow files", not (slow & quick))


def main():
    with tempfile.TemporaryDirectory(prefix="fms_start_") as tmp:
        base = Path(tmp)
        section_find_uv(base)
        section_install(base)
        section_ask()
        section_xcb(base)
        section_hsl()
        section_main()
    section_run_tests()
    print(f"{PASSED} checks passed")


if __name__ == "__main__":
    main()
