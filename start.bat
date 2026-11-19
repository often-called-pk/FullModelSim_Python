@echo off
rem One command setup for Windows. Extra arguments go to start.py.
cd /d "%~dp0"
py -3 --version >nul 2>&1
if not errorlevel 1 (
    py -3 start.py %*
    goto done
)
rem The Microsoft Store alias named python fails this check.
python --version >nul 2>&1
if not errorlevel 1 (
    python start.py %*
    goto done
)
where uv >nul 2>&1
if not errorlevel 1 (
    uv run --no-project python start.py %*
    goto done
)
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    "%USERPROFILE%\.local\bin\uv.exe" run --no-project python start.py %*
    goto done
)
echo error: Python and uv not found
echo fix: install Python 3.10 or newer from https://www.python.org/downloads/ and run this script again
pause
exit /b 1
:done
set RC=%errorlevel%
if not "%RC%"=="0" pause
exit /b %RC%
