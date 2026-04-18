@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "APP_NAME=luno_trader"
set "ENTRY_FILE=app.py"
set "DIST_DIR=dist"
set "BUILD_DIR=build"
set "SPEC_FILE=%APP_NAME%.spec"
set "ICON_FILE=%~dp0app_icon.ico"
set "PY_CMD="
call :resolve_python
if not defined PY_CMD (
    echo [ERROR] No working Python interpreter was found.
    echo.
    echo Try one of these:
    echo 1. Activate your venv, then run build_win.bat again.
    echo 2. Install Python from python.org and make sure it is available in PATH.
    echo 3. Edit this file and set PY_CMD to your full python.exe path.
    exit /b 1
)

call %PY_CMD% -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller is not installed for this Python environment.
    echo Installing PyInstaller...
    call %PY_CMD% -m pip install pyinstaller
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install PyInstaller automatically.
        echo Try running: %PY_CMD% -m pip install pyinstaller
        exit /b 1
    )

    call %PY_CMD% -m PyInstaller --version >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] PyInstaller still is not available after installation.
        exit /b 1
    )
)

set "DATA_ARGS="
call :add_data "luno_trader_config.json"
call :add_data "trade_audit_history.json"
call :add_data "license.key"
call :add_data "public_key.pem"
call :add_data "private_key.pem"
call :set_icon_arg

echo Building %APP_NAME%.exe from %ENTRY_FILE%...
call %PY_CMD% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name "%APP_NAME%" ^
  %ICON_ARG% ^
  %DATA_ARGS% ^
  "%ENTRY_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo Build complete.
echo EXE: "%~dp0%DIST_DIR%\%APP_NAME%\%APP_NAME%.exe"
exit /b 0

:set_icon_arg
if exist "%ICON_FILE%" (
    set "ICON_ARG=--icon ""%ICON_FILE%"""
) else (
    set "ICON_ARG="
)
goto :eof

:resolve_python
if defined VIRTUAL_ENV (
    call :try_python_path "%VIRTUAL_ENV%\Scripts\python.exe"
    if defined PY_CMD goto :eof
)

for %%P in (
    "%~dp0.venv\Scripts\python.exe"
    "%~dp0venv\Scripts\python.exe"
    "%~dp0..\.venv\Scripts\python.exe"
    "%~dp0..\venv\Scripts\python.exe"
    "%USERPROFILE%\Desktop\Projects\applicant-repo\venv\Scripts\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "%ProgramFiles%\Python39\python.exe"
) do (
    call :try_python_path "%%~P"
    if defined PY_CMD goto :eof
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py -3"
        goto :eof
    )
)

for %%P in (python python3) do (
    call :try_python_cmd %%P
    if defined PY_CMD goto :eof
)

goto :eof

:try_python_path
if not exist %1 goto :eof
call %1 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=%1"
)
goto :eof

:try_python_cmd
set "CANDIDATE=%~1"
where %CANDIDATE% >nul 2>nul || goto :eof
for /f "delims=" %%I in ('where %CANDIDATE% 2^>nul') do (
    call :try_python_path "%%~fI"
    if defined PY_CMD goto :eof
)
goto :eof

:add_data
if exist "%~1" (
    set "DATA_ARGS=%DATA_ARGS% --add-data ""%~1;."""
)
goto :eof
