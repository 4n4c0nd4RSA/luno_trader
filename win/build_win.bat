@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "APP_NAME=luno_trader"
set "ENTRY_FILE=app.py"
set "DIST_DIR=dist"
set "BUILD_DIR=build"
set "SPEC_FILE=%APP_NAME%.spec"
set "PACKAGE_DIR=%~dp0%DIST_DIR%\%APP_NAME%"
set "ICON_FILE=%~dp0app_icon.ico"
set "REQ_FILE=%~dp0..\requirements.txt"
set "VC_REDIST_FILE=%~dp0..\VC_redist.x64.exe"
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

echo Using Python:
call %PY_CMD% -c "import sys; print(sys.executable)"
echo.

call %PY_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pip is not available for this Python environment.
    echo Install pip or use a Python environment that includes pip, then run this build again.
    exit /b 1
)

if exist "%REQ_FILE%" (
    echo Installing required Python packages from "%REQ_FILE%"...
    call %PY_CMD% -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install the required Python packages.
        exit /b 1
    )
) else (
    echo [WARN] Requirements file was not found: "%REQ_FILE%"
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
call :add_data "app_icon.ico"
call :set_icon_arg
call :set_collect_args
call :ensure_not_running
if errorlevel 1 exit /b 1
call :remove_stale_onedir_runtime

echo Building %APP_NAME%.exe from %ENTRY_FILE%...
call %PY_CMD% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onefile ^
  --name "%APP_NAME%" ^
  --distpath "%PACKAGE_DIR%" ^
  --workpath "%BUILD_DIR%" ^
  %ICON_ARG% ^
  %DATA_ARGS% ^
  %COLLECT_ARGS% ^
  "%ENTRY_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

call :write_launcher
call :copy_support_files

echo.
echo Build complete.
echo EXE: "%PACKAGE_DIR%\%APP_NAME%.exe"
echo Launcher: "%PACKAGE_DIR%\Run %APP_NAME%.bat"
exit /b 0

:set_icon_arg
if exist "%ICON_FILE%" (
    set "ICON_ARG=--icon ""%ICON_FILE%"""
) else (
    set "ICON_ARG="
)
goto :eof

:set_collect_args
set "COLLECT_ARGS="
set "COLLECT_ARGS=%COLLECT_ARGS% --collect-all luno_python"
set "COLLECT_ARGS=%COLLECT_ARGS% --collect-all tzlocal"
set "COLLECT_ARGS=%COLLECT_ARGS% --hidden-import jwt"
set "COLLECT_ARGS=%COLLECT_ARGS% --hidden-import jwt.algorithms"
set "COLLECT_ARGS=%COLLECT_ARGS% --copy-metadata luno-python"
set "COLLECT_ARGS=%COLLECT_ARGS% --copy-metadata PyJWT"
set "COLLECT_ARGS=%COLLECT_ARGS% --copy-metadata cryptography"
set "COLLECT_ARGS=%COLLECT_ARGS% --copy-metadata tzlocal"
goto :eof

:ensure_not_running
tasklist /fi "imagename eq %APP_NAME%.exe" 2>nul | find /i "%APP_NAME%.exe" >nul
if not errorlevel 1 (
    echo [ERROR] %APP_NAME%.exe is currently running.
    echo Close the app from the Windows taskbar or Task Manager, then run this build again.
    exit /b 1
)
exit /b 0

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

:write_launcher
set "APP_DIST_DIR=%~dp0%DIST_DIR%\%APP_NAME%"
if not exist "%APP_DIST_DIR%" goto :eof
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo start "" "%APP_NAME%.exe"
) > "%APP_DIST_DIR%\Run %APP_NAME%.bat"
goto :eof

:copy_support_files
set "APP_DIST_DIR=%PACKAGE_DIR%"
if not exist "%APP_DIST_DIR%" goto :eof
if exist "%VC_REDIST_FILE%" (
    copy /Y "%VC_REDIST_FILE%" "%APP_DIST_DIR%\" >nul
)
call :copy_if_exists "luno_trader_config.json" "%APP_DIST_DIR%"
call :copy_if_exists "license.key" "%APP_DIST_DIR%"
call :copy_if_exists "public_key.pem" "%APP_DIST_DIR%"
call :copy_if_exists "private_key.pem" "%APP_DIST_DIR%"
(
    echo Trader for Luno Windows build
    echo.
    echo Run "Run %APP_NAME%.bat" from this folder to start the app.
    echo.
    echo The exe is a one-file PyInstaller build: Python and the required Python packages are embedded.
    echo Keep the config/license files beside the exe so the app can read and update them.
    echo If Windows reports a missing Microsoft C runtime DLL on an older PC, run VC_redist.x64.exe once and then start the app again.
) > "%APP_DIST_DIR%\README-WINDOWS.txt"
goto :eof

:remove_stale_onedir_runtime
if exist "%PACKAGE_DIR%\_internal" (
    rmdir /s /q "%PACKAGE_DIR%\_internal"
)
goto :eof

:copy_if_exists
if exist "%~1" (
    copy /Y "%~1" "%~2\" >nul
)
goto :eof
