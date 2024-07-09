@echo off
setlocal enabledelayedexpansion

:: Try to find Python 3.9 installation
for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i

:: If Python is not in PATH, look in common installation directories
if not defined PYTHON_PATH (
    if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
        set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python39\python.exe
    ) else if exist "C:\Program Files\Python39\python.exe" (
        set PYTHON_PATH=C:\Program Files\Python39\python.exe
    ) else if exist "C:\Python39\python.exe" (
        set PYTHON_PATH=C:\Python39\python.exe
    )
)

:: Check if Python 3.9 is found
if not defined PYTHON_PATH (
    echo Python 3.9 is not found in common locations or in PATH.
    echo Please install Python 3.9 or add it to your PATH.
    pause
    exit /b 1
)

:: Run the Python script in the background
start "" "%PYTHON_PATH%" .\luno_btc_zar_trader.py

:: Wait a moment for the script to start and potentially create the log file
timeout /t 2 >nul

:: Tail the log file
echo Tailing trading_bot.log...
echo Press Ctrl+C to stop tailing and exit.
powershell -command "Get-Content -Path trading_bot.log -Wait -Tail 10"

:: This part will only execute if the user presses Ctrl+C to stop tailing
echo Stopped tailing the log file.
echo The Python script may still be running in the background.
echo To stop it, you may need to close its window or use Task Manager.
pause