@echo off
setlocal enabledelayedexpansion

:: Set variables
set PYTHON_VERSION=3.9.13
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
set PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe

:: Download Python installer
echo Downloading Python %PYTHON_VERSION%...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')"
if %ERRORLEVEL% neq 0 (
    echo Failed to download Python installer.
    exit /b 1
)

:: Install Python
echo Installing Python %PYTHON_VERSION%...
start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %ERRORLEVEL% neq 0 (
    echo Failed to install Python.
    exit /b 1
)

:: Verify Python installation
python --version
if %ERRORLEVEL% neq 0 (
    echo Python installation failed or Python is not in PATH.
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo Failed to upgrade pip.
    exit /b 1
)

:: Install requirements
if exist requirements.txt (
    echo Installing requirements from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        exit /b 1
    )
) else (
    echo requirements.txt not found. Skipping package installation.
)

echo Setup completed successfully.