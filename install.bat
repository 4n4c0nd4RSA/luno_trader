@echo off
setlocal enabledelayedexpansion

:: Set variables
set PYTHON_VERSION=3.9.13
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
set PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
set GET_PIP_SCRIPT=get-pip.py
set VC_REDIST_INSTALLER=VC_redist.x64.exe

:: Install VC++ Redistributable
echo Installing Visual C++ Redistributable...
if exist "%VC_REDIST_INSTALLER%" (
    start /wait %VC_REDIST_INSTALLER% /quiet /norestart
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Visual C++ Redistributable.
        exit /b 1
    )
) else (
    echo VC_redist.x64.exe not found in the current directory. Skipping installation.
)

:: Download Python installer
echo Downloading Python %PYTHON_VERSION%...
powershell -Command "(New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')"
if %ERRORLEVEL% neq 0 (
    echo Failed to download Python installer.
    exit /b 1
)

:: Install Python
echo Installing Python %PYTHON_VERSION%...
start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_pip=1
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

:: Check if pip is installed
python -m pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo pip not found. Installing pip...
    echo Downloading get-pip.py...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

    REM Install pip
    echo Installing pip...
    python get-pip.py

    REM Clean up
    del get-pip.py
) else (
    echo pip is already installed.
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
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        exit /b 1
    )
) else (
    echo requirements.txt not found. Skipping package installation.
)

:: Set environment variables
echo.
echo Please enter your Luno API credentials:
set /p LUNO_API_KEY_ID=Enter your Luno API Key ID: 
set /p LUNO_API_KEY_SECRET=Enter your Luno API Key Secret: 

:: Set user environment variables
setx LUNO_API_KEY_ID "%LUNO_API_KEY_ID%"
setx LUNO_API_KEY_SECRET "%LUNO_API_KEY_SECRET%"

echo.
echo Environment variables have been set:
echo LUNO_API_KEY_ID: %LUNO_API_KEY_ID%
echo LUNO_API_KEY_SECRET: %LUNO_API_KEY_SECRET%

echo.
echo Setup completed successfully.
echo Please restart your command prompt or IDE for the environment variables to take effect.
pause