@echo off
setlocal enabledelayedexpansion

:: Find all luno_*.py files in the current directory
set "filelist="
for %%f in (luno_*.py) do (
    set "filelist=!filelist! %%f"
)

:: If no files are found, exit
if "%filelist%"=="" (
    echo No luno_*.py files found.
    exit /b
)

:: Display the list of files to the user
echo Select the files you want to run by entering their numbers separated by spaces:
set /a i=1
for %%f in (%filelist%) do (
    echo !i!. %%f
    set "file[!i!]=%%f"
    set /a i+=1
)

:: Get user input
set /p "choice=Enter your choices: "

:: Run the selected files in separate PowerShell windows
for %%i in (%choice%) do (
    if defined file[%%i] (
        set "filename=!file[%%i]!"
        set "params="

        :: Check if the file is luno_zar_trader.py and ask for True Trade
        if "!filename!"=="luno_zar_trader.py" (
            set /p "true_trade=Do you want to activate True Trade for luno_zar_trader.py? (y/n): "
            if /i "!true_trade!"=="y" (
                set "params=--true-trade"
            )
        )

        start powershell -NoExit -Command "python !filename! !params!"
    )
)

endlocal
