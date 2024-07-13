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
        start powershell -NoExit -Command "python !file[%%i]!"
    )
)

endlocal
