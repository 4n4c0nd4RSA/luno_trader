@echo off
setlocal enabledelayedexpansion

set "PAIR=XBTZAR"
set "THRESHOLD=0.1"
set "SHORT_THRESHOLD=0.11"
set "API_CALL_DELAY=60"
set "PERIOD=10"
set "SHORT_PERIOD=3"
set "RANGE=200"

echo Please provide the following values (press Enter to use the default):

set /p "PAIR=Enter the trading pair (default: %PAIR%): "
set /p "THRESHOLD=Enter the Market Perception threshold value (default: %THRESHOLD%): "
set /p "SHORT_THRESHOLD=Enter the Confidence threshold value (default: %SHORT_THRESHOLD%): "
set /p "API_CALL_DELAY=Enter the API call delay in seconds (default: %API_CALL_DELAY%): "
set /p "PERIOD=Enter the period value (default: %PERIOD%, max 24): "
set /p "SHORT_PERIOD=Enter the short period value (default: %SHORT_PERIOD%): "
set /p "RANGE=Enter the depth chart range in thousands (default: %RANGE%): "

(
echo PAIR = '%PAIR%'
echo THRESHOLD = %THRESHOLD%
echo SHORT_THRESHOLD = %SHORT_THRESHOLD%
echo API_CALL_DELAY = %API_CALL_DELAY%
echo PERIOD = %PERIOD%
echo SHORT_PERIOD = %SHORT_PERIOD%
echo RANGE = %RANGE%
) > config.py

echo Config file updated successfully.
pause