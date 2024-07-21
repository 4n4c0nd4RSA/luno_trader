@echo off
setlocal enabledelayedexpansion

set "PAIR=XBTZAR"
set "MARKET_PERCEPTION_THRESHOLD=0.1"
set "MARKET_MOMENTUM_INDICATOR_THRESHOLD=0.01"
set "PRICE_CONFIDENCE_THRESHOLD=0.11"
set "API_CALL_DELAY=60"
set "PERIOD=10"
set "SHORT_PERIOD=3"
set "RANGE=200"

echo Please provide the following values (press Enter to use the default):

set /p "PAIR=Enter the trading pair (default: %PAIR%): "
set /p "MARKET_PERCEPTION_THRESHOLD=Enter the Market Perception threshold value (default: %MARKET_PERCEPTION_THRESHOLD%): "
set /p "PRICE_CONFIDENCE_THRESHOLD=Enter the Price Confidence threshold value (default: %PRICE_CONFIDENCE_THRESHOLD%): "
set /p "MARKET_MOMENTUM_INDICATOR_THRESHOLD=Enter the Market Momentum Indicator threshold value (default: %MARKET_MOMENTUM_INDICATOR_THRESHOLD%): "
set /p "API_CALL_DELAY=Enter the API call delay in seconds (default: %API_CALL_DELAY%): "
set /p "PERIOD=Enter the period value (default: %PERIOD%, max 24): "
set /p "SHORT_PERIOD=Enter the short period value (default: %SHORT_PERIOD%): "
set /p "RANGE=Enter the depth chart range in thousands (default: %RANGE%): "

(
echo PAIR = '%PAIR%'
echo MARKET_PERCEPTION_THRESHOLD = %MARKET_PERCEPTION_THRESHOLD%
echo PRICE_CONFIDENCE_THRESHOLD = %PRICE_CONFIDENCE_THRESHOLD%
echo MARKET_MOMENTUM_INDICATOR_THRESHOLD = %MARKET_MOMENTUM_INDICATOR_THRESHOLD%
echo API_CALL_DELAY = %API_CALL_DELAY%
echo PERIOD = %PERIOD%
echo SHORT_PERIOD = %SHORT_PERIOD%
echo RANGE = %RANGE%
) > config.py

echo Config file updated successfully.
pause