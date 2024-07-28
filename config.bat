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
set "AVERAGE_WINDOW_SIZE=30"
set "MACD_FAST_PERIOD=120"
set "MACD_SLOW_PERIOD=260"
set "MACD_SIGNAL_PERIOD=90"
set "MACD_CANDLE_PERIOD=300"

echo Please provide the following values (press Enter to use the default):

set /p "PAIR=Enter the trading pair (default: %PAIR%): "
set /p "MARKET_PERCEPTION_THRESHOLD=Enter the Market Perception threshold value (default: %MARKET_PERCEPTION_THRESHOLD%): "
set /p "PRICE_CONFIDENCE_THRESHOLD=Enter the Price Confidence threshold value (default: %PRICE_CONFIDENCE_THRESHOLD%): "
set /p "MARKET_MOMENTUM_INDICATOR_THRESHOLD=Enter the Market Momentum Indicator threshold value (default: %MARKET_MOMENTUM_INDICATOR_THRESHOLD%): "
set /p "API_CALL_DELAY=Enter the API call delay in seconds (default: %API_CALL_DELAY%): "
set /p "PERIOD=Enter the period value (default: %PERIOD%, max 24): "
set /p "SHORT_PERIOD=Enter the short period value (default: %SHORT_PERIOD%): "
set /p "RANGE=Enter the depth chart range in thousands (default: %RANGE%): "
set /p "AVERAGE_WINDOW_SIZE=Enter the average window in transactions (default: %AVERAGE_WINDOW_SIZE%): "
set /p "MACD_FAST_PERIOD=Enter the MACD fast period (default: %MACD_FAST_PERIOD%): "
set /p "MACD_SLOW_PERIOD=Enter the MACD slow period (default: %MACD_SLOW_PERIOD%): "
set /p "MACD_SIGNAL_PERIOD=Enter the MACD signal period (default: %MACD_SIGNAL_PERIOD%): "
set /p "MACD_CANDLE_PERIOD=Enter the MACD candle period (default: %MACD_CANDLE_PERIOD%): "

(
echo PAIR = '%PAIR%'
echo MARKET_PERCEPTION_THRESHOLD = %MARKET_PERCEPTION_THRESHOLD%
echo PRICE_CONFIDENCE_THRESHOLD = %PRICE_CONFIDENCE_THRESHOLD%
echo MARKET_MOMENTUM_INDICATOR_THRESHOLD = %MARKET_MOMENTUM_INDICATOR_THRESHOLD%
echo API_CALL_DELAY = %API_CALL_DELAY%
echo PERIOD = %PERIOD%
echo SHORT_PERIOD = %SHORT_PERIOD%
echo RANGE = %RANGE%
echo AVERAGE_WINDOW_SIZE = %AVERAGE_WINDOW_SIZE%
echo MACD_FAST_PERIOD = %MACD_FAST_PERIOD%
echo MACD_SLOW_PERIOD = %MACD_SLOW_PERIOD%
echo MACD_SIGNAL_PERIOD = %MACD_SIGNAL_PERIOD%
echo MACD_CANDLE_PERIOD = %MACD_CANDLE_PERIOD%
) > config.py

echo Config file updated successfully.
pause