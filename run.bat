@echo off
cd /d "%~dp0"
echo Delivery Time Prediction - starting...
python run_all.py
if errorlevel 1 (
  echo.
  echo Something went wrong. Make sure Python 3.10+ is installed and "Add Python to PATH" was ticked.
)
pause
