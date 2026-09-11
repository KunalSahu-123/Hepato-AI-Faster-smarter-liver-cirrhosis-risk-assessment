@echo off
title Hepato AI - Liver Cirrhosis Prediction System
echo ============================================
echo   Hepato AI - Liver Cirrhosis Prediction
echo   saxenashvam0321@gmail.com
echo ============================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment found.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install dependencies
echo [2/3] Installing dependencies...
pip install -r requirements.txt --quiet

:: Run the app
echo [3/3] Starting the server...
echo.
echo   Open in browser: http://localhost:5000
echo   Admin login:     admin@liverai.com / Admin@123
echo   Press Ctrl+C to stop.
echo.
python app.py
pause
