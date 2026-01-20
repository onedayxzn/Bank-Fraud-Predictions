@echo off
REM Streamlit App Launcher for Bank Fraud Detection

echo.
echo ========================================
echo Bank Fraud Detection - Streamlit App
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if env folder exists
if not exist "env" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv env
    pause
    exit /b 1
)

REM Activate virtual environment
call env\Scripts\activate.bat

REM Check if streamlit is installed
python -m pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Streamlit...
    python -m pip install streamlit
)

REM Run streamlit app
echo.
echo Starting Streamlit application...
echo Browser will open automatically at http://localhost:8501
echo.
echo To stop the app, press Ctrl+C
echo.

streamlit run app.py

pause
