@echo off
REM Setup script for main_experiments virtual environment on Windows
REM This creates a lightweight Python virtual environment without conda

echo ===============================================
echo Setting up main_experiments virtual environment
echo ===============================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python version: 
python --version

REM Create virtual environment in the current directory
set ENV_NAME=venv_main_exp
echo Creating virtual environment: %ENV_NAME%

python -m venv %ENV_NAME%

REM Check if virtual environment was created successfully
if not exist "%ENV_NAME%" (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully

REM Activate virtual environment
call %ENV_NAME%\Scripts\activate.bat

echo Activated virtual environment
echo Python location:
where python

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.
echo To activate the environment in the future, run:
echo   %ENV_NAME%\Scripts\activate.bat
echo.
echo To deactivate the environment, run:
echo   deactivate
echo.
echo To test AEDAT4 loading, run:
echo   python time_offset_analysis.py --clean_file "E:\path\to\clean.aedat4" --flare_file "E:\path\to\flare.aedat4"
echo.
pause