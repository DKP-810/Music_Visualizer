@echo off
echo ========================================
echo Audio Visualizer - Installing Dependencies
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Installing required packages...
echo.
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Installation successful!
    echo ========================================
    echo.
    echo To run the visualizer, type: python visualizer.py
    echo To build an executable, type: python build_exe.py
    echo.
) else (
    echo.
    echo ========================================
    echo Installation failed!
    echo ========================================
    echo Please check the error messages above.
    echo.
)

pause
