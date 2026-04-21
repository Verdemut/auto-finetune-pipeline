@echo off
title Auto-Finetune Pipeline Installer
echo ========================================
echo Auto-Finetune Pipeline Installer
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo [OK] Python found

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
echo This may take a few minutes...

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU detected, installing CPU version...
    pip install -r requirements.txt
) else (
    echo [INFO] NVIDIA GPU detected, installing CUDA version...
    pip install -r requirements-cuda.txt
)

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Creating directories...
mkdir datasets 2>nul
mkdir outputs 2>nul
mkdir logs 2>nul
mkdir checkpoints 2>nul
mkdir exports 2>nul

echo.
echo Creating start scripts...
echo @echo off > start.bat
echo call venv\Scripts\activate >> start.bat
echo python webui_platform.py >> start.bat
echo pause >> start.bat

echo @echo off > start_colab.bat
echo call venv\Scripts\activate >> start_colab.bat
echo python webui_platform.py >> start_colab.bat
echo pause >> start_colab.bat

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo To start the application:
echo   1. Run start.bat
echo   2. Open browser to http://127.0.0.1:7860
echo.
echo For Google Colab:
echo   1. Run start_colab.bat
echo   2. Export your dataset and config
echo   3. Upload to Google Colab
echo.
pause