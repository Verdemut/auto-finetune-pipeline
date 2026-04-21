@echo off
title Auto-Finetune Pipeline

echo ========================================
echo Auto-Finetune Pipeline
echo ========================================
echo.

call venv\Scripts\activate.bat

echo Starting web interface...
echo Open http://127.0.0.1:7860 in your browser
echo.
echo Press Ctrl+C to stop
echo ========================================

python webui.py

pause