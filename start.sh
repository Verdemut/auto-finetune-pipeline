#!/bin/bash

echo "========================================"
echo "Auto-Finetune Pipeline"
echo "========================================"
echo ""

source venv/bin/activate

echo "Starting web interface..."
echo "Open http://127.0.0.1:7860 in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"

python webui.py