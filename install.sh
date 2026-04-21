#!/bin/bash

echo "========================================"
echo "Auto-Finetune Pipeline Installer"
echo "========================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"

echo ""
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected, installing CUDA version..."
    pip install -r requirements-cuda.txt
else
    echo "[INFO] No NVIDIA GPU detected, installing CPU version..."
    pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo ""
echo "Creating directories..."
mkdir -p datasets outputs logs checkpoints exports

echo ""
echo "Creating start script..."
cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python webui_platform.py
EOF

chmod +x start.sh

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "To start the application:"
echo "  1. Run: ./start.sh"
echo "  2. Open browser to http://127.0.0.1:7860"
echo ""
echo "For Google Colab:"
echo "  1. Export your dataset and config"
echo "  2. Upload to Google Colab"
echo ""