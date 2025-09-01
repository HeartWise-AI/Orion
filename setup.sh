#!/bin/bash

# Orion Setup Script - Migrated to UV
# This script sets up the development environment for Orion using UV

set -e  # Exit on error

echo "========================================="
echo "Orion Setup Script - UV Migration"
echo "========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    
    # Detect OS and install UV accordingly
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "Windows detected. Please run:"
        echo "  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
        exit 1
    else
        # macOS and Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    echo "UV installed successfully!"
else
    echo "UV is already installed: $(uv --version)"
fi

echo ""
echo "Creating Python 3.10 virtual environment..."
uv venv --python 3.10

echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Please run: .venv\\Scripts\\activate"
else
    echo "Please run: source .venv/bin/activate"
fi

echo ""
echo "Installing dependencies..."
source .venv/bin/activate 2>/dev/null || .venv/Scripts/activate 2>/dev/null || true

# Install main dependencies
uv pip install -r requirements.txt

echo ""
echo "Optional: Install development dependencies (y/n)?"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    uv pip install -r requirements-dev.txt
    echo "Installing pre-commit hooks..."
    pre-commit install
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate  # macOS/Linux"
echo "   .venv\\Scripts\\activate    # Windows"
echo ""
echo "2. Configure Weights & Biases:"
echo "   wandb login"
echo ""
echo "3. If using GPU, ensure correct PyTorch CUDA version:"
echo "   # For CUDA 11.8:"
echo "   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
echo "   # For CUDA 12.1:"
echo "   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "4. Start training:"
echo "   torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config_path=notebooks/config/config.yaml"
echo ""
echo "For more information, see README.md and MIGRATION_TO_UV.md"