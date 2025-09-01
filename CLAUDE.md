# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Orion is a PyTorch-based video training and evaluation framework for medical imaging, specifically designed for echocardiographic video analysis. It supports both regression and classification tasks using various state-of-the-art video models.

## Key Commands

### Setup and Installation (using UV - recommended)
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR
pip install uv

# Create virtual environment with UV
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# For development (includes linting and testing)
uv pip install -r requirements-dev.txt

# Install PyTorch with CUDA support (choose based on your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Login to Weights & Biases (required for experiment tracking)
wandb login
```

### Running Training
```bash
# Distributed training with torchrun (recommended)
torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config_path=notebooks/config/config.yaml

# Single GPU training with sweep configuration
python run_sweep.py
```

### Running Inference
```bash
python run_inference.py --config_path notebooks/config/config_x3d_multi_output_classification_eval.yaml --split inference --model_path outputs/outputs_folder_id/best.pt
```

### Code Quality and Formatting
```bash
# Install pre-commit hooks (after installing dev dependencies)
pre-commit install

# Run pre-commit checks manually
pre-commit run --all-files

# Run individual formatters
black . --line-length=98
isort . --profile=black --line-length=98
ruff check . --fix

# Run tests (if available)
pytest tests/

# The repository uses automatic formatting with:
# - black (line-length=98)
# - isort (profile=black)
# - ruff for linting
# - Various other formatters configured in .pre-commit-config.yaml
```

### Dependency Management with UV
```bash
# Add a new package
uv pip install package-name
# Then manually add to requirements.txt

# Upgrade all packages
uv pip install --upgrade -r requirements.txt

# Show installed packages
uv pip list

# Uninstall a package
uv pip uninstall package-name

# Clear UV cache (if needed)
uv cache clean

# Compile requirements for faster installs
uv pip compile requirements.txt -o requirements-lock.txt
```

## Architecture Overview

### Core Components

1. **Models** (`orion/models/`)
   - `x3d_multi.py`: X3D model implementation with multi-output head support
   - `videopairclassifier.py`: Siamese-style video pair classifier
   - `__init__.py`: Contains model loading logic, head replacement functions, and multi-output head implementation
   - Supports various pretrained models from PyTorchVideo and torchvision

2. **Datasets** (`orion/datasets/`)
   - `video.py`: Main video dataset implementation
   - `video_inference.py`: Inference-specific dataset handling
   - Handles medical video data with support for masking and preprocessing

3. **Training Pipeline** (`orion/utils/`)
   - `video_training_and_eval.py`: Main training loop with distributed training support
   - `video_run_inference.py`: Inference pipeline
   - `losses.py`: Loss function registry supporting various losses (MSE, MAE, Huber, etc.)
   - `plot.py`: Comprehensive metrics computation and visualization

4. **Configuration System**
   - YAML-based configuration files in `notebooks/config/`
   - Supports model-specific requirements (frame counts, image sizes)
   - Configurable loss functions, optimizers, and training parameters

### Model-Specific Requirements

| Model | Frame Count | Image Size | Learning Rate |
|-------|-------------|------------|---------------|
| x3d | Multiples of 8 | 224 or 256 | 1e-3 to 1e-4 |
| swin3d | 24 or 32 | Flexible | 1e-4 to 1e-5 |
| mvit | 16 | Flexible | 3e-5 |

### Multi-Output Classification

The framework supports multi-head classification where different heads can have different numbers of output classes. This is configured via the `head_structure` parameter in the config file:

```yaml
head_structure:
  contrast_agent: 1  # Binary classification
  main_structure: 5  # Multi-class classification
  stent_presence: 1  # Binary classification
```

### Loss Functions

The framework supports multiple loss functions through `LossRegistry`:
- Regression: MSE, MAE, Huber, SmoothL1
- Classification: CrossEntropy, BCEWithLogits, FocalLoss
- Custom weighted combinations for multi-output models

### Distributed Training

The codebase is designed for distributed training using PyTorch's DistributedDataParallel (DDP). It automatically handles:
- Multi-GPU synchronization
- Gradient accumulation
- Mixed precision training (AMP)
- Distributed evaluation with proper metric aggregation

## Important Notes

- Always ensure CUDA is available - the framework requires GPU for training
- Weights & Biases integration is essential for experiment tracking
- Pre-commit hooks automatically format code - let them run before committing
- The framework validates model configurations at runtime (e.g., frame counts for specific models)
- When modifying models, use the existing head replacement logic in `orion/models/__init__.py`
- Always actiave venv with source .venv/bin/activate