# Migration from Poetry to UV

This document outlines the migration from Poetry to UV for dependency management in the Orion project.

## Why UV?

UV is a fast Python package installer and resolver written in Rust. It offers:
- **10-100x faster** package installation compared to pip and Poetry
- Built-in virtual environment management
- Compatible with pip requirements files
- Better caching and parallel downloads
- Simpler configuration

## Migration Overview

### What Changed

1. **Configuration Files**
   - Added `uv.toml` for UV-specific settings
   - Added `.python-version` to specify Python version
   - Updated `pyproject.toml` to PEP 621 standard format
   - Split requirements into `requirements.txt` and `requirements-dev.txt`

2. **Removed Files** (can be deleted after migration)
   - `poetry.lock` - No longer needed with UV
   - Old Poetry-specific sections in `pyproject.toml`

3. **New Features**
   - Faster dependency resolution and installation
   - Better PyTorch installation with CUDA support
   - Simpler virtual environment management

## Installation Guide

### Prerequisites

Install UV if you haven't already:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Setting Up the Project

#### 1. Create Virtual Environment

```bash
# UV will automatically use Python 3.10 from .python-version
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
# Install main dependencies
uv pip install -r requirements.txt

# For development (includes linting and testing tools)
uv pip install -r requirements-dev.txt

# Or install from pyproject.toml directly
uv pip install -e .

# For development with all extras
uv pip install -e ".[dev]"
```

#### 3. Install PyTorch with CUDA (if needed)

```bash
# For CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Common Commands Comparison

| Task | Poetry | UV |
|------|--------|-----|
| Create venv | `poetry install` | `uv venv` |
| Install deps | `poetry install` | `uv pip install -r requirements.txt` |
| Add package | `poetry add package` | `uv pip install package` + add to requirements.txt |
| Add dev package | `poetry add -D package` | `uv pip install package` + add to requirements-dev.txt |
| Remove package | `poetry remove package` | `uv pip uninstall package` + remove from requirements.txt |
| Update deps | `poetry update` | `uv pip install --upgrade -r requirements.txt` |
| Show installed | `poetry show` | `uv pip list` |
| Run script | `poetry run python script.py` | `python script.py` (with venv activated) |

## Project Structure

```
orion/
├── .python-version        # Specifies Python 3.10
├── uv.toml               # UV configuration
├── pyproject.toml        # PEP 621 project metadata
├── requirements.txt      # Main dependencies
├── requirements-dev.txt  # Development dependencies
├── .venv/               # Virtual environment (created by UV)
└── orion/               # Source code
```

## Troubleshooting

### GPU/CUDA Issues

If you encounter CUDA-related errors:

1. Check your CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install the appropriate PyTorch version:
   ```bash
   # Uninstall existing torch
   uv pip uninstall torch torchvision
   
   # Install correct version (example for CUDA 11.8)
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Dependency Conflicts

UV provides better error messages for conflicts. If you encounter issues:

1. Clear UV cache:
   ```bash
   uv cache clean
   ```

2. Recreate virtual environment:
   ```bash
   rm -rf .venv
   uv venv
   uv pip install -r requirements.txt
   ```

### Missing Dependencies

If a package is missing after migration:

1. Check if it was in `poetry.lock` but not in `pyproject.toml`
2. Add it to `requirements.txt` or `requirements-dev.txt`
3. Install with: `uv pip install package-name`

## Benefits After Migration

1. **Speed**: Package installation is 10-100x faster
2. **Simplicity**: No lock files to manage
3. **Compatibility**: Works with standard pip requirements
4. **Reliability**: Better dependency resolution
5. **PyTorch**: Easier CUDA version management

## Rollback Plan

If you need to rollback to Poetry:

1. Keep `poetry.lock` file (don't delete immediately)
2. Restore original `pyproject.toml` from git
3. Run `poetry install`

## Next Steps

1. Delete Poetry-related files after confirming UV works:
   ```bash
   rm poetry.lock
   ```

2. Update CI/CD pipelines to use UV
3. Update team documentation
4. Consider adding UV to Docker images for faster builds

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)