## Orion Video Training and Evaluation

Orion is a Python-based application designed for video training and evaluation tasks. It supports both regression and classification tasks and can be used with a variety of models.

## Setup

### Prerequisites

1. Python 3.10 or later installed
2. UV package manager (see installation below)
3. CUDA-compatible GPU (recommended for training)

### Quick Start

1. **Install UV** (fast Python package manager):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/HeartWise-AI/Orion.git
   cd Orion
   ```

3. **Set up virtual environment and install dependencies**:
   ```bash
   # Create virtual environment (uses Python 3.10 by default)
   uv venv
   
   # Activate virtual environment
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .venv\Scripts\activate  # On Windows
   
   # Install dependencies
   uv pip install -r requirements.txt
   
   # For development (includes testing and linting tools)
   uv pip install -r requirements-dev.txt
   ```

4. **Configure PyTorch for your CUDA version** (if using GPU):
   ```bash
   # For CUDA 11.8
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Set up Weights & Biases** for experiment tracking:
   ```bash
   wandb login
   ```

### Alternative Setup (using pip directly)

If you prefer using pip without UV:
```bash
python -m venv orion-venv
source orion-venv/bin/activate  # On macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### Migration from Poetry

If you're migrating from an older version using Poetry, see [MIGRATION_TO_UV.md](MIGRATION_TO_UV.md) for detailed instructions

## Usage

You can use Orion either through a Python script 
```bash
python run_sweep.py
python run_inference.py
```

or build your own Jupyter notebook inspired by code samples in the `notebooks/` folder.

### Configuration file example

See `notebooks/config/` for different config.YAML file examples. Here are some key configuration parameters:

\[... keep the existing parameter descriptions ...\]

### Model-Specific Configurations

Different video models have specific requirements and recommendations for optimal performance. The table below summarizes the recommended frame sizes and learning rates for supported models:

| Model  | Recommended Frame Sizes | Recommended Learning Rate |
| ------ | ----------------------- | ------------------------- |
| x3d    | Multiples of 8          | 1e-3 to 1e-4              |
| swin3d | 24 or 32                | 1e-4 to 1e-5              |
| mvit   | 16                      | 3e-5 (0.00003)            |

#### Additional Model-Specific Notes:

1. x3d:

   - Supports various frame counts as long as they are multiples of 8.
   - The x3d_m variant specifically supports video sizes of either 224x224 or 256x256.

1. swin3d:

   - Only supports 24 or 32 frames.
   - Ensure your configuration matches one of these frame counts.

1. mvit:

   - Strictly requires 16 frames.
   - Adjust your data preprocessing to match this requirement.

When configuring your model in the YAML file, ensure that the `frames` and `resize` parameters align with these recommendations. The `lr` (learning rate) parameter should be set within the recommended range for best results.

### Using a Python Script

1. Prepare your configuration file in YAML format.

2. Run the main script with torchrun:

   ```bash
   torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config_path=notebooks/config/config.yaml
   ```

   This command runs the script in a distributed manner with 2 GPU processes on a single node.

### GPU Selection

You can specify which GPUs to use in multiple ways:

1. **Using the --gpu flag** (Recommended):
   ```bash
   # Single GPU
   torchrun --standalone --nnodes=1 --nproc-per-node=1 orion/utils/video_training_and_eval.py \
     --config_path=notebooks/config/config.yaml --gpu 0

   # Multiple GPUs
   torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py \
     --config_path=notebooks/config/config.yaml --gpu 0,1

   # All 4 GPUs
   torchrun --standalone --nnodes=1 --nproc-per-node=4 orion/utils/video_training_and_eval.py \
     --config_path=notebooks/config/config.yaml --gpu 0,1,2,3
   ```

2. **Using CUDA_VISIBLE_DEVICES environment variable**:
   ```bash
   # Single GPU
   CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 \
     orion/utils/video_training_and_eval.py --config_path=notebooks/config/config.yaml

   # Multiple GPUs
   CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
     orion/utils/video_training_and_eval.py --config_path=notebooks/config/config.yaml
   ```

   **Note**: Make sure `--nproc-per-node` matches the number of GPUs you're using.

### Using a Jupyter Notebook

1. Prepare your configuration file in YAML format.
2. In your Jupyter notebook, import necessary modules and set up environment variables.
3. Define a class for your command-line arguments.
4. Load your configuration, create transforms, initialize a wandb run, and run the main process.

### Using Weights & Biases Sweeps

1. Prepare your configuration and sweep configuration files in YAML format.
2. Run the `run_sweep.py` script to start the sweep agent (no argument needed - update the `sweep_conf_file_path` variable in the script to point to your sweep configuration file).
   ```bash
   python run_sweep.py
   ```

### Inference and Evaluation

To run inference and evaluation, use the `run_inference.py` script.

```bash
python run_inference.py --config_path notebooks/config/config_x3d_multi_output_classification_eval.yaml --splits inference --model_path outputs/outputs_folder_id/best.pt
```

The other arguments are optional: the data path (usually defined in the config file), the output directory (usually defined in the config file), the wandb id & resume flag (both not needed for inference).

### Note

This application uses distributed training, so it is designed to be run on multiple GPUs. If you are running it on a machine with a single GPU or CPU, you may need to modify the code accordingly.

## Validation of Model Configurations

The `build_model` function in `video_training_and_eval.py` includes checks to ensure that the frame and resize parameters in your configuration file are compatible with the chosen model. If these parameters don't meet the model-specific requirements, the function will raise a `ValueError` with an appropriate error message.

For example:

- For swin3d models, it checks if the frame count is either 24 or 32.
- For x3d models, it verifies that the frame count is a multiple of 8.
- For x3d_m specifically, it checks if the resize value is either 224 or 256.
- For mvit models, it ensures the frame count is exactly 16.

These checks help prevent configuration errors and ensure that your model is set up correctly for training or inference.
