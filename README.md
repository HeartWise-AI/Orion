## Orion Video Training and Evaluation

Orion is a Python-based application designed for video training and evaluation tasks. It supports both regression and classification tasks and can be used with a variety of models.

## Setup

1. Ensure you have Python 3.10 or later installed.

1. It is recommended to use Conda and create a new environment as such: `conda create -n orion python=3.10 anaconda`

1. `conda activate orion`

1. Install Poetry, a Python package manager. You can do this by running the following commands:

   ```bash
   python3 -m pip install --user pipx
   export PATH="$PATH:/root/.local/bin"
   source ~/.zshrc  # or source ~/.bashrc
   pipx install poetry
   ```

1. Then run `poetry install` to install all dependencies in your LOCAL environment.

1. Ensure you have an account on wandb.ai and that you've logged in using `wandb login`

## Usage

You can use Orion either through a Python script or directly in a Jupyter notebook.

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

1. Run the main script with torchrun:

   ```bash
   torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config notebooks/config/config.yaml
   ```

   This command runs the script in a distributed manner with 2 GPU processes on a single node.

### Using a Jupyter Notebook

1. Prepare your configuration file in YAML format.
1. In your Jupyter notebook, import necessary modules and set up environment variables.
1. Define a class for your command-line arguments.
1. Load your configuration, create transforms, initialize a wandb run, and run the main process.

### Using Weights & Biases Sweeps

1. Prepare your configuration and sweep configuration files in YAML format.
1. In your Jupyter notebook, import necessary modules, define a function to load YAML configurations, and run the sweep agent.

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
