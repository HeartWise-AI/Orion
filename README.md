## Orion Video Training and Evaluation

Orion is a Python-based application designed for video training and evaluation tasks. It supports both regression and classification tasks, and can be used with a variety of models.

## Setup

1. Ensure you have Python 3.10 or later installed.
1. It is recommended to use Conda and create a new environemnt as such : `conda create -n orion python=3.10 anaconda`
1. `conda activate orion`
1. Install Poetry, a Python package manager. You can do this by running the following commands:

`python3 -m pip install --user pipx` <br>

Reload terminal or re-login
`export PATH="$PATH:/root/.local/bin"`
`source ~/.zshrc` or `source ~/.bashrc`

Try using pipx by typing `pipx`. If command is not found re-export the PATH `export PATH="$HOME/.local/bin:$PATH" ` then reload the terminal.

`python3 -m pipx ensurepath`

`pipx install poetry`

Then `poetry install`

This will install all dependencies in your LOCAL environment.

5. Ensure you have an account on wandb.ai and that you logged in used `wandb login`

## Usage

You can use Orion either through a Python script or directly in a Jupyter notebook.

### Configuration file example

See `notebooks/config/` for different config.YAML file examples.

```
- num_epochs: The number of epochs for training.
- model_name: The architecture of the model to use. We use torchvision such 'x3d', 'r2+1d', or 'swin3d'. They are initializd with the Kinetics 400 weights.
- model_path: The path to the model file for retraining the model. If this is set to null, a new model will be created.
- use_amp: Whether to use automatic mixed precision for training. This can speed up training on GPUs.
- data_filename: The path to the data file. The data needs to be split using the alpha character : α - this is to prevent commas present in text reports for wrongly splitting the row.
- root: The root directory for the data.
- target_label: The label to predict.
- datapoint_loc_label: The label for the data point location of the MP4 or video files.
- label_loc_label: The definition of each label in textual format (i.e. normal or reduced).
- binary_threshold: The threshold for binary classification in a regression.
- frames: The number of frames to use for video data. If set to 32, it goes from frame 0 to 32 of the video.
- device: The device to use for training (e.g., cuda for GPU, cpu for CPU).
- period: How many frames you sample. If 1 it samples all frames up to "frames" parameter, if 2 it samples every other frame (0, 2, 4, 6).
- pretrained: Whether to use a pretrained model.
- output: The output directory for the model.
- n_train_patients: The number of patients to use for training.
- seed: The random seed for reproducibility.
- optimizer: The optimizer to use for training.
- weight_decay: The weight decay for the optimizer.
- num_workers: The number of workers ("processes") for data loading.
- batch_size: The batch size for training.
- lr: The learning rate for the optimizer.
- label_smoothing: Whether to use label smoothing.
- label_smoothing_value: The value for label smoothing.
- class_weights: The weights for the classes if imbalanced datasets. Put them as a dictionary [0, 2, 3] or a single value for binary classification [3]
- loss: The loss function to use.
- task: The task type (regression or classification).
- num_classes: The number of classes for classification.
- scheduler_type: The type of scheduler to use.
- factor: The factor for the scheduler determining how much you reduce the LR. if 0.5 you reduce it by half.
- patience: The patience for the scheduler.
- threshold: The threshold for the scheduler.
- lr_step_period: The learning rate step period for the scheduler.
- run_test: Whether to run a test after training.
- tag: The tag for the run.
- project: The project name for the run.
- entity: The entity name for the run.
- resume: Whether to resume training from a checkpoint.
- rand_augment: Whether to use random data augmentation. 
- resize: The size to resize the video to.
- apply_mask: Whether to apply a mask section to the videos to hide patient information.
- view_count: The view count for the videos for multi-view classification.
- save_best: The metric to use for saving the best model.
- metrics_control: The metrics control parameters.
- transforms: The transforms to apply to the image
```

### Using a Python Script

1. Prepare your configuration file in YAML format. This should include all the necessary parameters for your task, such as the task type, model name, number of epochs, batch size, and more. An example configuration file is provided in notebooks/config/config.yaml.

1. Run the main script with torchrun, passing the path to your Python script and the path to your configuration file as arguments. Here is an example command:

`torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config notebooks/config/config.yaml`

The command you provided is used to run a distributed training or evaluation script using PyTorch's `torchrun` utility. Here’s a detailed breakdown of the command:

1. **`torchrun`**: This is a utility provided by PyTorch to run distributed training scripts. It handles the initialization and execution of distributed training processes.

2. **`--standalone`**: This flag indicates that the script will run in a standalone mode, meaning it will not connect to an external cluster or other nodes.

3. **`--nnodes=1`**: This specifies the number of nodes (or clusters) to use for the training. In this case, it is set to 1, indicating that the script will run on a single node.

4. **`--nproc-per-node=2`**: This specifies the number of processes (or GPUs) to run on each node. Here, it is set to 2, meaning that 2 processes will be run on the single node specified by `--nnodes`.

5. **`orion/utils/video_training_and_eval.py`**: This is the path to the Python script that will be executed. This script likely contains the logic for training and evaluating a model.

6. **`--config notebooks/config/config.yaml`**: This is an argument passed to the script, specifying the path to a configuration file (`config.yaml`). This configuration file typically contains settings and hyperparameters needed for the training and evaluation process.

In summary, this command runs the `video_training_and_eval.py` script in a distributed manner with 2 GPU processes on a single node, using the configuration specified in `config.yaml`.

### Using a Jupyter Notebook

1. Prepare your configuration file in YAML format as described above.

1. In your Jupyter notebook, import the necessary modules and set up your environment variables:

```
from orion.utils import video_training_and_eval
import wandb
import yaml
import os
os.environ["WANDB_NOTEBOOK_NAME"] = "test_view_classifier.ipynb"
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
```

3. Define a class for your command-line arguments:

```
class Args:
    def __init__(self):
        self.local_rank = 0
        self.log_all = False
        self.epochs = 2
        self.batch = 32
```

4. Load your configuration from the YAML file, create the transforms, initialize a wandb run if logging, and run the main process:

```
args = Args()
with open('config/config.yaml', 'r') as file:
    config_defaults = yaml.safe_load(file)
transform_list = video_training_and_eval.create_transforms(config_defaults)
run = video_training_and_eval.setup_run(args, config_defaults)
video_training_and_eval.execute_run(config_defaults=config_defaults, transforms=transform_list, args=args, run=run)
```

### Using Weights & Biases Sweeps

1. Prepare your configuration file in YAML format as described above. Additionally, prepare a sweep configuration file in YAML format. This file should reference the initial model configuration file and include additional sweep parameters. An example sweep configuration file is provided in notebooks/config/sweep_config.yaml.

1. In your Jupyter notebook, import the necessary modules:

```
import wandb
import yaml
```

3. Define a function to load your YAML configuration:

```
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
```

4. Load your sweep configuration from the YAML file, create a sweep with wandb, and run the sweep agent:

```
sweep_conf_file_path = 'config/sweep_config.yaml'
sweep_conf = load_yaml_config(sweep_conf_file_path)
sweep_id = wandb.sweep(sweep_conf,project=sweep_conf['name'])
count = 1  # number of runs to execute
wandb.agent(
    sweep_id=sweep_id,
    count=count
)
```

### Note

This application uses distributed training, so it is designed to be run on multiple GPUs. If you are running it on a machine with a single GPU or CPU, you may need to modify the code accordingly.
