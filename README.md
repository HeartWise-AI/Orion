## Orion Video Training and Evaluation

Orion is a Python-based application designed for video training and evaluation tasks. It supports both regression and classification tasks, and can be used with a variety of models.
## Setup



1. Ensure you have Python 3.10 or later installed. Ensure you have an account on wandb.ai and that you logged in used `wandb login`
2. It is recommended to use Conda and create a new environemnt as such : `conda create -n orion python=3.10 anaconda`
3. `conda activate orion`
4. Install Poetry, a Python package manager. You can do this by running the following commands:

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

Then `poetry install` 

This will install all dependencies in your LOCAL environment. 

## Usage

You can use Orion either through a Python script or directly in a Jupyter notebook.
### Configuration file example
See `/config/` for different config.YAML file examples.

```
- num_epochs: The number of epochs for training.
- model_name: The name of the model to use.
- model_path: The path to the model file. If this is set to null, a new model will be created.
- use_amp: Whether to use automatic mixed precision for training. This can speed up training on GPUs.
- data_filename: The path to the data file.
- root: The root directory for the data.
- target_label: The label to predict.
- datapoint_loc_label: The label for the data point location.
- label_loc_label: The label for the label location.
- binary_threshold: The threshold for binary classification.
- frames: The number of frames to use for video data.
- device: The device to use for training (e.g., cuda for GPU, cpu for CPU).
- period: The period for the scheduler.
- pretrained: Whether to use a pretrained model.
- output: The output directory for the model.
- n_train_patients: The number of patients to use for training.
- seed: The random seed for reproducibility.
- optimizer: The optimizer to use for training.
- weight_decay: The weight decay for the optimizer.
- num_workers: The number of workers for data loading.
- batch_size: The batch size for training.
- lr: The learning rate for the optimizer.
- label_smoothing: Whether to use label smoothing.
- label_smoothing_value: The value for label smoothing.
- class_weights: The weights for the classes.
- loss: The loss function to use.
- task: The task type (regression or classification).
- num_classes: The number of classes for classification.
- scheduler_type: The type of scheduler to use.
- factor: The factor for the scheduler.
- patience: The patience for the scheduler.
- threshold: The threshold for the scheduler.
- lr_step_period: The learning rate step period for the scheduler.
- run_test: Whether to run a test after training.
- tag: The tag for the run.
- project: The project name for the run.
- entity: The entity name for the run.
- resume: Whether to resume training from a checkpoint.
- rand_augment: Whether to use random augmentation.
- resize: The size to resize the images to.
- apply_mask: Whether to apply a mask to the images.
- view_count: The view count for the images.
- save_best: The metric to use for saving the best model.
- metrics_control: The metrics control parameters.
- transforms: The transforms to apply to the image
```

### Using a Python Script

1. Prepare your configuration file in YAML format. This should include all the necessary parameters for your task, such as the task type, model name, number of epochs, batch size, and more. An example configuration file is provided in notebooks/config/config.yaml.

2. Run the main script with torchrun, passing the path to your Python script and the path to your configuration file as arguments. Here is an example command:

`torchrun --standalone --nnodes=1 --nproc-per-node=2 orion/utils/video_training_and_eval.py --config notebooks/config/config.yaml` 

### Using a Jupyter Notebook


1. Prepare your configuration file in YAML format as described above.

2. In your Jupyter notebook, import the necessary modules and set up your environment variables:

```
from orion.utils import video_training_and_eval
import wandb
import yaml
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

2. In your Jupyter notebook, import the necessary modules:

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

