import wandb
import yaml


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Load sweep configuration
sweep_conf_file_path = 'notebooks/config/sweep_config.yaml'
sweep_conf = load_yaml_config(sweep_conf_file_path)

# Number of runs to execute
count = 1

# Initialize the sweep
sweep_id = wandb.sweep(
    sweep=sweep_conf,
    project="DeepRV_V2",
    entity="jacques-delfrate"
)

# Start the sweep agent
wandb.agent(
    sweep_id=sweep_id,
    project="DeepRV_V2",
    entity="jacques-delfrate",
    count=count
)