import wandb
import yaml
import sys
import os


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Load sweep configuration
sweep_conf_file_path = 'notebooks/config/sweep_config.yaml'
sweep_conf = load_yaml_config(sweep_conf_file_path)

# Number of runs to execute - can be passed as command line argument
if len(sys.argv) > 1:
    count = int(sys.argv[1])
    print(f"Running {count} sweep experiments")
else:
    count = 1
    print("Running 1 sweep experiment (default). Pass a number as argument to run more.")

# Initialize the sweep
# Use project/entity from environment or defaults
project = os.getenv("WANDB_PROJECT", "orion-x3d-debug-binary")
entity = os.getenv("WANDB_ENTITY", "mhi_ai")

print(f"Starting W&B sweep for project: {project}, entity: {entity}")

sweep_id = wandb.sweep(
    sweep=sweep_conf,
    project=project,
    entity=entity
)

# Start the sweep agent
wandb.agent(
    sweep_id=sweep_id,
    project=project,
    entity=entity,
    count=count
)