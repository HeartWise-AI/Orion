import argparse
import os


def parse_args():
    """
    Parse arguments given to the script.

    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb.")
    
    # Use environment variable as default if available
    default_local_rank = int(os.environ.get('LOCAL_RANK', 0))

    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=default_local_rank, 
        metavar="N", 
        help="Local process rank."
    )
    parser.add_argument(
        "--log_all",
        action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )  
    # Add a new argument for the configuration file path
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the configuration file"
    )

    args = parser.parse_args()

    return args