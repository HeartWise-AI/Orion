import argparse
import os


def parse_args():
    """
    Parse arguments given to the script. Dynamically handles additional key-value pair arguments.

    Returns:
        args: The parsed arguments.
        additional_args: A dictionary of additional key-value pair arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb."
    )

    # Use environment variable as default if available
    default_local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parser.add_argument(
        "--local_rank",
        type=int,
        default=default_local_rank,
        metavar="N",
        help="Local process rank.",
    )
    parser.add_argument(
        "--log_all",
        action="store_true",
        help="Flag to log in all processes, otherwise only in rank0",
    )
    # Parse known arguments first
    args, unknown = parser.parse_known_args()

    # Process the additional arguments
    additional_args = {}
    for arg in unknown:
        if arg.startswith("--"):
            key, val = arg.split("=")
            additional_args[key.lstrip("--")] = val

    # If --config_path is not specified in the command line, check additional_args
    if "config_path" not in args:
        args.config_path = additional_args.get("config_path", "config.yaml")
    return args, additional_args
