import argparse

from dataclasses import dataclass

from orion.utils.video_run_inference import run_inference


@dataclass
class Args:
    config_path: str
    splits: list[str]
    log_wandb: bool
    model_path: str
    data_path: str
    output_dir: str
    wandb_id: str
    resume: bool

    @staticmethod
    def verify_splits(splits: list[str]) -> list[str]:
        valid_splits = {'val', 'test', 'inference'}
        invalid_splits = [split for split in splits if split not in valid_splits]
        if invalid_splits:
            raise ValueError(
                f"Invalid split(s): {invalid_splits}. "
                f"Splits must be one or more of: {', '.join(valid_splits)}"
            )
        return splits

    @classmethod
    def parse_config(cls) -> 'Args':
        parser = argparse.ArgumentParser(description="Run inference on a video dataset")
        parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
        parser.add_argument(
            "--split", 
            type=str, 
            nargs='+', 
            required=True, 
            help="Splits to run inference on (e.g., --split test val inference)"
        )
        parser.add_argument("--log_wandb", type=bool, default=False, help="Log to wandb")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model file")
        parser.add_argument("--data_path", type=str, default=None, help="Path to the data file")
        parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
        parser.add_argument("--wandb_id", type=str, default=None, help="Wandb id")
        parser.add_argument("--resume", type=bool, default=False, help="Resume training")
        args = parser.parse_args()
        
        # Verify splits before creating the dataclass
        verified_splits = cls.verify_splits(args.split)
        
        return cls(
            config_path=args.config_path,
            splits=verified_splits,
            log_wandb=args.log_wandb,
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            wandb_id=args.wandb_id,
            resume=args.resume
        )

def main(args: Args):
    df_dict = {}
    for split in args.splits:
        print(split)
        df_dict[split] = run_inference(
            config_path=args.config_path,
            split=split,
            log_wandb=args.log_wandb,
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            wandb_id=args.wandb_id,
            resume=args.resume,
        )
    return df_dict

if __name__ == "__main__":
    args: Args = Args.parse_config()
    main(args)
