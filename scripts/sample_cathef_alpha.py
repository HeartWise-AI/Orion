#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd


def sample_csv(
    input_path: Path,
    output_dir: Path,
    num_samples: int = 400,
    input_sep: str = "α",
    output_sep: str | None = None,
    random_seed: int = 42,
) -> Path:
    """Load an alpha-separated CSV and save a sampled subset.

    Args:
        input_path: Path to the source CSV file.
        output_dir: Directory to save the sampled CSV into.
        num_samples: Number of rows to sample (will cap at dataset size).
        input_sep: Delimiter used in the input CSV (default: Greek alpha 'α').
        output_sep: Delimiter for the output CSV (default: same as input).
        random_seed: Random seed for reproducibility.

    Returns:
        Path to the saved sampled CSV file.
    """

    if output_sep is None:
        output_sep = input_sep

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path, sep=input_sep, engine="python", encoding="utf-8")

    n = min(num_samples, len(df))
    if n <= 0:
        raise ValueError("No rows available to sample from the input CSV.")

    print(f"Sampling {n} of {len(df)} rows (seed={random_seed})…")
    sampled = df.sample(n=n, random_state=random_seed, replace=False)

    output_name = f"{input_path.stem}_sample_{n}.csv"
    output_path = output_dir / output_name

    print(f"Writing: {output_path}")
    sampled.to_csv(output_path, sep=output_sep, index=False, encoding="utf-8")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample rows from an alpha-separated CSV and write to the data directory."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=(
            "/media/data1/datasets/CathEF/"
            "CathEF_MHI_UCSF_2016-to-july-2022-and-2023-08-30-post-"
            "CathEF_alpha_filtered_30days.csv"
        ),
        help="Path to input CSV (default: CathEF alpha-separated dataset)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/volume/Orion/data",
        help="Output directory to place the sampled CSV (default: /volume/Orion/data)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=400,
        help="Number of rows to sample (default: 400)",
    )
    parser.add_argument(
        "--input-sep",
        type=str,
        default="α",
        help="Input CSV delimiter (default: Greek alpha 'α')",
    )
    parser.add_argument(
        "--output-sep",
        type=str,
        default=None,
        help=(
            "Output CSV delimiter (default: same as input). Use ',' for standard CSV."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve()

    output_path = sample_csv(
        input_path=input_path,
        output_dir=output_dir,
        num_samples=args.n,
        input_sep=args.input_sep,
        output_sep=args.output_sep,
        random_seed=args.seed,
    )

    print(f"Done. Sample saved to: {output_path}")


if __name__ == "__main__":
    main()


