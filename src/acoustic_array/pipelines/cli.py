"""Command-line entry point for reproducible reference experiments."""

import argparse
from pathlib import Path

from acoustic_array.pipelines.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Microphone-array engineering references")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    run = run_experiment(args.config, args.output_root)
    print(run)


if __name__ == "__main__":
    main()
