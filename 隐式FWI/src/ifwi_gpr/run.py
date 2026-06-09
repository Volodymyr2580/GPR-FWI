"""Command-line entry point for IFWI experiments."""

from __future__ import annotations

import argparse
import json

from ifwi_gpr.config import load_config
from ifwi_gpr.train import dry_run, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an IFWI GPR experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate network and config without running FDTD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run(config), indent=2))
        return
    run_dir = train_from_config(config)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()

