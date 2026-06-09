"""Unified command-line runner for migrated GPR-FWI experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig, describe_config, load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

SUPPORTED_EXPERIMENTS = {
    ("marmousi", "mode1", "traditional", "eps_only", "lbfgs", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.traditional_eps_only.runner",
        "main",
    ),
    ("marmousi", "mode1", "traditional", "eps_only", "rmsprop", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.traditional_eps_only.runner",
        "main",
    ),
    ("marmousi", "mode2", "traditional", "eps_only", "lbfgs", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.traditional_eps_only.runner",
        "main",
    ),
    ("marmousi", "mode2", "traditional", "eps_only", "rmsprop", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.traditional_eps_only.runner",
        "main",
    ),
    ("marmousi", "mode1", "unet", "eps_only", "adam", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.unet_eps_only.mode1.gpr_train_mpi",
        "main",
    ),
    ("marmousi", "mode1", "unet", "eps_only", "adam", "illumination"): (
        "gpr_inversion.experiments.marmousi.unet_eps_only.mode1_illumination.gpr_train_mpi",
        "main",
    ),
    ("marmousi", "mode2", "unet", "eps_only", "adam", "no_illumination"): (
        "gpr_inversion.experiments.marmousi.unet_eps_only.mode2.gpr_train_mpi",
        "main",
    ),
    ("marmousi", "mode2", "unet", "eps_only", "adam", "illumination"): (
        "gpr_inversion.experiments.marmousi.unet_eps_only.mode2_illumination.gpr_train_mpi",
        "main",
    ),
    ("overthrust", "mode2", "unet", "eps_then_sig", "adam", "illumination"): (
        "gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst",
        "main",
    ),
    ("overthrust", "mode2", "unet", "eps_then_sig", "adam", "no_illumination"): (
        "gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst",
        "main",
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a migrated GPR-FWI experiment from a config file.")
    parser.add_argument("--config", help="Path to a YAML experiment config.")
    parser.add_argument(
        "--list-supported",
        action="store_true",
        help="List config keys that have been connected to the unified runner.",
    )
    parser.add_argument(
        "--list-planned",
        action="store_true",
        help="List valid configs that are not connected to a migrated runner yet.",
    )
    parser.add_argument(
        "--check-configs",
        action="store_true",
        help="Parse every configs/*.yaml file and report whether it is valid.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the parsed config and exit without running inversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dispatch and print the command-equivalent arguments without running inversion.",
    )
    args = parser.parse_args(argv)

    if args.list_supported:
        print("Supported migrated experiments:")
        for key in sorted(SUPPORTED_EXPERIMENTS):
            print(f"  - {'/'.join(key)}")
        return
    if args.list_planned:
        print_planned_configs(CONFIG_DIR)
        return
    if args.check_configs:
        check_configs(CONFIG_DIR)
        return
    if not args.config:
        parser.error("--config is required unless a list/check option is used")

    config = load_experiment_config(args.config)
    print(describe_config(config))

    if args.describe:
        return

    experiment = SUPPORTED_EXPERIMENTS.get(config.key)
    if experiment is None:
        supported = ", ".join("/".join(key) for key in sorted(SUPPORTED_EXPERIMENTS))
        raise SystemExit(
            "This experiment is described by the unified config layer but has not been migrated yet.\n"
            f"Requested: {'/'.join(config.key)}\n"
            f"Supported now: {supported}"
        )

    translated_args = build_legacy_argv(config)
    if args.dry_run:
        print("\nTranslated legacy module arguments:")
        print(" ".join(translated_args))
        return

    module_name, function_name = experiment
    module = __import__(module_name, fromlist=[function_name])
    run_func = getattr(module, function_name)
    run_func(translated_args)


def build_legacy_argv(config: ExperimentConfig) -> list[str]:
    """Translate the unified config into the current migrated experiment's CLI."""
    if config.model == "marmousi" and config.parameterization == "traditional":
        return [
            "--acquisition-mode",
            config.acquisition_mode,
            "--optimizer",
            config.optimizer,
            "--data-path",
            config.data_path,
            "--output-dir",
            config.runtime.output_dir,
            "--learning-rate-eps",
            str(config.training.learning_rate_eps),
            "--alpha-tikhonov-eps",
            str(config.regularization.tikhonov_eps),
            "--num-epochs-stage1",
            str(config.training.num_epochs_stage1),
            "--fixed-top-rows",
            str(config.fixed_top_rows),
            "--snapshot-interval",
            str(config.snapshot_interval),
        ]

    if config.model == "marmousi" and config.parameterization == "unet":
        return [
            "--device",
            config.runtime.device,
            "--data-path",
            config.data_path,
            "--output-dir",
            config.runtime.output_dir,
            "--learning-rate-eps",
            str(config.training.learning_rate_eps),
            "--alpha-tikhonov-eps",
            str(config.regularization.tikhonov_eps),
            "--alpha-tikhonov-sig",
            str(config.regularization.tikhonov_sig),
            "--offset",
            str(config.training.offset),
            "--num-epochs-stage1",
            str(config.training.num_epochs_stage1),
            "--fixed-top-rows",
            str(config.fixed_top_rows),
            "--snapshot-interval",
            str(config.snapshot_interval),
        ]

    argv = [
        "--device",
        config.runtime.device,
        "--data-path",
        config.data_path,
        "--output-dir",
        config.runtime.output_dir,
        "--learning-rate-eps",
        str(config.training.learning_rate_eps),
        "--learning-rate-sig",
        str(config.training.learning_rate_sig),
        "--lr-eps-stage2",
        str(config.training.lr_eps_stage2),
        "--alpha-tv-eps",
        str(config.regularization.tv_eps),
        "--alpha-tv-sig",
        str(config.regularization.tv_sig),
        "--alpha-tikhonov-eps",
        str(config.regularization.tikhonov_eps),
        "--alpha-tikhonov-sig",
        str(config.regularization.tikhonov_sig),
        "--alpha-l1-data",
        str(config.training.alpha_l1_data),
        "--offset",
        str(config.training.offset),
        "--num-epochs-stage1",
        str(config.training.num_epochs_stage1),
        "--num-epochs-stage2",
        str(config.training.num_epochs_stage2),
        "--fixed-top-rows",
        str(config.fixed_top_rows),
        "--snapshot-interval",
        str(config.snapshot_interval),
    ]
    if config.illumination:
        argv.append("--illumination")
    if config.runtime.resume:
        argv.append("--resume")
    if config.runtime.stage1_checkpoint_path:
        argv.extend(["--stage1-checkpoint-path", config.runtime.stage1_checkpoint_path])
    return argv


def print_planned_configs(config_dir: Path) -> None:
    """Print valid configs that are not yet registered as supported experiments."""
    planned: list[ExperimentConfig] = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        config = load_experiment_config(config_path)
        if config.key not in SUPPORTED_EXPERIMENTS or config.migration_status == "planned":
            planned.append(config)

    if not planned:
        print("No planned configs found.")
        return

    print("Planned experiment configs:")
    for config in planned:
        print(f"  - {config.name}: {'/'.join(config.key)}")
        if config.source_path:
            print(f"    source: {config.source_path}")


def check_configs(config_dir: Path) -> None:
    """Validate all YAML configs without running any inversion."""
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"No config files found in {config_dir}")

    for config_path in config_paths:
        config = load_experiment_config(config_path)
        status = "supported" if config.key in SUPPORTED_EXPERIMENTS else "planned"
        print(f"OK {config_path.name}: {'/'.join(config.key)} [{status}]")


if __name__ == "__main__":
    main()
