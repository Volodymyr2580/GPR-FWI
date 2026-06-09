#!/usr/bin/env python3
"""Optuna launcher for the OverThrust mode2 eps-then-sig experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "overthrust" / "OverThrust.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overthrust" / "optuna_eps_then_sig"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search OverThrust mode2 eps/sig hyperparameters with Optuna."
    )
    parser.add_argument("--stage1-checkpoint-path", required=True)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mpi-processes", type=int, default=30)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--study-name", default="overthrust_mode2_eps_then_sig")
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL. Defaults to sqlite:///<output-dir>/optuna_study.db.",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--num-epochs-stage2", type=int, default=1000)
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--learning-rate-eps", type=float, default=1e-4)
    parser.add_argument("--lr-eps-stage2-min", type=float, default=1e-6)
    parser.add_argument("--lr-eps-stage2-max", type=float, default=1e-4)
    parser.add_argument("--learning-rate-sig-min", type=float, default=1e-6)
    parser.add_argument("--learning-rate-sig-max", type=float, default=1e-4)
    parser.add_argument("--alpha-tv-eps-min", type=float, default=1e-2)
    parser.add_argument("--alpha-tv-eps-max", type=float, default=5e-2)
    parser.add_argument("--alpha-tv-sig", type=float, default=3.5e-2)
    parser.add_argument("--search-alpha-tv-sig", action="store_true")
    parser.add_argument("--alpha-tv-sig-min", type=float, default=3e-2)
    parser.add_argument("--alpha-tv-sig-max", type=float, default=4e-2)
    parser.add_argument("--alpha-l1-data", type=float, default=0.0)
    parser.add_argument("--search-alpha-l1-data", action="store_true")
    parser.add_argument("--alpha-l1-data-max", type=float, default=0.5)
    parser.add_argument("--stage1-data-loss", choices=("l1", "l2", "mixed"), default="l1")
    parser.add_argument("--stage2-data-loss", choices=("l1", "l2", "mixed"), default="l2")
    parser.add_argument("--alpha-tikhonov-eps", type=float, default=0.0)
    parser.add_argument("--alpha-tikhonov-sig", type=float, default=0.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--fixed-top-rows", type=int, default=10)
    parser.add_argument("--fdtd-backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--no-illumination", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def suggest_parameters(trial, args: argparse.Namespace) -> dict[str, float]:
    params = {
        "learning_rate_eps": args.learning_rate_eps,
        "lr_eps_stage2": trial.suggest_float(
            "lr_eps_stage2",
            args.lr_eps_stage2_min,
            args.lr_eps_stage2_max,
            log=True,
        ),
        "learning_rate_sig": trial.suggest_float(
            "learning_rate_sig",
            args.learning_rate_sig_min,
            args.learning_rate_sig_max,
            log=True,
        ),
        "alpha_tv_eps": trial.suggest_float(
            "alpha_tv_eps",
            args.alpha_tv_eps_min,
            args.alpha_tv_eps_max,
        ),
        "alpha_tv_sig": args.alpha_tv_sig,
        "alpha_l1_data": args.alpha_l1_data,
    }
    if args.search_alpha_tv_sig:
        params["alpha_tv_sig"] = trial.suggest_float(
            "alpha_tv_sig",
            args.alpha_tv_sig_min,
            args.alpha_tv_sig_max,
        )
    if args.search_alpha_l1_data:
        params["alpha_l1_data"] = trial.suggest_float(
            "alpha_l1_data",
            0.0,
            args.alpha_l1_data_max,
        )
    return params


def build_command(args: argparse.Namespace, params: dict[str, float], trial_dir: Path) -> list[str]:
    command = [
        "mpirun",
        "-np",
        str(args.mpi_processes),
        args.python_executable,
        "-m",
        "gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst",
        "--stage1-checkpoint-path",
        args.stage1_checkpoint_path,
        "--resume",
        "--device",
        args.device,
        "--data-path",
        args.data_path,
        "--output-dir",
        str(trial_dir),
        "--learning-rate-eps",
        format(params["learning_rate_eps"], ".8g"),
        "--learning-rate-sig",
        format(params["learning_rate_sig"], ".8g"),
        "--lr-eps-stage2",
        format(params["lr_eps_stage2"], ".8g"),
        "--alpha-tv-eps",
        format(params["alpha_tv_eps"], ".8g"),
        "--alpha-tv-sig",
        format(params["alpha_tv_sig"], ".8g"),
        "--alpha-l1-data",
        format(params["alpha_l1_data"], ".8g"),
        "--stage1-data-loss",
        args.stage1_data_loss,
        "--stage2-data-loss",
        args.stage2_data_loss,
        "--alpha-tikhonov-eps",
        format(args.alpha_tikhonov_eps, ".8g"),
        "--alpha-tikhonov-sig",
        format(args.alpha_tikhonov_sig, ".8g"),
        "--offset",
        format(args.offset, ".8g"),
        "--num-epochs-stage2",
        str(args.num_epochs_stage2),
        "--fixed-top-rows",
        str(args.fixed_top_rows),
        "--fdtd-backend",
        args.fdtd_backend,
        "--snapshot-interval",
        str(args.snapshot_interval),
    ]
    if not args.no_illumination:
        command.append("--illumination")
    return command


def find_latest_metrics(trial_dir: Path) -> Path:
    candidates = sorted(
        trial_dir.rglob("metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No metrics.json found under {trial_dir}")
    return candidates[0]


def load_objective_value(metrics_path: Path) -> float:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    final_metrics = metrics.get("final") or {}
    value = final_metrics.get("data_misfit_mse")
    if value is None:
        raise KeyError(f"{metrics_path} does not contain final.data_misfit_mse")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Objective value is not finite: {value}")
    return value


def load_final_metrics(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return dict(metrics.get("final") or {})


def append_summary(summary_path: Path, row: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run `python -m pip install optuna` "
            "or use scripts/run_overthrust_optuna_search.sh with INSTALL_OPTUNA=1."
        ) from exc

    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_dir / 'optuna_study.db'}"

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    summary_path = output_dir / "trials_summary.jsonl"

    def objective(trial) -> float:
        params = suggest_parameters(trial, args)
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, params, trial_dir)
        command_text = " ".join(command)

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("command", command_text)
        append_summary(
            summary_path,
            {
                "trial": trial.number,
                "status": "started",
                "params": params,
                "trial_dir": str(trial_dir),
                "command": command_text,
            },
        )
        print(f"\n[trial {trial.number}] {params}", flush=True)
        print(f"[trial {trial.number}] {command_text}", flush=True)

        if args.dry_run:
            return float("inf")

        env = os.environ.copy()
        src_path = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_path
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            append_summary(
                summary_path,
                {
                    "trial": trial.number,
                    "status": "failed",
                    "returncode": completed.returncode,
                    "params": params,
                    "trial_dir": str(trial_dir),
                },
            )
            raise RuntimeError(f"Trial {trial.number} failed with return code {completed.returncode}")

        metrics_path = find_latest_metrics(trial_dir)
        final_metrics = load_final_metrics(metrics_path)
        objective_value = load_objective_value(metrics_path)
        trial.set_user_attr("metrics_path", str(metrics_path))
        for key in (
            "data_misfit_mse",
            "eps_model_loss",
            "sig_model_loss",
            "eps_residual_mae",
            "sig_residual_mae",
            "model_residual_score",
        ):
            if key in final_metrics:
                trial.set_user_attr(key, final_metrics[key])
        append_summary(
            summary_path,
            {
                "trial": trial.number,
                "status": "completed",
                "objective": objective_value,
                "final_metrics": final_metrics,
                "params": params,
                "trial_dir": str(trial_dir),
                "metrics_path": str(metrics_path),
            },
        )
        print(f"[trial {trial.number}] data_misfit_mse={objective_value:.8e}", flush=True)
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    print("\nBest trial")
    print(f"  number: {study.best_trial.number}")
    print(f"  data_misfit_mse: {study.best_value:.8e}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"  trial_dir: {study.best_trial.user_attrs.get('trial_dir', '')}")


if __name__ == "__main__":
    main()
