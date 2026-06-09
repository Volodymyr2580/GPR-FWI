#!/usr/bin/env python3
"""Optuna search using inverted-model MSE and SSIM as the objective."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overthrust" / "optuna_jax_model_quality"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search JAX OverThrust eps-then-sig hyperparameters using model MSE and SSIM."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mpi-processes", type=int, default=3)
    parser.add_argument("--study-name", default="overthrust_jax_model_quality")
    parser.add_argument("--storage", default="")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--num-epochs-stage1", type=int, default=5000)
    parser.add_argument("--num-epochs-stage2", type=int, default=2500)
    parser.add_argument("--stage1-checkpoint-path", default="")
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    parser.add_argument("--fixed-top-rows", type=int, default=20)
    parser.add_argument("--stage1-data-loss", choices=("l1", "l2", "mixed"), default="l1")
    parser.add_argument("--stage2-data-loss", choices=("l1", "l2", "mixed"), default="l2")
    parser.add_argument("--fdtd-backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--no-illumination", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax-device", default="auto")
    parser.add_argument("--auto-jax-shot-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-jax-shot-batch-size", dest="auto_jax_shot_batch_size", action="store_false")
    parser.add_argument("--max-jax-shot-batch-size", type=int, default=5)
    parser.add_argument("--jax-shot-batch-size", type=int, default=5)
    parser.add_argument("--learning-rate-eps", type=float, default=1e-4)
    parser.add_argument("--lr-eps-stage2", type=float, default=None)
    parser.add_argument("--lr-eps-stage2-min", type=float, default=1e-6)
    parser.add_argument("--lr-eps-stage2-max", type=float, default=1e-4)
    parser.add_argument("--learning-rate-sig-min", type=float, default=1e-6)
    parser.add_argument("--learning-rate-sig-max", type=float, default=1e-4)
    parser.add_argument("--alpha-tv-eps", type=float, default=None)
    parser.add_argument("--alpha-tv-eps-min", type=float, default=1e-2)
    parser.add_argument("--alpha-tv-eps-max", type=float, default=5e-2)
    parser.add_argument("--alpha-tv-sig-min", type=float, default=3e-2)
    parser.add_argument("--alpha-tv-sig-max", type=float, default=4e-2)
    parser.add_argument("--sigma-param", choices=("linear", "log"), default="linear")
    parser.add_argument("--sigma-tv-domain", choices=("physical", "log"), default="physical")
    parser.add_argument("--sigma-beta", type=float, default=None)
    parser.add_argument("--sigma-beta-min", type=float, default=0.2)
    parser.add_argument("--sigma-beta-max", type=float, default=1.0)
    parser.add_argument("--alpha-l1-data", type=float, default=None)
    parser.add_argument("--alpha-l1-data-max", type=float, default=0.5)
    parser.add_argument("--alpha-tikhonov-eps", type=float, default=0.0)
    parser.add_argument("--alpha-tikhonov-sig", type=float, default=0.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--eps-weight", type=float, default=1.0)
    parser.add_argument("--sig-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=1.0)
    parser.add_argument("--model-error-metric", choices=("nmse", "mse"), default="nmse")
    parser.add_argument("--eps-min-ssim", type=float, default=0.0)
    parser.add_argument("--eps-max-nmse", type=float, default=math.inf)
    parser.add_argument("--eps-ssim-penalty-weight", type=float, default=0.0)
    parser.add_argument("--eps-nmse-penalty-weight", type=float, default=0.0)
    parser.add_argument("--objective-region", choices=("active", "full"), default="active")
    parser.add_argument("--objective-source", choices=("final", "best-history"), default="final")
    parser.add_argument("--history-max-epoch", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def fixed_or_suggest_float(
    trial,
    name: str,
    fixed_value: float | None,
    low: float,
    high: float,
    *,
    log: bool = False,
) -> float:
    if fixed_value is not None:
        return float(fixed_value)
    if low == high:
        return float(low)
    return float(trial.suggest_float(name, low, high, log=log))


def suggest_parameters(trial, args: argparse.Namespace) -> dict[str, float]:
    return {
        "learning_rate_eps": args.learning_rate_eps,
        "lr_eps_stage2": fixed_or_suggest_float(
            trial,
            "lr_eps_stage2",
            args.lr_eps_stage2,
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
        "alpha_tv_eps": fixed_or_suggest_float(
            trial,
            "alpha_tv_eps",
            args.alpha_tv_eps,
            args.alpha_tv_eps_min,
            args.alpha_tv_eps_max,
        ),
        "alpha_tv_sig": trial.suggest_float(
            "alpha_tv_sig",
            args.alpha_tv_sig_min,
            args.alpha_tv_sig_max,
            log=True,
        ),
        "sigma_beta": fixed_or_suggest_float(
            trial,
            "sigma_beta",
            args.sigma_beta,
            args.sigma_beta_min,
            args.sigma_beta_max,
            log=True,
        ),
        "alpha_l1_data": fixed_or_suggest_float(
            trial,
            "alpha_l1_data",
            args.alpha_l1_data,
            0.0,
            args.alpha_l1_data_max,
        ),
    }


def build_command(args: argparse.Namespace, params: dict[str, float], trial_dir: Path) -> list[str]:
    command = [
        "mpirun",
        "-np",
        str(args.mpi_processes),
        args.python_executable,
        "-m",
        "gpr_inversion.experiments.overthrust.mode2_eps_then_sig.twopara_epsfirst",
        "--fdtd-backend",
        args.fdtd_backend,
        "--data-path",
        args.data_path,
        "--output-dir",
        str(trial_dir),
        "--fixed-top-rows",
        str(args.fixed_top_rows),
        "--num-epochs-stage1",
        str(args.num_epochs_stage1),
        "--num-epochs-stage2",
        str(args.num_epochs_stage2),
        "--stage1-data-loss",
        args.stage1_data_loss,
        "--stage2-data-loss",
        args.stage2_data_loss,
        "--device",
        args.device,
        "--jax-device",
        args.jax_device,
        "--max-jax-shot-batch-size",
        str(args.max_jax_shot_batch_size),
        "--jax-shot-batch-size",
        str(args.jax_shot_batch_size),
        "--snapshot-interval",
        str(args.snapshot_interval),
        "--learning-rate-eps",
        format(params["learning_rate_eps"], ".8g"),
        "--lr-eps-stage2",
        format(params["lr_eps_stage2"], ".8g"),
        "--learning-rate-sig",
        format(params["learning_rate_sig"], ".8g"),
        "--alpha-tv-eps",
        format(params["alpha_tv_eps"], ".8g"),
        "--alpha-tv-sig",
        format(params["alpha_tv_sig"], ".8g"),
        "--sigma-param",
        args.sigma_param,
        "--sigma-tv-domain",
        args.sigma_tv_domain,
        "--sigma-beta",
        format(params["sigma_beta"], ".8g"),
        "--alpha-l1-data",
        format(params["alpha_l1_data"], ".8g"),
        "--alpha-tikhonov-eps",
        format(args.alpha_tikhonov_eps, ".8g"),
        "--alpha-tikhonov-sig",
        format(args.alpha_tikhonov_sig, ".8g"),
        "--offset",
        format(args.offset, ".8g"),
    ]
    if not args.no_illumination:
        command.append("--illumination")
    if args.auto_jax_shot_batch_size:
        command.append("--auto-jax-shot-batch-size")
    if args.stage1_checkpoint_path:
        command.extend(
            [
                "--resume",
                "--stage1-checkpoint-path",
                args.stage1_checkpoint_path,
            ]
        )
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


def load_final_metrics(metrics_path: Path) -> dict[str, object]:
    payload = load_metrics_payload(metrics_path)
    return dict(payload.get("final") or {})


def load_metrics_payload(metrics_path: Path) -> dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as f:
        return dict(json.load(f))


def compute_objective(final_metrics: dict[str, object], args: argparse.Namespace) -> float:
    suffix = "_active" if args.objective_region == "active" else ""
    error_suffix = "_mse" if args.model_error_metric == "mse" else "_nmse"
    eps_error = float(final_metrics[f"eps_model{suffix}{error_suffix}"])
    sig_error = float(final_metrics[f"sig_model{suffix}{error_suffix}"])
    eps_ssim = float(final_metrics[f"eps_model{suffix}_ssim"])
    sig_ssim = float(final_metrics[f"sig_model{suffix}_ssim"])
    objective = (
        args.eps_weight * (args.mse_weight * eps_error + args.ssim_weight * (1.0 - eps_ssim))
        + args.sig_weight * (args.mse_weight * sig_error + args.ssim_weight * (1.0 - sig_ssim))
    )
    eps_nmse = float(final_metrics[f"eps_model{suffix}_nmse"])
    if args.eps_min_ssim > 0.0 and eps_ssim < args.eps_min_ssim:
        objective += args.eps_ssim_penalty_weight * (args.eps_min_ssim - eps_ssim)
    if math.isfinite(args.eps_max_nmse) and eps_nmse > args.eps_max_nmse:
        objective += args.eps_nmse_penalty_weight * (eps_nmse - args.eps_max_nmse)
    if not math.isfinite(objective):
        raise ValueError(f"Objective is not finite: {objective}")
    return float(objective)


def select_objective_metrics(
    metrics_payload: dict[str, object], args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object]]:
    final_metrics = dict(metrics_payload.get("final") or {})
    if args.objective_source == "final":
        return final_metrics, {"objective_source": "final"}

    history = [dict(row) for row in metrics_payload.get("history") or []]
    if args.history_max_epoch >= 0:
        history = [row for row in history if int(row.get("epoch", -1)) <= args.history_max_epoch]
    if not history:
        return final_metrics, {"objective_source": "final_no_history"}

    best_metrics = min(history, key=lambda row: compute_objective(row, args))
    selection = {
        "objective_source": "best-history",
        "objective_epoch": best_metrics.get("epoch"),
        "final_objective": compute_objective(final_metrics, args) if final_metrics else None,
    }
    return best_metrics, selection


def append_summary(summary_path: Path, row: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run `python -m pip install optuna` "
            "or use scripts/run_overthrust_model_quality_optuna.sh with INSTALL_OPTUNA=1."
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_dir / 'optuna_study.db'}"
    summary_path = output_dir / "trials_summary.jsonl"

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

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
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
        metrics_payload = load_metrics_payload(metrics_path)
        final_metrics = dict(metrics_payload.get("final") or {})
        objective_metrics, objective_selection = select_objective_metrics(metrics_payload, args)
        objective_value = compute_objective(objective_metrics, args)
        trial.set_user_attr("metrics_path", str(metrics_path))
        for key, value in objective_selection.items():
            trial.set_user_attr(key, value)
        for key in (
            "eps_model_active_nmse",
            "sig_model_active_nmse",
            "eps_model_active_ssim",
            "sig_model_active_ssim",
            "model_quality_objective_active",
            "eps_model_nmse",
            "sig_model_nmse",
            "eps_model_ssim",
            "sig_model_ssim",
            "model_quality_objective_full",
            "data_misfit_mse",
        ):
            if key in final_metrics:
                trial.set_user_attr(key, final_metrics[key])
        append_summary(
            summary_path,
            {
                "trial": trial.number,
                "status": "completed",
                "objective": objective_value,
                "objective_selection": objective_selection,
                "objective_metrics": objective_metrics,
                "final_metrics": final_metrics,
                "params": params,
                "trial_dir": str(trial_dir),
                "metrics_path": str(metrics_path),
            },
        )
        print(f"[trial {trial.number}] model_quality_objective={objective_value:.8e}", flush=True)
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    print("\nBest trial")
    print(f"  number: {study.best_trial.number}")
    print(f"  model_quality_objective: {study.best_value:.8e}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"  trial_dir: {study.best_trial.user_attrs.get('trial_dir', '')}")


if __name__ == "__main__":
    main()
