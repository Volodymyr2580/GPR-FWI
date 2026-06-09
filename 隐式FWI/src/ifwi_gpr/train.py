"""Minimal IFWI training loop."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ifwi_gpr.acquisition import build_acquisition
from ifwi_gpr.autograd import cpu_mpi_forward
from ifwi_gpr.config import require_section
from ifwi_gpr.grids import make_normalized_grid
from ifwi_gpr.io import save_snapshot, unique_run_dir, write_json
from ifwi_gpr.models import build_cross_shape_model
from ifwi_gpr.networks import IFWIFrInrNetwork, IFWINetwork
from ifwi_gpr.parameterization import ParameterBounds, ParameterStats
from ifwi_gpr.plotting import generate_run_figures
from ifwi_gpr.solver_bridge import CpuMpiSolverBridge, SolverSettings


def build_network(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = require_section(config, "model")
    network_cfg = require_section(config, "network")
    param_cfg = require_section(config, "parameters")
    parameter_mapping = build_parameter_mapping(param_cfg)
    architecture = str(network_cfg.get("architecture", "siren")).lower()
    shape = tuple(model_cfg["shape"])
    hidden_features = int(network_cfg.get("hidden_features", 128))
    hidden_layers = int(network_cfg.get("hidden_layers", 4))
    dropout = float(network_cfg.get("dropout", 0.0))

    if architecture in {"siren", "ifwi_siren"}:
        return IFWINetwork(
            parameter_mapping,
            shape=shape,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega0=float(network_cfg.get("omega0", 30.0)),
            dropout=dropout,
        ).to(device)

    if architecture in {"fr_inr", "fr-inr", "frinr"}:
        return IFWIFrInrNetwork(
            parameter_mapping,
            shape=shape,
            mode=str(network_cfg.get("mode", "sin+fr")),
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            high_freq_num=int(network_cfg.get("high_freq_num", 8)),
            low_freq_num=int(network_cfg.get("low_freq_num", 8)),
            phi_num=int(network_cfg.get("phi_num", 8)),
            alpha=float(network_cfg.get("alpha", 0.01)),
            first_omega0=float(network_cfg.get("first_omega0", network_cfg.get("omega0", 30.0))),
            hidden_omega0=float(network_cfg.get("hidden_omega0", network_cfg.get("omega0", 30.0))),
            dropout=dropout,
        ).to(device)

    raise ValueError(f"Unsupported network architecture: {architecture}")


def build_parameter_mapping(param_cfg: dict[str, Any]) -> ParameterBounds | ParameterStats:
    mapping = str(param_cfg.get("mapping", "bounds")).lower()
    if mapping in {"bounds", "bounded", "sigmoid"}:
        return ParameterBounds(
            epsilon_min=float(param_cfg["epsilon_min"]),
            epsilon_max=float(param_cfg["epsilon_max"]),
            sigma_min=float(param_cfg["sigma_min"]),
            sigma_max=float(param_cfg["sigma_max"]),
        )
    if mapping in {"standardized", "mean_std", "stats"}:
        return ParameterStats(
            epsilon_mean=float(param_cfg["epsilon_mean"]),
            epsilon_std=float(param_cfg["epsilon_std"]),
            sigma_mean=float(param_cfg["sigma_mean"]),
            sigma_std=float(param_cfg["sigma_std"]),
            epsilon_min=_optional_float(param_cfg, "epsilon_min"),
            epsilon_max=_optional_float(param_cfg, "epsilon_max"),
            sigma_min=_optional_float(param_cfg, "sigma_min"),
            sigma_max=_optional_float(param_cfg, "sigma_max"),
        )
    raise ValueError(f"Unsupported parameter mapping: {mapping}")


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    if key not in config:
        return None
    return float(config[key])


def build_true_model(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model_cfg = require_section(config, "model")
    kind = model_cfg.get("kind")
    if kind != "cross_shape":
        raise ValueError("Only 'cross_shape' is implemented in this first scaffold")
    return build_cross_shape_model(
        tuple(model_cfg["shape"]),
        epsilon_background=float(model_cfg.get("epsilon_background", 4.0)),
        sigma_background=float(model_cfg.get("sigma_background", 0.003)),
    )


def build_initial_model(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build the pre-inversion model that FR-INR should represent first."""
    model_cfg = require_section(config, "model")
    initial_cfg = config.get("initial", {})
    if initial_cfg is not None and not isinstance(initial_cfg, dict):
        raise ValueError("Config section 'initial' must be a mapping when provided")

    shape = tuple(model_cfg["shape"])
    epsilon = _load_initial_parameter(
        initial_cfg,
        value_key="epsilon",
        path_key="epsilon_path",
        default_value=float(model_cfg.get("epsilon_background", 4.0)),
        shape=shape,
    )
    sigma = _load_initial_parameter(
        initial_cfg,
        value_key="sigma",
        path_key="sigma_path",
        default_value=float(model_cfg.get("sigma_background", 0.003)),
        shape=shape,
    )
    return epsilon, sigma


def _load_initial_parameter(
    initial_cfg: dict[str, Any],
    *,
    value_key: str,
    path_key: str,
    default_value: float,
    shape: tuple[int, ...],
) -> np.ndarray:
    path = initial_cfg.get(path_key)
    if path:
        array = np.load(Path(path))
        if tuple(array.shape) != shape:
            raise ValueError(
                f"Initial parameter '{path_key}' has shape {tuple(array.shape)}, "
                f"expected {shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Initial parameter '{path_key}' contains non-finite values")
        return array.astype(np.float32, copy=False)

    value = float(initial_cfg.get(value_key, default_value))
    return np.full(shape, value, dtype=np.float32)


def build_solver(config: dict[str, Any]) -> tuple[CpuMpiSolverBridge, SolverSettings]:
    model_cfg = require_section(config, "model")
    solver_cfg = require_section(config, "solver")
    bridge = CpuMpiSolverBridge(solver_cfg.get("reference_project"))
    settings = SolverSettings(
        dt=float(solver_cfg["dt"]),
        dx=float(model_cfg["dx"]),
        dz=float(model_cfg["dz"]),
        npml=int(solver_cfg["npml"]),
        freq=float(solver_cfg["freq"]),
        steps=int(solver_cfg["steps"]),
        gradient_normalization=str(solver_cfg.get("gradient_normalization", "none")),
        epsilon_gradient_weight=float(solver_cfg.get("epsilon_gradient_weight", 1.0)),
        sigma_gradient_weight=float(solver_cfg.get("sigma_gradient_weight", 1.0)),
        epsilon_gradient_mode=str(solver_cfg.get("epsilon_gradient_mode", "reference")),
    )
    return bridge, settings


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    """Instantiate the network and make one parameterized model without FDTD."""
    device = torch.device(config.get("device", "cpu"))
    torch.manual_seed(int(config.get("seed", 2025)))
    network = build_network(config, device)
    coordinates = make_normalized_grid(tuple(require_section(config, "model")["shape"]), device=device)
    epsilon, sigma = network(coordinates)
    return {
        "epsilon_shape": list(epsilon.shape),
        "sigma_shape": list(sigma.shape),
        "epsilon_finite": bool(torch.isfinite(epsilon).all().item()),
        "sigma_finite": bool(torch.isfinite(sigma).all().item()),
        "parameter_count": sum(p.numel() for p in network.parameters()),
    }


def train_from_config(config: dict[str, Any]) -> Path:
    """Run a small IFWI experiment from a config file."""
    device = torch.device(config.get("device", "cpu"))
    torch.manual_seed(int(config.get("seed", 2025)))
    np.random.seed(int(config.get("seed", 2025)))

    run_dir = unique_run_dir(config.get("output_dir", "runs/ifwi_run"))
    write_json(run_dir / "config.json", config)

    network = build_network(config, device)
    model_shape = tuple(require_section(config, "model")["shape"])
    coordinates = make_normalized_grid(model_shape, device=device)
    epsilon_true_np, sigma_true_np = build_true_model(config)
    epsilon_initial_target_np, sigma_initial_target_np = build_initial_model(config)
    np.save(run_dir / "true_epsilon.npy", epsilon_true_np)
    np.save(run_dir / "true_sigma.npy", sigma_true_np)
    np.save(run_dir / "initial_target_epsilon.npy", epsilon_initial_target_np)
    np.save(run_dir / "initial_target_sigma.npy", sigma_initial_target_np)

    train_cfg = require_section(config, "training")
    pretrain_history = _pretrain_network(
        network,
        coordinates,
        epsilon_initial_target_np,
        sigma_initial_target_np,
        train_cfg,
        device,
    )

    network.eval()
    with torch.no_grad():
        epsilon_initial, sigma_initial = network(coordinates)
        epsilon_initial_np = epsilon_initial.detach().cpu().numpy()
        sigma_initial_np = sigma_initial.detach().cpu().numpy()
    np.save(run_dir / "initial_epsilon.npy", epsilon_initial_np)
    np.save(run_dir / "initial_sigma.npy", sigma_initial_np)

    bridge, settings = build_solver(config)

    acquisition_cfg = require_section(config, "acquisition")
    source_list, receiver_list = build_acquisition(acquisition_cfg, model_shape)

    d_obs_np = bridge.forward(
        epsilon_true_np,
        sigma_true_np,
        source_list,
        receiver_list,
        settings,
        save_wavefield=False,
    )
    d_obs = torch.as_tensor(d_obs_np, dtype=torch.float32, device=device)
    initial_synthetic_np = bridge.forward(
        epsilon_initial_np,
        sigma_initial_np,
        source_list,
        receiver_list,
        settings,
        save_wavefield=False,
    )
    np.save(run_dir / "observed_data.npy", d_obs_np)
    np.save(run_dir / "initial_synthetic_data.npy", initial_synthetic_np)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        betas=(0.9, 0.95),
    )
    scheduler = _build_scheduler(optimizer, train_cfg)
    epochs = int(train_cfg.get("epochs", 1))
    snapshot_interval = int(train_cfg.get("snapshot_interval", 1))
    fixed_epsilon = torch.as_tensor(
        epsilon_initial_target_np,
        dtype=torch.float32,
        device=device,
    )
    fixed_sigma = torch.as_tensor(
        sigma_initial_target_np,
        dtype=torch.float32,
        device=device,
    )
    freeze_epsilon = bool(train_cfg.get("freeze_epsilon", False))
    freeze_sigma = bool(train_cfg.get("freeze_sigma", False))
    freeze_epsilon_epochs = int(train_cfg.get("freeze_epsilon_epochs", 0))
    freeze_sigma_epochs = int(train_cfg.get("freeze_sigma_epochs", 0))
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_epsilon_np: np.ndarray | None = None
    best_sigma_np: np.ndarray | None = None
    network.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epsilon, sigma = network(coordinates)
        current_freeze_epsilon = _is_parameter_frozen(
            epoch,
            freeze_always=freeze_epsilon,
            freeze_epochs=freeze_epsilon_epochs,
        )
        current_freeze_sigma = _is_parameter_frozen(
            epoch,
            freeze_always=freeze_sigma,
            freeze_epochs=freeze_sigma_epochs,
        )
        epsilon, sigma = _apply_fixed_parameters(
            epsilon,
            sigma,
            fixed_epsilon=fixed_epsilon,
            fixed_sigma=fixed_sigma,
            freeze_epsilon=current_freeze_epsilon,
            freeze_sigma=current_freeze_sigma,
        )
        d_syn = cpu_mpi_forward(
            epsilon,
            sigma,
            bridge,
            source_list,
            receiver_list,
            settings,
        )
        loss = F.mse_loss(d_syn, d_obs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.detach().cpu().item())
        history.append({"epoch": epoch, "loss": loss_value})
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_epsilon_np = epsilon.detach().cpu().numpy()
            best_sigma_np = sigma.detach().cpu().numpy()
        if epoch == 1 or epoch % snapshot_interval == 0 or epoch == epochs:
            with torch.no_grad():
                eps_np = epsilon.detach().cpu().numpy()
                sig_np = sigma.detach().cpu().numpy()
            save_snapshot(run_dir, epoch, eps_np, sig_np)

    network.eval()
    with torch.no_grad():
        epsilon_final, sigma_final = network(coordinates)
        epsilon_final, sigma_final = _apply_fixed_parameters(
            epsilon_final,
            sigma_final,
            fixed_epsilon=fixed_epsilon,
            fixed_sigma=fixed_sigma,
            freeze_epsilon=freeze_epsilon,
            freeze_sigma=freeze_sigma,
        )
        epsilon_final_np = epsilon_final.detach().cpu().numpy()
        sigma_final_np = sigma_final.detach().cpu().numpy()
    np.save(run_dir / "final_epsilon.npy", epsilon_final_np)
    np.save(run_dir / "final_sigma.npy", sigma_final_np)
    if best_epsilon_np is None or best_sigma_np is None:
        best_epsilon_np = epsilon_final_np
        best_sigma_np = sigma_final_np
    np.save(run_dir / "best_epsilon.npy", best_epsilon_np)
    np.save(run_dir / "best_sigma.npy", best_sigma_np)

    final_synthetic_np = bridge.forward(
        epsilon_final_np,
        sigma_final_np,
        source_list,
        receiver_list,
        settings,
        save_wavefield=False,
    )
    best_synthetic_np = bridge.forward(
        best_epsilon_np,
        best_sigma_np,
        source_list,
        receiver_list,
        settings,
        save_wavefield=False,
    )
    np.save(run_dir / "final_synthetic_data.npy", final_synthetic_np)
    np.save(run_dir / "best_synthetic_data.npy", best_synthetic_np)

    final_training_loss = (
        history[-1]
        if history
        else {
            "epoch": 0,
            "loss": _data_misfit(final_synthetic_np, d_obs_np)["mse"],
        }
    )
    metrics = {
        "pretrain_history": pretrain_history,
        "history": history,
        "final": final_training_loss,
        "best": {"epoch": best_epoch, "loss": best_loss},
        "data_misfit": {
            "observed_power": _array_power(d_obs_np),
            "initial": _data_misfit(initial_synthetic_np, d_obs_np),
            "best": _data_misfit(best_synthetic_np, d_obs_np),
            "final": _data_misfit(final_synthetic_np, d_obs_np),
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    generate_run_figures(
        run_dir,
        epsilon_true=epsilon_true_np,
        sigma_true=sigma_true_np,
        epsilon_initial=epsilon_initial_np,
        sigma_initial=sigma_initial_np,
        epsilon_final=epsilon_final_np,
        sigma_final=sigma_final_np,
        metrics=metrics,
        dx=settings.dx,
        dz=settings.dz,
        observed_data=d_obs_np,
        synthetic_data=final_synthetic_np,
        dt=settings.dt,
    )
    shutil.copyfile(Path(__file__).resolve(), run_dir / "train_source_snapshot.py")
    return run_dir


def _apply_fixed_parameters(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    *,
    fixed_epsilon: torch.Tensor,
    fixed_sigma: torch.Tensor,
    freeze_epsilon: bool,
    freeze_sigma: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freeze_epsilon:
        epsilon = fixed_epsilon
    if freeze_sigma:
        sigma = fixed_sigma
    return epsilon, sigma


def _is_parameter_frozen(
    epoch: int,
    *,
    freeze_always: bool,
    freeze_epochs: int,
) -> bool:
    return freeze_always or (freeze_epochs > 0 and epoch <= freeze_epochs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.MultiStepLR | None:
    milestones = train_cfg.get("learning_rate_milestones", [])
    if not milestones:
        return None
    if not isinstance(milestones, list):
        raise ValueError("training.learning_rate_milestones must be a list")
    gamma = float(train_cfg.get("learning_rate_gamma", 0.5))
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(item) for item in milestones],
        gamma=gamma,
    )


def _pretrain_network(
    network: torch.nn.Module,
    coordinates: torch.Tensor,
    epsilon_target_np: np.ndarray,
    sigma_target_np: np.ndarray,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float | int]]:
    epochs = int(train_cfg.get("pretrain_epochs", 0))
    if epochs <= 0:
        return []

    epsilon_target = torch.as_tensor(epsilon_target_np, dtype=torch.float32, device=device)
    sigma_target = torch.as_tensor(sigma_target_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(train_cfg.get("pretrain_learning_rate", train_cfg.get("learning_rate", 1e-4))),
        betas=(0.9, 0.95),
    )
    history: list[dict[str, float | int]] = []
    restore_best = bool(train_cfg.get("restore_best_pretrain", False))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    if bool(train_cfg.get("pretrain_dropout", True)):
        network.train()
    else:
        network.eval()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epsilon, sigma = network(coordinates)
        epsilon_scale, sigma_scale = _parameter_scales(network)
        loss = F.mse_loss(
            (epsilon - epsilon_target) / epsilon_scale,
            torch.zeros_like(epsilon_target),
        ) + F.mse_loss(
            (sigma - sigma_target) / sigma_scale,
            torch.zeros_like(sigma_target),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite pretrain loss at epoch {epoch}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if restore_best and loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                key: value.detach().clone()
                for key, value in network.state_dict().items()
            }
        history.append({"epoch": epoch, "loss": loss_value})
    if restore_best and best_state is not None:
        network.load_state_dict(best_state)
    return history


def _parameter_scales(network: torch.nn.Module) -> tuple[float, float]:
    bounds = getattr(network, "bounds", None)
    mapping = getattr(network, "parameter_mapping", bounds)
    if isinstance(mapping, ParameterStats):
        return max(float(mapping.epsilon_std), 1e-12), max(float(mapping.sigma_std), 1e-12)
    if not isinstance(mapping, ParameterBounds):
        return 1.0, 1.0
    epsilon_scale = max(float(mapping.epsilon_max - mapping.epsilon_min), 1e-12)
    sigma_scale = max(float(mapping.sigma_max - mapping.sigma_min), 1e-12)
    return epsilon_scale, sigma_scale


def _array_power(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(np.mean(values * values))


def _data_misfit(synthetic: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    residual = np.asarray(synthetic, dtype=np.float64) - np.asarray(
        observed,
        dtype=np.float64,
    )
    mse = float(np.mean(residual * residual))
    observed_power = max(_array_power(observed), 1e-30)
    return {
        "mse": mse,
        "relative_mse": mse / observed_power,
        "rmse": float(np.sqrt(mse)),
    }
