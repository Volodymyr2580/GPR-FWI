"""Experiment configuration loading and validation.

The project uses a small YAML subset so users can keep experiment settings in
readable text files without adding another runtime dependency. Supported YAML
features are enough for this repository's configs: nested mappings by
indentation, strings, numbers, booleans, and blank/comment lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_CHOICES = {"marmousi", "overthrust"}
ACQUISITION_CHOICES = {"mode1", "mode2"}
PARAMETERIZATION_CHOICES = {"traditional", "unet"}
INVERSION_CHOICES = {"eps_only", "sig_only", "twopara", "eps_then_sig", "sig_then_eps"}
OPTIMIZER_CHOICES = {"lbfgs", "rmsprop", "adam"}


@dataclass(frozen=True)
class RegularizationConfig:
    tv_eps: float = 0.0
    tv_sig: float = 0.0
    tikhonov_eps: float = 0.0
    tikhonov_sig: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate_eps: float = 1e-4
    learning_rate_sig: float = 1e-5
    lr_eps_stage2: float = 1e-6
    num_epochs_stage1: int = 2500
    num_epochs_stage2: int = 12501
    alpha_l1_data: float = 0.0
    offset: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    mpi_processes: int = 20
    output_dir: str = "outputs"
    resume: bool = False
    stage1_checkpoint_path: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    acquisition_mode: str
    parameterization: str
    inversion_strategy: str
    optimizer: str
    regularization: RegularizationConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    illumination: bool = False
    fixed_top_rows: int = 10
    snapshot_interval: int = 1000
    data_path: str = ""
    migration_status: str = "planned"
    source_path: str = ""

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.model,
            self.acquisition_mode,
            self.parameterization,
            self.inversion_strategy,
            self.optimizer,
            "illumination" if self.illumination else "no_illumination",
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config file."""
    config_path = Path(path)
    raw = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent
    return build_experiment_config(raw, base_dir=base_dir)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple YAML subset used by repository configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"YAML indentation must use multiples of 2 spaces at line {line_number}")
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Expected 'key: value' at line {line_number}: {raw_line}")
        key, value_text = stripped.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not key:
            raise ValueError(f"Empty YAML key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_text == "":
            value: dict[str, Any] = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _parse_scalar(value_text)

    return root


def build_experiment_config(raw: dict[str, Any], base_dir: Path | None = None) -> ExperimentConfig:
    """Convert parsed YAML into typed config objects."""
    regularization_raw = _mapping(raw.get("regularization", {}), "regularization")
    tv_raw = _mapping(regularization_raw.get("tv", {}), "regularization.tv")
    tikhonov_raw = _mapping(regularization_raw.get("tikhonov", {}), "regularization.tikhonov")
    training_raw = _mapping(raw.get("training", {}), "training")
    runtime_raw = _mapping(raw.get("runtime", {}), "runtime")

    output_dir = str(runtime_raw.get("output_dir", "outputs"))
    data_path = str(raw.get("data_path", ""))
    source_path = str(raw.get("source_path", ""))
    if base_dir is not None:
        output_dir = _resolve_repo_relative(output_dir, base_dir)
        if data_path:
            data_path = _resolve_repo_relative(data_path, base_dir)
        if source_path:
            source_path = _resolve_repo_relative(source_path, base_dir)

    config = ExperimentConfig(
        name=str(raw.get("name", "unnamed_experiment")),
        model=str(raw.get("model", "")).lower(),
        acquisition_mode=str(raw.get("acquisition_mode", "")).lower(),
        parameterization=str(raw.get("parameterization", "")).lower(),
        inversion_strategy=str(raw.get("inversion_strategy", "")).lower(),
        optimizer=str(raw.get("optimizer", "")).lower(),
        illumination=bool(raw.get("illumination", False)),
        fixed_top_rows=int(raw.get("fixed_top_rows", 10)),
        snapshot_interval=int(raw.get("snapshot_interval", 1000)),
        data_path=data_path,
        migration_status=str(raw.get("migration_status", "planned")).lower(),
        source_path=source_path,
        regularization=RegularizationConfig(
            tv_eps=float(tv_raw.get("eps", 0.0)),
            tv_sig=float(tv_raw.get("sig", 0.0)),
            tikhonov_eps=float(tikhonov_raw.get("eps", 0.0)),
            tikhonov_sig=float(tikhonov_raw.get("sig", 0.0)),
        ),
        training=TrainingConfig(
            learning_rate_eps=float(training_raw.get("learning_rate_eps", 1e-4)),
            learning_rate_sig=float(training_raw.get("learning_rate_sig", 1e-5)),
            lr_eps_stage2=float(training_raw.get("lr_eps_stage2", 1e-6)),
            num_epochs_stage1=int(training_raw.get("num_epochs_stage1", 2500)),
            num_epochs_stage2=int(training_raw.get("num_epochs_stage2", 12501)),
            alpha_l1_data=float(training_raw.get("alpha_l1_data", 0.0)),
            offset=float(training_raw.get("offset", 0.0)),
        ),
        runtime=RuntimeConfig(
            device=str(runtime_raw.get("device", "cuda:0")),
            mpi_processes=int(runtime_raw.get("mpi_processes", 20)),
            output_dir=output_dir,
            resume=bool(runtime_raw.get("resume", False)),
            stage1_checkpoint_path=str(runtime_raw.get("stage1_checkpoint_path", "")),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Raise a helpful error when a config uses unsupported values."""
    _require_choice("model", config.model, MODEL_CHOICES)
    _require_choice("acquisition_mode", config.acquisition_mode, ACQUISITION_CHOICES)
    _require_choice("parameterization", config.parameterization, PARAMETERIZATION_CHOICES)
    _require_choice("inversion_strategy", config.inversion_strategy, INVERSION_CHOICES)
    _require_choice("optimizer", config.optimizer, OPTIMIZER_CHOICES)
    if config.fixed_top_rows < 0:
        raise ValueError("fixed_top_rows must be >= 0")
    if config.snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be > 0")
    if config.training.num_epochs_stage1 < 0 or config.training.num_epochs_stage2 < 0:
        raise ValueError("num_epochs_stage1 and num_epochs_stage2 must be >= 0")
    if config.training.alpha_l1_data < 0:
        raise ValueError("alpha_l1_data must be >= 0")
    if config.migration_status not in {"supported", "planned"}:
        raise ValueError("migration_status must be either 'supported' or 'planned'")


def describe_config(config: ExperimentConfig) -> str:
    """Return a concise human-readable summary for config checks."""
    lines = [
        f"Experiment: {config.name}",
        f"  model: {config.model}",
        f"  acquisition_mode: {config.acquisition_mode}",
        f"  parameterization: {config.parameterization}",
        f"  inversion_strategy: {config.inversion_strategy}",
        f"  optimizer: {config.optimizer}",
        f"  illumination: {config.illumination}",
        f"  fixed_top_rows: {config.fixed_top_rows}",
        f"  snapshot_interval: {config.snapshot_interval}",
        f"  migration_status: {config.migration_status}",
        f"  source_path: {config.source_path}",
        f"  data_path: {config.data_path}",
        f"  output_dir: {config.runtime.output_dir}",
        "  regularization:",
        f"    tv eps/sig: {config.regularization.tv_eps} / {config.regularization.tv_sig}",
        f"    tikhonov eps/sig: {config.regularization.tikhonov_eps} / {config.regularization.tikhonov_sig}",
        "  training:",
        f"    epochs stage1/stage2: {config.training.num_epochs_stage1} / {config.training.num_epochs_stage2}",
        f"    lr eps/sig/stage2-eps: {config.training.learning_rate_eps} / {config.training.learning_rate_sig} / {config.training.lr_eps_stage2}",
        f"    alpha_l1_data: {config.training.alpha_l1_data}",
        f"  runtime device/mpi: {config.runtime.device} / {config.runtime.mpi_processes}",
    ]
    return "\n".join(lines)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(part in value for part in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_choice(field: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"{field} must be one of: {valid}. Got: {value!r}")


def _resolve_repo_relative(path_text: str, base_dir: Path) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
