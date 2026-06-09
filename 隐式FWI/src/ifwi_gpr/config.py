"""Configuration loading for IFWI experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config, with optional YAML support if PyYAML is installed."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Use JSON configs or install PyYAML."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config suffix: {config_path.suffix}")


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is required")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum config shape and annotate experiment intent."""
    for section in ("model", "network", "parameters", "solver", "acquisition", "training"):
        require_section(config, section)

    experiment = config.get("experiment")
    if experiment is None:
        config["experiment"] = infer_experiment_metadata(config)
        return config
    if not isinstance(experiment, dict):
        raise ValueError("Config section 'experiment' must be a mapping when provided")
    experiment.setdefault("track", infer_experiment_track(config))
    experiment.setdefault("purpose", infer_experiment_purpose(config))
    return config


def infer_experiment_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Infer whether a config is a smoke setup or a paper-aligned template."""
    return {
        "track": infer_experiment_track(config),
        "purpose": infer_experiment_purpose(config),
    }


def infer_experiment_track(config: dict[str, Any]) -> str:
    model_cfg = require_section(config, "model")
    kind = str(model_cfg.get("kind", "")).lower()
    if kind in {"tiny_smooth", "cross_shape"}:
        return "smoke"
    return "paper"


def infer_experiment_purpose(config: dict[str, Any]) -> str:
    name = str(config.get("name", "")).lower()
    if "smoke" in name or "tiny" in name or "cross" in name:
        return "chain_validation"
    return "paper_reproduction"
