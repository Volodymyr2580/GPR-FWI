"""Metrics used to compare IFWI reconstructions with ground truth."""

from __future__ import annotations

import numpy as np


def r2_score(predicted: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination for one parameter map."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    residual = truth - pred
    centered = truth - np.mean(truth)
    denominator = float(np.sum(centered * centered))
    if denominator <= 1e-30:
        return 1.0 if float(np.sum(residual * residual)) <= 1e-30 else 0.0
    return float(1.0 - np.sum(residual * residual) / denominator)


def snr_db(predicted: np.ndarray, target: np.ndarray) -> float:
    """Signal-to-noise ratio in dB using target energy over residual energy."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    signal = float(np.sum(truth * truth))
    noise = float(np.sum((truth - pred) ** 2))
    if noise <= 1e-30:
        return float("inf")
    return float(10.0 * np.log10(max(signal, 1e-30) / noise))


def mape(predicted: np.ndarray, target: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean absolute percentage error in percent."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    denominator = np.maximum(np.abs(truth), eps)
    return float(np.mean(np.abs((truth - pred) / denominator)) * 100.0)


def ssim_global(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    c1: float = 0.01,
    c2: float = 0.03,
) -> float:
    """Global SSIM variant matching the paper's scalar metric style."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(truth))
    var_x = float(np.var(pred))
    var_y = float(np.var(truth))
    cov_xy = float(np.mean((pred - mu_x) * (truth - mu_y)))
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(numerator / max(denominator, 1e-30))


def parameter_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return the paper-style metric bundle for one parameter map."""
    return {
        "r2": r2_score(predicted, target),
        "snr_db": snr_db(predicted, target),
        "mape_percent": mape(predicted, target),
        "ssim": ssim_global(predicted, target),
    }
