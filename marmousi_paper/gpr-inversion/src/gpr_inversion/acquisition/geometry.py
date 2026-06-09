"""Source and receiver geometry for GPR-FWI acquisition modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionGeometry:
    """Discrete source and receiver positions on the model grid."""

    mode: str
    sources: list[tuple[int, int]]
    receivers: list[tuple[int, int]]


def build_acquisition_geometry(
    mode: str,
    lateral_size: int,
    source_depth_index: int = 0,
    source_step: int = 5,
    receiver_step: int = 2,
) -> AcquisitionGeometry:
    """Build mode1 or mode2 acquisition geometry.

    `mode1` is self-transmit/self-receive: each shot uses one receiver at the
    same grid location. `mode2` uses the same source row but a fixed
    multi-offset receiver line.
    """
    normalized_mode = mode.lower()
    if normalized_mode not in {"mode1", "mode2"}:
        raise ValueError("mode must be 'mode1' or 'mode2'")
    if lateral_size <= 0:
        raise ValueError("lateral_size must be positive")
    if source_step <= 0 or receiver_step <= 0:
        raise ValueError("source_step and receiver_step must be positive")

    sources = [(source_depth_index, i) for i in range(0, lateral_size, source_step)]
    if normalized_mode == "mode1":
        receivers = sources.copy()
    else:
        receivers = [(source_depth_index, i) for i in range(0, lateral_size, receiver_step)]

    return AcquisitionGeometry(mode=normalized_mode, sources=sources, receivers=receivers)
