"""Bridge to the verified CPU/MPI GPR FDTD solver from the reference project."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np


DEFAULT_REFERENCE_PROJECT = Path("E:/sci_research/GPR/marmousi_paper/gpr-inversion")
REFERENCE_MODULE = "gpr_inversion.experiments.overthrust.mode2_eps_then_sig"


@dataclass(frozen=True)
class SolverSettings:
    """Numerical settings needed by the CPU/MPI solver."""

    dt: float
    dx: float
    dz: float
    npml: int
    freq: float
    steps: int
    gradient_normalization: str = "none"
    epsilon_gradient_weight: float = 1.0
    sigma_gradient_weight: float = 1.0
    epsilon_gradient_mode: str = "reference"


class CpuMpiSolverBridge:
    """Thin wrapper around the existing CPU/MPI forward and gradient functions."""

    def __init__(self, reference_project: str | Path = DEFAULT_REFERENCE_PROJECT) -> None:
        self.reference_project = Path(reference_project)
        self.reference_src = self.reference_project / "src"
        self._forward_model = None
        self._compute_gradient = None
        self._add_cpml = None
        self._update_h = None
        self._update_e = None

    def ensure_loaded(self) -> None:
        if (
            self._forward_model is not None
            and self._compute_gradient is not None
            and self._add_cpml is not None
            and self._update_h is not None
            and self._update_e is not None
        ):
            return
        if not self.reference_src.exists():
            raise FileNotFoundError(
                f"Reference project src directory not found: {self.reference_src}"
            )
        src_text = str(self.reference_src)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)

        forward_module = import_module(f"{REFERENCE_MODULE}.forward")
        gradient_module = import_module(f"{REFERENCE_MODULE}.gradient")
        cpml_module = import_module("gpr_inversion.common.Add_CPML")
        time_loop_module = import_module(f"{REFERENCE_MODULE}.Time_loop")
        self._forward_model = forward_module.forward_model
        self._compute_gradient = gradient_module.compute_gradient
        self._add_cpml = cpml_module.Add_CPML
        self._update_h = time_loop_module.update_H
        self._update_e = time_loop_module.update_E

    def forward(
        self,
        epsilon: np.ndarray,
        sigma: np.ndarray,
        source_list: list[tuple[int, int]],
        receiver_list: list[tuple[int, int]],
        settings: SolverSettings,
        *,
        save_wavefield: bool = True,
        return_illumination: bool = False,
    ):
        self.ensure_loaded()
        return self._forward_model(
            np.asarray(epsilon),
            np.asarray(sigma),
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            save_wavefield=save_wavefield,
            return_illumination=return_illumination,
        )

    def gradient(
        self,
        epsilon: np.ndarray,
        sigma: np.ndarray,
        residual: np.ndarray,
        source_list: list[tuple[int, int]],
        receiver_list: list[tuple[int, int]],
        settings: SolverSettings,
        *,
        wavefield_data,
        sigma_required_gradient: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.ensure_loaded()
        epsilon_array = np.asarray(epsilon)
        sigma_array = np.asarray(sigma)
        residual_array = np.asarray(residual)
        if _normalized_mode(settings.epsilon_gradient_mode) not in {
            "reference",
            "default",
            "continuous",
            "current",
            "none",
        }:
            return self._compute_gradient_with_custom_epsilon(
                epsilon_array,
                sigma_array,
                residual_array,
                source_list,
                receiver_list,
                settings,
                wavefield_data=wavefield_data,
                sigma_required_gradient=sigma_required_gradient,
            )
        return self._compute_gradient(
            epsilon_array,
            sigma_array,
            residual_array,
            source_list,
            receiver_list,
            settings.dt,
            settings.dx,
            settings.dz,
            settings.npml,
            settings.freq,
            settings.steps,
            sigma_required_gradient=sigma_required_gradient,
            wavefield_data=wavefield_data,
        )

    def _compute_gradient_with_custom_epsilon(
        self,
        epsilon: np.ndarray,
        sigma: np.ndarray,
        residual: np.ndarray,
        source_list: list[tuple[int, int]],
        receiver_list: list[tuple[int, int]],
        settings: SolverSettings,
        *,
        wavefield_data,
        sigma_required_gradient: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        mode = _normalized_mode(settings.epsilon_gradient_mode)
        if mode not in {
            "negative_discrete_ca_cb_adj_plus1",
            "neg_discrete_ca_cb_adj_plus1",
            "negative_discrete",
        }:
            raise ValueError(
                f"Unsupported epsilon_gradient_mode: {settings.epsilon_gradient_mode}"
            )
        if wavefield_data is None:
            raise ValueError("wavefield_data is required for custom epsilon gradients")

        xl, zl = epsilon.shape
        npml = settings.npml
        ep0 = 8.841941282883074e-12
        epsilon_extended = np.pad(epsilon, ((npml, npml), (npml, npml)), "edge")
        sigma_extended = np.pad(sigma, ((npml, npml), (npml, npml)), "edge")
        mu_extended = np.ones((xl + 2 * npml, zl + 2 * npml))
        cpml_params = self._add_cpml(
            xl,
            zl,
            sigma_extended.copy(),
            epsilon_extended.copy(),
            mu_extended.copy(),
            settings.dx,
            settings.dz,
            settings.dt,
        )
        epsilon_abs = epsilon_extended * ep0
        denominator = epsilon_abs + sigma_extended * settings.dt / 2.0
        dca_deps = ep0 * sigma_extended * settings.dt / (denominator * denominator)
        dcb_deps = -ep0 / (denominator * denominator)

        grad_eps = np.zeros_like(epsilon, dtype=np.float64)
        grad_sig = np.zeros_like(sigma, dtype=np.float64) if sigma_required_gradient else None
        wavefield_array = np.asarray(wavefield_data)
        for shot_idx, _source_pos in enumerate(source_list):
            forward_wavefield = np.asarray(wavefield_array[shot_idx])
            shot_residual = residual[shot_idx, :, :]
            adjoint_list = self._reverse_time_loop_reference_variant(
                epsilon_extended=epsilon_extended,
                sigma_extended=sigma_extended,
                mu_extended=mu_extended,
                cpml_params=cpml_params,
                receiver_list=receiver_list,
                residual_data=shot_residual,
                settings=settings,
                shape=(xl, zl),
            )
            for k in range(1, settings.steps - 1):
                adjoint_idx = k + 1
                wavefield_idx = k
                if adjoint_idx >= len(adjoint_list) or wavefield_idx >= len(forward_wavefield):
                    continue
                e_old = forward_wavefield[wavefield_idx - 1]
                e_new = forward_wavefield[wavefield_idx]
                curl_h_dt = (e_new - cpml_params.ca * e_old) / cpml_params.cb
                discrete_derivative = dca_deps * e_old + dcb_deps * curl_h_dt
                adjoint_field = adjoint_list[adjoint_idx]
                valid = (
                    -adjoint_field[npml : npml + xl, npml : npml + zl]
                    * discrete_derivative[npml : npml + xl, npml : npml + zl]
                )
                grad_eps += valid
                if grad_sig is not None:
                    sigma_adjoint_field = adjoint_list[k]
                    grad_sig += (
                        sigma_adjoint_field[npml : npml + xl, npml : npml + zl]
                        * forward_wavefield[k][npml : npml + xl, npml : npml + zl]
                    )

        mask = np.ones_like(grad_eps)
        mask[:10, :] = 0.0
        grad_eps *= mask
        if grad_sig is not None:
            grad_sig *= mask
        return grad_eps.astype(epsilon.dtype, copy=False), (
            None if grad_sig is None else grad_sig.astype(sigma.dtype, copy=False)
        )

    def _reverse_time_loop_reference_variant(
        self,
        *,
        epsilon_extended: np.ndarray,
        sigma_extended: np.ndarray,
        mu_extended: np.ndarray,
        cpml_params,
        receiver_list: list[tuple[int, int]],
        residual_data: np.ndarray,
        settings: SolverSettings,
        shape: tuple[int, int],
    ) -> list[np.ndarray]:
        ep0 = 8.841941282883074e-12
        mu0 = 1.2566370614359173e-06
        xl, zl = shape
        npml = settings.npml
        epsilon_abs = epsilon_extended.copy() * ep0
        mu_abs = mu_extended.copy() * mu0
        ey = np.zeros((xl + 2 * npml, zl + 2 * npml))
        hz = np.zeros_like(ey)
        hx = np.zeros_like(ey)
        memory_dey_dx = np.zeros((2 * npml, zl + 2 * npml))
        memory_dey_dz = np.zeros((xl + 2 * npml, 2 * npml))
        memory_dhz_dx = np.zeros((2 * npml, zl + 2 * npml))
        memory_dhx_dz = np.zeros((xl + 2 * npml, 2 * npml))
        receiver_list_extended = [(rec[0] + npml, rec[1] + npml) for rec in receiver_list]
        adjoint_sampled = []
        for tt in range(settings.steps):
            time_index = settings.steps - tt - 1
            for rec_idx, rec_pos in enumerate(receiver_list_extended):
                ey[rec_pos[0], rec_pos[1]] -= residual_data[rec_idx, time_index]
            hz, hx = self._update_h(
                xl,
                zl,
                settings.dx,
                settings.dz,
                settings.dt,
                sigma_extended.copy(),
                epsilon_abs,
                mu_abs,
                npml,
                cpml_params.a_x_half,
                cpml_params.a_z_half,
                cpml_params.b_x_half,
                cpml_params.b_z_half,
                cpml_params.k_x_half,
                cpml_params.k_z_half,
                hz,
                hx,
                ey,
                memory_dey_dx,
                memory_dey_dz,
            )
            ey = self._update_e(
                xl,
                zl,
                settings.dx,
                settings.dz,
                settings.dt,
                cpml_params.ca_r,
                cpml_params.cb,
                npml,
                cpml_params.a_x,
                cpml_params.a_z,
                cpml_params.b_x,
                cpml_params.b_z,
                cpml_params.k_x,
                cpml_params.k_z,
                hz,
                hx,
                ey,
                memory_dhz_dx,
                memory_dhx_dz,
            )
            adjoint_sampled.append(ey.copy())
        return adjoint_sampled[::-1]


def _normalized_mode(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")
