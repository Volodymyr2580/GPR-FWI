"""Unit tests for the isolated JAX FDTD lab."""

from __future__ import annotations

import unittest

import numpy as np

from .cases import build_tiny_case
from .cpu_baseline import (
    run_cpu_adjoint_gradient,
    run_cpu_baseline,
    run_cpu_forward_with_illumination,
    run_cpu_forward_with_wavefield,
    summarize_array,
)
from .gradient_check import (
    run_epsilon_directional_gradient_check,
    run_l1_residual_backward_check,
    run_network_sigma_directional_gradient_check,
    run_sigma_directional_gradient_check,
)
from .jax_adjoint import run_jax_adjoint_gradient
from .jax_forward import (
    describe_jax_environment,
    is_jax_available,
    run_jax_forward,
    run_jax_forward_with_illumination,
    run_jax_forward_with_illumination_batched,
    run_jax_forward_with_wavefield,
)


class TinyFdtdLabTests(unittest.TestCase):
    def test_tiny_case_shape(self):
        case = build_tiny_case(steps=8)
        self.assertEqual(case.epsilon.shape, (20, 40))
        self.assertEqual(case.sigma.shape, (20, 40))
        self.assertEqual(len(case.sources), 2)
        self.assertEqual(len(case.receivers), 4)
        self.assertEqual(case.steps, 8)

    def test_cpu_baseline_runs_and_is_finite(self):
        case = build_tiny_case(steps=8)
        data = run_cpu_baseline(case)
        summary = summarize_array(data)
        self.assertEqual(summary.shape, (2, 4, 8))
        self.assertTrue(summary.finite)
        self.assertGreaterEqual(summary.maximum, summary.minimum)

    def test_jax_environment_descriptor_is_safe_without_jax(self):
        env = describe_jax_environment()
        self.assertEqual(env.available, is_jax_available())
        if not env.available:
            self.assertEqual(env.devices, ())
            self.assertIsNone(env.default_backend)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_forward_matches_cpu_shape_and_finiteness(self):
        case = build_tiny_case(steps=8, dtype=np.float32)
        cpu_data = run_cpu_baseline(case)
        jax_data = run_jax_forward(case)
        self.assertEqual(jax_data.shape, cpu_data.shape)
        self.assertTrue(np.isfinite(jax_data).all())
        diff = jax_data - cpu_data
        max_abs_error = np.max(np.abs(diff))
        relative_l2_error = np.linalg.norm(diff.ravel()) / max(
            np.linalg.norm(cpu_data.ravel()), 1e-30
        )
        self.assertLess(max_abs_error, 1e-4)
        self.assertLess(relative_l2_error, 1e-5)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_forward_respects_requested_step_count(self):
        case = build_tiny_case(steps=500, dtype=np.float32)
        jax_data = run_jax_forward(case)
        self.assertEqual(jax_data.shape, (2, 4, 500))
        self.assertTrue(np.isfinite(jax_data).all())

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_wavefield_matches_original_cpu_wavefield(self):
        case = build_tiny_case(steps=24, dtype=np.float32)
        cpu_data, cpu_wavefield = run_cpu_forward_with_wavefield(case)
        jax_data, jax_wavefield = run_jax_forward_with_wavefield(case)
        self.assertEqual(jax_data.shape, cpu_data.shape)
        self.assertEqual(jax_wavefield.shape, cpu_wavefield.shape)
        self.assertTrue(np.isfinite(jax_wavefield).all())

        diff = jax_wavefield - cpu_wavefield
        max_abs_error = np.max(np.abs(diff))
        relative_l2_error = np.linalg.norm(diff.ravel()) / max(
            np.linalg.norm(cpu_wavefield.ravel()), 1e-30
        )
        self.assertLess(max_abs_error, 1e-3)
        self.assertLess(relative_l2_error, 1e-5)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_illumination_matches_original_cpu_illumination(self):
        case = build_tiny_case(steps=24, dtype=np.float32)
        cpu_data, cpu_illumination = run_cpu_forward_with_illumination(case)
        jax_data, jax_illumination = run_jax_forward_with_illumination(case)
        batched_data, batched_illumination = run_jax_forward_with_illumination_batched(
            case,
            shot_batch_size=1,
        )

        self.assertEqual(jax_data.shape, cpu_data.shape)
        self.assertEqual(jax_illumination.shape, cpu_illumination.shape)
        self.assertTrue(np.isfinite(jax_illumination).all())
        self.assertTrue(np.isfinite(batched_illumination).all())

        illum_relative_l2 = np.linalg.norm((jax_illumination - cpu_illumination).ravel()) / max(
            np.linalg.norm(cpu_illumination.ravel()),
            1e-30,
        )
        batched_relative_l2 = np.linalg.norm((batched_illumination - jax_illumination).ravel()) / max(
            np.linalg.norm(jax_illumination.ravel()),
            1e-30,
        )
        data_relative_l2 = np.linalg.norm((batched_data - jax_data).ravel()) / max(
            np.linalg.norm(jax_data.ravel()),
            1e-30,
        )
        self.assertLess(illum_relative_l2, 1e-5)
        self.assertLess(batched_relative_l2, 3e-7)
        self.assertLess(data_relative_l2, 3e-7)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_adjoint_gradient_matches_original_cpu_gradient(self):
        case = build_tiny_case(steps=24, dtype=np.float32)
        cpu_data, cpu_wavefield = run_cpu_forward_with_wavefield(case)
        residual = (cpu_data * 0.05).astype(np.float32)
        cpu_grad_eps, cpu_grad_sig = run_cpu_adjoint_gradient(
            case,
            residual,
            wavefield_data=cpu_wavefield,
            sigma_required_gradient=True,
        )
        jax_grad_eps, jax_grad_sig = run_jax_adjoint_gradient(
            case,
            residual,
            sigma_required_gradient=True,
        )

        self.assertTrue(np.isfinite(jax_grad_eps).all())
        self.assertTrue(np.isfinite(jax_grad_sig).all())
        self.assertGreater(np.linalg.norm(cpu_grad_eps.ravel()), 0.0)
        self.assertGreater(np.linalg.norm(cpu_grad_sig.ravel()), 0.0)

        eps_relative_l2 = np.linalg.norm((jax_grad_eps - cpu_grad_eps).ravel()) / max(
            np.linalg.norm(cpu_grad_eps.ravel()),
            1e-30,
        )
        sig_relative_l2 = np.linalg.norm((jax_grad_sig - cpu_grad_sig).ravel()) / max(
            np.linalg.norm(cpu_grad_sig.ravel()),
            1e-30,
        )
        self.assertLess(eps_relative_l2, 1e-5)
        self.assertLess(sig_relative_l2, 1e-5)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_illumination_corrected_gradient_matches_cpu_path(self):
        case = build_tiny_case(steps=24, dtype=np.float32)
        cpu_data, cpu_wavefield = run_cpu_forward_with_wavefield(case)
        _cpu_data_illum, cpu_illumination = run_cpu_forward_with_illumination(case)
        _jax_data_illum, jax_illumination = run_jax_forward_with_illumination_batched(
            case,
            shot_batch_size=1,
        )
        residual = (cpu_data * 0.05).astype(np.float32)
        cpu_grad_eps, cpu_grad_sig = run_cpu_adjoint_gradient(
            case,
            residual,
            wavefield_data=cpu_wavefield,
            sigma_required_gradient=True,
        )
        jax_grad_eps, jax_grad_sig = run_jax_adjoint_gradient(
            case,
            residual,
            sigma_required_gradient=True,
        )

        cpu_denom = cpu_illumination + 1e-8 * np.max(cpu_illumination)
        jax_denom = jax_illumination + 1e-8 * np.max(jax_illumination)
        cpu_grad_eps_corrected = cpu_grad_eps / cpu_denom
        cpu_grad_sig_corrected = cpu_grad_sig / cpu_denom
        jax_grad_eps_corrected = jax_grad_eps / jax_denom
        jax_grad_sig_corrected = jax_grad_sig / jax_denom

        eps_relative_l2 = np.linalg.norm((jax_grad_eps_corrected - cpu_grad_eps_corrected).ravel()) / max(
            np.linalg.norm(cpu_grad_eps_corrected.ravel()),
            1e-30,
        )
        sig_relative_l2 = np.linalg.norm((jax_grad_sig_corrected - cpu_grad_sig_corrected).ravel()) / max(
            np.linalg.norm(cpu_grad_sig_corrected.ravel()),
            1e-30,
        )
        self.assertLess(eps_relative_l2, 2e-5)
        self.assertLess(sig_relative_l2, 2e-5)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_epsilon_gradient_matches_finite_difference_direction(self):
        case = build_tiny_case(steps=8, dtype=np.float32)
        result = run_epsilon_directional_gradient_check(case)
        self.assertTrue(result.finite)
        self.assertLess(result.relative_error, 5e-2)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_sigma_gradient_matches_finite_difference_direction(self):
        case = build_tiny_case(steps=8, dtype=np.float32)
        result = run_sigma_directional_gradient_check(case)
        self.assertTrue(result.finite)
        self.assertLess(result.relative_error, 5e-2)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_l1_residual_backward_has_expected_zero_subgradient(self):
        result = run_l1_residual_backward_check(zero_subgradient=True)
        self.assertTrue(result.finite)
        self.assertLess(result.max_abs_error, 1e-7)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_l1_sigma_gradient_matches_finite_difference_direction(self):
        case = build_tiny_case(steps=8, dtype=np.float32)
        result = run_sigma_directional_gradient_check(case, loss_norm="l1")
        self.assertTrue(result.finite)
        self.assertLess(result.relative_error, 8e-2)

    @unittest.skipUnless(is_jax_available(), "JAX is not installed in this environment")
    def test_jax_network_sigma_gradient_reaches_parameters(self):
        case = build_tiny_case(steps=8, dtype=np.float32)
        result = run_network_sigma_directional_gradient_check(case)
        self.assertTrue(result.finite)
        self.assertLess(result.relative_error, 8e-2)


if __name__ == "__main__":
    unittest.main()
