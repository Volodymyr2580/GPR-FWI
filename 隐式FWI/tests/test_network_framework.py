import unittest
import importlib.util
import tempfile
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed in this environment")
else:
    import numpy as np
    import torch

from tests import _bootstrap  # noqa: F401
from ifwi_gpr.grids import make_normalized_grid
from ifwi_gpr.networks import IFWIFrInrNetwork, IFWINetwork
from ifwi_gpr.parameterization import ParameterBounds, ParameterStats
from ifwi_gpr.train import (
    _apply_fixed_parameters,
    _is_parameter_frozen,
    build_initial_model,
    dry_run,
)


class NetworkFrameworkTests(unittest.TestCase):
    def test_ifwi_network_outputs_physical_maps(self):
        shape = (11, 13)
        bounds = ParameterBounds(
            epsilon_min=1.0,
            epsilon_max=8.0,
            sigma_min=1e-4,
            sigma_max=1e-2,
        )
        net = IFWINetwork(
            bounds,
            shape=shape,
            hidden_features=16,
            hidden_layers=2,
            omega0=20.0,
            dropout=0.2,
        )
        coords = make_normalized_grid(shape)
        epsilon, sigma = net(coords)

        self.assertEqual(tuple(epsilon.shape), shape)
        self.assertEqual(tuple(sigma.shape), shape)
        self.assertTrue(torch.isfinite(epsilon).all())
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertGreaterEqual(float(epsilon.detach().min()), bounds.epsilon_min)
        self.assertLessEqual(float(epsilon.detach().max()), bounds.epsilon_max)
        self.assertGreaterEqual(float(sigma.detach().min()), bounds.sigma_min)
        self.assertLessEqual(float(sigma.detach().max()), bounds.sigma_max)

    def test_dropout_is_disabled_in_eval_mode(self):
        shape = (7, 9)
        bounds = ParameterBounds(1.0, 8.0, 1e-4, 1e-2)
        net = IFWINetwork(
            bounds,
            shape=shape,
            hidden_features=16,
            hidden_layers=2,
            dropout=0.5,
        )
        coords = make_normalized_grid(shape)
        net.eval()
        eps_a, sig_a = net(coords)
        eps_b, sig_b = net(coords)
        self.assertTrue(torch.allclose(eps_a, eps_b))
        self.assertTrue(torch.allclose(sig_a, sig_b))

    def test_fr_inr_network_outputs_physical_maps(self):
        shape = (9, 10)
        bounds = ParameterBounds(1.0, 8.0, 1e-4, 1e-2)
        net = IFWIFrInrNetwork(
            bounds,
            shape=shape,
            mode="sin+fr",
            hidden_features=16,
            hidden_layers=3,
            high_freq_num=3,
            low_freq_num=2,
            phi_num=4,
            alpha=0.01,
            dropout=0.1,
        )
        coords = make_normalized_grid(shape)
        epsilon, sigma = net(coords)

        self.assertEqual(tuple(epsilon.shape), shape)
        self.assertEqual(tuple(sigma.shape), shape)
        self.assertTrue(torch.isfinite(epsilon).all())
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertGreaterEqual(float(epsilon.detach().min()), bounds.epsilon_min)
        self.assertLessEqual(float(epsilon.detach().max()), bounds.epsilon_max)
        self.assertGreaterEqual(float(sigma.detach().min()), bounds.sigma_min)
        self.assertLessEqual(float(sigma.detach().max()), bounds.sigma_max)

    def test_fr_inr_network_accepts_standardized_paper_mapping(self):
        shape = (9, 10)
        stats = ParameterStats(
            epsilon_mean=4.0,
            epsilon_std=1.0,
            sigma_mean=0.003,
            sigma_std=0.0015,
        )
        net = IFWIFrInrNetwork(
            stats,
            shape=shape,
            mode="sin+fr",
            hidden_features=16,
            hidden_layers=3,
            high_freq_num=3,
            low_freq_num=2,
            phi_num=4,
            alpha=0.01,
            dropout=0.1,
        )
        coords = make_normalized_grid(shape)
        epsilon, sigma = net(coords)

        self.assertEqual(tuple(epsilon.shape), shape)
        self.assertEqual(tuple(sigma.shape), shape)
        self.assertTrue(torch.isfinite(epsilon).all())
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertIsNone(net.bounds)
        self.assertIs(net.parameter_mapping, stats)

    def test_dry_run_accepts_fr_inr_architecture(self):
        config = {
            "seed": 2025,
            "device": "cpu",
            "model": {"shape": [5, 6]},
            "network": {
                "architecture": "fr_inr",
                "mode": "sin+fr",
                "hidden_layers": 2,
                "hidden_features": 8,
                "high_freq_num": 2,
                "low_freq_num": 2,
                "phi_num": 4,
                "alpha": 0.01,
                "omega0": 10.0,
                "dropout": 0.0,
            },
            "parameters": {
                "epsilon_min": 1.0,
                "epsilon_max": 8.0,
                "sigma_min": 1e-4,
                "sigma_max": 1e-2,
            },
        }
        result = dry_run(config)
        self.assertEqual(result["epsilon_shape"], [5, 6])
        self.assertEqual(result["sigma_shape"], [5, 6])
        self.assertTrue(result["epsilon_finite"])
        self.assertTrue(result["sigma_finite"])
        self.assertGreater(result["parameter_count"], 0)

    def test_dry_run_reports_shapes(self):
        config = {
            "seed": 2025,
            "device": "cpu",
            "model": {"shape": [5, 6]},
            "network": {
                "hidden_layers": 2,
                "hidden_features": 8,
                "omega0": 10.0,
                "dropout": 0.0,
            },
            "parameters": {
                "epsilon_min": 1.0,
                "epsilon_max": 8.0,
                "sigma_min": 1e-4,
                "sigma_max": 1e-2,
            },
        }
        result = dry_run(config)
        self.assertEqual(result["epsilon_shape"], [5, 6])
        self.assertEqual(result["sigma_shape"], [5, 6])
        self.assertTrue(result["epsilon_finite"])
        self.assertTrue(result["sigma_finite"])

    def test_build_initial_model_loads_parameter_maps(self):
        shape = (5, 6)
        epsilon = np.full(shape, 4.2, dtype=np.float32)
        sigma = np.full(shape, 0.004, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            epsilon_path = Path(tmp_dir) / "epsilon.npy"
            sigma_path = Path(tmp_dir) / "sigma.npy"
            np.save(epsilon_path, epsilon)
            np.save(sigma_path, sigma)

            loaded_epsilon, loaded_sigma = build_initial_model(
                {
                    "model": {"shape": list(shape)},
                    "initial": {
                        "epsilon_path": str(epsilon_path),
                        "sigma_path": str(sigma_path),
                    },
                }
            )

        self.assertTrue(np.allclose(loaded_epsilon, epsilon))
        self.assertTrue(np.allclose(loaded_sigma, sigma))
        self.assertEqual(loaded_epsilon.dtype, np.float32)
        self.assertEqual(loaded_sigma.dtype, np.float32)

    def test_build_initial_model_rejects_wrong_shape_parameter_maps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            epsilon_path = Path(tmp_dir) / "epsilon.npy"
            np.save(epsilon_path, np.zeros((4, 6), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "expected"):
                build_initial_model(
                    {
                        "model": {"shape": [5, 6]},
                        "initial": {"epsilon_path": str(epsilon_path)},
                    }
                )

    def test_apply_fixed_parameters_can_freeze_sigma_only(self):
        epsilon = torch.full((3, 4), 4.2)
        sigma = torch.full((3, 4), 0.004)
        fixed_epsilon = torch.full((3, 4), 4.0)
        fixed_sigma = torch.full((3, 4), 0.003)

        out_epsilon, out_sigma = _apply_fixed_parameters(
            epsilon,
            sigma,
            fixed_epsilon=fixed_epsilon,
            fixed_sigma=fixed_sigma,
            freeze_epsilon=False,
            freeze_sigma=True,
        )

        self.assertTrue(torch.allclose(out_epsilon, epsilon))
        self.assertTrue(torch.allclose(out_sigma, fixed_sigma))

    def test_is_parameter_frozen_supports_staged_freezing(self):
        self.assertTrue(
            _is_parameter_frozen(3, freeze_always=False, freeze_epochs=5)
        )
        self.assertFalse(
            _is_parameter_frozen(6, freeze_always=False, freeze_epochs=5)
        )
        self.assertTrue(
            _is_parameter_frozen(6, freeze_always=True, freeze_epochs=0)
        )


if __name__ == "__main__":
    unittest.main()
