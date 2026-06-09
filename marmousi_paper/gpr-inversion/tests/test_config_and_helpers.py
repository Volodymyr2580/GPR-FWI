import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gpr_inversion.acquisition import build_acquisition_geometry
from gpr_inversion.config import build_experiment_config, parse_simple_yaml
from gpr_inversion.io import ensure_unique_dir, save_model_snapshot
from gpr_inversion.models import build_initial_model, build_sigma_from_epsilon


class ConfigAndHelperTests(unittest.TestCase):
    def test_parse_config_subset(self):
        raw = parse_simple_yaml(
            """
            name: sample
            model: overthrust
            acquisition_mode: mode2
            parameterization: unet
            inversion_strategy: eps_then_sig
            optimizer: adam
            illumination: true
            regularization:
              tv:
                eps: 0.0
                sig: 5e-4
            training:
              alpha_l1_data: 0.2
            """
        )
        config = build_experiment_config(raw)
        self.assertEqual(config.model, "overthrust")
        self.assertEqual(config.acquisition_mode, "mode2")
        self.assertTrue(config.illumination)
        self.assertEqual(config.key[-1], "illumination")
        self.assertEqual(config.regularization.tv_sig, 5e-4)
        self.assertEqual(config.training.alpha_l1_data, 0.2)

    def test_acquisition_modes(self):
        mode1 = build_acquisition_geometry("mode1", lateral_size=10, source_step=5)
        self.assertEqual(mode1.sources, mode1.receivers)

        mode2 = build_acquisition_geometry("mode2", lateral_size=10, source_step=5, receiver_step=2)
        self.assertEqual(mode2.sources, [(0, 0), (0, 5)])
        self.assertEqual(mode2.receivers, [(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)])

    def test_model_initialization(self):
        model = np.arange(12, dtype=np.float32).reshape(3, 4) + 3.0
        sigma = build_sigma_from_epsilon(model)
        self.assertTrue(np.allclose(sigma, model * 1e-3))
        uniform = build_initial_model(model, "uniform", constant_value=5.0)
        self.assertTrue(np.allclose(uniform, 5.0))
        linear = build_initial_model(model, "linear")
        self.assertEqual(linear.shape, model.shape)

    def test_output_helpers_do_not_overwrite(self):
        with mock.patch("pathlib.Path.mkdir") as mkdir_mock:
            unique = ensure_unique_dir(Path("src"))

        self.assertEqual(unique.name, "src_run001")
        mkdir_mock.assert_called_once()

        with mock.patch("pathlib.Path.mkdir"):
            with mock.patch("numpy.save") as save_mock:
                saved = save_model_snapshot("run", 0, np.ones((2, 2)), sigma=np.zeros((2, 2)))

        self.assertEqual(len(saved), 2)
        self.assertEqual(save_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
