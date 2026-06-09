import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import _bootstrap  # noqa: F401
from ifwi_gpr.plotting import generate_run_figures


class PlottingTests(unittest.TestCase):
    def test_generate_run_figures_writes_expected_files(self):
        shape = (5, 6)
        epsilon_true = np.full(shape, 4.0, dtype=np.float32)
        sigma_true = np.full(shape, 0.003, dtype=np.float32)
        epsilon_initial = np.full(shape, 3.8, dtype=np.float32)
        sigma_initial = np.full(shape, 0.0025, dtype=np.float32)
        epsilon_final = np.full(shape, 4.2, dtype=np.float32)
        sigma_final = np.full(shape, 0.0035, dtype=np.float32)
        metrics = {
            "pretrain_history": [{"epoch": 1, "loss": 0.2}, {"epoch": 2, "loss": 0.1}],
            "history": [{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}],
            "final": {"epoch": 2, "loss": 0.5},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            figures_dir = generate_run_figures(
                tmp_dir,
                epsilon_true=epsilon_true,
                sigma_true=sigma_true,
                epsilon_initial=epsilon_initial,
                sigma_initial=sigma_initial,
                epsilon_final=epsilon_final,
                sigma_final=sigma_final,
                metrics=metrics,
                dx=0.1,
                dz=0.1,
            )

            expected = [
                "parameter_maps.png",
                "parameter_maps.pdf",
                "error_maps.png",
                "error_maps.pdf",
                "loss_curve.png",
                "loss_curve.pdf",
            ]
            for name in expected:
                self.assertTrue((Path(figures_dir) / name).exists(), msg=name)


if __name__ == "__main__":
    unittest.main()
