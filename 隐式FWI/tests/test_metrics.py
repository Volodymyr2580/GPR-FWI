import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from ifwi_gpr.metrics import mape, parameter_metrics, r2_score, snr_db, ssim_global


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction_metrics(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = target.copy()

        self.assertAlmostEqual(r2_score(predicted, target), 1.0)
        self.assertTrue(np.isinf(snr_db(predicted, target)))
        self.assertAlmostEqual(mape(predicted, target), 0.0)
        self.assertAlmostEqual(ssim_global(predicted, target), 1.0)

    def test_parameter_metrics_returns_expected_keys(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = np.array([[1.1, 1.9], [3.2, 3.8]])
        result = parameter_metrics(predicted, target)

        self.assertEqual(
            set(result),
            {"r2", "snr_db", "mape_percent", "ssim"},
        )


if __name__ == "__main__":
    unittest.main()
