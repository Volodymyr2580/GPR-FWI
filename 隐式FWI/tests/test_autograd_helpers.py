import unittest

import numpy as np

from tests import _bootstrap  # noqa: F401
from ifwi_gpr.autograd import normalize_parameter_gradients


class AutogradHelperTests(unittest.TestCase):
    def test_match_rms_balances_parameter_gradient_channels(self):
        grad_eps = np.full((4, 4), 100.0)
        grad_sig = np.full((4, 4), 0.01)

        balanced_eps, balanced_sig = normalize_parameter_gradients(
            grad_eps,
            grad_sig,
            mode="match_rms",
        )

        eps_rms = float(np.sqrt(np.mean(balanced_eps * balanced_eps)))
        sig_rms = float(np.sqrt(np.mean(balanced_sig * balanced_sig)))
        self.assertAlmostEqual(eps_rms, sig_rms, places=7)

    def test_none_mode_keeps_relative_scale_with_weights(self):
        grad_eps = np.full((2, 2), 2.0)
        grad_sig = np.full((2, 2), 3.0)

        scaled_eps, scaled_sig = normalize_parameter_gradients(
            grad_eps,
            grad_sig,
            mode="none",
            epsilon_weight=0.5,
            sigma_weight=2.0,
        )

        self.assertTrue(np.allclose(scaled_eps, 1.0))
        self.assertTrue(np.allclose(scaled_sig, 6.0))


if __name__ == "__main__":
    unittest.main()
