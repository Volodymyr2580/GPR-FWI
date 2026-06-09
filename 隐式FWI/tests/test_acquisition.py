import unittest

from tests import _bootstrap  # noqa: F401
from ifwi_gpr.acquisition import build_acquisition, make_perimeter_points


class AcquisitionTests(unittest.TestCase):
    def test_make_perimeter_points_evenly_samples_boundary(self):
        points = make_perimeter_points((5, 5), 8)
        self.assertEqual(
            points,
            [
                (0, 0),
                (0, 2),
                (0, 4),
                (2, 4),
                (4, 4),
                (4, 2),
                (4, 0),
                (2, 0),
            ],
        )

    def test_build_acquisition_accepts_explicit_lists(self):
        sources, receivers = build_acquisition(
            {
                "sources": [[0, 1], [2, 3]],
                "receivers": [[4, 5]],
            },
            (10, 10),
        )
        self.assertEqual(sources, [(0, 1), (2, 3)])
        self.assertEqual(receivers, [(4, 5)])

    def test_build_acquisition_accepts_perimeter_counts(self):
        sources, receivers = build_acquisition(
            {
                "kind": "perimeter",
                "source_count": 4,
                "receiver_count": 8,
            },
            (5, 5),
        )
        self.assertEqual(len(sources), 4)
        self.assertEqual(len(receivers), 8)
        self.assertEqual(sources[0], (0, 0))
        self.assertEqual(receivers[0], (0, 0))


if __name__ == "__main__":
    unittest.main()
