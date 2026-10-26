import asyncio
import unittest

from organism.calibrate import curve, knee, llm_curve, membw_gbps
from organism.llm import FakeLLM


class KneeTest(unittest.TestCase):
    def test_linear_goes_to_end(self):
        self.assertEqual(knee([(1, 10.0), (2, 20.0), (4, 40.0), (6, 60.0)]), 6)

    def test_saturation_stops_at_plateau(self):
        self.assertEqual(knee([(1, 10.0), (2, 19.0), (4, 20.0), (6, 20.0)]), 2)

    def test_flat_gives_first(self):
        self.assertEqual(knee([(1, 10.0), (2, 10.0), (4, 10.0)]), 1)

    def test_empty(self):
        self.assertEqual(knee([]), 1)

    def test_single(self):
        self.assertEqual(knee([(3, 5.0)]), 3)

    def test_gain_threshold(self):
        pts = [(1, 10.0), (2, 11.5), (4, 12.0)]
        self.assertEqual(knee(pts, gain=0.1), 2)
        self.assertEqual(knee(pts, gain=0.5), 1)


class MeasureTest(unittest.TestCase):
    def test_membw_positive(self):
        self.assertGreater(membw_gbps(8, runs=2), 0)

    def test_curve_shape(self):
        pts = curve((4, 1, 2, 2), tasks=8, buf_mb=1, repeats=1)
        self.assertEqual([n for n, _ in pts], [1, 2, 4])
        self.assertTrue(all(rate > 0 for _, rate in pts))

    def test_curve_knee_in_range(self):
        pts = curve((1, 2), tasks=8, buf_mb=1, repeats=1)
        self.assertIn(knee(pts), (1, 2))

    def test_llm_curve_fake(self):
        pts = asyncio.run(llm_curve(FakeLLM(delay_s=0.01), slots=(2, 1, 4), per_slot=2))
        self.assertEqual([n for n, _ in pts], [1, 2, 4])
        self.assertTrue(all(rate > 0 for _, rate in pts))
        self.assertGreater(pts[-1][1], pts[0][1])


if __name__ == "__main__":
    unittest.main()
