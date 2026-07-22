from __future__ import annotations

import unittest

from detector import Detection
from pedestrian_violation import (
    PedestrianViolationMonitor,
    parse_zone_points,
    scale_zone_points,
)


def person(box) -> Detection:
    return Detection(
        label="person",
        score=0.9,
        box=box,
        type_name="person",
        full_text="行人 90%",
    )


class PedestrianViolationMonitorTest(unittest.TestCase):
    def test_default_zone_and_scaling(self) -> None:
        points = parse_zone_points("")
        self.assertEqual(len(points), 4)
        scaled = scale_zone_points(points, 1280, 720, normalized=True)
        self.assertEqual(len(scaled), 4)
        self.assertTrue(all(0 <= x < 1280 and 0 <= y < 720 for x, y in scaled))

    def test_requires_confirmation_and_clears_after_exit(self) -> None:
        monitor = PedestrianViolationMonitor(
            [(0.4, 0.5), (0.8, 0.5), (0.8, 1.0), (0.4, 1.0)],
            confirmation_hits=2,
            clear_hits=2,
            hold_seconds=1.0,
        )
        inside = person((40, 40, 20, 50))
        outside = person((20, 40, 20, 50))

        monitor.update([inside], 1, 100, 100, now=1.0)
        _, violations = monitor.annotate([inside], now=1.0)
        self.assertEqual(violations, [])

        monitor.update([inside], 2, 100, 100, now=1.1)
        annotated, violations = monitor.annotate([inside], now=1.1)
        self.assertEqual(len(violations), 1)
        self.assertEqual(annotated[0].type_name, "行人违法")
        self.assertIn("进入机动车道", annotated[0].full_text)

        monitor.update([outside], 3, 100, 100, now=1.2)
        _, violations = monitor.annotate([outside], now=1.2)
        self.assertEqual(len(violations), 1)

        monitor.update([outside], 4, 100, 100, now=1.3)
        annotated, violations = monitor.annotate([outside], now=1.3)
        self.assertEqual(violations, [])
        self.assertNotEqual(annotated[0].type_name, "行人违法")


if __name__ == "__main__":
    unittest.main()
