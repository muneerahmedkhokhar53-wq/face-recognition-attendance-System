import unittest

from attendance_logic import resolve_known_identity


class ResolveKnownIdentityTests(unittest.TestCase):
    def setUp(self):
        self.labels = {0: "23MTE053", 1: "Student B"}

    def test_accepts_known_label_below_threshold(self):
        self.assertEqual(
            resolve_known_identity(0, 42.5, self.labels, 80.0),
            "23MTE053",
        )

    def test_rejects_score_equal_to_threshold(self):
        self.assertIsNone(resolve_known_identity(0, 80.0, self.labels, 80.0))

    def test_rejects_score_above_threshold(self):
        self.assertIsNone(resolve_known_identity(0, 95.0, self.labels, 80.0))

    def test_rejects_unknown_label(self):
        self.assertIsNone(resolve_known_identity(99, 20.0, self.labels, 80.0))

    def test_rejects_missing_or_empty_name(self):
        self.assertIsNone(resolve_known_identity(2, 20.0, {2: ""}, 80.0))
        self.assertIsNone(resolve_known_identity(2, 20.0, {}, 80.0))

    def test_rejects_invalid_and_non_finite_scores(self):
        for score in ("not-a-score", None, float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score):
                self.assertIsNone(resolve_known_identity(0, score, self.labels, 80.0))

    def test_rejects_invalid_label_or_threshold(self):
        self.assertIsNone(resolve_known_identity("not-a-label", 20.0, self.labels, 80.0))
        self.assertIsNone(resolve_known_identity(0, 20.0, self.labels, "invalid"))


if __name__ == "__main__":
    unittest.main()
