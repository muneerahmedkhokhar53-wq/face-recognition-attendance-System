import unittest

from attendance_logic import is_valid_student_name, load_marked_students, mark_attendance, resolve_known_identity, update_confirmation


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

    def test_rejects_negative_score(self):
        self.assertIsNone(resolve_known_identity(0, -1, self.labels, 80.0))

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
        self.assertIsNone(resolve_known_identity(0, 20.0, self.labels, 0))

    def test_valid_student_names(self):
        for name in ("Muneer Ahmed", "23MTE053", "student-01", "Ahmed_2.0"):
            with self.subTest(name=name):
                self.assertTrue(is_valid_student_name(name))

    def test_rejects_unsafe_student_names(self):
        for name in ("", "   ", ".", "..", "../outside", "name/subdir", r"name\\subdir", "name:", "x" * 65, None):
            with self.subTest(name=name):
                self.assertFalse(is_valid_student_name(name))


class AttendanceCsvTests(unittest.TestCase):
    def test_csv_helper_imports(self):
        from attendance_logic import load_marked_students, mark_attendance
        self.assertTrue(callable(load_marked_students))
        self.assertTrue(callable(mark_attendance))


    def test_csv_created_and_duplicate_prevented(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attendance.csv"
            today = "2026-10-29"
            self.assertEqual(load_marked_students(path, today), set())
            self.assertTrue(mark_attendance(path, "Alice", today, "09:00:00"))
            self.assertFalse(mark_attendance(path, "Alice", today, "09:30:00"))
            with path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.reader(file))), 2)
            self.assertEqual(load_marked_students(path, today), {"Alice"})

    def test_malformed_header_is_reported(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "attendance.csv"
            path.write_text("Student,Day\\nAlice,2026-10-29\\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Name, Date, and Time"):
                load_marked_students(path, "2026-10-29")
            with self.assertRaisesRegex(ValueError, "Name, Date, and Time"):
                mark_attendance(path, "Bob", "2026-10-29", "10:00:00")



class RecognitionConfirmationTests(unittest.TestCase):
    def test_requires_consecutive_matching_frames(self):
        candidate, count, confirmed = update_confirmation(None, 0, "Alice", 3)
        self.assertEqual((candidate, count, confirmed), ("Alice", 1, False))
        candidate, count, confirmed = update_confirmation(candidate, count, "Alice", 3)
        self.assertEqual((candidate, count, confirmed), ("Alice", 2, False))
        candidate, count, confirmed = update_confirmation(candidate, count, "Alice", 3)
        self.assertEqual((candidate, count, confirmed), ("Alice", 3, True))

    def test_identity_change_restarts_confirmation(self):
        candidate, count, _ = update_confirmation("Alice", 5, "Bob", 3)
        self.assertEqual((candidate, count), ("Bob", 1))
        candidate, count, confirmed = update_confirmation(candidate, count, "Bob", 3)
        self.assertEqual((candidate, count, confirmed), ("Bob", 2, False))

    def test_missing_or_unknown_frame_resets_candidate(self):
        for observed in (None, "", "   "):
            with self.subTest(observed=observed):
                self.assertEqual(
                    update_confirmation("Alice", 2, observed, 3),
                    (None, 0, False),
                )

    def test_invalid_confirmation_length_is_rejected(self):
        for frames in (0, -1, 1.5, None):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    update_confirmation(None, 0, "Alice", frames)


if __name__ == "__main__":
    unittest.main()
