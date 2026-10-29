"""Dependency-free helpers for attendance identity and CSV handling."""
import csv
import math
import re
from pathlib import Path

ATTENDANCE_COLUMNS = ("Name", "Date", "Time")


def is_valid_student_name(name):
    """Allow simple labels while preventing path separators and traversal."""
    if not isinstance(name, str):
        return False
    if name != name.strip():
        return False
    name = name.strip()
    if not name or name in {".", ".."} or len(name) > 64:
        return False
    if name.endswith((".", " ")):
        return False
    # Windows reserves these device names, even when followed by an extension.
    device_name = name.split(".", 1)[0].upper()
    reserved_names = {"CON", "PRN", "AUX", "NUL"}
    reserved_names.update(f"COM{i}" for i in range(1, 10))
    reserved_names.update(f"LPT{i}" for i in range(1, 10))
    if device_name in reserved_names:
        return False
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}", name) is not None


def resolve_known_identity(label, score, labels, confidence_threshold):
    """Return a known enrolled name only when LBPH's score passes the threshold."""
    try:
        label_id = int(label)
        numeric_score = float(score)
        threshold = float(confidence_threshold)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_score) or not math.isfinite(threshold):
        return None
    if numeric_score < 0 or threshold <= 0 or numeric_score >= threshold:
        return None
    name = labels.get(label_id)
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def load_marked_students(path, today):
    """Create the CSV if needed and return names already marked for today."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(ATTENDANCE_COLUMNS)
        return set()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not set(ATTENDANCE_COLUMNS).issubset(reader.fieldnames):
                raise ValueError("Attendance CSV needs Name, Date, and Time columns.")
            return {
                row["Name"].strip()
                for row in reader
                if row.get("Name") and row.get("Date", "").strip() == today
            }
    except csv.Error as exc:
        raise ValueError(f"Attendance CSV is malformed: {exc}") from exc


def mark_attendance(path, name, date, time):
    """Append once per name/date. Return False when the record already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(ATTENDANCE_COLUMNS)
    try:
        with path.open("r+", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not set(ATTENDANCE_COLUMNS).issubset(reader.fieldnames):
                raise ValueError("Attendance CSV needs Name, Date, and Time columns.")
            for row in reader:
                if row.get("Name", "").strip() == name.strip() and row.get("Date", "").strip() == date.strip():
                    return False
            file.seek(0, 2)
            csv.writer(file).writerow([name, date, time])
        return True
    except csv.Error as exc:
        raise ValueError(f"Attendance CSV is malformed: {exc}") from exc


def update_confirmation(candidate_name, candidate_count, observed_name, required_frames):
    """Advance multi-frame identity confirmation.

    Returns (candidate_name, candidate_count, confirmed). An absent/invalid
    observation resets the candidate so frames from separate appearances
    cannot be combined.
    """
    if not isinstance(required_frames, int) or required_frames < 1:
        raise ValueError("required_frames must be a positive integer")
    if not isinstance(observed_name, str) or not observed_name.strip():
        return None, 0, False

    observed_name = observed_name.strip()
    if observed_name == candidate_name:
        try:
            candidate_count = max(0, int(candidate_count)) + 1
        except (TypeError, ValueError, OverflowError):
            candidate_count = 1
    else:
        candidate_count = 1

    return observed_name, candidate_count, candidate_count >= required_frames
