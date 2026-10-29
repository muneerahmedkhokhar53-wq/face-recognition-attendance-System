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
    name = name.strip()
    if not name or name in {".", ".."} or len(name) > 64:
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
                if row.get("Name") and row.get("Date") == today
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
                if row.get("Name") == name and row.get("Date") == date:
                    return False
            file.seek(0, 2)
            csv.writer(file).writerow([name, date, time])
        return True
    except csv.Error as exc:
        raise ValueError(f"Attendance CSV is malformed: {exc}") from exc
