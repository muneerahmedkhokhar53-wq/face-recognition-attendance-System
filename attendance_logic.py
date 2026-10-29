"""Dependency-free helpers for attendance identity validation."""

import re


def is_valid_student_name(name):
    """Allow simple labels while preventing path separators and traversal."""
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name or name in {".", ".."} or len(name) > 64:
        return False
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}", name) is not None


def resolve_known_identity(label, score, labels, confidence_threshold):
    """Return a known enrolled name only when LBPH's score passes the threshold.

    LBPH uses a distance-like score, so lower scores indicate a closer match.
    Unknown labels, malformed/non-finite scores, and threshold failures are rejected.
    """
    try:
        label_id = int(label)
        numeric_score = float(score)
        threshold = float(confidence_threshold)
    except (TypeError, ValueError, OverflowError):
        return None

    if numeric_score != numeric_score or numeric_score in (float("inf"), float("-inf")):
        return None
    if threshold != threshold or threshold in (float("inf"), float("-inf")):
        return None
    if numeric_score < 0 or threshold <= 0 or numeric_score >= threshold:
        return None

    name = labels.get(label_id)
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()
