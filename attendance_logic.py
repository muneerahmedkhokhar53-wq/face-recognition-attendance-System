"""Small, dependency-free helpers for attendance identity validation."""


def resolve_known_identity(label, score, labels, confidence_threshold):
    """Return a known enrolled name only when LBPH's score passes the threshold.

    LBPH uses a distance-like score, so lower scores indicate a closer match.
    Unknown labels, malformed scores, non-finite scores, and threshold failures
    must never be accepted as an attendance identity.
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
    if numeric_score >= threshold:
        return None

    name = labels.get(label_id)
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()
