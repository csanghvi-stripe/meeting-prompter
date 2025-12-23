"""Score normalization for ColBERT MaxSim scores."""

import math


def normalize_maxsim(raw_score: float, center: float = 25.0, scale: float = 5.0) -> float:
    """
    Normalize ColBERT MaxSim score to 0-1 confidence range.

    ColBERT's MaxSim scores are unbounded (typically 15-40 range for good matches).
    We use a sigmoid transformation to map them to a stable 0-1 range.

    Score interpretation (empirical):
    - < 15: Poor match
    - 15-25: Moderate match
    - 25-35: Good match
    - > 35: Excellent match

    Args:
        raw_score: The raw MaxSim score from ColBERT
        center: The score value that maps to 0.5 confidence (default 25.0)
        scale: Controls the steepness of the sigmoid (default 5.0)

    Returns:
        Normalized confidence score between 0.0 and 1.0
    """
    # Sigmoid transformation: 1 / (1 + e^(-(x - center) / scale))
    normalized = 1 / (1 + math.exp(-(raw_score - center) / scale))
    return max(0.0, min(1.0, normalized))


def format_confidence(score: float) -> str:
    """
    Format confidence score for display with color indicators.

    Args:
        score: Normalized confidence score (0-1)

    Returns:
        Formatted string with emoji indicator
    """
    percentage = score * 100
    if percentage >= 50:
        return f"[green]{percentage:.0f}%[/green]"
    elif percentage >= 30:
        return f"[yellow]{percentage:.0f}%[/yellow]"
    else:
        return f"[red]{percentage:.0f}%[/red]"


# Calibration constants - adjust based on empirical testing
CONFIDENCE_THRESHOLD = 0.30  # Minimum confidence to return a result
HIGH_CONFIDENCE = 0.70  # Threshold for "high confidence" answers
