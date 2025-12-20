"""Vibe Check - Emotional category detection from transcript"""
from typing import Dict, List, Tuple

# Emotional category keywords
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "Excited": [
        "amazing", "fantastic", "love", "great", "wow", "excited", "awesome",
        "incredible", "brilliant", "excellent", "thrilled", "wonderful",
        "perfect", "outstanding", "impressive", "phenomenal",
    ],
    "Frustrated": [
        "frustrated", "annoying", "problem", "issue", "difficult", "struggle",
        "stuck", "broken", "failing", "doesn't work", "wrong", "confused",
        "complicated", "impossible", "nightmare", "terrible", "awful",
    ],
    "Uncertain": [
        "maybe", "not sure", "possibly", "might", "unclear", "confused",
        "uncertain", "don't know", "wondering", "perhaps", "could be",
        "hard to say", "depends", "unsure", "questionable",
    ],
    "Confident": [
        "definitely", "absolutely", "certain", "clearly", "sure", "know",
        "confident", "guarantee", "proven", "obvious", "undoubtedly",
        "without doubt", "certainly", "precisely", "exactly",
    ],
    "Engaged": [
        "interesting", "tell me more", "how", "why", "what if", "curious",
        "intrigued", "fascinating", "learn", "understand", "explain",
        "dig deeper", "explore", "consider", "think about",
    ],
}

# Emoji mappings for display
VIBE_EMOJI: Dict[str, str] = {
    "Excited": "🔥",
    "Frustrated": "😤",
    "Uncertain": "🤔",
    "Confident": "💪",
    "Engaged": "👀",
    "Neutral": "😐",
}


def analyze_vibe(transcript: str) -> Dict:
    """
    Analyze transcript for emotional categories.

    Returns:
        dict with 'dominant' emotion, 'emoji', and 'scores' breakdown
    """
    if not transcript or not transcript.strip():
        return {
            "dominant": "Neutral",
            "emoji": VIBE_EMOJI["Neutral"],
            "scores": {k: 0.0 for k in EMOTION_KEYWORDS},
        }

    words = transcript.lower().split()
    text_lower = transcript.lower()

    scores: Dict[str, int] = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Check for multi-word phrases
            if ' ' in keyword:
                score += text_lower.count(keyword)
            else:
                score += sum(1 for w in words if keyword in w)
        scores[emotion] = score

    # Find dominant emotion
    total = sum(scores.values())
    if total == 0:
        dominant = "Neutral"
        normalized = {k: 0.0 for k in scores}
    else:
        dominant = max(scores, key=scores.get)
        normalized = {k: v / total for k, v in scores.items()}

    return {
        "dominant": dominant,
        "emoji": VIBE_EMOJI.get(dominant, "😐"),
        "scores": normalized,
        "raw_scores": scores,
    }


def get_vibe_summary(vibe_result: Dict) -> str:
    """Format vibe result for display"""
    emoji = vibe_result["emoji"]
    dominant = vibe_result["dominant"]

    # Get top 2 non-zero emotions for context
    scores = vibe_result.get("raw_scores", {})
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [e for e, s in sorted_emotions[:2] if s > 0]

    if len(top_emotions) > 1:
        return f"{emoji} {dominant} (with hints of {top_emotions[1]})"
    return f"{emoji} {dominant}"
