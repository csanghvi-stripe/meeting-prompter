"""Question Detector - Identifies questions in transcribed text"""
import re
from typing import List, Tuple, Optional


# Question patterns to detect
QUESTION_PATTERNS = [
    # Direct question words
    r'\b(what|how|why|when|where|who|which|whose|whom)\b.*\?',
    r'\b(what|how|why|when|where|who|which|whose|whom)\s+(?:is|are|do|does|did|can|could|would|will|should|has|have|had)\b',

    # Yes/No question starters
    r'\b(is|are|do|does|did|can|could|would|will|should|has|have|had)\s+(?:it|this|that|there|you|we|they|the)\b',

    # Common question phrases
    r'\b(can you|could you|would you|tell me|explain|describe)\b',
    r'\b(what about|how about|what if)\b',

    # Technical/sales questions
    r'\b(how does|how do|how can|how would)\b',
    r'\b(what\'s the|what is the|what are the)\b',
    r'\b(does it|can it|will it|is it)\b',
    r'\b(do you|can you|could you)\s+(?:support|offer|provide|have|integrate)\b',
]

# Keywords that often appear in customer questions
QUESTION_KEYWORDS = [
    'pricing', 'cost', 'price', 'license', 'subscription',
    'integrate', 'integration', 'api', 'sdk', 'compatibility',
    'security', 'privacy', 'compliance', 'gdpr', 'hipaa', 'soc2',
    'performance', 'latency', 'speed', 'benchmark',
    'support', 'documentation', 'training', 'onboarding',
    'difference', 'compare', 'versus', 'vs', 'better',
    'feature', 'capability', 'limitation', 'roadmap',
    'example', 'demo', 'proof', 'case study',
]


def detect_questions(text: str) -> List[Tuple[str, float]]:
    """
    Detect questions in transcribed text.

    Returns:
        List of (question_text, confidence_score) tuples
    """
    if not text or not text.strip():
        return []

    questions = []
    sentences = _split_sentences(text)

    for sentence in sentences:
        score = _score_question(sentence)
        if score > 0.3:  # Threshold for question detection
            questions.append((sentence.strip(), score))

    return questions


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, merging continuations"""
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', text)

    # Merge back sentences that are clearly continuations of previous
    continuation_starters = [
        'help me', 'tell me', 'and', 'also', 'but', 'because',
        'so', 'then', 'or', 'that', 'which', 'where', 'when',
        'understand', 'explain', 'describe'
    ]

    merged = []
    current = ""
    for part in parts:
        part_lower = part.lower().strip()
        # If part starts with continuation words, merge with previous
        if current and any(part_lower.startswith(c) for c in continuation_starters):
            current = current + " " + part
        else:
            if current:
                merged.append(current)
            current = part
    if current:
        merged.append(current)

    # Handle long sentences
    result = []
    for s in merged:
        if len(s) > 100:
            # Only split on ? if there's more content after
            sub_parts = re.split(r'(?<=[?])\s+', s)
            result.extend(sub_parts)
        else:
            result.append(s)

    return [s.strip() for s in result if s.strip()]


def _score_question(sentence: str) -> float:
    """
    Score how likely a sentence is a COMPLETE question (0.0 to 1.0)
    """
    sentence_lower = sentence.lower().strip()
    words = sentence_lower.split()
    score = 0.0

    # Reject too short (likely incomplete)
    if len(words) < 5:
        return 0.0

    # Reject fragments that are clearly incomplete
    # Note: 'me' and 'you' are VALID endings (e.g., "Can you help me?")
    incomplete_endings = ['the', 'a', 'an', 'to', 'of', 'for', 'with', 'about',
                         'how', 'what', 'is', 'are', 'does', 'do', 'can', 'could', 'would']
    last_word = words[-1].rstrip('?.,!') if words else ""
    if last_word in incomplete_endings:
        return 0.0

    # Reject if it's just "can you tell me" type fragments
    fragment_patterns = [
        r'^can you tell me\??$',
        r'^tell me\??$',
        r'^can you explain\??$',
        r'^what about\??$',
        r'^how about\??$',
        r'^okay,?\s*(can you|tell me)?\??$',
        r'^so,?\s*(tell me|can you)?\??$',
    ]
    for pattern in fragment_patterns:
        if re.match(pattern, sentence_lower):
            return 0.0

    # Check for question mark (strong signal)
    if '?' in sentence:
        score += 0.5

    # Check question patterns
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, sentence_lower):
            score += 0.3
            break

    # Check for question keywords (business/technical topics)
    keyword_count = sum(1 for kw in QUESTION_KEYWORDS if kw in sentence_lower)
    if keyword_count > 0:
        score += min(0.3, keyword_count * 0.1)

    # Check for question word at start
    first_word = words[0] if words else ""
    if first_word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']:
        score += 0.2
    elif first_word in ['can', 'could', 'would', 'does', 'do', 'is', 'are']:
        score += 0.1

    # Bonus for complete-sounding questions (has subject and verb)
    if len(words) >= 7:
        score += 0.1

    return min(1.0, score)


def extract_question_topic(question: str) -> Optional[str]:
    """
    Extract the main topic/subject of a question for RAG lookup.
    """
    question_lower = question.lower()

    # Look for keyword matches
    for kw in QUESTION_KEYWORDS:
        if kw in question_lower:
            return kw

    # Extract noun phrases after "about", "regarding", etc.
    about_match = re.search(r'\b(?:about|regarding|for|with)\s+(\w+(?:\s+\w+)?)', question_lower)
    if about_match:
        return about_match.group(1)

    # Extract object of "what is/are"
    what_match = re.search(r'\bwhat\s+(?:is|are)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', question_lower)
    if what_match:
        return what_match.group(1)

    # Extract object of "how does/do"
    how_match = re.search(r'\bhow\s+(?:does|do|can|would)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', question_lower)
    if how_match:
        return how_match.group(1)

    return None


def is_question(text: str) -> bool:
    """Simple check if text contains a question"""
    questions = detect_questions(text)
    return len(questions) > 0


def get_primary_question(text: str) -> Optional[Tuple[str, float]]:
    """Get the most likely question from text"""
    questions = detect_questions(text)
    if not questions:
        return None
    # Return highest scoring question
    return max(questions, key=lambda x: x[1])
