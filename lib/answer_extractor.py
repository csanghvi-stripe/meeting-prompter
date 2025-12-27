"""
Answer Extractor - Extracts answers from context without hallucination.

Design Philosophy:
- Extract, don't generate: Pull sentences from context rather than creating new text
- LLM only for polish: Use the model to rephrase, not to add information
- Fail gracefully: If we can't find an answer, say so honestly

This approach is more reliable than generative QA because:
1. Small models (1-3B) are weak at following "only use this context" instructions
2. Extraction structurally prevents hallucination
3. Answers are traceable to source text
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ScoredSentence:
    """A sentence with its relevance score."""
    text: str
    score: float
    position: int  # Position in document (earlier = higher weight for definitions)


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences, preserving structure.

    Handles:
    - Standard sentence endings (. ! ?)
    - Bullet points and list items
    - Markdown headers
    """
    # Split on sentence boundaries
    # But don't split on abbreviations like "e.g." or "i.e."
    text = re.sub(r'\bi\.e\.', 'IE_PLACEHOLDER', text)
    text = re.sub(r'\be\.g\.', 'EG_PLACEHOLDER', text)

    # Split on . ! ? followed by space and capital, or on newlines for bullets
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\s*\n', text)

    # Restore abbreviations
    sentences = [s.replace('IE_PLACEHOLDER', 'i.e.').replace('EG_PLACEHOLDER', 'e.g.')
                 for s in sentences]

    # Clean and filter
    cleaned = []
    for s in sentences:
        s = s.strip()
        # Skip very short sentences
        if len(s) < 20:
            continue
        # Skip markdown headers alone (they'll be part of content)
        if s.startswith('#') and len(s) < 50:
            continue
        cleaned.append(s)

    return cleaned


def score_sentence(sentence: str, question: str, position: int, total: int) -> float:
    """
    Score a sentence's relevance to a question.

    Scoring factors:
    1. Keyword overlap - sentences containing question words score higher
    2. Definition patterns - sentences with "is", "stands for", "means" score higher
    3. Position bonus - earlier sentences often contain definitions
    4. Length penalty - very long sentences are less likely to be direct answers
    """
    sentence_lower = sentence.lower()
    question_lower = question.lower()

    # Extract question keywords (excluding stop words)
    stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'how', 'does', 'do', 'can',
                  'you', 'me', 'tell', 'explain', 'about', 'of', 'for', 'to', 'in',
                  'this', 'that', 'these', 'those', 'some', 'any', 'with'}
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower)) - stop_words
    sentence_words = set(re.findall(r'\b\w{3,}\b', sentence_lower))

    score = 0.0

    # 1. Keyword overlap (primary signal)
    if question_words:
        overlap = len(question_words & sentence_words)
        keyword_score = overlap / len(question_words)
        score += keyword_score * 0.5

    # 2. Definition patterns (strong signal for "what is X" questions)
    definition_patterns = [
        r'\bstands for\b',
        r'\bis defined as\b',
        r'\bmeans\b',
        r'\brefers to\b',
        r'\b(?:is|are) (?:a|an|the)\b',  # "X is a ..."
        r'^[A-Z][A-Za-z]+ (?:is|are)\b',  # Sentence starting with "X is"
    ]

    for pattern in definition_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            score += 0.2
            break

    # 3. Position bonus (earlier = more likely to be definition)
    position_score = 1.0 - (position / max(total, 1)) * 0.3
    score += position_score * 0.2

    # 4. Length penalty (very long sentences are less focused)
    word_count = len(sentence.split())
    if word_count > 50:
        score *= 0.8
    elif word_count < 10:
        score *= 0.9  # Too short might be incomplete

    # 5. Bonus for sentences containing key question terms exactly
    for word in question_words:
        if len(word) >= 4 and word in sentence_lower:
            score += 0.1

    return min(score, 1.0)


def extract_answer(context: str, question: str, max_sentences: int = 3) -> Tuple[str, float]:
    """
    Extract the most relevant sentences from context to answer the question.

    Args:
        context: The retrieved context from RAG
        question: The user's question
        max_sentences: Maximum sentences to extract

    Returns:
        Tuple of (extracted_answer, confidence_score)
    """
    if not context or not question:
        return "", 0.0

    # Split context into sentences
    sentences = extract_sentences(context)

    if not sentences:
        return "", 0.0

    # Score each sentence
    scored = []
    for i, sentence in enumerate(sentences):
        score = score_sentence(sentence, question, i, len(sentences))
        scored.append(ScoredSentence(text=sentence, score=score, position=i))

    # Sort by score descending
    scored.sort(key=lambda x: x.score, reverse=True)

    # Take top sentences
    top_sentences = scored[:max_sentences]

    # If best score is too low, we don't have a good answer
    if not top_sentences or top_sentences[0].score < 0.2:
        return "", top_sentences[0].score if top_sentences else 0.0

    # Re-sort by position to maintain document order
    top_sentences.sort(key=lambda x: x.position)

    # Combine into answer
    answer = " ".join(s.text for s in top_sentences)

    # Calculate confidence as average of top sentence scores
    avg_score = sum(s.score for s in top_sentences) / len(top_sentences)

    return answer, avg_score


def format_as_bullets(text: str) -> str:
    """
    Format extracted text as bullet points for readability.

    Splits on sentence boundaries and adds bullets.
    """
    sentences = extract_sentences(text)

    if not sentences:
        return text

    # Take first 3 sentences max
    sentences = sentences[:3]

    # Format as bullets
    bullets = ["• " + s.strip() for s in sentences]

    return "\n".join(bullets)


class AnswerExtractor:
    """
    Extracts answers from context with optional LLM polishing.

    The extraction happens without LLM - we use the LLM only to
    optionally rephrase the extracted text for natural speech.
    """

    def __init__(self, llm=None):
        """
        Initialize the extractor.

        Args:
            llm: Optional LLM for polishing (llama_cpp.Llama instance)
        """
        self.llm = llm

    def extract(self, context: str, question: str, polish: bool = False) -> Tuple[str, float]:
        """
        Extract answer from context.

        Args:
            context: Retrieved context from RAG
            question: User's question
            polish: Whether to use LLM to rephrase for natural speech

        Returns:
            Tuple of (answer, confidence)
        """
        # Step 1: Extract relevant sentences (no LLM)
        extracted, confidence = extract_answer(context, question)

        if not extracted or confidence < 0.2:
            return "[I don't have specific information on that in my documents]", confidence

        # Step 2: Format as bullets
        formatted = format_as_bullets(extracted)

        # Step 3: Optional LLM polish (rephrase only, don't add info)
        if polish and self.llm:
            polished = self._polish(formatted, question)
            if polished:
                return polished, confidence

        return formatted, confidence

    def _polish(self, extracted: str, question: str) -> Optional[str]:
        """
        Use LLM to rephrase extracted text for natural speech.

        The LLM is only allowed to rephrase - not add information.
        """
        if not self.llm:
            return None

        prompt = f"""Rephrase this text to sound natural when spoken aloud.
Do not add any new information. Only rephrase what's given.

Text to rephrase:
{extracted}

Rephrased (keep it brief, 2-3 sentences):"""

        try:
            response = self.llm(
                prompt,
                max_tokens=100,
                temperature=0.1,
                stop=["\n\n", "---", "Text:"],
            )
            result = response['choices'][0]['text'].strip()

            # Validate the polish didn't add hallucination
            # Check that key terms from extracted appear in result
            extracted_words = set(re.findall(r'\b\w{4,}\b', extracted.lower()))
            result_words = set(re.findall(r'\b\w{4,}\b', result.lower()))

            # If result lost too many key terms, use original
            overlap = len(extracted_words & result_words) / max(len(extracted_words), 1)
            if overlap < 0.3:
                return None

            return result

        except Exception:
            return None
