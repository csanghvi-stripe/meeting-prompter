"""Answer Generator - Uses LFM2 text model to generate answers from RAG context"""
import re
from pathlib import Path
from typing import Optional, Set
from llama_cpp import Llama


def extract_key_terms(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract key terms from text for grounding validation.

    Returns a set of significant words (lowercased) that appear in the text.
    Filters out common stop words and short words.
    """
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'only', 'same', 'than', 'very', 'just',
        'also', 'now', 'here', 'there', 'then', 'once', 'from', 'into',
        'with', 'about', 'against', 'between', 'through', 'during', 'before',
        'after', 'above', 'below', 'under', 'over', 'again', 'further',
        'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although',
        'while', 'if', 'unless', 'until', 'since', 'when', 'where', 'whether',
        'not', 'no', 'yes', 'they', 'them', 'their', 'theirs', 'themselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'we', 'us', 'our', 'ours', 'ourselves', 'i', 'me', 'my', 'mine',
        'myself', 'one', 'ones', 'any', 'many', 'much', 'like', 'make',
        'made', 'use', 'used', 'using', 'get', 'got', 'getting', 'take',
        'took', 'taking', 'need', 'needs', 'needed', 'want', 'wants', 'wanted',
    }

    # Extract words, lowercase, filter by length and stop words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {w for w in words if len(w) >= min_length and w not in stop_words}


def validate_answer_grounding(answer: str, context: str, min_overlap: float = 0.15) -> bool:
    """
    Validate that an answer is grounded in the provided context.

    An answer is considered grounded if a sufficient percentage of its
    key terms also appear in the context.

    Args:
        answer: The generated answer to validate
        context: The source context the answer should be derived from
        min_overlap: Minimum fraction of answer terms that must appear in context

    Returns:
        True if answer appears grounded in context, False otherwise
    """
    if not answer or not context:
        return False

    answer_terms = extract_key_terms(answer)
    context_terms = extract_key_terms(context)

    if not answer_terms:
        return True  # No significant terms to validate

    # Calculate overlap
    overlap = answer_terms & context_terms
    overlap_ratio = len(overlap) / len(answer_terms)

    return overlap_ratio >= min_overlap


# Response format templates - designed for small model instruction following
# Key insight: Small models follow "summarize this" better than "answer using only this"
SALES_PROMPT_TEMPLATE = """Summarize this text to answer the question.

TEXT:
{context}

QUESTION: {question}

SUMMARY (2-3 bullets from the text above):
•"""


DETAILED_PROMPT_TEMPLATE = """Summarize this text to answer the question.

TEXT:
{context}

QUESTION: {question}

SUMMARY:"""


class AnswerGenerator:
    """Generates answers to questions using LFM2 text model and RAG context"""

    # Response modes for different meeting contexts
    MODE_QUICK = "quick"      # Fast 2-bullet response
    MODE_DETAILED = "detailed"  # Structured 3-part response

    def __init__(self, model_path: Path, n_ctx: int = 1024, mode: str = "quick"):
        """
        Initialize the answer generator.

        Args:
            model_path: Path to LFM2-1.2B GGUF model
            n_ctx: Context window size (smaller = more stable)
            mode: Response mode - "quick" (2 bullets) or "detailed" (3 parts)
        """
        self.model_path = model_path
        self.llm: Optional[Llama] = None
        self.n_ctx = n_ctx
        self._call_count = 0
        self.mode = mode

    def load(self):
        """Load the model (lazy loading for memory efficiency)"""
        if self.llm is None:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,  # Use GPU (Metal on Mac)
                verbose=False,
            )

    def _reset_if_needed(self):
        """Reset model state periodically to prevent KV cache corruption"""
        self._call_count += 1
        # Reset every 10 calls to prevent state corruption
        if self._call_count >= 10 and self.llm is not None:
            try:
                self.llm.reset()
            except:
                # If reset fails, reload the model
                self.llm = None
                self.load()
            self._call_count = 0

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 150,
    ) -> str:
        """
        Generate a structured, sales-friendly answer to a question.

        The response is formatted for easy verbal delivery in meetings:
        - Quick mode: 2-3 bullet points
        - Detailed mode: SHORT ANSWER / WHY IT MATTERS / PROOF POINT

        Includes grounding validation to ensure the answer is derived from
        the provided context, not hallucinated.

        Args:
            question: The customer's question
            context: Relevant context from RAG (Liquid docs)
            max_tokens: Maximum tokens in response (default 150 for structured output)

        Returns:
            Structured answer string optimized for reading aloud
        """
        self.load()
        self._reset_if_needed()

        # Build prompt for structured Q&A
        prompt = self._build_prompt(question, context)

        # Stop sequences for clean output
        stop_sequences = ["Question:", "Customer:", "Context:", "TEXT:", "\n\n\n", "---"]

        # Retry logic for llama_decode errors
        for attempt in range(2):
            try:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    temperature=0.1,  # Very low for factual, grounded responses
                    top_p=0.9,        # Nucleus sampling for coherence
                    repeat_penalty=1.1,  # Discourage repetition
                )
                raw_answer = response['choices'][0]['text'].strip()
                answer = self._clean_answer(raw_answer)

                # Validate answer is grounded in context
                if not validate_answer_grounding(answer, context, min_overlap=0.15):
                    # Answer not grounded - fall back to context excerpt
                    return self._create_fallback_answer(context)

                return answer

            except Exception as e:
                if attempt == 0 and "llama_decode" in str(e):
                    # Reset and retry once
                    try:
                        self.llm.reset()
                    except:
                        self.llm = None
                        self.load()
                    continue
                return "[Let me get back to you on that after the call]"

        return "[Let me get back to you on that after the call]"

    def _create_fallback_answer(self, context: str) -> str:
        """
        Create a fallback answer by extracting key points from context.

        Used when model-generated answer fails grounding validation.
        """
        # Extract first 2-3 sentences from context as fallback
        sentences = context.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

        if not sentences:
            return "[Let me check our documentation on that]"

        # Take first 2 meaningful sentences
        fallback_sentences = sentences[:2]
        result = ". ".join(fallback_sentences)

        if not result.endswith('.'):
            result += "."

        return f"• {result}"

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build a sales-optimized prompt for structured answer generation.

        Uses different templates based on mode:
        - quick: Bullet point format for fast responses
        - detailed: Labeled sections (SHORT ANSWER / WHY IT MATTERS / PROOF POINT)
        """
        # Allow more context since we're now getting multi-chunk results
        max_context_chars = 800
        if len(context) > max_context_chars:
            # Try to truncate at a sentence boundary
            truncated = context[:max_context_chars]
            last_period = truncated.rfind('.')
            if last_period > max_context_chars * 0.7:
                context = truncated[:last_period + 1]
            else:
                context = truncated + "..."

        # Truncate question if too long
        if len(question) > 150:
            question = question[:150] + "..."

        # Select template based on mode
        if self.mode == self.MODE_DETAILED:
            return DETAILED_PROMPT_TEMPLATE.format(context=context, question=question)
        else:
            return SALES_PROMPT_TEMPLATE.format(context=context, question=question)

    def _clean_answer(self, answer: str) -> str:
        """
        Clean up the generated answer while preserving structure.

        Keeps bullet points and labeled sections intact for readability.
        """
        if not answer:
            return "[Let me follow up on that after the call]"

        # Split into lines and clean each
        lines = answer.split('\n')
        cleaned_lines = []
        seen = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Normalize bullet points for consistency
            if line.startswith('-'):
                line = '•' + line[1:]
            elif line.startswith('*'):
                line = '•' + line[1:]

            # Skip duplicate lines
            line_lower = line.lower()
            if line_lower in seen:
                continue
            seen.add(line_lower)

            cleaned_lines.append(line)

        # Join with newlines to preserve structure
        result = '\n'.join(cleaned_lines)

        # Ensure we have something useful
        if not result or len(result) < 10:
            return "[Let me follow up on that after the call]"

        return result

    def set_mode(self, mode: str) -> None:
        """
        Switch response mode.

        Args:
            mode: "quick" for bullet points, "detailed" for labeled sections
        """
        if mode in (self.MODE_QUICK, self.MODE_DETAILED):
            self.mode = mode

    def generate_coaching_tip(
        self,
        transcript: str,
        context: str,
        vibe: str,
    ) -> str:
        """
        Generate a coaching tip for the AE/SE based on conversation context.

        Args:
            transcript: Recent conversation transcript
            context: Relevant RAG context
            vibe: Current emotional vibe of the conversation

        Returns:
            Coaching tip string
        """
        self.load()

        prompt = f"""You are coaching a sales engineer during a customer call. Based on the conversation and the customer's emotional state, provide a brief, actionable tip.

Recent conversation:
{transcript[:500]}

Customer mood: {vibe}

Relevant product info:
{context[:500]}

Coaching tip (one sentence):"""

        try:
            response = self.llm(
                prompt,
                max_tokens=75,
                stop=["\n\n", "---"],
                temperature=0.5,
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return ""


def test_answer_generator():
    """Test the answer generator with sales-optimized prompts"""
    from pathlib import Path

    model_path = Path("models/LFM2-1.2B-Q4_K_M.gguf")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print("Loading answer generator...")
    gen = AnswerGenerator(model_path, mode="quick")
    gen.load()
    print("Model loaded!")

    # Test question
    question = "How does Liquid AI handle real-time audio processing?"
    context = """Liquid Neural Networks (LNNs) use continuous-time dynamics based on
    ordinary differential equations (ODEs). LFM2-Audio is an end-to-end audio foundation
    model that processes audio-to-intelligence in a single unified pass, achieving
    sub-100ms latency for real-time applications."""

    print(f"\nQuestion: {question}")
    print(f"Context: {context[:100]}...")

    # Test quick mode (bullet points)
    print("\n" + "="*50)
    print("QUICK MODE (bullet points):")
    print("="*50)
    gen.set_mode("quick")
    answer = gen.generate_answer(question, context)
    print(answer)

    # Test detailed mode (labeled sections)
    print("\n" + "="*50)
    print("DETAILED MODE (labeled sections):")
    print("="*50)
    gen.set_mode("detailed")
    answer = gen.generate_answer(question, context)
    print(answer)


if __name__ == "__main__":
    test_answer_generator()
