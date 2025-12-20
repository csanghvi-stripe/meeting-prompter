"""Answer Generator - Uses LFM2 text model to generate answers from RAG context"""
from pathlib import Path
from typing import Optional
from llama_cpp import Llama


class AnswerGenerator:
    """Generates answers to questions using LFM2 text model and RAG context"""

    def __init__(self, model_path: Path, n_ctx: int = 1024):
        """
        Initialize the answer generator.

        Args:
            model_path: Path to LFM2-1.2B GGUF model
            n_ctx: Context window size (smaller = more stable)
        """
        self.model_path = model_path
        self.llm: Optional[Llama] = None
        self.n_ctx = n_ctx
        self._call_count = 0

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

    def clean_question(self, garbled_question: str) -> str:
        """
        Clean garbled transcriptions using regex + LLM.

        Fixes issues like:
        - "How does liquid Does Liquid AI work" -> "How does Liquid AI work?"
        - Duplicate words, grammar errors, incomplete phrases

        Args:
            garbled_question: Raw transcription that may have errors

        Returns:
            Cleaned question string
        """
        import re

        # Step 1: Basic regex cleanup (fast, reliable)
        cleaned = garbled_question

        # Remove consecutive duplicate words/phrases (case-insensitive)
        # Handle "word Word" and "word word" patterns
        cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)

        # Handle "phrase Phrase" patterns like "tell me tell me"
        cleaned = re.sub(r'\b(\w+\s+\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)

        # Handle "What is what is" pattern (same words different case)
        cleaned = re.sub(r'\b(what|how|why|when|where|who|can|does|is)\s+(is|are|do|does|can)\s+\1\s+\2\b',
                        r'\1 \2', cleaned, flags=re.IGNORECASE)

        # Remove common filler patterns
        filler_patterns = [
            r'\b(um|uh|like|you know|I mean)\b\s*',
            r'\bthe the\b', r'\ba a\b', r'\ban an\b',
            r'\bof of\b', r'\bto to\b', r'\bis is\b',
            r'\bOr the\b', r'\bThen the\b', r'\bAnd the\b',
            r'^(so|well|okay|oh|right|yeah)\s+',  # At start of sentence
            r'^(well\s+)?(okay\s+)?(so\s+)?',  # Combo at start
        ]
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Fix capitalization after cleanup
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()

        # Ensure ends with ?
        if cleaned and not cleaned.endswith('?'):
            cleaned = cleaned.rstrip('.!') + '?'

        # Step 2: Additional regex patterns for common transcription errors

        # Fix "L Those" -> "Liquid" (common mishearing)
        cleaned = re.sub(r'\bL\s+Those\b', 'Liquid', cleaned, flags=re.IGNORECASE)

        # Fix duplicate "liquid liquid" -> "Liquid"
        cleaned = re.sub(r'\b(liquid)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bLiquid\s+liquid\b', 'Liquid', cleaned, flags=re.IGNORECASE)

        # Fix "open" or "open." -> "OpenAI" when comparing
        if 'different' in cleaned.lower() or 'compare' in cleaned.lower():
            cleaned = re.sub(r'\bthan\s+open\b\.?', 'than OpenAI', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\bfrom\s+open\b\.?', 'from OpenAI', cleaned, flags=re.IGNORECASE)

        # Remove sentence fragments after a question mark
        if '?' in cleaned:
            cleaned = cleaned.split('?')[0] + '?'

        # Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned and not cleaned.endswith('?'):
            cleaned = cleaned.rstrip('.!') + '?'

        return cleaned if len(cleaned) > 5 else garbled_question

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 100,
    ) -> str:
        """
        Generate an answer to a question based on RAG context.

        Args:
            question: The customer's question
            context: Relevant context from RAG (Liquid docs)
            max_tokens: Maximum tokens in response

        Returns:
            Generated answer string
        """
        self.load()
        self._reset_if_needed()

        # Build prompt for Q&A
        prompt = self._build_prompt(question, context)

        # Retry logic for llama_decode errors
        for attempt in range(2):
            try:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    stop=["Question:", "\n\n\n", "---"],
                    temperature=0.3,  # Lower for more factual responses
                )
                answer = response['choices'][0]['text'].strip()
                return self._clean_answer(answer)
            except Exception as e:
                if attempt == 0 and "llama_decode" in str(e):
                    # Reset and retry once
                    try:
                        self.llm.reset()
                    except:
                        self.llm = None
                        self.load()
                    continue
                return f"[Unable to generate answer: {e}]"

        return "[Unable to generate answer after retry]"

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for answer generation"""
        # Truncate context to fit in smaller context window
        max_context_chars = 600
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."

        # Truncate question if too long
        if len(question) > 200:
            question = question[:200] + "..."

        prompt = f"""Answer the question using the context. Be concise.

Context: {context}

Question: {question}

Answer:"""
        return prompt

    def _clean_answer(self, answer: str) -> str:
        """Clean up the generated answer"""
        # Remove any trailing incomplete sentences
        if answer and not answer[-1] in '.!?':
            last_period = answer.rfind('.')
            if last_period > len(answer) // 2:
                answer = answer[:last_period + 1]

        # Remove any repeated phrases
        lines = answer.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            line_clean = line.strip().lower()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line.strip())

        return ' '.join(unique_lines)

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
    """Test the answer generator"""
    from pathlib import Path

    model_path = Path("models/LFM2-1.2B-Q4_K_M.gguf")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print("Loading answer generator...")
    gen = AnswerGenerator(model_path)
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
    print("\nGenerating answer...")

    answer = gen.generate_answer(question, context)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    test_answer_generator()
