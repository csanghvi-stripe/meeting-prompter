"""Answer Generator - Uses LFM2 text model to generate answers from RAG context"""
from pathlib import Path
from typing import Optional
from llama_cpp import Llama


class AnswerGenerator:
    """Generates answers to questions using LFM2 text model and RAG context"""

    def __init__(self, model_path: Path, n_ctx: int = 2048):
        """
        Initialize the answer generator.

        Args:
            model_path: Path to LFM2-1.2B GGUF model
            n_ctx: Context window size
        """
        self.model_path = model_path
        self.llm: Optional[Llama] = None
        self.n_ctx = n_ctx

    def load(self):
        """Load the model (lazy loading for memory efficiency)"""
        if self.llm is None:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,  # Use GPU (Metal on Mac)
                verbose=False,
            )

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 150,
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

        # Build prompt for Q&A
        prompt = self._build_prompt(question, context)

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
            return f"[Unable to generate answer: {e}]"

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for answer generation"""
        # Truncate context if too long
        max_context_chars = 1500
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."

        prompt = f"""You are a helpful sales engineer for Liquid AI. Answer the customer's question using the provided context. Be concise, accurate, and helpful. If the context doesn't contain enough information, say so.

Context from Liquid AI documentation:
{context}

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
