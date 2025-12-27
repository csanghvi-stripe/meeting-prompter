"""Hybrid Answerer: Extraction for grounding + LLM for fluency.

This module implements a two-stage answer pipeline:
1. Extract relevant sentences from RAG context (grounding)
2. Generate fluent answer with LFM2-1.2B-RAG (synthesis)

The hybrid approach combines the reliability of extraction (no hallucination)
with the fluency of LLM generation (natural language).
"""

from pathlib import Path
from typing import Tuple

from lib.answer_extractor import extract_answer, format_as_bullets
from lib.rag_generator import RAGAnswerGenerator


class HybridAnswerer:
    """
    Two-stage answer pipeline combining extraction and generation.

    Stage 1 (Grounding): Extract relevant sentences from retrieved context
    Stage 2 (Generation): Use LFM2-1.2B-RAG to synthesize a fluent answer

    The extraction stage ensures the answer is grounded in the source material,
    while the generation stage produces natural, conversational responses.

    Attributes:
        use_generation: Whether to use LLM generation (True) or extraction only
        generator: RAGAnswerGenerator instance for LLM generation
    """

    def __init__(self, model_path: Path, use_generation: bool = True):
        """
        Initialize the hybrid answerer.

        Args:
            model_path: Path to LFM2-1.2B-RAG GGUF model
            use_generation: Enable LLM generation (default True).
                           Set False to use extraction-only mode.
        """
        self.use_generation = use_generation
        self.generator = None

        if use_generation:
            if not model_path.exists():
                print(f"[Hybrid Answerer] Warning: Model not found at {model_path}")
                print("[Hybrid Answerer] Falling back to extraction-only mode")
                self.use_generation = False
            else:
                self.generator = RAGAnswerGenerator(model_path)

    def answer(
        self,
        question: str,
        rag_context: str,
        min_extraction_confidence: float = 0.25,
    ) -> Tuple[str, float, str]:
        """
        Generate answer using hybrid pipeline.

        Pipeline:
        1. Extract relevant sentences from rag_context
        2. If extraction confidence is too low, return "no match"
        3. If generation enabled, generate fluent answer from extracted context
        4. If generation fails, fall back to extracted bullets

        Args:
            question: User's question
            rag_context: Raw context from ColBERT retrieval
            min_extraction_confidence: Minimum extraction score to proceed (0-1)

        Returns:
            Tuple of (answer, confidence, method_used)
            - answer: The generated/extracted answer string
            - confidence: Extraction confidence score (0-1)
            - method_used: "hybrid", "extraction", or "no_match"
        """
        # Stage 1: Extract relevant sentences (grounding)
        extracted, extraction_confidence = extract_answer(
            rag_context, question, max_sentences=3
        )

        # If extraction confidence too low, we don't have good context
        if extraction_confidence < min_extraction_confidence or not extracted:
            return (
                "[I don't have specific information on that in my documents]",
                extraction_confidence,
                "no_match",
            )

        # Stage 2: Generate fluent answer (if enabled)
        if self.use_generation and self.generator:
            try:
                generated = self.generator.generate(question, extracted)

                # Validate generation succeeded
                if generated and not generated.startswith("["):
                    return (generated, extraction_confidence, "hybrid")

            except Exception as e:
                print(f"[Hybrid Answerer] Generation failed: {e}")
                # Fall through to extraction fallback

        # Fallback: Return extracted bullets
        formatted = format_as_bullets(extracted)
        return (formatted, extraction_confidence, "extraction")

    def set_generation_mode(self, enabled: bool) -> None:
        """
        Enable or disable LLM generation.

        Args:
            enabled: True to enable generation, False for extraction-only
        """
        if enabled and self.generator is None:
            print("[Hybrid Answerer] Cannot enable generation: model not loaded")
            return
        self.use_generation = enabled


def test_hybrid_answerer():
    """Test the hybrid answerer with sample data."""
    from pathlib import Path

    model_path = Path("models/LFM2-1.2B-RAG-Q4_K_M.gguf")

    print("Initializing Hybrid Answerer...")
    answerer = HybridAnswerer(model_path, use_generation=True)

    # Sample context from LEAP documentation
    context = """## The LEAP Platform Components

LEAP stands for Liquid Edge AI Platform and transforms cutting-edge AI research
into deployable business solutions. The platform includes a Model Library with
models ranging from LFM2-350M through 8.3B with optimized variants.

---

## Fine-Tuning CLI

The Fine-Tuning CLI provides LoRA adapters, data pipeline tools, training
infrastructure, and an evaluation suite. Users can fine-tune models on their
own data with minimal code changes.

---

## Edge SDK

The Edge SDK is cross-platform supporting macOS, Windows, Linux, iOS, and Android.
It enables deployment of LFM models directly on edge devices with optimized inference."""

    test_cases = [
        "What is LEAP?",
        "What platforms does the Edge SDK support?",
        "How can I fine-tune a model?",
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)

        answer, confidence, method = answerer.answer(question, context)

        print(f"Method: {method}")
        print(f"Confidence: {confidence:.0%}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    test_hybrid_answerer()
