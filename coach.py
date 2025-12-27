#!/usr/bin/env python3
"""
Real-Time Meeting Intelligence Agent

Powered by LFM2-Audio running 100% locally on Mac.

Features:
- Zero-Latency Multimodality: Audio-to-intelligence in a single pass
- Constant-Memory RAG: Local document retrieval
- Privacy Moat: All processing stays on-device
"""
import os
# Suppress tokenizer parallelism warning (must be before any HuggingFace imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.lfm2_wrapper import LFM2Wrapper
from lib.audio_capture import AudioCapture, list_audio_devices
from lib.vibe_check import analyze_vibe, get_vibe_summary
from lib.rag_engine import RAGEngine, format_confidence
from lib.question_detector import detect_questions, get_primary_question, get_question_score
from lib.hybrid_answerer import HybridAnswerer
from lib.question_buffer import QuestionBuffer, BufferConfig
from lib.dashboard import (
    display_header,
    display_status,
    display_update,
    display_summary,
)

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
RUNNER_DIR = BASE_DIR / "runners" / "macos-arm64"
DOCS_DIR = BASE_DIR / "docs"  # Load all PDFs from this directory
RAG_MODEL = MODEL_DIR / "LFM2-1.2B-RAG-Q4_K_M.gguf"  # RAG-specialized model for answer generation
OUTPUT_FILE = BASE_DIR / "output" / "live_analytics.txt"


class MeetingIntelligence:
    """Real-time meeting intelligence agent with Q&A"""

    def __init__(self, audio_device: str = "BlackHole 2ch"):
        """
        Initialize the meeting intelligence agent.

        Args:
            audio_device: Audio input device name
        """
        display_header()

        # Initialize components
        display_status("Loading LFM2-Audio model...")
        self.lfm2 = LFM2Wrapper(MODEL_DIR, RUNNER_DIR)
        display_status("LFM2-Audio ready")

        display_status("Loading RAG engine...")
        self.rag = RAGEngine(DOCS_DIR)
        display_status("RAG engine ready")

        # Hybrid answerer - extraction for grounding + LLM for fluency
        display_status("Loading LFM2-1.2B-RAG model...")
        self.answerer = HybridAnswerer(
            model_path=RAG_MODEL,
            use_generation=True,  # Always use generation by default
        )
        display_status("Hybrid answerer ready (extraction + generation)")

        self.audio = AudioCapture(device=audio_device)

        # State tracking
        self.transcript_buffer = []
        self.chunk_count = 0
        self.vibe_counts = defaultdict(int)
        self.confidence_sum = 0.0
        self.last_answer = ""  # Cache last generated answer

        # Thread-safe question buffering with time-based pause detection
        self.question_buffer = QuestionBuffer(BufferConfig(
            pause_threshold=1.5,      # 1.5 seconds of silence triggers flush
            max_buffer_time=8.0,      # 8 second max buffer time
            min_words=4,              # Minimum words for valid question
            confidence_threshold=0.3  # Minimum question confidence
        ))

        # Wire up silence callback from AudioCapture to QuestionBuffer
        self.audio.on_silence = self._on_silence

        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _is_hallucination(self, text: str) -> bool:
        """
        Detect LFM2-Audio hallucination patterns.

        The model tends to hallucinate these patterns when given noise/silence.
        """
        text_lower = text.lower().strip()

        # Common hallucination starters - vague statements that don't relate to context
        hallucination_starters = [
            "i don't know what",
            "i'm not sure what",
            "she chose",
            "he chose",
            "they chose",
            "it's just the",
            "it's going to be",
            "that's going to be",
            "you're going to",
            "we're going to",
            "the one that was",
            "the reason why",
            "i think it's",
            "i think that",
            "i guess",
            "i suppose",
            "maybe it's",
            "perhaps it's",
            "it seems like",
            "it looks like",
            "sort of",
            "kind of like",
        ]

        for starter in hallucination_starters:
            if text_lower.startswith(starter):
                return True

        # Third-person statements are usually hallucinations (not questions to us)
        third_person_starters = ["she ", "he ", "they ", "it was ", "there was "]
        for starter in third_person_starters:
            if text_lower.startswith(starter):
                return True

        # Very vague questions without any specific topic
        vague_questions = [
            "can you explain to me",
            "can you tell me",
            "can you help me",
            "tell me about",
            "explain to me",
            "what do you mean",
            "what does that mean",
        ]
        text_clean = text_lower.rstrip('?.,!')
        if text_clean in vague_questions:
            return True

        # Repetitive/circular phrases (hallucination symptom)
        words = text_lower.split()
        if len(words) >= 6:
            # Check for repeated 3-word sequences
            for i in range(len(words) - 5):
                seq = " ".join(words[i:i+3])
                rest = " ".join(words[i+3:])
                if seq in rest:
                    return True

        return False

    def _is_noise(self, text: str) -> bool:
        """Check if text is just filler words/noise that should be ignored"""
        text_lower = text.lower().strip()
        words = text_lower.split()

        # Too short to be meaningful
        if len(words) < 3:
            return True

        # Check for hallucination patterns first
        if self._is_hallucination(text):
            return True

        # Common filler phrases to ignore
        noise_phrases = [
            "yeah", "yeah yeah", "yeah yeah yeah",
            "um", "uh", "uh huh", "um um",
            "okay", "ok", "oh", "oh well",
            "i don't know", "i dunno",
            "and then", "to that one", "so",
            "a", "the", "is it", "it is",
            "hmm", "hm", "ah", "eh",
            "right", "right right",
            "sure", "sure sure",
            "well", "well well",
            "you know", "like",
        ]

        # Check if entire text matches a noise phrase
        text_clean = text_lower.rstrip('.,?!')
        if text_clean in noise_phrases:
            return True

        # Check if all words are filler words
        filler_words = {'yeah', 'um', 'uh', 'okay', 'ok', 'oh', 'well', 'so', 'like',
                        'right', 'hmm', 'hm', 'ah', 'eh', 'a', 'the', 'and', 'then',
                        'i', "don't", 'know', 'just', 'to', 'that', 'one', 'it', 'is'}
        meaningful_words = [w for w in words if w.rstrip('.,?!') not in filler_words]

        # If less than 2 meaningful words, it's noise
        if len(meaningful_words) < 2:
            return True

        return False

    def _on_silence(self, timestamp: float):
        """Called when silence is detected in audio stream"""
        # Let buffer check if pause threshold reached
        question = self.question_buffer.on_silence(timestamp)
        if question:
            self._process_complete_question(question)

    def _normalize_text(self, text: str) -> str:
        """
        Light regex normalization - preserves content, only fixes obvious issues.
        Does NOT extract or interpret - just cleans duplicate words and known mishearings.
        """
        import re

        result = text

        # 1. Fix consecutive duplicate words only
        result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)

        # 2. Fix known mishearings (conservative, specific patterns)
        replacements = [
            (r'\bL\s+Those\b', 'Liquid'),
            (r'\bLiquid\s+AI\s+Liquid\s+AI\b', 'Liquid AI'),
            (r'\bliquid\s+liquid\b', 'Liquid'),
        ]
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # 3. Clean whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        # 4. Capitalize first letter
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

        # 5. Add ? only if clearly a question (don't force it)
        question_starters = ['how', 'what', 'why', 'when', 'where', 'who', 'which',
                            'can', 'could', 'would', 'help', 'tell', 'explain', 'does', 'do', 'is', 'are']
        if result and not result.endswith('?') and not result.endswith('.'):
            if any(result.lower().startswith(w) for w in question_starters):
                result = result.rstrip('.!,') + '?'

        return result

    def process_chunk(self, audio_path: Path, timestamp: float = None):
        """Process a single audio chunk with intelligent question buffering"""
        import time as time_module
        timestamp = timestamp or time_module.time()

        try:
            # 1. Transcribe audio
            text = self.lfm2.transcribe(audio_path)

            # 2. Handle empty/error transcriptions
            if not text or text.startswith("["):
                # Transcription failed or returned noise marker
                # Force flush buffer if we have content (indicates pause)
                question = self.question_buffer.force_flush()
                if question:
                    self._process_complete_question(question)
                return

            # 3. Handle filler/noise
            if self._is_noise(text):
                print(f"\r🎧 Listening...                                        ", end="", flush=True)
                # Noise also indicates pause - check buffer
                question = self.question_buffer.force_flush()
                if question:
                    self._process_complete_question(question)
                return

            # 4. Add to transcript buffer for context
            self.transcript_buffer.append(text)
            self.chunk_count += 1

            # 5. Add to question buffer - may return complete question
            question = self.question_buffer.add_chunk(text, timestamp)

            if question:
                # Buffer returned a complete question
                self._process_complete_question(question)
            else:
                # Show what we're buffering
                status = self.question_buffer.get_status()
                if status["is_buffering"]:
                    preview = status["text_preview"]
                    score = get_question_score(preview) if preview else 0
                    if score > 0.2:
                        print(f"\r🎤 \"{preview[:60]}...\" ", end="", flush=True)
                    else:
                        print(f"\r🎧 \"{text[:60]}\"" + ("..." if len(text) > 60 else ""), end="", flush=True)
                else:
                    print(f"\r🎧 \"{text[:60]}\"" + ("..." if len(text) > 60 else ""), end="", flush=True)

                # Log non-question speech to file
                log_timestamp = time.strftime('%H:%M:%S')
                with open(OUTPUT_FILE, "a") as f:
                    f.write(f"[{log_timestamp}] {text}\n")

        except Exception as e:
            print(f"\nError processing chunk: {e}")

    def _process_complete_question(self, full_question: str):
        """Process a complete question from the buffer and generate answer"""
        import time as time_module

        if not full_question:
            return

        # Skip if too short
        if len(full_question.split()) < 4:
            print(f"\r🎧 Listening... (too short)", end="", flush=True)
            return

        # Skip hallucinations
        if self._is_hallucination(full_question):
            print(f"\r🎧 Listening... (filtered)", end="", flush=True)
            return

        # Check question confidence - must look like an actual question
        question_score = get_question_score(full_question)
        if question_score < 0.25:
            # Log but don't process - not a real question
            log_timestamp = time.strftime('%H:%M:%S')
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"[{log_timestamp}] {full_question}\n")
            print(f"\r🎧 \"{full_question[:50]}...\"" if len(full_question) > 50 else f"\r🎧 \"{full_question}\"", end="", flush=True)
            return

        print(f"\n\n⏳ Processing: \"{full_question[:50]}...\"" if len(full_question) > 50 else f"\n\n⏳ Processing: \"{full_question}\"")

        # Light normalization - preserves content, only fixes obvious issues
        cleaned_question = self._normalize_text(full_question)

        # Get context using cleaned question for better RAG matching
        full_context = " ".join(self.transcript_buffer[-10:])
        rag_context, confidence, source_file = self.rag.query(cleaned_question)
        vibe = analyze_vibe(full_context)

        # Skip if confidence too low (question doesn't match docs)
        if confidence < 0.30:
            print(f"\n⚠️  No match in documents ({confidence:.0%})")
            print(f"   \"{cleaned_question[:50]}...\"")
            print(f"🎧 Listening...")
            return

        # Generate answer using hybrid pipeline (extraction + LFM2-1.2B-RAG)
        t_start = time_module.time()
        answer, extraction_confidence, method = self.answerer.answer(
            question=cleaned_question,
            rag_context=rag_context,
        )
        t_answer = time_module.time() - t_start

        if answer and not answer.startswith("[I don't") and method != "no_match":
            # Clear screen and show Q&A
            print("\033[2J\033[H")  # Clear screen
            print("━" * 70)
            print("  🎯 MEETING INTELLIGENCE AGENT")
            print("━" * 70)

            # Show CLEANED question (not raw transcription)
            print(f"\n❓ QUESTION:")
            if len(cleaned_question) > 65:
                q_words = cleaned_question.split()
                q_line = "   "
                for word in q_words:
                    if len(q_line) + len(word) > 65:
                        print(q_line)
                        q_line = "   " + word
                    else:
                        q_line += " " + word if q_line != "   " else word
                if q_line.strip():
                    print(q_line)
            else:
                print(f"   {cleaned_question}")

            print(f"\n💡 ANSWER:")
            # Display answer preserving bullet structure
            answer_lines = answer.split('\n')
            for ans_line in answer_lines:
                ans_line = ans_line.strip()
                if not ans_line:
                    continue
                # Check if it's a bullet point or labeled section
                is_bullet = ans_line.startswith('•') or ans_line.startswith('-')
                is_label = any(ans_line.startswith(label) for label in
                              ['SHORT ANSWER:', 'WHY IT MATTERS:', 'PROOF POINT:'])

                if is_bullet or is_label:
                    print()  # Blank line before each bullet/section

                # Word wrap this line
                words = ans_line.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) > 65:
                        print(line)
                        line = "   " + word
                    else:
                        line += " " + word if line != "   " else word
                if line.strip():
                    print(line)

            # Source citation
            print(f"\n📄 Source: {source_file}")
            print(f"   \"{self.rag.get_context_preview(rag_context, 120)}\"")
            conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"📊 Confidence: [{conf_bar}] {confidence:.0%}")
            print(f"🎭 Vibe: {vibe['emoji']} {vibe['dominant']}")

            # Show which method was used
            method_label = "LFM2-RAG" if method == "hybrid" else "Extraction"
            print(f"⚡ Method: {method_label} ({t_answer:.1f}s)")

            print("\n" + "━" * 70)
            print("🎧 Listening for next question... (Ctrl+C to stop)")

            # Log Q&A to file
            timestamp = time.strftime('%H:%M:%S')
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"\n[{timestamp}] ❓ Q: {cleaned_question}\n")
                f.write(f"[{timestamp}] 💡 A: {answer}\n")
                f.write(f"[{timestamp}] 📄 Source: {source_file} | Confidence: {confidence:.0%}\n\n")
        else:
            print(f"\r🎧 Listening... (no answer found)", end="", flush=True)

    def run(self):
        """Start real-time processing"""
        display_status(f"Listening to {self.audio.device}...")
        display_status(f"Output file: {OUTPUT_FILE}")
        print(f"\n📝 Mode: Hybrid (extraction + LFM2-1.2B-RAG generation)")
        print(f"   Press Ctrl+C to stop\n")

        self._running = True
        try:
            self.audio.start_stream(self.process_chunk)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.show_summary()

    def show_summary(self):
        """Display meeting summary"""
        if self.chunk_count > 0:
            avg_conf = self.confidence_sum / self.chunk_count
            display_summary(self.chunk_count, dict(self.vibe_counts), avg_conf)


def test_transcription(audio_file: Path):
    """Test mode: transcribe a single audio file with Q&A"""
    display_header()
    display_status("Test mode: Single file transcription with Q&A")

    lfm2 = LFM2Wrapper(MODEL_DIR, RUNNER_DIR)
    rag = RAGEngine(DOCS_DIR)
    answerer = HybridAnswerer(RAG_MODEL, use_generation=True)

    # Resolve to absolute path
    audio_file = audio_file.resolve()
    display_status(f"Transcribing: {audio_file}")
    text = lfm2.transcribe(audio_file)

    print(f"\n📝 Transcription:\n{text}\n")

    vibe = analyze_vibe(text)
    print(f"🎭 Vibe: {get_vibe_summary(vibe)}")

    context, confidence, source_file = rag.query(text)
    print(f"📊 RAG Confidence: {format_confidence(confidence)}")
    print(f"📄 Source: {source_file}")
    print(f"📄 Context: {rag.get_context_preview(context, 200)}")

    # Check for questions and generate answers
    question_result = get_primary_question(text)
    if question_result:
        question_text, score = question_result
        print(f"\n❓ Question Detected (confidence: {score:.0%}):\n   {question_text}")

        if score > 0.3:
            print("\n💡 ANSWER (Hybrid Pipeline):")
            answer, extraction_conf, method = answerer.answer(question_text, context)
            print(f"   {answer}")
            print(f"\n📄 Source: {source_file}")
            print(f"⚡ Method: {'LFM2-RAG' if method == 'hybrid' else 'Extraction'}")
    else:
        print("\n❓ No question detected in transcript")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Meeting Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coach.py                  # Live meeting mode (BlackHole)
  python coach.py --mic            # Test mode (speak into microphone)
  python coach.py --test audio.wav # Test with audio file
  python coach.py --list-devices   # List available audio devices

Answer Mode:
  Hybrid mode - extraction for grounding + LFM2-1.2B-RAG for fluent answers.
  Stage 1: Extract relevant sentences from documents (grounding)
  Stage 2: Generate fluent answer with RAG-specialized model
        """,
    )
    parser.add_argument(
        "--device", "-d",
        default="BlackHole 2ch",
        help="Audio input device name (default: BlackHole 2ch)",
    )
    parser.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Use MacBook microphone instead of BlackHole (for testing)",
    )
    parser.add_argument(
        "--test", "-t",
        type=Path,
        help="Test mode: transcribe a single audio file",
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio devices and exit",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.test:
        if not args.test.exists():
            print(f"Error: Audio file not found: {args.test}")
            sys.exit(1)
        test_transcription(args.test)
        return

    # Determine audio device
    if args.mic:
        audio_device = "MacBook Pro Microphone"
        print("\n🎤 MIC MODE: Speak into your microphone to test")
        print("   (Use without --mic for live meetings with BlackHole)\n")
    else:
        audio_device = args.device
        print("\n🔊 MEETING MODE: Capturing from", audio_device)
        print("   (Use --mic to test with your microphone)\n")

    # Start live processing
    agent = MeetingIntelligence(audio_device=audio_device)
    agent.run()


if __name__ == "__main__":
    main()
