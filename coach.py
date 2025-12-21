#!/usr/bin/env python3
"""
Real-Time Meeting Intelligence Agent

Powered by LFM2-Audio running 100% locally on Mac.

Features:
- Zero-Latency Multimodality: Audio-to-intelligence in a single pass
- Constant-Memory RAG: Local document retrieval
- Privacy Moat: All processing stays on-device
"""
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
from lib.question_detector import detect_questions, get_primary_question
from lib.answer_generator import AnswerGenerator
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
TEXT_MODEL = MODEL_DIR / "LFM2-1.2B-Q4_K_M.gguf"
OUTPUT_FILE = BASE_DIR / "output" / "live_analytics.txt"


class MeetingIntelligence:
    """Real-time meeting intelligence agent with Q&A"""

    def __init__(self, audio_device: str = "BlackHole 2ch"):
        display_header()

        # Initialize components
        display_status("Loading LFM2-Audio model...")
        self.lfm2 = LFM2Wrapper(MODEL_DIR, RUNNER_DIR)
        display_status("LFM2-Audio ready")

        display_status("Loading RAG engine...")
        self.rag = RAGEngine(DOCS_DIR)
        display_status("RAG engine ready")

        display_status("Loading LFM2 text model for Q&A...")
        self.answer_gen = AnswerGenerator(TEXT_MODEL)
        self.answer_gen.load()
        display_status("Q&A engine ready")

        self.audio = AudioCapture(device=audio_device)

        # State tracking
        self.transcript_buffer = []
        self.chunk_count = 0
        self.vibe_counts = defaultdict(int)
        self.confidence_sum = 0.0
        self.last_answer = ""  # Cache last generated answer

        # Question buffering - accumulate until pause detected
        self.question_chunks = []  # Accumulate multiple chunks
        self.silence_count = 0  # Track consecutive short/noise chunks
        self.is_buffering_question = False
        self.buffer_start_time = 0

        # Context buffer - keep recent non-question chunks for merging
        self.context_buffer = []
        self.MAX_CONTEXT_CHUNKS = 3

        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _is_noise(self, text: str) -> bool:
        """Check if text is just filler words/noise that should be ignored"""
        text_lower = text.lower().strip()
        words = text_lower.split()

        # Too short to be meaningful
        if len(words) < 3:
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

    def _looks_like_question_start(self, text: str) -> bool:
        """Check if text looks like the start of a question"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        if not words:
            return False

        # Question starters
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who', 'which',
                            'can', 'could', 'would', 'will', 'does', 'do', 'is', 'are',
                            'tell', 'explain', 'describe', 'help']

        first_word = words[0].rstrip('.,?!')
        return first_word in question_starters

    def _has_question_ending(self, text: str) -> bool:
        """Check if text has a clear question ending"""
        text = text.strip()
        # Ends with question mark
        if text.endswith('?'):
            return True
        # Long enough sentence ending with period (likely complete thought)
        if text.endswith('.') and len(text.split()) >= 8:
            return True
        return False

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

    def process_chunk(self, audio_path: Path):
        """Process a single audio chunk with intelligent question buffering"""
        try:
            import time as time_module

            # 1. Transcribe audio
            t_start = time_module.time()
            text = self.lfm2.transcribe(audio_path)
            t_transcribe = time_module.time() - t_start

            if not text or text.startswith("["):
                self.silence_count += 1
                self._check_and_process_buffer()
                return

            # Check if this is noise/filler
            is_noise = self._is_noise(text)

            if is_noise:
                self.silence_count += 1
                self._check_and_process_buffer()
                print(f"\r🎧 Listening...                                        ", end="", flush=True)
                return

            # Meaningful content - reset silence counter
            self.silence_count = 0

            # Add to transcript buffer
            self.transcript_buffer.append(text)
            self.chunk_count += 1

            # Check if we should start buffering a question
            if not self.is_buffering_question:
                # Keep recent chunks in context buffer (for merging with questions)
                self.context_buffer.append(text)
                self.context_buffer = self.context_buffer[-self.MAX_CONTEXT_CHUNKS:]

                if self._looks_like_question_start(text):
                    # Start buffering - include recent context for multi-part questions
                    self.is_buffering_question = True
                    # Use context buffer to catch "Can you help me? Help me understand..."
                    self.question_chunks = list(self.context_buffer)
                    self.buffer_start_time = time_module.time()
                    full_so_far = " ".join(self.question_chunks)
                    print(f"\r🎤 Question: \"{full_so_far[:70]}\"" + ("..." if len(full_so_far) > 70 else ""), end="", flush=True)

                    # If it already looks complete, process it
                    if self._has_question_ending(text) and len(text.split()) >= 6:
                        self._process_question_buffer()
                    return
                else:
                    # Not a question, just show what we heard
                    print(f"\r🎧 \"{text[:60]}\"" + ("..." if len(text) > 60 else ""), end="", flush=True)

                    # Log to file
                    timestamp = time.strftime('%H:%M:%S')
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(f"[{timestamp}] {text}\n")
                    return

            # We're buffering a question - add this chunk
            self.question_chunks.append(text)
            full_question = " ".join(self.question_chunks)
            print(f"\r🎤 Question: \"{full_question[:70]}\"" + ("..." if len(full_question) > 70 else ""), end="", flush=True)

            # Check if question is now complete
            if self._has_question_ending(text):
                self._process_question_buffer()
            # Or if we've been buffering too long (timeout after 10 seconds)
            elif time_module.time() - self.buffer_start_time > 10:
                self._process_question_buffer()

        except Exception as e:
            print(f"\nError processing chunk: {e}")

    def _check_and_process_buffer(self):
        """Check if we should process buffered question due to silence"""
        # If we're buffering and hit silence, process the question
        if self.is_buffering_question and self.question_chunks and self.silence_count >= 1:
            self._process_question_buffer()

    def _process_question_buffer(self):
        """Process accumulated question chunks and generate answer"""
        import time as time_module

        if not self.question_chunks:
            self.is_buffering_question = False
            return

        # Combine all chunks into full question
        full_question = " ".join(self.question_chunks)

        # Reset buffer state
        self.question_chunks = []
        self.is_buffering_question = False
        self.context_buffer = []  # Clear context after processing

        # Skip if too short
        if len(full_question.split()) < 4:
            print(f"\r🎧 Listening... (question too short: \"{full_question}\")", end="", flush=True)
            return

        print(f"\n\n⏳ Processing: \"{full_question[:50]}...\"" if len(full_question) > 50 else f"\n\n⏳ Processing: \"{full_question}\"")

        # Light normalization - preserves content, only fixes obvious issues
        cleaned_question = self._normalize_text(full_question)

        # Get context using cleaned question for better RAG matching
        full_context = " ".join(self.transcript_buffer[-10:])
        rag_context, confidence, source_file = self.rag.query(cleaned_question)
        vibe = analyze_vibe(full_context)

        # Skip if confidence too low (question doesn't match docs)
        if confidence < 0.05:
            print(f"\n⚠️  No match in documents ({confidence:.0%})")
            print(f"   \"{cleaned_question[:50]}...\"")
            print(f"🎧 Listening...")
            return

        # Generate answer using cleaned question
        t_start = time_module.time()
        answer = self.answer_gen.generate_answer(cleaned_question, rag_context)
        t_answer = time_module.time() - t_start

        if answer and not answer.startswith("["):
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
            # Word wrap the answer
            words = answer.split()
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

            print("\n" + "━" * 70)
            print("🎧 Listening for next question... (Ctrl+C to stop)")

            # Log Q&A to file
            timestamp = time.strftime('%H:%M:%S')
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"\n[{timestamp}] ❓ Q: {cleaned_question}\n")
                f.write(f"[{timestamp}] 💡 A: {answer}\n")
                f.write(f"[{timestamp}] 📄 Source: {source_file} | Confidence: {confidence:.0%}\n\n")
        else:
            print(f"\r🎧 Listening... (couldn't generate answer)", end="", flush=True)

    def run(self):
        """Start real-time processing"""
        display_status(f"Listening to {self.audio.device}...")
        display_status(f"Output file: {OUTPUT_FILE}")
        print()  # Empty line before updates

        try:
            self.audio.start_stream(self.process_chunk)
        except KeyboardInterrupt:
            pass
        finally:
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

    display_status("Loading Q&A engine...")
    answer_gen = AnswerGenerator(TEXT_MODEL)
    answer_gen.load()
    display_status("Q&A engine ready")

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
            print("\n💡 SUGGESTED ANSWER:")
            answer = answer_gen.generate_answer(question_text, context)
            print(f"   {answer}")
            print(f"\n📄 Source: {source_file}")
    else:
        print("\n❓ No question detected in transcript")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Meeting Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coach.py                       # Live meeting mode (BlackHole)
  python coach.py --mic                 # Test mode (speak into microphone)
  python coach.py --test audio.wav      # Test with audio file
  python coach.py --list-devices        # List available audio devices
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
