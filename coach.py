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
DOCS_PATH = BASE_DIR / "docs" / "LiquidAI_Technical_Whitepaper.pdf"
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
        self.rag = RAGEngine(DOCS_PATH)
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

        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def process_chunk(self, audio_path: Path):
        """Process a single audio chunk"""
        try:
            # 1. Transcribe audio
            text = self.lfm2.transcribe(audio_path)

            if not text or text.startswith("["):
                return  # Skip errors/empty

            self.transcript_buffer.append(text)
            self.chunk_count += 1

            # Use rolling window of last 10 chunks for context
            full_context = " ".join(self.transcript_buffer[-10:])

            # 2. Analyze vibe
            vibe = analyze_vibe(full_context)
            self.vibe_counts[vibe["dominant"]] += 1

            # 3. Query RAG for context
            rag_context, confidence = self.rag.query(full_context)
            self.confidence_sum += confidence

            # 4. Detect questions and generate answers
            question_result = get_primary_question(text)
            answer = ""
            if question_result:
                question_text, question_score = question_result
                if question_score > 0.4:  # High confidence question
                    answer = self.answer_gen.generate_answer(question_text, rag_context)
                    self.last_answer = answer

            # 5. Update display
            display_update(
                transcript=text,
                vibe=vibe["dominant"],
                vibe_emoji=vibe["emoji"],
                confidence=confidence,
                context_preview=self.rag.get_context_preview(rag_context),
            )

            # 6. Show answer if question was detected
            if answer:
                print(f"\n\n💡 SUGGESTED ANSWER:\n{answer}\n")

            # 7. Log to file
            timestamp = time.strftime('%H:%M:%S')
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"[{timestamp}] {text} | Vibe: {vibe['dominant']} | Conf: {confidence:.0%}\n")
                if answer:
                    f.write(f"[{timestamp}] 💡 ANSWER: {answer}\n")

        except Exception as e:
            print(f"\nError processing chunk: {e}")

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
    rag = RAGEngine(DOCS_PATH)

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

    context, confidence = rag.query(text)
    print(f"📊 RAG Confidence: {format_confidence(confidence)}")
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
    else:
        print("\n❓ No question detected in transcript")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Meeting Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coach.py                       # Start live capture from BlackHole
  python coach.py --device "MacBook Pro Microphone"  # Use different device
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

    # Start live processing
    agent = MeetingIntelligence(audio_device=args.device)
    agent.run()


if __name__ == "__main__":
    main()
