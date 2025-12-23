"""Question Buffer - Thread-safe buffering with time-based pause detection"""
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BufferConfig:
    """Configuration for question buffering behavior"""
    pause_threshold: float = 1.5      # Seconds of silence to trigger flush
    max_buffer_time: float = 8.0      # Max seconds before forced flush
    min_words: int = 4                # Minimum words for valid question
    confidence_threshold: float = 0.3  # Minimum question confidence
    min_start_score: float = 0.1      # Minimum score to start buffering


class QuestionBuffer:
    """
    Thread-safe question buffering with time-based pause detection.

    Solves the "second question triggers first" problem by using actual
    timestamps instead of chunk counts for pause detection.
    """

    def __init__(self, config: BufferConfig = None):
        self.config = config or BufferConfig()
        self._lock = threading.Lock()
        self._chunks: List[Tuple[str, float]] = []  # (text, timestamp)
        self._buffer_start: float = 0
        self._last_speech: float = 0
        self._is_buffering: bool = False

    def add_chunk(self, text: str, timestamp: float = None) -> Optional[str]:
        """
        Add a transcribed chunk and return complete question if ready.

        Args:
            text: Transcribed text from audio chunk
            timestamp: Time when the chunk was recorded (defaults to now)

        Returns:
            Complete question text if buffer should be flushed, None otherwise
        """
        timestamp = timestamp or time.time()

        with self._lock:
            # Check if we should flush BEFORE adding new chunk
            if self._should_flush(timestamp):
                question = self._flush_locked()
                # Check if new chunk looks like a question before starting new buffer
                if self._looks_like_question_start(text):
                    self._start_new_buffer(text, timestamp)
                return question

            # Add to buffer
            if not self._is_buffering:
                # Only start buffering if text looks like question start
                if self._looks_like_question_start(text):
                    self._start_new_buffer(text, timestamp)
                # Otherwise, don't buffer - return None
            else:
                self._chunks.append((text, timestamp))
                self._last_speech = timestamp

            # Check if question is complete based on confidence
            if self._is_question_complete():
                return self._flush_locked()

            return None

    def _looks_like_question_start(self, text: str) -> bool:
        """Check if text looks like the start of a question worth buffering."""
        if not text or len(text.split()) < 2:
            return False

        # Import here to avoid circular imports
        from lib.question_detector import _score_question

        score = _score_question(text)
        return score >= self.config.min_start_score

    def on_silence(self, timestamp: float = None) -> Optional[str]:
        """
        Called when silence is detected (no speech in audio chunk).

        Args:
            timestamp: Time of the silence detection

        Returns:
            Complete question if buffer should be flushed
        """
        timestamp = timestamp or time.time()

        with self._lock:
            if self._should_flush(timestamp):
                return self._flush_locked()
            return None

    def force_flush(self) -> Optional[str]:
        """
        Force flush the buffer regardless of timing.

        Use when transcription returns noise/errors and we want to
        process whatever we have buffered.

        Returns:
            Buffered text if any, None if buffer empty
        """
        with self._lock:
            if self._is_buffering and self._chunks:
                return self._flush_locked()
            return None

    def _should_flush(self, current_time: float) -> bool:
        """Check if buffer should be flushed based on timing."""
        if not self._is_buffering or not self._chunks:
            return False

        # Pause detection: time since last speech
        pause_duration = current_time - self._last_speech
        if pause_duration >= self.config.pause_threshold:
            return True

        # Timeout: max buffer time exceeded
        buffer_duration = current_time - self._buffer_start
        if buffer_duration >= self.config.max_buffer_time:
            return True

        return False

    def _is_question_complete(self) -> bool:
        """
        Check if the buffered text represents a complete question.

        Uses confidence scoring and structural analysis instead of
        relying on punctuation (which transcription often omits).
        """
        if not self._chunks:
            return False

        # Combine all buffered text
        text = self._get_combined_text()
        words = text.split()

        # Too short to be complete
        if len(words) < self.config.min_words:
            return False

        # Import here to avoid circular imports
        from lib.question_detector import _score_question

        score = _score_question(text)

        # High confidence = definitely complete
        if score >= 0.6:
            return True

        # Medium confidence with good length = probably complete
        if score >= self.config.confidence_threshold and len(words) >= 8:
            return True

        return False

    def _start_new_buffer(self, text: str, timestamp: float):
        """Initialize a new buffer with the first chunk."""
        self._chunks = [(text, timestamp)]
        self._buffer_start = timestamp
        self._last_speech = timestamp
        self._is_buffering = True

    def _flush_locked(self) -> str:
        """
        Flush buffer and return combined text.

        Must be called while holding _lock.
        """
        text = self._get_combined_text()
        self._chunks = []
        self._is_buffering = False
        self._buffer_start = 0
        self._last_speech = 0
        return text

    def _get_combined_text(self) -> str:
        """Combine all chunk texts into single string."""
        if not self._chunks:
            return ""
        texts = [chunk[0] for chunk in self._chunks]
        return " ".join(texts)

    @property
    def is_buffering(self) -> bool:
        """Check if currently buffering (thread-safe)."""
        with self._lock:
            return self._is_buffering

    @property
    def buffer_duration(self) -> float:
        """Get current buffer duration in seconds (thread-safe)."""
        with self._lock:
            if not self._is_buffering:
                return 0.0
            return time.time() - self._buffer_start

    def get_status(self) -> dict:
        """Get buffer status for debugging (thread-safe)."""
        with self._lock:
            return {
                "is_buffering": self._is_buffering,
                "chunk_count": len(self._chunks),
                "buffer_duration": time.time() - self._buffer_start if self._is_buffering else 0,
                "text_preview": self._get_combined_text()[:50] if self._chunks else "",
            }
