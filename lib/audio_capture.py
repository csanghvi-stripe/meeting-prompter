"""Audio Capture - Real-time streaming from BlackHole with chunking"""
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioCapture:
    """Continuous audio capture with chunking for real-time processing"""

    def __init__(
        self,
        device: str = "BlackHole 2ch",
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        overlap: float = 0.5,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self.buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.callback: Optional[Callable] = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}")

        audio_data = indata.flatten().astype(np.float32)

        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, audio_data])

            # Check if we have enough for a chunk
            while len(self.buffer) >= self.chunk_samples:
                chunk = self.buffer[:self.chunk_samples].copy()
                self.buffer = self.buffer[self.step_samples:]  # Keep overlap

                # Process chunk in separate thread to avoid blocking audio
                if self.callback:
                    threading.Thread(
                        target=self._process_chunk,
                        args=(chunk,),
                        daemon=True
                    ).start()

    def _process_chunk(self, chunk: np.ndarray):
        """Save chunk to temp file and call callback"""
        try:
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = Path(f.name)

            sf.write(temp_path, chunk, self.sample_rate)

            # Call the processing callback
            if self.callback:
                self.callback(temp_path)

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Chunk processing error: {e}")

    def start_stream(self, callback: Callable[[Path], None]):
        """Start continuous audio capture"""
        self.callback = callback
        self.running = True

        # Find the device index
        devices = sd.query_devices()
        device_idx = None
        for i, dev in enumerate(devices):
            if self.device.lower() in dev['name'].lower():
                device_idx = i
                break

        if device_idx is None:
            raise RuntimeError(f"Audio device '{self.device}' not found. Available: {[d['name'] for d in devices]}")

        print(f"Starting audio capture from: {devices[device_idx]['name']}")

        with sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        ):
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.running = False

    def stop(self):
        """Stop audio capture"""
        self.running = False


def list_audio_devices():
    """List available audio devices"""
    print("Available audio devices:")
    for i, dev in enumerate(sd.query_devices()):
        print(f"  [{i}] {dev['name']} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")
