"""LFM2-Audio Wrapper - Subprocess interface to llama.cpp"""
import subprocess
from pathlib import Path
from typing import Optional


class LFM2Wrapper:
    """Wrapper for llama-lfm2-audio binary"""

    def __init__(self, model_dir: Path, runner_dir: Path):
        self.model = model_dir / "LFM2-Audio-1.5B-Q8_0.gguf"
        self.mmproj = model_dir / "mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf"
        self.decoder = model_dir / "audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf"
        self.runner = runner_dir / "lfm2-audio-macos-arm64" / "llama-lfm2-audio"

        # Validate files exist
        for f in [self.model, self.mmproj, self.decoder, self.runner]:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text using ASR"""
        cmd = [
            str(self.runner),
            "-m", str(self.model),
            "--mmproj", str(self.mmproj),
            "-mv", str(self.decoder),
            "--audio", str(audio_path),
            "-sys", "Perform ASR.",  # Required exact prompt for ASR mode
            "--temp", "0",  # Deterministic output
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                cwd=self.runner.parent,  # Run from runner directory for dylib loading
            )
            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            return "[Transcription timeout]"
        except Exception as e:
            return f"[Error: {e}]"

    def _parse_output(self, raw: bytes) -> str:
        """Filter llama.cpp logging, extract clean transcription"""
        lines = raw.decode('utf-8', errors='replace').split('\n')

        # Skip llama.cpp verbose logging
        skip_keywords = [
            'loading', 'loaded', 'gguf', 'tensors', 'model', 'backend',
            'metal', 'gpu', 'cpu', 'simd', 'memory', 'init', 'build',
            'llama_', 'ggml_', 'load_', 'mtmd', 'sampler', 'token',
            'system_info', 'n_ctx', 'n_batch', 'flash_attn',
            'encoding audio', 'audio slice', 'audio decoded', 'decoding audio',
            'clip_', 'alloc_', 'print_info', 'common_init', 'main:',
        ]

        clean_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            if any(kw in line_lower for kw in skip_keywords):
                continue
            # Skip lines that look like logging (contain timestamps, brackets, etc.)
            if line.startswith('[') or line.startswith('llama') or line.startswith('---'):
                continue
            # Skip timing info like "47 ms"
            if line_lower.endswith(' ms') or line_lower.endswith(' ms)'):
                continue
            clean_lines.append(line.strip())

        return ' '.join(clean_lines).strip()
