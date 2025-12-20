# Meeting Prompter

**Real-Time Meeting Intelligence Agent** powered by [Liquid AI's LFM2-Audio](https://www.liquid.ai/) running 100% locally on Mac.

> Transform customer calls into coaching opportunities with zero-latency, privacy-first AI.

## What It Does

During a live customer meeting, the agent:

1. **Transcribes speech in real-time** (~200ms latency) using LFM2-Audio
2. **Detects customer questions** automatically from the conversation
3. **Generates suggested answers** by querying your product documentation
4. **Analyzes emotional tone** (Vibe Check) to help you read the room
5. **Provides confidence scores** showing how well answers match your docs

All processing happens **locally on your Mac** - no audio ever leaves your machine.

## Key Value Propositions

| Feature | Benefit |
|---------|---------|
| **Zero-Latency Multimodality** | Audio-to-intelligence in a single pass. No chaining Whisper→GPT. Get coaching tips while the customer is still speaking. |
| **Constant-Memory RAG** | Liquid Neural Networks maintain constant memory even during hour-long meetings. No slowdown as context grows. |
| **Privacy Moat** | 100% local processing. Extract emotional intelligence and technical accuracy without any data leaving your device. |

## Demo

```
============================================================
  Real-Time Meeting Intelligence Agent
  Powered by LFM2-Audio | 100% Local Processing
============================================================

[STATUS] LFM2-Audio ready
[STATUS] RAG engine ready (55 chunks loaded)
[STATUS] Q&A engine ready
[STATUS] Listening to BlackHole 2ch...

[CPU: 45% RAM: 62%] 🔥 Excited      Conf: 35% │ How does Liquid AI handle...

💡 SUGGESTED ANSWER:
   Liquid AI's architecture, including the Liquid Time-constant Model,
   is optimized for real-time audio processing with sub-100ms latency...
```

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **8GB+ RAM** (16GB recommended)
- **Python 3.10+**
- **BlackHole** virtual audio driver (for meeting audio capture)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/csanghvi-stripe/meeting-prompter.git
cd meeting-prompter
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Models

The models will be downloaded automatically on first run, or you can pre-download them:

```bash
# Create models directory
mkdir -p models

# Download LFM2-Audio models from HuggingFace
# See: https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF
```

Required model files in `models/`:
- `LFM2-Audio-1.5B-Q8_0.gguf` (1.2 GB) - Audio transcription
- `mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf` (317 MB) - Audio encoder
- `audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf` (358 MB) - Audio decoder
- `LFM2-1.2B-Q4_K_M.gguf` (730 MB) - Text generation for Q&A

### 4. Download llama.cpp Runner

```bash
# Create runners directory
mkdir -p runners/macos-arm64

# Download from HuggingFace
# See: https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF/tree/main/runners
```

### 5. Install BlackHole (for meeting audio capture)

```bash
brew install blackhole-2ch
```

Configure your Mac's audio:
1. Open **Audio MIDI Setup**
2. Create a **Multi-Output Device** combining your speakers + BlackHole
3. Set this as your default output
4. Meeting audio will now route to both your ears and the agent

## Usage

### Test with Audio File

```bash
./venv/bin/python3 coach.py --test audio.wav
```

### List Audio Devices

```bash
./venv/bin/python3 coach.py --list-devices
```

### Start Live Capture

```bash
# Default: capture from BlackHole
./venv/bin/python3 coach.py

# Or specify a device
./venv/bin/python3 coach.py --device "MacBook Pro Microphone"
```

### During a Meeting

1. Start a video call (Zoom, Meet, Teams, etc.)
2. Run the agent: `./venv/bin/python3 coach.py`
3. Watch for:
   - Real-time transcription
   - Vibe indicators (Excited/Frustrated/Uncertain/Confident/Engaged)
   - Suggested answers when questions are detected
4. Press `Ctrl+C` to stop and see meeting summary

## Project Structure

```
meeting-prompter/
├── coach.py                    # Main entry point
├── lib/
│   ├── lfm2_wrapper.py        # LFM2-Audio subprocess interface
│   ├── audio_capture.py       # BlackHole streaming + chunking
│   ├── question_detector.py   # Automatic question detection
│   ├── answer_generator.py    # LFM2 text model for Q&A
│   ├── vibe_check.py          # Emotional category detection
│   ├── rag_engine.py          # Lightweight RAG for docs
│   └── dashboard.py           # Terminal display
├── models/                     # GGUF model files
├── runners/                    # llama.cpp binaries
├── docs/                       # Your product documentation (PDFs)
└── output/                     # Live transcription logs
```

## Customization

### Add Your Own Documentation

Place PDF files in the `docs/` directory. The RAG engine will automatically index them for context retrieval.

### Tune Question Detection

Edit `lib/question_detector.py` to add industry-specific keywords:

```python
QUESTION_KEYWORDS = [
    'pricing', 'integration', 'security', 'compliance',
    # Add your product-specific terms
]
```

### Adjust Vibe Categories

Edit `lib/vibe_check.py` to customize emotional detection:

```python
EMOTION_KEYWORDS = {
    "Excited": ["amazing", "fantastic", "love", ...],
    # Add your own categories
}
```

## Performance

Tested on MacBook Pro M4 Pro (16GB):

| Operation | Latency |
|-----------|---------|
| Audio transcription | ~200ms per 2s chunk |
| Question detection | <10ms |
| RAG lookup | <50ms |
| Answer generation | ~500-1000ms |
| **Total response time** | **<1.5s** |

## Troubleshooting

### "Audio device not found"

```bash
./venv/bin/python3 coach.py --list-devices
```

Make sure BlackHole is installed and visible.

### "Model not found"

Ensure all GGUF files are in the `models/` directory and the llama.cpp runner is in `runners/macos-arm64/`.

### High CPU/Memory Usage

The agent uses ~4GB RAM when both models are loaded. Close other memory-intensive applications.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Liquid AI](https://www.liquid.ai/) for LFM2-Audio and LFM2 models
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for efficient local inference
- [BlackHole](https://existential.audio/blackhole/) for virtual audio routing
