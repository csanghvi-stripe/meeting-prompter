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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 MEETING INTELLIGENCE AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION:
   How does Liquid AI handle edge deployment?

💡 ANSWER:
   Liquid AI models are optimized for edge devices with 2x faster
   inference and 90% less memory usage compared to traditional
   transformer architectures...

📄 Source: LiquidAI_Technical_Whitepaper.pdf
📊 Confidence: [████████░░░░░░░░░░░░] 40%
🎭 Vibe: 👀 Engaged

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎧 Listening for next question... (Ctrl+C to stop)
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

### Testing the Agent

There are two ways to test the agent before using it in a live meeting:

#### Option 1: Create Test Audio with macOS `say` Command

Generate a test audio file from text, then run the agent:

```bash
# Step 1: Create AIFF file with text-to-speech
say -o test.aiff "How does Liquid AI compare to OpenAI?"

# Step 2: Convert to WAV format (16kHz, 16-bit - required by LFM2-Audio)
afconvert test.aiff -o test.wav -d LEI16@16000 -f WAVE

# Step 3: Run the agent
./venv/bin/python3 coach.py --test test.wav
```

Or as a one-liner:
```bash
say -o q.aiff "What makes your product different?" && afconvert q.aiff -o q.wav -d LEI16@16000 -f WAVE && ./venv/bin/python3 coach.py --test q.wav
```

#### Option 2: Use an Existing Audio File

If you have a WAV file from a recorded meeting or call:

```bash
./venv/bin/python3 coach.py --test your_audio.wav
```

**Note:** Audio files must be WAV format. To convert other formats:
```bash
# Convert MP3 to WAV (requires ffmpeg)
ffmpeg -i audio.mp3 -ar 16000 -ac 1 audio.wav

# Convert M4A to WAV (using macOS afconvert)
afconvert audio.m4a -o audio.wav -d LEI16@16000 -f WAVE
```

### List Audio Devices

```bash
./venv/bin/python3 coach.py --list-devices
```

### Live Meeting Mode (Zoom, Meet, Teams, etc.)

To capture audio from video calls, you need to route meeting audio through BlackHole:

#### Step 1: Install BlackHole (one-time setup)

```bash
brew install blackhole-2ch
```

#### Step 2: Create Multi-Output Device (one-time setup)

This lets you hear meeting audio AND send it to the agent:

1. Open **Audio MIDI Setup** (Applications → Utilities → Audio MIDI Setup)
2. Click the **+** button at bottom-left → **Create Multi-Output Device**
3. Check both:
   - ✅ **BlackHole 2ch**
   - ✅ **MacBook Pro Speakers** (or your headphones)
4. Right-click the Multi-Output Device → **Use This Device For Sound Output** (optional)

#### Step 3: Configure Zoom/Meet Audio

**Option A: Per-app (recommended)**
- In Zoom: Click `^` next to mic → **Speaker** → Select **"Multi-Output Device"**
- In Google Meet: Settings → Audio → Speaker → **"Multi-Output Device"**

**Option B: System-wide**
- System Settings → Sound → Output → Select **"Multi-Output Device"**

#### Step 4: Start the Agent

```bash
source venv/bin/activate
python coach.py
```

You should see:
```
[STATUS] Listening to BlackHole 2ch...
[STATUS] Output file: /path/to/output/live_analytics.txt
```

#### Step 5: Join Your Meeting

Join your Zoom/Meet/Teams call. As people speak, you'll see:
- Real-time transcription
- Vibe indicators (Excited/Frustrated/Uncertain/Confident/Engaged)
- Suggested answers when questions are detected

Press `Ctrl+C` to stop and see the meeting summary.

#### Troubleshooting: No Audio Detected

If you're not seeing transcriptions, verify audio is flowing:

```bash
# Test if BlackHole is receiving audio
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds... (play audio in your meeting)')
rec = sd.rec(int(3*16000), samplerate=16000, channels=1, device='BlackHole 2ch')
sd.wait()
level = np.max(np.abs(rec))
print(f'Audio level: {level:.4f}')
print('✅ Audio detected!' if level > 0.001 else '❌ No audio - check Multi-Output Device setup')
"
```

Common fixes:
- Ensure Zoom/Meet speaker is set to **"Multi-Output Device"**
- Make sure BlackHole 2ch is checked in the Multi-Output Device settings
- Restart the meeting app after changing audio settings

## Architecture

### Processing Pipeline

```
Audio Chunk (2s)
      ↓
┌─────────────────┐
│ Audio Quality   │ → Skip if too quiet (prevents hallucinations)
│ Check           │
└─────────────────┘
      ↓
┌─────────────────┐
│ LFM2-Audio      │ → Transcribe speech to text (~200ms)
└─────────────────┘
      ↓
┌─────────────────┐
│ Question Buffer │ → Accumulate until complete thought
└─────────────────┘
      ↓
┌─────────────────┐
│ Normalize       │ → Fix ASR stutters, strip filler
└─────────────────┘
      ↓
┌─────────────────┐
│ RAG Query       │ → Search ALL docs, return best match + source
└─────────────────┘
      ↓
┌─────────────────┐
│ Confidence Gate │ → Skip if <5% match (irrelevant question)
└─────────────────┘
      ↓
┌─────────────────┐
│ Answer Generate │ → LFM2-1.2B with RAG context
└─────────────────┘
      ↓
┌─────────────────┐
│ Display         │ → Question, Answer, Source, Confidence
└─────────────────┘
```

### Key Features

1. **Multi-Document RAG**
   - Loads ALL PDFs from `docs/` directory
   - Tracks source file for each chunk
   - Cites source document with every answer

2. **Audio Quality Check**
   - Skips quiet/silent audio chunks
   - Prevents hallucinations from background noise

3. **Confidence-Based Filtering**
   - Questions that don't match documents are skipped
   - Only generates answers when confidence ≥5%

4. **Source Citations**
   - Every answer shows which document it came from
   - Builds trust and enables verification

## Project Structure

```
meeting-prompter/
├── coach.py                    # Main orchestrator
├── lib/
│   ├── lfm2_wrapper.py        # LFM2-Audio subprocess interface
│   ├── audio_capture.py       # BlackHole streaming + chunking
│   ├── question_detector.py   # Question detection + sentence merging
│   ├── answer_generator.py    # LFM2 text model for Q&A
│   ├── vibe_check.py          # Emotional category detection
│   ├── rag_engine.py          # Lightweight BM25-style RAG
│   └── dashboard.py           # Terminal display
├── models/                     # GGUF model files
├── runners/                    # llama.cpp binaries
├── docs/                       # Your product documentation (PDFs)
└── output/                     # Live transcription logs
```

## Customization

### Add Your Own Documentation

Place PDF files in the `docs/` directory. The RAG engine will automatically:
- Load **all PDFs** in the directory
- Chunk and index each document
- Track which document each chunk came from
- Cite sources in answers

```bash
# Example: Add multiple documents
cp product_guide.pdf docs/
cp technical_specs.pdf docs/
cp faq.pdf docs/

# Restart the agent to reload
python coach.py --mic
```

The agent works with **any domain** - just provide relevant documents.

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
