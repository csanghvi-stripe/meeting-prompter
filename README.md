# Meeting Intelligence CLI

A real-time meeting assistant that listens to audio, transcribes speech, and answers questions using your documentation. Everything runs locally on your Mac using [LFM2-Audio](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF) for transcription and extraction-based RAG for answers.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd meeting-intelligence-cli
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download models (see Models section below)

# Run with your microphone
python coach.py --mic
```

Speak a question like *"How does Liquid AI handle edge deployment?"* and watch it generate an answer from your docs.

## Architecture

The system uses extraction-based RAG with ColBERT semantic search:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Audio Pipeline                               │
│                                                                      │
│  Microphone → Audio Capture → LFM2-Audio-1.5B → Question Detector   │
│                 (4s chunks)    (transcription)    (pattern matching) │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline (Extraction-Based)                 │
│                                                                      │
│  Question → ColBERT Retriever → Answer Extractor → Terminal UI      │
│              (semantic search)   (sentence scoring)                  │
│              (top-3 chunks)      (no LLM generation)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Extraction Over Generation?

Small LLMs (1-3B parameters) are unreliable at following "only use this context" instructions. They often hallucinate or ignore the provided context. Our solution:

| Approach | Problem | Our Solution |
|----------|---------|--------------|
| LLM Generation | Ignores context, hallucinates | **Sentence Extraction** - pull directly from docs |
| Single embedding | Loses semantic nuance | **ColBERT MaxSim** - token-level matching |
| Random chunk splits | Loses document structure | **Section-aware chunking** - split on ## headers |
| Single result | May miss relevant context | **Multi-chunk retrieval** - combine top-3 results |

### Components

| Component | Model/Tool | What it does |
|-----------|------------|--------------|
| **Audio Capture** | sounddevice | Streams 4-second chunks from mic or BlackHole |
| **Transcription** | LFM2-Audio-1.5B | Speech-to-text via llama.cpp subprocess (~300ms) |
| **Hallucination Filter** | Pattern matching | Filters common LFM2-Audio hallucination patterns |
| **Question Detection** | Regex + scoring | Identifies questions by structure, keywords, and confidence |
| **Question Buffer** | Time-based | Buffers speech chunks, flushes on 1.5s pause |
| **RAG Search** | LFM2-ColBERT-350M | Section-aware semantic search, returns top-3 chunks (~100ms) |
| **Answer Extraction** | Sentence scoring | Extracts relevant sentences from context (no LLM) |
| **Vibe Check** | Keyword scoring | Detects emotional tone (Excited/Frustrated/etc.) |

### Why Two Models Instead of Three?

- **LFM2-Audio-1.5B** is a multimodal model that directly processes audio waveforms. It runs as a subprocess via the `llama-lfm2-audio` binary.

- **LFM2-ColBERT-350M** provides semantic document retrieval. Unlike keyword search, it understands that "neural network alternatives" should match content about "LFM architecture" even without exact word overlap.

- **Answer Extraction** replaces LLM generation. Instead of asking a small LLM to "answer using only this context" (which they're bad at), we score and extract sentences directly from the retrieved context. This structurally prevents hallucination.

This separation keeps each component focused on what it does best - and avoids the hallucination problems of small generative models.

---

## RAG: Why ColBERT?

### The Problem with Keyword Search

Traditional keyword matching (Jaccard similarity) fails for semantic queries:

| Query | Keyword Result | ColBERT Result |
|-------|----------------|----------------|
| "What is Liquid AI?" | Found | Found (77%) |
| "neural network alternatives" | **MISS** (no keyword overlap) | **Found (74%)** |
| "compete with OpenAI" | **MISS** (no keyword overlap) | **Found (76%)** |

### How ColBERT Works

ColBERT uses **late interaction** - instead of compressing documents into single vectors, it creates **one vector per token**:

```
Traditional Embeddings:
  Document → [single 768-dim vector]
  Query    → [single 768-dim vector]
  Score    = cosine_similarity

ColBERT (Late Interaction):
  Document → [[vec1], [vec2], [vec3], ...] (128-dim per token)
  Query    → [[vec1], [vec2], [vec3], ...]
  Score    = MaxSim (find best match for each query token)
```

**MaxSim** finds semantic connections at the token level. "Neural" matches "model", "alternatives" matches "architecture" - even without exact keyword overlap.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   RAGEngine                         │
│                                                     │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │    ColBERT      │    │    Jaccard      │        │
│  │   (primary)     │    │   (fallback)    │        │
│  └────────┬────────┘    └─────────────────┘        │
│           │                                         │
│           ▼                                         │
│  ┌─────────────────┐                               │
│  │ LFM2-ColBERT    │  • 353M parameters            │
│  │    -350M        │  • 128-dim per token          │
│  └────────┬────────┘  • MaxSim scoring             │
│           │                                         │
│           ▼                                         │
│  ┌─────────────────┐                               │
│  │  PLAID Index    │  • Persisted to disk          │
│  │                 │  • ~6s load time              │
│  └─────────────────┘  • ~20MB for 76 chunks        │
└─────────────────────────────────────────────────────┘
```

### Section-Aware Chunking

Documents are chunked with awareness of document structure:

1. **Markdown files** are split by `##` and `###` headers first
2. Each section is then chunked into 400-token segments (ColBERT max is 512)
3. Section headers are prepended to each chunk for context
4. 50-token overlap preserves context at boundaries

This ensures that when you ask "What is LEAP?", you get chunks from the LEAP section, not random nearby text.

### Multi-Chunk Retrieval

Instead of returning a single best match, we:
1. Retrieve top-3 results from ColBERT
2. Filter to chunks within 80% of top confidence
3. Combine into richer context for answer extraction

### Fallback

If ColBERT fails to load (missing dependencies, memory pressure), the system falls back to keyword-based Jaccard similarity automatically.

## Usage

### Microphone Mode (easiest way to test)

```bash
python coach.py --mic
```

Uses your Mac's built-in microphone. Just speak a question and the CLI will:
1. Transcribe your speech
2. Search your docs for relevant context
3. Generate and display an answer with source citation

### Test with Audio File

```bash
python coach.py --test audio.wav
```

Or generate test audio with macOS text-to-speech:

```bash
say -o test.aiff "What makes Liquid AI different from other AI models?"
afconvert test.aiff -o test.wav -d LEI16@16000 -f WAVE
python coach.py --test test.wav
```

### Live Meeting Mode (Zoom/Meet/Teams)

```bash
python coach.py
```

Captures from BlackHole virtual audio device. Requires setup (see below).

### List Audio Devices

```bash
python coach.py --list-devices
```

## Models

### Local Models (download to `models/`)

| Model | Size | Purpose |
|-------|------|---------|
| [LFM2-Audio-1.5B-Q8_0.gguf](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF) | 1.2 GB | Speech-to-text |
| mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf | 317 MB | Audio encoder |
| audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf | 358 MB | Audio decoder |

Also download the llama.cpp runner from [LFM2-Audio runners](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF/tree/main/runners) → `runners/macos-arm64/`

### HuggingFace Models (auto-downloaded)

| Model | Size | Purpose |
|-------|------|---------|
| [LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 1.4 GB | Semantic document retrieval |

ColBERT is downloaded automatically on first run via the `pylate` library. The PLAID index is built once and cached in `data/colbert_index/`.

**Note**: LFM2-1.2B text model is no longer required. Answers are extracted directly from retrieved context without LLM generation.

## Live Meeting Setup (BlackHole)

To capture audio from video calls:

1. Install BlackHole: `brew install blackhole-2ch`

2. Create Multi-Output Device:
   - Open Audio MIDI Setup (Applications > Utilities)
   - Click + > Create Multi-Output Device
   - Check both BlackHole 2ch and your speakers

3. Set your meeting app's speaker to "Multi-Output Device"

4. Run: `python coach.py`

## Adding Your Own Docs

Drop PDF or Markdown files in `docs/`. The CLI loads all documents at startup.

```bash
cp your-product-docs.pdf docs/
python coach.py --mic
```

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MEETING INTELLIGENCE AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

? QUESTION:
   How does Liquid AI handle edge deployment?

ANSWER:
   Liquid AI models are optimized for edge devices with 2x faster
   inference and 90% less memory usage compared to traditional
   transformer architectures.

Source: LiquidAI_Technical_Whitepaper.pdf
Confidence: [########............] 40%
Vibe: Engaged

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Listening for next question... (Ctrl+C to stop)
```

## Project Structure

```
meeting-intelligence-cli/
├── coach.py                  # Main entry point
├── lib/
│   ├── lfm2_wrapper.py       # LFM2-Audio subprocess wrapper
│   ├── audio_capture.py      # Streaming audio capture (with level detection)
│   ├── question_detector.py  # Question pattern matching + scoring
│   ├── question_buffer.py    # Time-based question buffering
│   ├── answer_extractor.py   # Sentence extraction (no LLM)
│   ├── answer_generator.py   # LLM generation (legacy, unused)
│   ├── rag_engine.py         # Document search (ColBERT + fallback)
│   ├── vibe_check.py         # Tone detection
│   └── colbert/              # Semantic retrieval module
│       ├── __init__.py
│       ├── retriever.py      # ColBERT model + PLAID index
│       ├── chunker.py        # Section-aware document chunking
│       ├── index_manager.py  # Index persistence/cache
│       └── normalizer.py     # MaxSim score normalization
├── models/                   # GGUF model files
├── runners/                  # llama.cpp binaries
├── docs/                     # Your documentation (PDF + Markdown)
├── data/                     # Generated index files (gitignored)
│   └── colbert_index/        # PLAID index cache
└── output/                   # Transcription logs
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 8GB+ RAM (ColBERT ~1.5GB + LFM2-Audio ~2GB = ~3.5GB total)
- Python 3.10+

## Troubleshooting

**No audio detected**: Check that your mic is working (`--list-devices`) or that BlackHole is configured correctly for meeting mode.

**Model not found**: Ensure all GGUF files are in `models/` and the runner binary is in `runners/macos-arm64/`.

**Garbled transcriptions**: LFM2-Audio sometimes hallucinates on background noise. The system filters common patterns, but noisy environments may cause issues.

**Wrong answers returned**: If answers seem unrelated to questions, delete `data/colbert_index/` and restart to rebuild the index with section-aware chunking.

**ColBERT not loading**: If you see "using Jaccard fallback", check that `pylate` is installed (`pip install pylate`). On 8GB Macs, set `RAG_USE_FALLBACK=1` to use lighter keyword search.

**Slow first startup**: First run downloads the ColBERT model (~1.4GB) and builds the PLAID index (~30-60s). Subsequent runs load from cache (~6s).

## License

MIT
