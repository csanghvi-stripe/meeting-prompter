# Meeting Intelligence CLI - Architecture & Implementation

## Overview

This document describes the architecture decisions and implementation details for the Meeting Intelligence CLI, a real-time audio Q&A system using Liquid AI models.

## Design Philosophy

### Core Principle: Hybrid RAG (Extraction + Generation)

Small LLMs (1-3B parameters) are unreliable at following "only use this context" instructions when given raw retrieved chunks. **Our solution**: Extract first, then generate.

```
Traditional RAG (Problematic):
  Question → Retriever → Raw Context → LLM → Answer
                                        ↑
                                  (hallucination risk)

Hybrid RAG (Our Approach):
  Question → Retriever → Sentence Extractor → LFM2-1.2B-RAG → Fluent Answer
               ↓              ↓                    ↓
          (ColBERT)     (grounding)         (RAG-specialized)
```

**Why this works**:
- **Extraction stage** filters irrelevant text and provides grounded context
- **LFM2-1.2B-RAG** is specifically trained on 1M+ RAG samples to follow context
- **Fallback** to extraction-only if generation fails

---

## Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AUDIO PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Microphone ──► Audio Capture ──► LFM2-Audio ──► Hallucination      │
│                 (4s chunks)       (transcribe)    Filter            │
│                                                      │               │
│                                                      ▼               │
│                                               Question Buffer        │
│                                               (time-based flush)     │
│                                                      │               │
│                                                      ▼               │
│                                               Question Detector      │
│                                               (pattern + scoring)    │
│                                                                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      HYBRID RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Question ──► ColBERT ──► Sentence Extraction ──► LFM2-1.2B-RAG     │
│               (top-3)     (grounded context)      (fluent answer)   │
│                                                                      │
│  Stage 1: RETRIEVAL        Stage 2: GROUNDING     Stage 3: GENERATION│
│  - Semantic search         - Score sentences      - ChatML format    │
│  - Multi-chunk combine     - Extract top-3        - RAG-optimized    │
│  - Section-aware           - Validate relevance   - Synthesize       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Audio Capture (`lib/audio_capture.py`)

- Captures 4-second audio chunks from microphone
- Checks audio level (RMS + peak) to filter silence
- Saves as 16kHz WAV for LFM2-Audio

#### 2. LFM2-Audio Transcription (`lib/lfm2_wrapper.py`)

- Runs as subprocess via `llama-lfm2-audio` binary
- Processes audio → text in ~300ms
- Known issue: hallucinates on background noise

#### 3. Hallucination Filter (`coach.py:_is_hallucination`)

LFM2-Audio produces predictable hallucination patterns on noise:
- "I don't know what's going to..."
- "She chose the one that was..."
- "Can you explain to me?"

We filter these with pattern matching before question detection.

#### 4. Question Buffer (`lib/question_buffer.py`)

Problem: Speech spans multiple audio chunks. Solution: Buffer chunks and flush on pause.

```python
class QuestionBuffer:
    pause_threshold: float = 1.5   # Seconds of silence to flush
    max_buffer_time: float = 8.0   # Maximum buffer before forced flush
```

#### 5. Question Detector (`lib/question_detector.py`)

Scores text for "question-ness" using:
- Interrogative patterns (what, how, why, etc.)
- Question mark presence
- Word count and structure

Returns confidence score 0-1.

#### 6. ColBERT Retriever (`lib/colbert/`)

**Why ColBERT over dense embeddings?**

ColBERT uses "late interaction" - one vector per token instead of one per document:

```
Dense Embedding:
  "What is LEAP?" → [single 768-dim vector]
  Problem: Loses token-level semantics

ColBERT (Late Interaction):
  "What is LEAP?" → [[vec_what], [vec_is], [vec_leap], [vec_?]]
  MaxSim finds best match for EACH query token
```

**Section-Aware Chunking** (`lib/colbert/chunker.py`):

```python
def chunk_markdown(self, md_path: Path) -> List[Chunk]:
    # 1. Split by ## and ### headers
    sections = self._split_by_headers(full_text)

    # 2. Prepend header to each section's chunks
    for section_header, section_text in sections:
        section_with_header = f"{section_header}\n\n{section_text}"
        chunks = self.chunk_text(section_with_header, md_path.name)
```

This ensures "What is LEAP?" retrieves from the LEAP section, not nearby text.

**Multi-Chunk Retrieval** (`lib/rag_engine.py`):

```python
def _query_colbert(self, text: str) -> Tuple[str, float, str]:
    results = self._colbert.query(text, k=3)  # Get top 3

    # Combine chunks with similar confidence
    for chunk_text, confidence, source in results:
        if confidence >= top_confidence * 0.8:
            combined_chunks.append(chunk_text)

    return "\n\n---\n\n".join(combined_chunks), top_confidence, top_source
```

#### 7. Answer Extractor (`lib/answer_extractor.py`)

Scores sentences for relevance to ensure grounded context for the LLM:

```python
def score_sentence(sentence: str, question: str, position: int, total: int) -> float:
    """Score sentence relevance to question."""
    score = 0.0
    # 1. Keyword overlap (primary signal)
    # 2. Definition patterns ("X is...", "X stands for...")
    # 3. Position bonus (earlier = more likely definition)
    return score
```

#### 8. RAG Answer Generator (`lib/rag_generator.py`)

Uses LFM2-1.2B-RAG with ChatML format:

```python
RAG_PROMPT_TEMPLATE = """<|im_start|>user
Use the following context to answer the question. Be concise and direct.
Only use information from the provided context.

CONTEXT:
{context}

QUESTION: {question}<|im_end|>
<|im_start|>assistant
"""
```

Key settings:
- `temperature=0` for factual responses
- `max_tokens=200` for concise answers
- Model reset before each generation to prevent KV cache issues

#### 9. Hybrid Answerer (`lib/hybrid_answerer.py`)

Orchestrates the two-stage pipeline:

```python
def answer(self, question: str, rag_context: str) -> Tuple[str, float, str]:
    # Stage 1: Extract grounded context
    extracted, confidence = extract_answer(rag_context, question)

    # Stage 2: Generate fluent answer
    if self.use_generation:
        answer = self.generator.generate(question, extracted)
        return (answer, confidence, "hybrid")

    # Fallback: Return extracted bullets
    return (format_as_bullets(extracted), confidence, "extraction")
```

---

## Key Improvements Implemented

| Issue | Before | After |
|-------|--------|-------|
| Choppy answers | Bullet points only | Fluent LLM-generated answers |
| LLM ignores context | Generated irrelevant answers | Extraction-grounded generation |
| Wrong section retrieved | Random chunks near answer | Section-aware chunking with headers |
| Single result misses info | One best match | Top-3 combined for richer context |
| Audio hallucinations | Processed as questions | Pattern-filtered before detection |
| Chunk boundaries | Random 400-char splits | Split on markdown headers first |
| Thread safety | Race conditions in buffer | Lock-protected QuestionBuffer |
| Pause detection | Counted chunks (broken) | Time-based (1.5s threshold) |

---

## Configuration

### Audio Capture
```python
CHUNK_DURATION = 4.0      # Seconds per audio chunk
SAMPLE_RATE = 16000       # Hz (required by LFM2-Audio)
RMS_THRESHOLD = 0.005     # Minimum audio level
PEAK_THRESHOLD = 0.02     # Minimum peak level
```

### Question Buffer
```python
PAUSE_THRESHOLD = 1.5     # Seconds of silence to flush
MAX_BUFFER_TIME = 8.0     # Maximum buffer duration
MIN_WORDS = 4             # Minimum words for valid question
```

### ColBERT Chunking
```python
TARGET_TOKENS = 400       # Target tokens per chunk (max 512)
OVERLAP_TOKENS = 50       # Token overlap between chunks
MIN_TOKENS = 30           # Minimum tokens to keep chunk
```

### Answer Extraction
```python
MAX_SENTENCES = 3         # Maximum sentences to extract
MIN_CONFIDENCE = 0.25     # Minimum score to proceed to generation
```

### RAG Generation
```python
MODEL = "LFM2-1.2B-RAG-Q4_K_M.gguf"  # Q4_K_M quantization (700MB)
N_CTX = 2048              # Context window size
MAX_TOKENS = 200          # Maximum tokens in response
TEMPERATURE = 0           # Greedy decoding for factual answers
```

---

## File Overview

| File | Purpose |
|------|---------|
| `coach.py` | Main entry point, orchestrates pipeline |
| `lib/audio_capture.py` | Microphone streaming, level detection |
| `lib/lfm2_wrapper.py` | LFM2-Audio subprocess management |
| `lib/question_buffer.py` | Time-based speech buffering |
| `lib/question_detector.py` | Question pattern matching |
| `lib/answer_extractor.py` | Sentence extraction for grounding |
| `lib/rag_generator.py` | LFM2-1.2B-RAG answer generation |
| `lib/hybrid_answerer.py` | Two-stage pipeline (extraction → generation) |
| `lib/rag_engine.py` | RAG orchestration (ColBERT + fallback) |
| `lib/colbert/retriever.py` | ColBERT model + PLAID index |
| `lib/colbert/chunker.py` | Section-aware document chunking |
| `lib/colbert/index_manager.py` | Index persistence/caching |
| `lib/colbert/normalizer.py` | MaxSim score normalization |
| `lib/vibe_check.py` | Emotional tone detection |

---

## Re-indexing Documents

When documents change or chunking strategy is updated:

```bash
# Delete existing index
rm -rf data/colbert_index/

# Restart - index rebuilds automatically
python coach.py --mic
```

---

## Future Improvements

1. **Streaming transcription**: Process audio in real-time instead of 4s chunks
2. **Multi-document citations**: Show which document each sentence came from
3. **Confidence calibration**: Better threshold tuning for extraction/generation confidence
4. **Query expansion**: Use question keywords to improve retrieval
5. **Caching**: Cache frequent question-answer pairs
6. **Answer validation**: Cross-check generated answer against extracted context
