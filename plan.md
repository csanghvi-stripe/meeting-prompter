# Meeting Intelligence CLI - Architecture & Implementation

## Overview

This document describes the architecture decisions and implementation details for the Meeting Intelligence CLI, a real-time audio Q&A system using Liquid AI models.

## Design Philosophy

### Core Principle: Extraction Over Generation

Small LLMs (1-3B parameters) are unreliable at following instructions like "only use this context". They frequently:
- Ignore provided context entirely
- Hallucinate information not in the source
- Mix context with training data

**Our solution**: Don't generate answers. Extract them.

```
Traditional RAG:
  Question → Retriever → Context → LLM → Generated Answer
                                    ↑
                              (hallucination risk)

Extraction-Based RAG:
  Question → Retriever → Context → Sentence Scorer → Extracted Answer
                                          ↑
                                    (no hallucination possible)
```

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
│                          RAG PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Question ──► ColBERT Retriever ──► Answer Extractor ──► Display    │
│               (semantic search)      (sentence scoring)              │
│               (top-3 chunks)         (no LLM needed)                 │
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

**Core insight**: We can extract the answer without an LLM.

```python
def score_sentence(sentence: str, question: str, position: int, total: int) -> float:
    """Score sentence relevance to question."""
    score = 0.0

    # 1. Keyword overlap (primary signal)
    overlap = len(question_words & sentence_words)
    score += (overlap / len(question_words)) * 0.5

    # 2. Definition patterns ("X is...", "X stands for...")
    if re.search(r'\bstands for\b', sentence):
        score += 0.2

    # 3. Position bonus (earlier = more likely definition)
    score += (1.0 - position/total) * 0.2

    return score
```

Process:
1. Split context into sentences
2. Score each sentence against question
3. Take top 3 by score
4. Re-sort by document position
5. Format as bullet points

**No LLM = No Hallucination**

---

## Key Improvements Implemented

| Issue | Before | After |
|-------|--------|-------|
| LLM ignores context | Generated irrelevant answers | Extracts from context directly |
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
MIN_CONFIDENCE = 0.2      # Minimum score to return answer
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
| `lib/answer_extractor.py` | Sentence extraction (no LLM) |
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
3. **Confidence calibration**: Better threshold tuning for extraction confidence
4. **Query expansion**: Use question keywords to improve retrieval
5. **Caching**: Cache frequent question-answer pairs
