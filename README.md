# Video Understanding and Question Answering using Visual-Only RAG (No Audio)

This project implements a **visual-only** Retrieval-Augmented Generation (RAG) pipeline for video QA using:
- **Qwen-VL** for per-frame visual understanding
- **CLIP** for multimodal embeddings (text + representative frame image)
- **FAISS (in-memory)** for temporary vector indexing
- **Gemini** for grounded answer generation from retrieved visual chunks

> No audio is processed anywhere in this pipeline.

---

## A) Step-by-step system architecture

1. **Video ingestion**
   - Input: local video path.
   - Validation checks verify file access and readable metadata.

2. **Frame extraction with timestamps**
   - Frames are sampled every `FRAME_INTERVAL_SEC` (default 1.5s).
   - Each saved frame keeps a timestamp from the original FPS.

3. **Per-frame visual analysis via Qwen-VL**
   - For each frame, Qwen returns structured JSON:
     - scene description
     - objects
     - actions/interactions
     - facial expressions/emotions
     - OCR text
     - environment context
   - Responses are normalized to avoid schema drift.

4. **Chronological merge and semantic chunking**
   - Frame analyses are kept in temporal order.
   - Consecutive frames are grouped using object/action/environment similarity (Jaccard).
   - Each chunk represents a coherent short scene.

5. **Chunk enrichment**
   - An LLM pass generates for each chunk:
     - concise summary
     - keywords
     - enhanced text with expanded abbreviations and clarified references

6. **Multimodal embedding generation (CLIP)**
   - Text embedding from chunk summary + keywords + enhanced text.
   - Image embedding from a representative chunk frame.
   - Final vector = normalized average(text_vector, image_vector).

7. **Temporary vector indexing**
   - Vectors are stored in a **FAISS in-memory index** (`IndexFlatIP`).
   - No persistent database is required.

8. **RAG query pipeline**
   - User asks a question.
   - Query is embedded and top-k chunks are retrieved.
   - Low-score chunks are filtered by threshold.
   - Retrieved context is sent to Gemini with strict grounding instructions.
   - If evidence is insufficient: return exactly
     - `The information is not available in the video.`

9. **Explainability output**
   - Returned context includes chunk IDs, timestamps, similarity scores, summaries, and keywords.

---

## B) Modular folder structure

```text
rag_model/
├── README.md
├── requirements.txt
├── .env.example
└── src/
    └── visual_rag/
        ├── __init__.py
        ├── api.py              # Optional FastAPI interface
        ├── chunker.py          # Scene-based semantic chunking
        ├── config.py           # Environment-driven settings
        ├── embeddings.py       # CLIP text/image embedding logic
        ├── frame_extractor.py  # Video -> timestamped frames
        ├── main.py             # CLI entrypoint
        ├── models.py           # Dataclasses for frames/chunks/results
        ├── qwen_analyzer.py    # Qwen-VL frame understanding
        ├── rag_pipeline.py     # End-to-end indexing + QA orchestration
        └── vector_store.py     # In-memory FAISS retrieval layer
```

---

## C) Complete Python implementation

All modules are included under `src/visual_rag/`.

### Run as CLI

```bash
python -m src.visual_rag.main index /path/to/video.mp4
python -m src.visual_rag.main ask /path/to/video.mp4 "What text appears on the whiteboard and when?"
```

### Run FastAPI demo interface (optional)

```bash
uvicorn src.visual_rag.api:app --host 0.0.0.0 --port 8000
```

Then:
- `POST /index` with JSON `{"video_path": "/path/to/video.mp4"}`
- `POST /ask` with JSON `{"question": "What is the person doing at the desk?"}`

---

## D) Required dependencies (`requirements.txt`)

See `requirements.txt`.

---

## E) Environment variable configuration (`.env`)

Copy `.env.example` to `.env` and fill API keys.

---

## F) Sample user query demonstration

Example:

1. Index video:
```bash
python -m src.visual_rag.main index ./samples/office_scene.mp4
```

2. Ask question:
```bash
python -m src.visual_rag.main ask ./samples/office_scene.mp4 "When does the presenter point to the chart, and what chart labels are visible?"
```

Expected behavior:
- Returns answer with chunk citations/timestamps when visual evidence exists.
- Returns `The information is not available in the video.` when evidence is absent.

---

## G) Error handling and edge cases

- Invalid/unreadable video path -> raises descriptive `ValueError`.
- Missing/invalid FPS metadata -> extraction error.
- No extracted frames (e.g., extreme interval, short clip) -> explicit failure.
- Qwen malformed JSON -> safe schema fallback with empty fields.
- Retrieval before indexing -> runtime guard.
- Empty or weak retrieval (below threshold) -> deterministic fallback message.
- Missing API keys -> initialization failure with clear instructions.

---

## H) Optimization techniques

- **Batching**: CLIP text/image encoding supports configurable `BATCH_SIZE`.
- **GPU usage**: CLIP automatically uses CUDA when available (`DEVICE=cuda`).
- **Caching opportunities**:
  - Save per-frame Qwen outputs keyed by image hash.
  - Cache chunk embeddings keyed by chunk text signature.
- **Cost control**:
  - Increase frame interval for long videos.
  - Set `MAX_FRAMES` for bounded indexing.
  - Use threshold-based retrieval filtering before Gemini call.

---

## I) Scalability suggestions

- Parallelize Qwen frame analysis with worker pools.
- Use shot-boundary detection before frame sampling for better chunk quality.
- Move from in-memory FAISS to sharded ANN service for multi-video corpora.
- Add metadata filters (video ID, scene tags, timestamp ranges) before vector search.
- Add async job queue for indexing large uploads.

---

## J) Optional interface

A minimal **FastAPI** app is included in `src/visual_rag/api.py`.

