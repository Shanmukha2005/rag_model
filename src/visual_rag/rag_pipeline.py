from __future__ import annotations

from dataclasses import dataclass
from typing import List

import google.generativeai as genai
from openai import OpenAI

from .chunker import SemanticChunker
from .config import Settings
from .embeddings import ClipEmbedder
from .frame_extractor import extract_frames
from .models import FrameRecord, RetrievalResult, SemanticChunk
from .qwen_analyzer import QwenVisionAnalyzer
from .vector_store import FaissStore


@dataclass
class PipelineArtifacts:
    frames: List[FrameRecord]
    chunks: List[SemanticChunk]


class VisualRAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.qwen = QwenVisionAnalyzer(
            model_name=settings.qwen_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.llm_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
        self.chunker = SemanticChunker(self.llm_client, settings.qwen_model_name)
        self.embedder = ClipEmbedder(settings.clip_model_name, device=settings.device)
        self.vector_store: FaissStore | None = None
        self.artifacts: PipelineArtifacts | None = None

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for grounded answer generation.")
        genai.configure(api_key=settings.gemini_api_key)
        self.gemini = genai.GenerativeModel(settings.gemini_model_name)

    def index_video(self, video_path: str) -> PipelineArtifacts:
        frame_dir = self.settings.workspace_dir / "frames"
        frame_tuples = extract_frames(
            video_path=video_path,
            output_dir=str(frame_dir),
            interval_sec=self.settings.frame_interval_sec,
            max_frames=self.settings.max_frames,
        )

        frames = self.qwen.batch_analyze(frame_tuples)
        chunks = self.chunker.chunk_frames(frames)
        if not chunks:
            raise ValueError("No semantic chunks generated from video frames.")

        embeddings = self.embedder.embed_chunks(chunks, batch_size=self.settings.batch_size)
        self.vector_store = FaissStore(dim=embeddings.shape[1])
        self.vector_store.add(chunks, embeddings)

        self.artifacts = PipelineArtifacts(frames=frames, chunks=chunks)
        return self.artifacts

    def retrieve(self, question: str) -> List[RetrievalResult]:
        if not self.vector_store:
            raise RuntimeError("Video not indexed yet. Call index_video() first.")
        qvec = self.embedder.embed_query(question)
        return self.vector_store.search(qvec, top_k=self.settings.top_k)

    def answer_question(self, question: str) -> dict:
        results = self.retrieve(question)
        valid = [r for r in results if r.score >= self.settings.min_retrieval_score]

        if not valid:
            return {
                "answer": "The information is not available in the video.",
                "context": [],
            }

        context = []
        context_text_blocks = []
        for item in valid:
            chunk = item.chunk
            context.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "time_window": chunk.display_window,
                    "score": round(item.score, 4),
                    "summary": chunk.summary,
                    "keywords": chunk.keywords,
                }
            )
            context_text_blocks.append(
                f"[Chunk {chunk.chunk_id} | {chunk.display_window}]\n"
                f"Summary: {chunk.summary}\n"
                f"Keywords: {', '.join(chunk.keywords)}\n"
                f"Details: {chunk.enhanced_text}"
            )

        grounding_prompt = (
            "You are a visual-only RAG answerer. Use ONLY the retrieved context below. "
            "If the answer is missing, reply exactly: 'The information is not available in the video.' "
            "Do not use outside knowledge. Always cite chunk IDs and timestamps."
        )
        gemini_output = self.gemini.generate_content(
            grounding_prompt
            + "\n\n"
            + f"Question: {question}\n\nRetrieved context:\n"
            + "\n\n".join(context_text_blocks)
        )
        answer = (gemini_output.text or "").strip() or "The information is not available in the video."

        return {
            "answer": answer,
            "context": context,
        }
