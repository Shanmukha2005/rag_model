from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .rag_pipeline import VisualRAGPipeline

app = FastAPI(title="Visual-Only Video RAG")
pipeline: VisualRAGPipeline | None = None


class IndexRequest(BaseModel):
    video_path: str


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = VisualRAGPipeline(settings)


@app.post("/index")
def index_video(payload: IndexRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    try:
        artifacts = pipeline.index_video(payload.video_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "frames": len(artifacts.frames),
        "chunks": len(artifacts.chunks),
    }


@app.post("/ask")
def ask_question(payload: AskRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    try:
        return pipeline.answer_question(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
