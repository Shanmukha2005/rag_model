from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FrameRecord:
    """Stores metadata and model analysis for a single extracted frame."""

    frame_id: int
    timestamp_sec: float
    image_path: str
    analysis: Dict[str, Any]

    @property
    def merged_text(self) -> str:
        """Return a retrieval-friendly text representation of the frame analysis."""
        parts: List[str] = [
            f"Timestamp: {self.timestamp_sec:.2f}s",
            f"Scene: {self.analysis.get('scene_description', 'N/A')}",
            f"Objects: {', '.join(self.analysis.get('objects', [])) or 'N/A'}",
            f"Actions: {', '.join(self.analysis.get('actions', [])) or 'N/A'}",
            f"Expressions: {', '.join(self.analysis.get('facial_expressions', [])) or 'N/A'}",
            f"OCR: {self.analysis.get('ocr_text', 'N/A')}",
            f"Environment: {self.analysis.get('environment_context', 'N/A')}",
        ]
        return "\n".join(parts)


@dataclass
class SemanticChunk:
    """Represents a grouped set of consecutive related frames."""

    chunk_id: int
    start_sec: float
    end_sec: float
    frame_ids: List[int]
    text: str
    summary: str
    keywords: List[str]
    enhanced_text: str
    image_paths: List[str] = field(default_factory=list)

    @property
    def display_window(self) -> str:
        return f"{self.start_sec:.2f}s → {self.end_sec:.2f}s"


@dataclass
class RetrievalResult:
    chunk: SemanticChunk
    score: float
