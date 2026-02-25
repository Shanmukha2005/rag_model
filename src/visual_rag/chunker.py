from __future__ import annotations

from typing import List

from openai import OpenAI

from .models import FrameRecord, SemanticChunk


def _frame_signature(frame: FrameRecord) -> set[str]:
    items = set(frame.analysis.get("objects", []))
    items.update(frame.analysis.get("actions", []))
    items.update([frame.analysis.get("environment_context", "")])
    return {x.lower().strip() for x in items if x and x.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticChunker:
    def __init__(self, llm_client: OpenAI, model_name: str, similarity_threshold: float = 0.25):
        self.llm_client = llm_client
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

    def _summarize_chunk(self, text: str) -> tuple[str, List[str], str]:
        prompt = (
            "You receive chronological visual observations from video frames. "
            "Return JSON with keys: summary (max 2 sentences), keywords (array up to 8), "
            "enhanced_text (coherent paragraph with clarified references and expanded abbreviations). "
            "Ground only in provided text."
        )
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        )

        import json

        parsed = json.loads(response.choices[0].message.content or "{}")
        summary = str(parsed.get("summary", "")).strip()
        keywords = [str(k).strip() for k in parsed.get("keywords", []) if str(k).strip()]
        enhanced = str(parsed.get("enhanced_text", "")).strip()
        return summary, keywords, enhanced

    def chunk_frames(self, frames: List[FrameRecord]) -> List[SemanticChunk]:
        if not frames:
            return []

        groups: List[List[FrameRecord]] = [[frames[0]]]
        prev_sig = _frame_signature(frames[0])

        for fr in frames[1:]:
            cur_sig = _frame_signature(fr)
            if _jaccard(prev_sig, cur_sig) >= self.similarity_threshold:
                groups[-1].append(fr)
            else:
                groups.append([fr])
            prev_sig = cur_sig

        chunks: List[SemanticChunk] = []
        for idx, group in enumerate(groups):
            chunk_text = "\n\n".join(fr.merged_text for fr in group)
            summary, keywords, enhanced = self._summarize_chunk(chunk_text)
            chunks.append(
                SemanticChunk(
                    chunk_id=idx,
                    start_sec=group[0].timestamp_sec,
                    end_sec=group[-1].timestamp_sec,
                    frame_ids=[fr.frame_id for fr in group],
                    text=chunk_text,
                    summary=summary,
                    keywords=keywords,
                    enhanced_text=enhanced or chunk_text,
                    image_paths=[fr.image_path for fr in group],
                )
            )
        return chunks
