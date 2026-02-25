from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from .models import FrameRecord


PROMPT = """You are a strict visual analyst.
Inspect the image and return JSON with keys:
scene_description (string),
objects (array of strings),
actions (array of strings),
facial_expressions (array of strings),
ocr_text (string),
environment_context (string).
Use only visible evidence from the image. No speculation.
If unknown, use empty values.
Return ONLY valid JSON."""


class QwenVisionAnalyzer:
    """Qwen-VL analysis via OpenAI-compatible endpoint."""

    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for Qwen analysis.")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    @staticmethod
    def _to_data_url(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower().replace(".", "") or "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{suffix};base64,{b64}"

    def analyze_frame(self, frame_id: int, timestamp_sec: float, image_path: str) -> FrameRecord:
        image_url = self._to_data_url(image_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this video frame."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        try:
            parsed: Dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            parsed = {
                "scene_description": "",
                "objects": [],
                "actions": [],
                "facial_expressions": [],
                "ocr_text": "",
                "environment_context": "",
            }

        # Normalize schema
        normalized = {
            "scene_description": str(parsed.get("scene_description", "")).strip(),
            "objects": [str(x).strip() for x in parsed.get("objects", []) if str(x).strip()],
            "actions": [str(x).strip() for x in parsed.get("actions", []) if str(x).strip()],
            "facial_expressions": [
                str(x).strip() for x in parsed.get("facial_expressions", []) if str(x).strip()
            ],
            "ocr_text": str(parsed.get("ocr_text", "")).strip(),
            "environment_context": str(parsed.get("environment_context", "")).strip(),
        }

        return FrameRecord(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            image_path=image_path,
            analysis=normalized,
        )

    def batch_analyze(self, frames: List[tuple[int, float, str]]) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        for frame_id, ts, frame_path in frames:
            records.append(self.analyze_frame(frame_id, ts, frame_path))
        return records
