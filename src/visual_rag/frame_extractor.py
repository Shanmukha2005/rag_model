from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 1.5,
    max_frames: int = 0,
) -> List[Tuple[int, float, str]]:
    """
    Extract frames at fixed time intervals.

    Returns tuples: (frame_id, timestamp_sec, frame_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Video FPS metadata is invalid or missing.")

    interval_frames = max(int(round(interval_sec * fps)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extracted: List[Tuple[int, float, str]] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % interval_frames == 0:
            timestamp = frame_idx / fps
            frame_path = out / f"frame_{saved_idx:06d}_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted.append((saved_idx, timestamp, str(frame_path)))
            saved_idx += 1
            if max_frames > 0 and saved_idx >= max_frames:
                break

        frame_idx += 1
        if total_frames > 0 and frame_idx >= total_frames:
            break

    cap.release()
    if not extracted:
        raise ValueError("No frames extracted. Check interval and video content.")
    return extracted
