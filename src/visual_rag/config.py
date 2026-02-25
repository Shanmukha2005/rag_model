from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    # Core runtime
    workspace_dir: Path = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
    frame_interval_sec: float = float(os.getenv("FRAME_INTERVAL_SEC", "1.5"))
    max_frames: int = int(os.getenv("MAX_FRAMES", "0"))  # 0 = no limit

    # Model names / endpoints
    qwen_model_name: str = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

    # API keys (if using remote inference)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.18"))

    # Compute
    device: str = os.getenv("DEVICE", "cuda")
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
settings.workspace_dir.mkdir(parents=True, exist_ok=True)
