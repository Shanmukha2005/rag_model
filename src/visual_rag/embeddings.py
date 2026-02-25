from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .models import SemanticChunk


class ClipEmbedder:
    """Creates multimodal embeddings by averaging CLIP text and image vectors."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            text_features = self.model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            vectors.append(text_features.cpu().numpy())
        return np.vstack(vectors)

    @torch.no_grad()
    def embed_images(self, image_paths: List[str], batch_size: int = 8) -> np.ndarray:
        vectors = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            vectors.append(image_features.cpu().numpy())
        return np.vstack(vectors)

    def embed_chunks(self, chunks: List[SemanticChunk], batch_size: int = 8) -> np.ndarray:
        texts = [f"{c.summary}\nKeywords: {', '.join(c.keywords)}\n{c.enhanced_text}" for c in chunks]
        text_vecs = self.embed_texts(texts, batch_size=batch_size)

        representative_images = [c.image_paths[0] for c in chunks]
        img_vecs = self.embed_images(representative_images, batch_size=batch_size)

        merged = (text_vecs + img_vecs) / 2.0
        norms = np.linalg.norm(merged, axis=1, keepdims=True)
        return merged / np.clip(norms, 1e-12, None)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_texts([query], batch_size=1)
        return vec[0]
