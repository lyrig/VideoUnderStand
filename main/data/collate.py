from __future__ import annotations
from typing import List, Dict, Any, Optional
from PIL import Image

from .jsonl_dataset import Sample

def load_image(path: Optional[str]) -> Optional[Image.Image]:
    if path is None:
        return None
    return Image.open(path).convert("RGB")

def collate_samples(samples: List[Sample]) -> Dict[str, Any]:
    ids = [s.id for s in samples]
    media_types = [s.media_type for s in samples]
    images = [load_image(s.image) for s in samples]
    videos = [s.video for s in samples]
    prompts = [s.prompt for s in samples]
    answers = [s.answer for s in samples]
    metas = [s.meta or {} for s in samples]
    return {
        "ids": ids,
        "media_types": media_types,
        "images": images,
        "videos": videos,
        "prompts": prompts,
        "answers": answers,
        "metas": metas,
    }
