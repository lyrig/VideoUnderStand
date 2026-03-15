from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset

QUESTION_TEMPLATE = (
    "{question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', "
    "'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final "
    "answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}


@dataclass
class Sample:
    id: str
    media_type: str
    image: Optional[str]
    video: Optional[str]
    prompt: str
    answer: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Sample] = []
        for obj in self._load_objects(jsonl_path):
            self.items.append(self._to_sample(obj, jsonl_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]

    def _load_objects(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                data = json.load(f)
                if isinstance(data, list):
                    return data
                raise ValueError(f"Expected a JSON list in {path}.")

            objs: List[Dict[str, Any]] = []
            for line in f:
                if not line.strip():
                    continue
                objs.append(json.loads(line))
            return objs

    def _to_sample(self, obj: Dict[str, Any], source_path: str) -> Sample:
        if "prompt" in obj:
            media_type = obj.get("media_type") or ("video" if obj.get("video") else "image")
            return Sample(
                id=str(obj.get("id", len(self.items))),
                media_type=media_type,
                image=self._resolve_media_path(obj.get("image"), source_path),
                video=self._resolve_media_path(obj.get("video"), source_path),
                prompt=obj["prompt"],
                answer=obj.get("answer"),
                meta={k: v for k, v in obj.items() if k not in ("id", "media_type", "image", "video", "prompt", "answer")},
            )

        if {"problem", "data_type", "path"}.issubset(obj):
            prompt = self._build_videor1_prompt(obj)
            answer = self._build_videor1_answer(obj)
            media_type = obj["data_type"]
            media_path = self._resolve_media_path(obj["path"], source_path)
            return Sample(
                id=str(obj.get("id", len(self.items))),
                media_type=media_type,
                image=media_path if media_type == "image" else None,
                video=media_path if media_type == "video" else None,
                prompt=prompt,
                answer=answer,
                meta={k: v for k, v in obj.items() if k not in ("id", "problem", "data_type", "path", "process", "solution")},
            )

        raise ValueError("Unsupported sample format. Expected either native {prompt,...} or Video-R1-style fields.")

    def _build_videor1_prompt(self, obj: Dict[str, Any]) -> str:
        question = obj["problem"]
        if obj.get("problem_type") == "multiple choice":
            options = obj.get("options", [])
            if options:
                question = question + "\nOptions:\n" + "\n".join(str(op) for op in options)
        suffix = TYPE_TEMPLATE.get(obj.get("problem_type", "free-form"), TYPE_TEMPLATE["free-form"])
        return QUESTION_TEMPLATE.format(question=question) + suffix

    def _build_videor1_answer(self, obj: Dict[str, Any]) -> Optional[str]:
        process = (obj.get("process") or "").strip()
        solution = (obj.get("solution") or "").strip()
        if process and solution:
            return process + "\n" + solution
        if solution:
            return solution
        if process:
            return process
        return obj.get("answer")

    def _resolve_media_path(self, path: Optional[str], source_path: str) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(os.path.dirname(source_path), path))
