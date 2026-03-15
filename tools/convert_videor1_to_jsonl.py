from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def load_objects(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}.")
            return data
        return [json.loads(line) for line in f if line.strip()]


def build_question(obj: Dict[str, Any]) -> str:
    question = obj["problem"]
    if obj.get("problem_type") == "multiple choice" and obj.get("options"):
        question = question + "\nOptions:\n" + "\n".join(str(x) for x in obj["options"])
    suffix = TYPE_TEMPLATE.get(obj.get("problem_type", "free-form"), TYPE_TEMPLATE["free-form"])
    return QUESTION_TEMPLATE.format(question=question) + suffix


def build_answer(obj: Dict[str, Any], mode: str) -> str | None:
    process = (obj.get("process") or "").strip()
    solution = (obj.get("solution") or "").strip()
    if mode == "rl":
        return solution or obj.get("answer")
    if process and solution:
        return process + "\n" + solution
    return solution or process or obj.get("answer")


def convert(records: Iterable[Dict[str, Any]], source_path: str, mode: str) -> Iterable[Dict[str, Any]]:
    base_dir = os.path.dirname(source_path)
    for idx, obj in enumerate(records):
        media_type = obj["data_type"]
        rel_path = obj["path"]
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(base_dir, rel_path))
        item = {
            "id": str(obj.get("id", idx)),
            "media_type": media_type,
            "image": abs_path if media_type == "image" else None,
            "video": abs_path if media_type == "video" else None,
            "prompt": build_question(obj),
            "answer": build_answer(obj, mode),
            "problem_type": obj.get("problem_type"),
            "source_path": rel_path,
        }
        yield item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video-R1 JSON/JSONL file")
    ap.add_argument("--output", required=True, help="Converted JSONL path")
    ap.add_argument("--mode", choices=["sft", "rl"], default="sft", help="sft keeps process+solution, rl keeps final solution")
    args = ap.parse_args()

    records = load_objects(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in convert(records, args.input, args.mode):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
