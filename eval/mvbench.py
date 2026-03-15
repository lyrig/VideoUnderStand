import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VIDEOUNDERSTAND_ROOT = REPO_ROOT / "VideoUnderStand"
if str(VIDEOUNDERSTAND_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOUNDERSTAND_ROOT))

from main.cli.common import build_vismem_config, load_yaml
from main.model.model import VisMemModel
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl

from dataset.video_transforms import (
    GroupCenterCrop,
    GroupNormalize,
    GroupScale,
    Stack,
    ToTorchFormatTensor,
)


DEFAULT_QWEN_CONFIG = VIDEOUNDERSTAND_ROOT / "configs" / "vismem_qwen25vl7b.yaml"
DEFAULT_SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question.\n"
)

RAW_DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "your_data_path/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "your_data_path/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "your_data_path/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "your_data_path/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "your_data_path/sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "your_data_path/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "your_data_path/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "your_data_path/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "your_data_path/perception/videos/", "video", False),
    # "Fine-grained Pose": ("fine_grained_pose.json", "your_data_path/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "your_data_path/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "your_data_path/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "your_data_path/tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "your_data_path/clevrer/video_validation/", "video", False),
}


class MVBenchDataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=16, resolution=224):
        self.data_list = []
        for task_type, task_meta in data_list.items():
            with open(Path(data_dir) / task_meta[0], "r", encoding="utf-8") as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append(
                    {
                        "task_type": task_type,
                        "prefix": task_meta[1],
                        "data_type": task_meta[2],
                        "bound": task_meta[3],
                        "data": data,
                    }
                )

        self.decord_method = {
            "video": self.read_video,
            "gif": self.read_gif,
            "frame": self.read_frame,
        }
        self.num_segments = num_segments
        self.transform = T.Compose(
            [
                GroupScale(int(resolution), interpolation=InterpolationMode.BICUBIC),
                GroupCenterCrop(resolution),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        return np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(self.num_segments)
            ]
        )

    def read_video(self, video_path, bound=None):
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        images_group = [Image.fromarray(vr[frame_index].numpy()) for frame_index in frame_indices]
        return self.transform(images_group)

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(str(video_path))
        max_frame = len(gif) - 1
        frame_indices = set(self.get_index(bound, fps, max_frame, first_idx=0).tolist())
        images_group = []
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                images_group.append(Image.fromarray(img))
        return self.transform(images_group)

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)
        images_group = [
            Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg")).convert("RGB")
            for frame_index in frame_indices
        ]
        return self.transform(images_group)

    def qa_template(self, data):
        question = f"Question: {data['question']}\nOptions:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, candidate in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {candidate}\n"
            if candidate == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        bound = None
        if sample["bound"]:
            bound = (sample["data"]["start"], sample["data"]["end"])
        video_path = os.path.join(sample["prefix"], sample["data"]["video"])
        torch_imgs = self.decord_method[sample["data_type"]](video_path, bound)
        question, answer = self.qa_template(sample["data"])
        return {
            "video": torch_imgs,
            "question": question,
            "answer": answer,
            "task_type": sample["task_type"],
            "video_path": video_path,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run MVBench evaluation with Qwen-VL or VisMem.")
    parser.add_argument("--config", default=str(DEFAULT_QWEN_CONFIG), help="Path to VideoUnderStand yaml config.")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Qwen-VL model path or HF id. Overrides yaml config if provided.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Optional VisMem checkpoint directory or main.pt path. Leave empty to evaluate raw Qwen-VL.",
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing MVBench json annotations.")
    parser.add_argument("--data_root", required=True, help="Root directory used to replace 'your_data_path'.")
    parser.add_argument("--output_path", default="mvbench_results.json", help="Where to save detailed results.")
    parser.add_argument(
        "--leaderboard_path",
        default="upload_leaderboard.json",
        help="Where to save per-task accuracy summary.",
    )
    parser.add_argument("--num_frames", type=int, default=16, help="Number of sampled frames per video.")
    parser.add_argument("--resolution", type=int, default=224, help="Frame resolution for MVBench decoding.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p.")
    parser.add_argument("--enable_vismem", action="store_true", help="Enable VisMem memory insertion during generate.")
    parser.add_argument("--reverse_mem_type", action="store_true", help="Use reversed memory type in VisMem generate.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--verbose", action="store_true", help="Print prompts and predictions.")
    return parser.parse_args()


def build_data_list(data_root):
    data_root = str(Path(data_root).resolve()).replace("\\", "/")
    return {
        task_type: (meta[0], meta[1].replace("your_data_path", data_root), meta[2], meta[3])
        for task_type, meta in RAW_DATA_LIST.items()
    }


def load_model(args):
    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path

    viscfg = build_vismem_config(cfg_dict)
    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust,
    )
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)

    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path).resolve()
        if ckpt_path.is_dir():
            ckpt_file = ckpt_path / "main.pt"
        else:
            ckpt_file = ckpt_path
        if not ckpt_file.exists():
            raise FileNotFoundError(f"VisMem checkpoint not found: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu")
        if "vismem_state" in state:
            vismem.load_state_dict(state["vismem_state"], strict=False)
        else:
            vismem.load_state_dict(state, strict=False)
        print(f"Loaded VisMem checkpoint from {ckpt_file}")

    vismem.eval()
    return vismem


def build_prompt(question):
    return (
        DEFAULT_SYSTEM_PROMPT
        + question
        + "\nOnly give the best option. Start your answer with the option letter, for example: (A)"
    )


def normalize_prediction(text):
    line = text.strip().splitlines()[0].strip()
    if not line.startswith("("):
        for option in ["A", "B", "C", "D", "E", "F"]:
            if line.upper().startswith(option):
                return f"({option})"
    return line


def infer_mvbench(model, sample, args):
    prompt = build_prompt(sample["question"])
    pred = model.generate(
        images=None,
        videos=[sample["video_path"]],
        prompts=[prompt],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_vismem=args.enable_vismem,
        reverse_mem_type=args.reverse_mem_type,
    )[0]
    pred = normalize_prediction(pred)
    if args.verbose:
        print(prompt)
        print(f"PRED: {pred}")
        print(f"GT: {sample['answer']}")
    return pred


def check_ans(pred, gt):
    pred_list = pred.lower().split(" ")
    gt_list = gt.lower().split(" ")
    if not pred_list or not gt_list:
        return False
    pred_option = pred_list[0]
    gt_option = gt_list[0]
    return pred_option.replace(".", "") in gt_option or gt_option in pred_option


def evaluate(model, dataset, args):
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    iterator = dataset if args.max_samples is None else [dataset[i] for i in range(min(args.max_samples, len(dataset)))]

    for example in tqdm(iterator):
        task_type = example["task_type"]
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]
        acc_dict[task_type][1] += 1
        total += 1

        pred = infer_mvbench(model, example, args)
        gt = example["answer"]
        res_list.append(
            {
                "task_type": task_type,
                "video_path": example["video_path"],
                "question": example["question"],
                "pred": pred,
                "gt": gt,
            }
        )

        if check_ans(pred, gt):
            acc_dict[task_type][0] += 1
            correct += 1

        print(f"Part Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100:.2f}%")
        print(f"Total Acc: {correct / total * 100:.2f}%")
        print("-" * 30, task_type, "-" * 30)

    final_res = {task_type: c / n * 100 for task_type, (c, n) in acc_dict.items()}
    final_res["Avg"] = correct / total * 100 if total else 0.0
    return acc_dict, res_list, final_res


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_root = Path(args.data_root).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"MVBench json directory not found: {data_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"MVBench data root not found: {data_root}")

    model = load_model(args)
    dataset = MVBenchDataset(
        data_dir=data_dir,
        data_list=build_data_list(data_root),
        num_segments=args.num_frames,
        resolution=args.resolution,
    )
    print(f"Loaded {len(dataset)} MVBench samples.")

    acc_dict, res_list, final_res = evaluate(model, dataset, args)

    output_path = Path(args.output_path).resolve()
    leaderboard_path = Path(args.leaderboard_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"acc_dict": acc_dict, "res_list": res_list}, f, ensure_ascii=False, indent=2)
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(final_res, f, ensure_ascii=False, indent=2)

    print("Final results:")
    print(json.dumps(final_res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
