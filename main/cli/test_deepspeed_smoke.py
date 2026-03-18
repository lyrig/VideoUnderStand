from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim
from accelerate import Accelerator, DeepSpeedPlugin
from PIL import Image

from main.cli.common import build_vismem_config, load_yaml
from main.cli.train_stage1 import apply_cuda_visible_devices, enable_memory_saving
from main.model.model import VisMemModel
from main.trainer.stage1_memory_formation import stage1_loss
from main.utils.misc import ensure_dir, set_seed, to_torch_dtype
from main.utils.qwen_vl import build_processor_inputs, load_qwen25vl


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Smoke test DeepSpeed loading and a few training steps for VideoUnderStand."
    )
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_jsonl", default=None, help="Optional existing JSONL. If omitted, a synthetic sample is generated.")
    ap.add_argument("--steps", type=int, default=1, help="Number of training steps to run.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--zero_stage", type=int, default=2)
    ap.add_argument("--mixed_precision", default="bf16")
    ap.add_argument("--cuda_visible_devices", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--former_backend",
        choices=["tiny_transformer", "lora_llm"],
        default=None,
        help="Optional override for vismem.former_backend. tiny_transformer is useful for lightweight smoke tests.",
    )
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--prompt", default="Describe the image briefly.")
    ap.add_argument("--answer", default="<think>This is a smoke test.</think><answer>A simple synthetic image.</answer>")
    return ap


def make_synthetic_sample(output_dir: str, image_size: int, prompt: str, answer: str) -> str:
    data_dir = Path(output_dir) / "smoke_data"
    ensure_dir(str(data_dir))

    image_path = data_dir / "synthetic.png"
    jsonl_path = data_dir / "synthetic.jsonl"

    img = Image.new("RGB", (image_size, image_size), color=(64, 128, 192))
    img.save(image_path)

    sample: Dict[str, Any] = {
        "id": "smoke-0",
        "media_type": "image",
        "image": image_path.name,
        "prompt": prompt,
        "answer": answer,
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return str(jsonl_path)


def load_first_sample(jsonl_path: str) -> Dict[str, Any]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"No samples found in {jsonl_path}.")
    obj = json.loads(first_line)
    image = obj.get("image")
    if image and not os.path.isabs(image):
        image = os.path.normpath(os.path.join(os.path.dirname(jsonl_path), image))
    return {
        "image": Image.open(image).convert("RGB") if image else None,
        "video": obj.get("video"),
        "prompt": obj["prompt"],
        "answer": obj.get("answer") or "<answer>synthetic</answer>",
    }


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    args = build_parser().parse_args()
    apply_cuda_visible_devices(args.cuda_visible_devices)
    ensure_dir(args.output_dir)

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict.setdefault("model", {})["model_name_or_path"] = args.model_name_or_path
    if args.former_backend is not None:
        vismem_cfg = cfg_dict.setdefault("vismem", cfg_dict.get("main", {}))
        vismem_cfg["former_backend"] = args.former_backend

    set_seed(args.seed)
    grad_accum = max(1, int(cfg_dict.get("training", {}).get("grad_accum", 1)))
    ds_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=grad_accum,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=args.mixed_precision,
        deepspeed_plugin=ds_plugin,
    )

    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))
    model_name = cfg_dict["model"]["model_name_or_path"]

    base_model, tokenizer, processor = load_qwen25vl(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust,
    )
    vismem = VisMemModel(base_model, tokenizer, processor, build_vismem_config(cfg_dict))
    enable_memory_saving(vismem)

    for p in vismem.base_model.parameters():
        p.requires_grad = False

    trainable = [p for p in vismem.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check former_backend and vismem config.")
    lr = args.lr if args.lr is not None else float(cfg_dict.get("training", {}).get("lr", 2e-4))
    opt = optim.AdamW(trainable, lr=lr)

    jsonl_path = args.train_jsonl or make_synthetic_sample(
        args.output_dir,
        image_size=args.image_size,
        prompt=args.prompt,
        answer=args.answer,
    )
    sample = load_first_sample(jsonl_path)

    vismem, opt = accelerator.prepare(vismem, opt)

    total_params, trainable_params = count_parameters(accelerator.unwrap_model(vismem))
    accelerator.print(f"Model loaded: {model_name}")
    accelerator.print(f"Sample source: {jsonl_path}")
    accelerator.print(
        f"Parameters: total={total_params:,}, trainable={trainable_params:,}, zero_stage={args.zero_stage}, mixed_precision={args.mixed_precision}"
    )

    vismem.train()
    final_loss_mem = None
    final_loss_base = None
    for step in range(args.steps):
        inputs = build_processor_inputs(
            processor,
            prompt=sample["prompt"],
            image=sample["image"],
            video=sample["video"],
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(accelerator.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with accelerator.accumulate(vismem):
            loss_mem, loss_base = stage1_loss(vismem.base_model, vismem, inputs, sample["answer"])
            loss = loss_mem - loss_base.detach()
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)

        final_loss_mem = float(loss_mem.detach().float().cpu())
        final_loss_base = float(loss_base.detach().float().cpu())
        accelerator.print(
            f"step={step + 1}/{args.steps} loss_mem={final_loss_mem:.6f} loss_base={final_loss_base:.6f}"
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report_path = Path(args.output_dir) / "smoke_test_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name_or_path": model_name,
                    "sample_source": jsonl_path,
                    "steps": args.steps,
                    "zero_stage": args.zero_stage,
                    "mixed_precision": args.mixed_precision,
                    "former_backend": accelerator.unwrap_model(vismem).former_backend,
                    "loss_mem": final_loss_mem,
                    "loss_base": final_loss_base,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        accelerator.print(f"Smoke test completed. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
