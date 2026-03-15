from __future__ import annotations
import argparse
import os
from typing import Any, Dict

from tqdm import tqdm
import torch
import torch.optim as optim

from main.utils.logging import get_logger
from main.utils.misc import set_seed, to_torch_dtype, ensure_dir
from main.utils.qwen_vl import load_qwen25vl, build_processor_inputs
from main.model.model import VisMemModel
from main.data.jsonl_dataset import JsonlVLDataset
from main.data.collate import collate_samples
from main.trainer.stage1_memory_formation import stage1_loss
from main.cli.common import load_yaml, build_vismem_config

logger = get_logger("main.train_stage1")
_USE_CONFIG_DEVICE_MAP = object()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cuda_visible_devices", default=None)
    return ap


def apply_cuda_visible_devices(cuda_visible_devices: str | None) -> None:
    if cuda_visible_devices is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    logger.info("Using CUDA_VISIBLE_DEVICES=%s", cuda_visible_devices)


def enable_memory_saving(vismem: VisMemModel) -> None:
    if hasattr(vismem.base_model.config, "use_cache"):
        vismem.base_model.config.use_cache = False
    if hasattr(vismem.base_model, "gradient_checkpointing_enable"):
        vismem.base_model.gradient_checkpointing_enable()
    if hasattr(vismem.base_model, "enable_input_require_grads"):
        vismem.base_model.enable_input_require_grads()


def build_stage1_components(
    args: argparse.Namespace,
    *,
    device_map_override: Any = _USE_CONFIG_DEVICE_MAP,
):
    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    set_seed(int(cfg_dict.get("training", {}).get("seed", 42)))

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    if device_map_override is _USE_CONFIG_DEVICE_MAP:
        device_map = cfg_dict["model"].get("device_map", "auto")
    else:
        device_map = device_map_override
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust,
    )
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    enable_memory_saving(vismem)

    for p in vismem.base_model.parameters():
        p.requires_grad = False

    trainable = [p for p in vismem.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else float(cfg_dict.get("training", {}).get("lr", 2e-4))
    opt = optim.AdamW(trainable, lr=lr)
    grad_accum = max(1, int(cfg_dict.get("training", {}).get("grad_accum", 1)))

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)
    return cfg_dict, vismem, tokenizer, processor, opt, grad_accum, ds


def save_stage1_checkpoint(
    output_dir: str,
    epoch: int,
    vismem_model,
    tokenizer,
    cfg_dict: Dict[str, Any],
    *,
    state_dict: Dict[str, torch.Tensor] | None = None,
) -> None:
    ckpt = os.path.join(output_dir, f"epoch{epoch}")
    ensure_dir(ckpt)
    if state_dict is None:
        state_dict = vismem_model.state_dict()
    torch.save({"vismem_state": state_dict, "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
    tokenizer.save_pretrained(ckpt)


def main():
    args = build_parser().parse_args()
    apply_cuda_visible_devices(args.cuda_visible_devices)

    cfg_dict, vismem, tokenizer, processor, opt, grad_accum, ds = build_stage1_components(args)

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Stage1 epoch {epoch}")
        opt.zero_grad(set_to_none=True)
        for i in pbar:
            batch = collate_samples([ds[i]])
            img = batch["images"][0]
            video = batch["videos"][0]
            prompt = batch["prompts"][0]
            answer = batch["answers"][0]
            if answer is None:
                continue

            inputs = build_processor_inputs(
                processor,
                prompt=prompt,
                image=img,
                video=video,
                add_generation_prompt=True,
            )
            inputs = {k: v.to(vismem.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            loss_mem, loss_base = stage1_loss(vismem.base_model, vismem, inputs, answer)
            loss = (loss_mem - loss_base.detach()) / grad_accum
            loss.backward()

            should_step = ((i + 1) % grad_accum == 0) or (i == len(ds) - 1)
            if should_step:
                opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(
                {
                    "loss_mem": float(loss_mem.detach().cpu()),
                    "loss_base": float(loss_base.detach().cpu()),
                }
            )

        save_stage1_checkpoint(args.output_dir, epoch, vismem, tokenizer, cfg_dict)

    logger.info("Stage1 done.")


if __name__ == "__main__":
    main()
