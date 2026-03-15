from __future__ import annotations

from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from main.data.collate import collate_samples
from main.trainer.stage1_memory_formation import stage1_loss
from main.utils.qwen_vl import build_processor_inputs
from main.cli.train_stage1 import (
    build_parser,
    apply_cuda_visible_devices,
    build_stage1_components,
    save_stage1_checkpoint,
)
from main.cli.common import load_yaml


def main():
    ap = build_parser()
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--mixed_precision", default="bf16")
    ap.add_argument("--zero_stage", type=int, default=2)
    args = ap.parse_args()

    apply_cuda_visible_devices(args.cuda_visible_devices)

    cfg_dict_for_runtime = load_yaml(args.config)
    grad_accum = max(1, int(cfg_dict_for_runtime.get("training", {}).get("grad_accum", 1)))
    ds_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=grad_accum,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=args.mixed_precision,
        deepspeed_plugin=ds_plugin,
    )

    cfg_dict, vismem, tokenizer, processor, opt, grad_accum_cfg, ds = build_stage1_components(
        args,
        device_map_override=None,
    )
    if grad_accum_cfg != grad_accum:
        accelerator.print(f"Warning: config grad_accum mismatch: loaded {grad_accum_cfg}, runtime {grad_accum}.")

    dataloader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=True,
    )

    vismem, opt, dataloader = accelerator.prepare(vismem, opt, dataloader)

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(
            dataloader,
            desc=f"Stage1 epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in pbar:
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
            inputs = {
                k: v.to(accelerator.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with accelerator.accumulate(vismem):
                loss_mem, loss_base = stage1_loss(vismem.base_model, vismem, inputs, answer)
                loss = loss_mem - loss_base.detach()
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process:
                pbar.set_postfix(
                    {
                        "loss_mem": float(loss_mem.detach().float().cpu()),
                        "loss_base": float(loss_base.detach().float().cpu()),
                    }
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_stage1_checkpoint(
                args.output_dir,
                epoch,
                accelerator.unwrap_model(vismem),
                tokenizer,
                cfg_dict,
                state_dict=accelerator.get_state_dict(vismem),
            )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
