from __future__ import annotations
from typing import Dict, Any
import torch


def _get_model_device_dtype(base_model):
    emb = base_model.get_input_embeddings()
    return emb.weight.device, emb.weight.dtype


def _build_qwen_multimodal_input_embeds(base_model, inputs: Dict[str, Any]) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    device, dtype = _get_model_device_dtype(base_model)
    inputs_embeds = base_model.get_input_embeddings()(input_ids.to(device))

    model = getattr(base_model, "model", None)
    if model is None:
        return inputs_embeds

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is not None and hasattr(model, "get_image_features"):
        image_embeds = model.get_image_features(pixel_values.to(device), image_grid_thw)
        mask = input_ids.to(device) == base_model.config.image_token_id
        if int(mask.sum().item()) != int(image_embeds.shape[0]):
            raise ValueError(
                f"Image features and image tokens do not match: tokens={int(mask.sum().item())}, "
                f"features={int(image_embeds.shape[0])}"
            )
        image_embeds = image_embeds.to(device=device, dtype=dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds)

    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")
    if pixel_values_videos is not None and hasattr(model, "get_video_features"):
        video_embeds = model.get_video_features(pixel_values_videos.to(device), video_grid_thw)
        mask = input_ids.to(device) == base_model.config.video_token_id
        if int(mask.sum().item()) != int(video_embeds.shape[0]):
            raise ValueError(
                f"Video features and video tokens do not match: tokens={int(mask.sum().item())}, "
                f"features={int(video_embeds.shape[0])}"
            )
        video_embeds = video_embeds.to(device=device, dtype=dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds)

    return inputs_embeds


def stage1_loss(base_model, vismem_model, inputs: Dict[str, Any], target_text: str):

    tokenizer = vismem_model.tokenizer
    model_device, model_dtype = _get_model_device_dtype(base_model)

    # Encode target
    tgt_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model_device)

    input_ids = inputs["input_ids"].to(model_device)
    prefix_len = input_ids.size(1)
    target_embeds = base_model.get_input_embeddings()(tgt_ids).to(dtype=model_dtype)

    multimodal_inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prefix_embeds = _build_qwen_multimodal_input_embeds(base_model, multimodal_inputs).to(dtype=model_dtype)

    full_ids = torch.cat([input_ids, tgt_ids], dim=1)
    full_attn = torch.ones_like(full_ids, dtype=torch.long, device=model_device)
    labels = full_ids.clone()
    labels[:, :prefix_len] = -100

    with torch.no_grad():
        out = base_model(
            input_ids=full_ids,
            attention_mask=full_attn,
            labels=labels,
            pixel_values=multimodal_inputs.get("pixel_values"),
            pixel_values_videos=multimodal_inputs.get("pixel_values_videos"),
            image_grid_thw=multimodal_inputs.get("image_grid_thw"),
            video_grid_thw=multimodal_inputs.get("video_grid_thw"),
            second_per_grid_ts=multimodal_inputs.get("second_per_grid_ts"),
        )
        loss_base = out.loss

    base_out = base_model(**multimodal_inputs, output_hidden_states=True)
    hidden = base_out.hidden_states[-1]  # (B,T,D)
    # Build H
    H = hidden
    M = vismem_model.form_memory(H, mem_type="short")  # or "long" / both, configurable

    mem_placeholders = torch.full(
        (input_ids.size(0), M.size(1)),
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        device=model_device,
        dtype=input_ids.dtype,
    )
    full_ids_with_mem = torch.cat([input_ids, mem_placeholders, tgt_ids], dim=1)
    inp_embeds = torch.cat([prefix_embeds, M.to(dtype=model_dtype), target_embeds], dim=1)
    attn2 = torch.ones(inp_embeds.size()[:-1], device=model_device, dtype=torch.long)

    labels2 = full_ids_with_mem.clone()
    labels2[:, : prefix_len + M.size(1)] = -100

    out2 = base_model(
        input_ids=full_ids_with_mem,
        inputs_embeds=inp_embeds,
        attention_mask=attn2,
        labels=labels2,
        image_grid_thw=multimodal_inputs.get("image_grid_thw"),
        video_grid_thw=multimodal_inputs.get("video_grid_thw"),
        second_per_grid_ts=multimodal_inputs.get("second_per_grid_ts"),
    )
    loss_mem = out2.loss

    # Optimize
    return loss_mem, loss_base
