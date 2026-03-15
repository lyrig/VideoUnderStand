from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn.functional as F


def _has_visual_inputs(inputs: Dict[str, Any]) -> bool:
    return inputs.get("pixel_values") is not None or inputs.get("pixel_values_videos") is not None


def _require_model_attr(model, attr: str):
    value = getattr(model, attr, None)
    if value is None:
        raise RuntimeError(
            f"Multimodal stage1 training requires base_model.{attr} to keep the forward path consistent."
        )
    return value


def _unwrap_feature_tensor(features, feature_name: str) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "last_hidden_state") and isinstance(features.last_hidden_state, torch.Tensor):
        return features.last_hidden_state
    if hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        return features.pooler_output
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        return features[0]
    raise RuntimeError(
        f"{feature_name} must be a Tensor-compatible output, got {type(features).__name__} instead."
    )


def _build_prompt_inputs_embeds(base_model, inputs: Dict[str, Any]) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    embeds = base_model.get_input_embeddings()(input_ids)

    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        get_image_features = _require_model_attr(base_model, "get_image_features")
        image_embeds = _unwrap_feature_tensor(
            get_image_features(pixel_values, inputs.get("image_grid_thw")),
            "image features",
        )
        image_token_id = base_model.config.image_token_id
        image_mask = input_ids == image_token_id
        n_image_tokens = int(image_mask.sum().item())
        n_image_features = int(image_embeds.shape[0])
        if n_image_tokens != n_image_features:
            raise RuntimeError(
                "Inconsistent multimodal forward: image token count does not match image feature count "
                f"({n_image_tokens} != {n_image_features})."
            )
        embeds = embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(embeds),
            image_embeds.to(device=embeds.device, dtype=embeds.dtype),
        )

    pixel_values_videos = inputs.get("pixel_values_videos")
    if pixel_values_videos is not None:
        get_video_features = _require_model_attr(base_model, "get_video_features")
        video_embeds = _unwrap_feature_tensor(
            get_video_features(pixel_values_videos, inputs.get("video_grid_thw")),
            "video features",
        )
        video_token_id = base_model.config.video_token_id
        video_mask = input_ids == video_token_id
        n_video_tokens = int(video_mask.sum().item())
        n_video_features = int(video_embeds.shape[0])
        if n_video_tokens != n_video_features:
            raise RuntimeError(
                "Inconsistent multimodal forward: video token count does not match video feature count "
                f"({n_video_tokens} != {n_video_features})."
            )
        embeds = embeds.masked_scatter(
            video_mask.unsqueeze(-1).expand_as(embeds),
            video_embeds.to(device=embeds.device, dtype=embeds.dtype),
        )

    return embeds


def _build_full_position_ids(
    base_model,
    inputs: Dict[str, Any],
    continuation_len: int,
    *,
    device: torch.device,
) -> torch.LongTensor:
    get_rope_index = _require_model_attr(base_model, "get_rope_index")
    attention_mask = inputs.get("attention_mask")
    position_ids, _ = get_rope_index(
        inputs.get("input_ids"),
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        inputs.get("second_per_grid_ts"),
        attention_mask,
    )
    if continuation_len == 0:
        return position_ids.to(device=device)

    batch_size = position_ids.shape[1]
    next_position = position_ids.max(dim=-1, keepdim=True).values + 1
    continuation_offsets = torch.arange(continuation_len, device=position_ids.device, dtype=position_ids.dtype)
    continuation_position_ids = next_position + continuation_offsets.view(1, 1, -1)
    continuation_position_ids = continuation_position_ids.expand(3, batch_size, -1)
    return torch.cat([position_ids, continuation_position_ids], dim=-1).to(device=device)


def stage1_loss(base_model, vismem_model, inputs: Dict[str, Any], target_text: str):

    tokenizer = vismem_model.tokenizer
    device = vismem_model.device
    prompt_attention_mask = inputs.get("attention_mask")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=inputs["input_ids"].device)

    # Encode target
    tgt_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    prompt_emb = _build_prompt_inputs_embeds(base_model, inputs)
    tgt_emb = base_model.get_input_embeddings()(tgt_ids.to(inputs["input_ids"].device))
    tgt_emb = tgt_emb.to(device=prompt_emb.device, dtype=prompt_emb.dtype)

    with torch.no_grad():
        full_embeds = torch.cat([prompt_emb, tgt_emb], dim=1)
        full_attn = torch.cat(
            [
                prompt_attention_mask.to(device=prompt_emb.device),
                torch.ones_like(tgt_ids, dtype=torch.long, device=prompt_emb.device),
            ],
            dim=1,
        )
        full_position_ids = _build_full_position_ids(base_model, inputs, tgt_emb.size(1), device=prompt_emb.device)
        labels = torch.cat([inputs["input_ids"], tgt_ids], dim=1).to(device=prompt_emb.device)
        labels[:, :inputs["input_ids"].size(1)] = -100
        out = base_model(inputs_embeds=full_embeds, attention_mask=full_attn, position_ids=full_position_ids)
        logits = out.logits[:, :-1, :]
        loss_base = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)

    base_out = base_model(**inputs, output_hidden_states=True)
    hidden = base_out.hidden_states[-1]  # (B,T,D)
    # Build H
    H = hidden
    M = vismem_model.form_memory(H, mem_type="short")  # or "long" / both, configurable

    # Feed
    emb = prompt_emb
    M = M.to(device=emb.device, dtype=emb.dtype)
    inp_embeds = torch.cat([emb, M, tgt_emb], dim=1)
    attn2 = torch.cat(
        [
            prompt_attention_mask.to(device=inp_embeds.device),
            torch.ones(
                (inputs["input_ids"].size(0), M.size(1) + tgt_emb.size(1)),
                device=inp_embeds.device,
                dtype=torch.long,
            ),
        ],
        dim=1,
    )
    position_ids2 = _build_full_position_ids(
        base_model,
        inputs,
        M.size(1) + tgt_emb.size(1),
        device=inp_embeds.device,
    )

    labels2 = torch.cat(
        [
            inputs["input_ids"].to(device),
            torch.full((inputs["input_ids"].size(0), M.size(1)), -100, device=device, dtype=torch.long),
            tgt_ids.to(device),
        ],
        dim=1,
    ).to(inp_embeds.device)
    labels2[:, :inputs["input_ids"].size(1) + M.size(1)] = -100

    if _has_visual_inputs(inputs):
        expected_prompt_len = inputs["input_ids"].size(1)
        if emb.size(1) != expected_prompt_len:
            raise RuntimeError(
                "Inconsistent multimodal forward: prompt embeddings length diverged from input_ids length "
                f"({emb.size(1)} != {expected_prompt_len})."
            )

    out2 = base_model(
        inputs_embeds=inp_embeds,
        attention_mask=attn2,
        position_ids=position_ids2,
    )
    logits2 = out2.logits[:, :-1, :]
    loss_mem = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), labels2[:, 1:].reshape(-1), ignore_index=-100)

    # Optimize
    return loss_mem, loss_base
