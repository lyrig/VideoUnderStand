from __future__ import annotations
from typing import Any, Dict, Optional
from main.constants import ALL_SPECIAL_TOKENS
import torch


def add_tokens(tokenizer):
    special = {"additional_special_tokens": ALL_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special)
    return tokenizer


def init_token_embeddings(model, tokenizer, init_from_token: str | None = None, noise_std: float = 1e-3):
    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        return
    w_in = emb_layer.weight

    init_id = None
    if init_from_token is not None:
        init_id = tokenizer.convert_tokens_to_ids(init_from_token)
    if init_id is None or init_id == tokenizer.unk_token_id:
        init_id = tokenizer.eos_token_id
    if init_id is None:
        return

    with torch.no_grad():
        ref = w_in[init_id].detach().clone()
        for tok in ALL_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                continue
            w_in[tid].copy_(ref + torch.randn_like(ref) * noise_std)

        out_layer = getattr(model, "get_output_embeddings", lambda: None)()
        if out_layer is not None and out_layer.weight.data_ptr() != w_in.data_ptr():
            w_out = out_layer.weight
            for tok in ALL_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid == tokenizer.unk_token_id:
                    continue
                w_out[tid].copy_(w_in[tid])


def load_qwen25vl(model_name_or_path: str, torch_dtype=None, device_map="auto", trust_remote_code=True):
    from transformers import AutoTokenizer, AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    except Exception:
        from transformers import AutoModelForCausalLM as AutoModelClass
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except Exception:
        Qwen2VLForConditionalGeneration = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception:
        Qwen2_5_VLForConditionalGeneration = None

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    old_vocab = len(tokenizer)
    tokenizer = add_tokens(tokenizer)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }

    model_name_lower = model_name_or_path.lower()
    if "qwen2.5-vl" in model_name_lower:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "Your transformers installation does not provide Qwen2_5_VLForConditionalGeneration. "
                "Please install a transformers version that supports Qwen2.5-VL."
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    elif "qwen2-vl" in model_name_lower:
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError(
                "Your transformers installation does not provide Qwen2VLForConditionalGeneration. "
                "Please install a transformers version that supports Qwen2-VL."
            )
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    else:
        model = AutoModelClass.from_pretrained(model_name_or_path, **model_kwargs)

    # resize embeddings after adding tokens
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if len(tokenizer) > old_vocab:
        init_token_embeddings(model, tokenizer, init_from_token=None, noise_std=1e-3)

    return model, tokenizer, processor


def build_messages(prompt: str, image: Optional[Any] = None, video: Optional[Any] = None, answer: Optional[str] = None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if video is not None:
        content.append({"type": "video", "video": video})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def build_processor_inputs(
    processor,
    prompt: str,
    image: Optional[Any] = None,
    video: Optional[Any] = None,
    answer: Optional[str] = None,
    add_generation_prompt: Optional[bool] = None,
):
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise ImportError(
            "qwen_vl_utils.process_vision_info is required for Qwen2/2.5-VL multimodal inputs."
        ) from exc

    if add_generation_prompt is None:
        add_generation_prompt = answer is None

    messages = build_messages(prompt=prompt, image=image, video=video, answer=answer)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
