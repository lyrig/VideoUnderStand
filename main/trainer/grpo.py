from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

@dataclass
class GRPOBatch:
    input_ids: torch.LongTensor          # (B, T)
    attention_mask: torch.LongTensor     # (B, T)
    labels: torch.LongTensor             # (B, T) with -100 for prompt positions

def sequence_logprobs(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    # gather label logp
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)

def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:

    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)  # (B,T)
    return kl.mean(dim=-1)

class SimpleGRPOTrainer:
    def __init__(self, model, ref_model=None, kl_beta: float = 0.02):
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta

    def _sequence_logprobs_from_prompt(self, model, prompts_inputs: Dict[str, Any], sampled_ids: torch.LongTensor) -> torch.Tensor:
        prompt_kwargs = dict(prompts_inputs)
        out = model(**prompt_kwargs, use_cache=True, output_hidden_states=False)
        past = out.past_key_values
        cur_logits = out.logits[:, -1, :]
        token_logps = []

        for pos in range(sampled_ids.size(1)):
            token = sampled_ids[:, pos]
            logp = F.log_softmax(cur_logits, dim=-1).gather(-1, token.unsqueeze(-1)).squeeze(-1)
            token_logps.append(logp)
            out = model(
                input_ids=token.unsqueeze(-1),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                attention_mask=None,
            )
            past = out.past_key_values
            cur_logits = out.logits[:, -1, :]

        return torch.stack(token_logps, dim=1).sum(dim=-1)

    def loss_from_samples(self, prompts_inputs: Dict[str, Any], sampled_ids: torch.LongTensor, rewards: torch.Tensor) -> torch.Tensor:
        logp = self._sequence_logprobs_from_prompt(self.model.base_model, prompts_inputs, sampled_ids)

        adv = rewards - rewards.mean()
        pg_loss = -(adv.detach() * logp).mean()

        if self.ref_model is None or self.kl_beta <= 0:
            return pg_loss

        with torch.no_grad():
            ref_logp = self._sequence_logprobs_from_prompt(self.ref_model, prompts_inputs, sampled_ids)
        kl = logp - ref_logp
        return pg_loss + self.kl_beta * kl.mean()
