from __future__ import annotations
import torch
import torch.nn as nn


class QueryBuilder(nn.Module):

    def __init__(self, hidden_size: int, query_len: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.0, ff_mult: int = 4, max_len: int = 30720):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_len = query_len
        self.q_init = nn.Parameter(torch.randn(query_len, hidden_size) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)

    def _ensure_pos_emb_len(self, target_len: int, device: torch.device, dtype: torch.dtype) -> None:
        cur_len = self.pos_emb.size(1)
        if target_len <= cur_len:
            return

        new_len = max(target_len, cur_len * 2)
        new_pos = torch.randn(1, new_len, self.hidden_size, device=device, dtype=dtype) * 0.02
        with torch.no_grad():
            new_pos[:, :cur_len, :] = self.pos_emb.detach().to(device=device, dtype=dtype)
        self.pos_emb = nn.Parameter(new_pos)

    def forward(self, H: torch.Tensor, H_key_padding_mask: torch.BoolTensor | None = None) -> torch.Tensor:

        B, L, D = H.shape
        model_dtype = self.q_init.dtype
        H_for_query = H.to(dtype=model_dtype)
        q = self.q_init.to(device=H.device).unsqueeze(0).expand(B, -1, -1)  # (B,K,D)
        x = torch.cat([H_for_query, q], dim=1)  # (B, L+K, D)
        self._ensure_pos_emb_len(x.size(1), device=H.device, dtype=model_dtype)
        x = x + self.pos_emb[:, : x.size(1), :].to(device=H.device, dtype=model_dtype)


        total = L + self.query_len
        attn_mask = torch.zeros(total, total, device=H.device, dtype=torch.float32)
        attn_mask[:L, L:] = float("-inf")

        src_kpm = None
        if H_key_padding_mask is not None:
            q_kpm = torch.zeros((B, self.query_len), device=H.device, dtype=torch.bool)
            src_kpm = torch.cat([H_key_padding_mask, q_kpm], dim=1)
        out = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_kpm)
        out = self.ln(out)
        Q = out[:, -self.query_len:, :].to(dtype=H.dtype)
        return Q
