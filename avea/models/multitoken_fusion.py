"""Multi-token multimodal fusion transformer student.

Unlike CrossAttentionStudent (which collapses each modality to a single token,
making attention a no-op), this model keeps a SEQUENCE per modality:

    visual_tokens : [B, Nv, 768]   (Nv per-frame SigLIP 2 embeddings)
    audio_tokens  : [B, Na, 512]   (Na CLAP embeddings over audio windows)

It builds one joint sequence [CLS, v_1..v_Nv, a_1..a_Na], adds learnable
positional + modality-type embeddings, runs a few standard self-attention
TransformerEncoder layers (so every token attends to every other token, across
modalities), and reads out the CLS token -> 2048-d ImageBind target.

A fully-zeroed modality (modality dropout at train time, or a single-modality
query at eval time) is handled gracefully: those positions still carry their
type/positional embeddings but contribute no content.
"""

import torch
import torch.nn as nn


class MultiTokenFusionTransformer(nn.Module):
    def __init__(
        self,
        visual_dim=768,
        audio_dim=512,
        d_model=512,
        target_dim=2048,
        num_heads=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        num_visual_tokens=5,
        num_audio_tokens=5,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.num_audio_tokens = num_audio_tokens

        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        seq_len = 1 + num_visual_tokens + num_audio_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Modality type ids: 0 = CLS, 1 = vision, 2 = audio.
        self.type_emb = nn.Embedding(3, d_model)
        type_ids = torch.tensor(
            [0] + [1] * num_visual_tokens + [2] * num_audio_tokens,
            dtype=torch.long,
        )
        self.register_buffer("type_ids", type_ids)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_dim),
        )

        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, visual_tokens, audio_tokens):
        # visual_tokens: [B, Nv, 768]   audio_tokens: [B, Na, 512]
        B = visual_tokens.size(0)

        v = self.visual_proj(visual_tokens)      # [B, Nv, d]
        a = self.audio_proj(audio_tokens)        # [B, Na, d]
        cls = self.cls.expand(B, -1, -1)         # [B, 1, d]

        x = torch.cat([cls, v, a], dim=1)        # [B, seq, d]
        x = x + self.pos + self.type_emb(self.type_ids).unsqueeze(0)

        x = self.encoder(x)                      # [B, seq, d]
        return self.head(x[:, 0])                # CLS token -> [B, target_dim]
