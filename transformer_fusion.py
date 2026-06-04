import torch
import torch.nn as nn

class CrossAttentionStudent(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=512, d_model=1024, target_dim=2048, num_heads=8):
        super().__init__()
        """
        visual_dim: 768 (SigLIP 2 Base)
        audio_dim:  512 (LAION-CLAP)
        d_model:    1024 (Shared attention space)
        target_dim: 2048 (ImageBind full-video target space)
        """

        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        self.vis_queries_aud = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.aud_queries_vis = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        
        # We concatenate the two attended vectors, so the input is d_model * 2 (2048)
        self.final_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, target_dim)
        )


    def forward(self, visual_feat, audio_feat):
        # vis_feat: [B, 768] from SigLIP
        # aud_feat: [B, 512] from CLAP
        
        # Step 1: Project to d_model and add sequence dimension (Length = 1)
        # Shape becomes [B, 1, 1024]
        v = self.visual_proj(visual_feat).unsqueeze(1)
        a = self.audio_proj(audio_feat).unsqueeze(1)

        # Step 2: Bi-Directional Cross Attention
        # v_attn: What visual concepts need to listen for in the audio
        v_attn, _ = self.vis_queries_aud(query=v, key=a, value=a)
        
        # a_attn: What acoustic events need to look for in the frame
        a_attn, _ = self.aud_queries_vis(query=a, key=v, value=v)

        # Step 3: Add & Norm
        v_attn = self.norm_v(v + v_attn).squeeze(1)  # [B, 1024]
        a_attn = self.norm_a(a + a_attn).squeeze(1)  # [B, 1024]

        # step 4: Concatenate attended features and project to target space
        combined = torch.cat([v_attn, a_attn], dim=-1)
        output = self.final_projection(combined)  # [B, 2048]

        return output





