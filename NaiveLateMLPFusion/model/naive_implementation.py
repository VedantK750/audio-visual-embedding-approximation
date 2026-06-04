import torch
import torch.nn as nn

class NaiveLateFusionMLP(nn.Module):
    def __init__(self, student_dim=1280, target_dim=2048, hidden_dim=1024):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, target_dim)
        )
        
    def forward(self, visual_feat, audio_feat):
        # visual_feat: [batch_size, 512]
        # audio_feat:  [batch_size, 768]
        x = torch.cat([visual_feat, audio_feat], dim=-1) # Output: [batch_size, 1280]
        return self.pipeline(x) # Output: [batch_size, 2048]
    

