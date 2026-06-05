"""Dataset of precomputed multi-token features for the fusion transformer.

Loads per-clip token sequences (multiple SigLIP 2 frame tokens + multiple CLAP
audio-window tokens) plus the 2048-d teacher target. Produced by
scripts/precompute_tokens.py and stored under:

    processed_vggsound/<split>/siglip2_tokens/<label>/<clip_id>.npy   # [Nv, 768]
    processed_vggsound/<split>/clap_tokens/<label>/<clip_id>.npy      # [Na, 512]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenSequenceDataset(Dataset):
    def __init__(self, root_dir, split):
        teacher_root = os.path.join(root_dir, split, "teacher_embeddings")
        siglip_root = os.path.join(root_dir, split, "siglip2_tokens")
        clap_root = os.path.join(root_dir, split, "clap_tokens")

        self.samples = []
        for label in sorted(os.listdir(teacher_root)):
            label_dir = os.path.join(teacher_root, label)
            if not os.path.isdir(label_dir):
                continue
            for file in sorted(os.listdir(label_dir)):
                if not file.endswith(".npy"):
                    continue
                visual_path = os.path.join(siglip_root, label, file)
                audio_path = os.path.join(clap_root, label, file)
                teacher_path = os.path.join(label_dir, file)
                if not (os.path.exists(visual_path) and os.path.exists(audio_path)):
                    continue
                self.samples.append({
                    "visual": visual_path,
                    "audio": audio_path,
                    "teacher": teacher_path,
                    "label": label,
                })

        print(f"Loaded {len(self.samples)} token-sequence samples from {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "visual": torch.from_numpy(np.load(s["visual"])).float(),     # [Nv, 768]
            "audio": torch.from_numpy(np.load(s["audio"])).float(),       # [Na, 512]
            "teacher": torch.from_numpy(np.load(s["teacher"])).float(),   # [2048]
            "label": s["label"],
        }
