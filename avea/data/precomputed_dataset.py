"""Dataset that loads the precomputed SigLIP 2 + CLAP feature vectors.

Used by the transformer trainer so each epoch is just tiny tensor loads instead
of decoding frames / resampling audio and running the encoders.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedEmbeddingDataset(Dataset):
    """Loads the cached SigLIP 2 + CLAP feature vectors and the teacher target.
    Mirrors the teacher_embeddings/ layout used by the precompute script."""

    def __init__(self, root_dir, split):
        teacher_root = os.path.join(root_dir, split, "teacher_embeddings")
        siglip_root = os.path.join(root_dir, split, "siglip2_embeddings")
        clap_root = os.path.join(root_dir, split, "clap_embeddings")

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
                })

        print(f"Loaded {len(self.samples)} precomputed samples from {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "visual": torch.from_numpy(np.load(s["visual"])).float(),     # [768]
            "audio": torch.from_numpy(np.load(s["audio"])).float(),       # [512]
            "teacher": torch.from_numpy(np.load(s["teacher"])).float(),   # [2048]
        }
