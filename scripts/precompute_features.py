"""Precompute & cache student-encoder embeddings to disk.

New student encoders (both frozen):
    Vision : SigLIP 2  (google/siglip2-base-patch16-224)   -> 768-d
    Audio  : LAION-CLAP (laion/clap-htsat-unfused)          -> 512-d

Because the encoders are frozen, each clip's embedding is identical every epoch,
so we compute them ONCE here and save them. Training then loads tiny vectors
instead of decoding frames/resampling audio and running ViT/CLAP every epoch.

Output layout (mirrors the existing teacher_embeddings/):
    processed_vggsound/<split>/siglip2_embeddings/<label>/<clip_id>.npy   # [768]
    processed_vggsound/<split>/clap_embeddings/<label>/<clip_id>.npy      # [512]

Embeddings are saved UNFUSED (vision and audio separate) so you can change the
fusion/MLP later without recomputing. Re-running skips clips already saved
(delete the dirs or set FORCE=1 to recompute).

Run:  python3 precompute_student_embeddings.py
Env:  FRAME_MODE=middle|mean   FORCE=1   SPLITS="train test"
"""

import os
import glob

import numpy as np
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor


ROOT_DIR = "/home/uasdtu/audio_visual_embedding_approximation/processed_vggsound"
SIGLIP_ID = "google/siglip2-base-patch16-224"
CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000  # CLAP's feature extractor expects 48 kHz; source audio is 16 kHz

FRAME_MODE = os.environ.get("FRAME_MODE", "middle")   # "middle" or "mean"
FORCE = os.environ.get("FORCE", "0") == "1"
SPLITS = os.environ.get("SPLITS", "train test").split()


def gather_records(root_dir, split):
    """One record per clip that has frames + audio + teacher embedding."""
    teacher_root = os.path.join(root_dir, split, "teacher_embeddings")
    records = []
    for label in sorted(os.listdir(teacher_root)):
        label_dir = os.path.join(teacher_root, label)
        if not os.path.isdir(label_dir):
            continue
        for file in sorted(os.listdir(label_dir)):
            if not file.endswith(".npy"):
                continue
            clip_id = file.replace(".npy", "")
            frame_dir = os.path.join(root_dir, split, "frames", label, clip_id)
            audio_path = os.path.join(root_dir, split, "audio", label, f"{clip_id}.wav")
            frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            if len(frame_paths) == 0 or not os.path.exists(audio_path):
                continue
            records.append({
                "label": label,
                "clip_id": clip_id,
                "frame_paths": frame_paths,
                "audio_path": audio_path,
            })
    print(f"[{split}] {len(records)} clips")
    return records


class SigLIP2Vision:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(SIGLIP_ID)
        self.model = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, frame_paths):
        if FRAME_MODE == "mean":
            frames = [Image.open(p).convert("RGB") for p in frame_paths]
        else:  # middle frame only (matches the original student pipeline)
            frames = [Image.open(frame_paths[len(frame_paths) // 2]).convert("RGB")]

        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)   # [num_frames, 768]
        return feats.mean(dim=0).cpu().numpy()            # [768]


class CLAPAudio:
    def __init__(self, device):
        self.device = device
        self.processor = ClapProcessor.from_pretrained(CLAP_ID)
        self.model = ClapModel.from_pretrained(CLAP_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)        # [C, N]
        waveform = waveform.mean(dim=0)                   # mono [N]
        if sr != CLAP_SR:
            waveform = torchaudio.functional.resample(waveform, sr, CLAP_SR)

        inputs = self.processor(
            audios=waveform.numpy(),
            sampling_rate=CLAP_SR,
            return_tensors="pt",
        ).to(self.device)
        feats = self.model.get_audio_features(**inputs)   # [1, 512] (already L2-normalized)
        return feats.squeeze(0).cpu().numpy()             # [512]


def out_path(root_dir, split, kind, label, clip_id):
    d = os.path.join(root_dir, split, kind, label)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{clip_id}.npy")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  FRAME_MODE={FRAME_MODE}  FORCE={FORCE}  SPLITS={SPLITS}")

    vision = SigLIP2Vision(device)
    audio = CLAPAudio(device)

    for split in SPLITS:
        records = gather_records(ROOT_DIR, split)
        n_done = n_skipped = 0

        for r in tqdm(records, desc=f"encode[{split}]"):
            v_path = out_path(ROOT_DIR, split, "siglip2_embeddings", r["label"], r["clip_id"])
            a_path = out_path(ROOT_DIR, split, "clap_embeddings", r["label"], r["clip_id"])

            if not FORCE and os.path.exists(v_path) and os.path.exists(a_path):
                n_skipped += 1
                continue

            np.save(v_path, vision.encode(r["frame_paths"]).astype(np.float32))
            np.save(a_path, audio.encode(r["audio_path"]).astype(np.float32))
            n_done += 1

        print(f"[{split}] saved {n_done}, skipped {n_skipped} (already cached)")

    print("\nDone. New student_dim = 768 (SigLIP2) + 512 (CLAP) = 1280")


if __name__ == "__main__":
    main()
