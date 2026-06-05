"""Precompute MULTI-TOKEN student features for the fusion transformer.

Unlike precompute_features.py (one pooled vector per modality), this saves a
short SEQUENCE per modality so the transformer has real tokens to attend over:

    siglip2_tokens/<label>/<clip_id>.npy   # [NUM_VISUAL_TOKENS, 768]  (one per frame)
    clap_tokens/<label>/<clip_id>.npy      # [NUM_AUDIO_TOKENS, 512]   (one per audio window)

Both encoders are frozen, so this is a one-off. Resumable (skips cached clips;
FORCE=1 to recompute).

Run:  python3 scripts/precompute_tokens.py
Env:  FORCE=1   SPLITS="train test"   NUM_AUDIO_TOKENS=5
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob

import numpy as np
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor, ClapModel, ClapProcessor

ROOT_DIR = "processed_vggsound"
SIGLIP_ID = "google/siglip2-base-patch16-224"
CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000

NUM_VISUAL_TOKENS = int(os.environ.get("NUM_VISUAL_TOKENS", "5"))  # one per frame
NUM_AUDIO_TOKENS = int(os.environ.get("NUM_AUDIO_TOKENS", "5"))    # audio windows
FORCE = os.environ.get("FORCE", "0") == "1"
SPLITS = os.environ.get("SPLITS", "train test").split()


def gather_records(root_dir, split):
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


def _fix_length(seq, n):
    """Pad (repeat last) or truncate a [T, D] array to exactly n rows."""
    if len(seq) == n:
        return seq
    if len(seq) > n:
        return seq[:n]
    pad = np.repeat(seq[-1:], n - len(seq), axis=0)
    return np.concatenate([seq, pad], axis=0)


class SigLIP2Tokens:
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(SIGLIP_ID)
        self.model = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, frame_paths):
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)        # [num_frames, 768]
        feats = feats.cpu().numpy()
        return _fix_length(feats, NUM_VISUAL_TOKENS).astype(np.float32)


class CLAPTokens:
    def __init__(self, device):
        self.device = device
        self.processor = ClapProcessor.from_pretrained(CLAP_ID)
        self.model = ClapModel.from_pretrained(CLAP_ID).to(device).eval()

    @torch.no_grad()
    def encode(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)             # [C, N]
        waveform = waveform.mean(dim=0)                        # mono [N]
        if sr != CLAP_SR:
            waveform = torchaudio.functional.resample(waveform, sr, CLAP_SR)
        waveform = waveform.numpy()

        windows = np.array_split(waveform, NUM_AUDIO_TOKENS)   # K temporal chunks
        toks = []
        for w in windows:
            inputs = self.processor(audio=w, sampling_rate=CLAP_SR, return_tensors="pt").to(self.device)
            feat = self.model.get_audio_features(**inputs)     # [1, 512]
            toks.append(feat.squeeze(0).cpu().numpy())
        return np.stack(toks, axis=0).astype(np.float32)       # [K, 512]


def out_path(root_dir, split, kind, label, clip_id):
    d = os.path.join(root_dir, split, kind, label)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{clip_id}.npy")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  Nv={NUM_VISUAL_TOKENS}  Na={NUM_AUDIO_TOKENS}  "
          f"FORCE={FORCE}  SPLITS={SPLITS}")

    vision = SigLIP2Tokens(device)
    audio = CLAPTokens(device)

    for split in SPLITS:
        records = gather_records(ROOT_DIR, split)
        n_done = n_skipped = 0

        for r in tqdm(records, desc=f"encode[{split}]"):
            v_path = out_path(ROOT_DIR, split, "siglip2_tokens", r["label"], r["clip_id"])
            a_path = out_path(ROOT_DIR, split, "clap_tokens", r["label"], r["clip_id"])

            if not FORCE and os.path.exists(v_path) and os.path.exists(a_path):
                n_skipped += 1
                continue

            np.save(v_path, vision.encode(r["frame_paths"]))
            np.save(a_path, audio.encode(r["audio_path"]))
            n_done += 1

        print(f"[{split}] saved {n_done}, skipped {n_skipped} (already cached)")

    print(f"\nDone. Token shapes: vision [{NUM_VISUAL_TOKENS}, 768], audio [{NUM_AUDIO_TOKENS}, 512]")


if __name__ == "__main__":
    main()
