"""Data loading + query extraction, shared by every evaluation script.

The `records` list (one entry per clip, in a fixed order) is the single source
of truth for ordering, so every model's queries stay index-aligned with the
teacher gallery.
"""

import os
import glob

import numpy as np
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

from imagebind import data
from imagebind.models.imagebind_model import ModalityType


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def gather_test_records(root_dir, split="test"):
    """Walk the given split and return one record per clip that has all of
    frames + audio + teacher embedding."""

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
            teacher_path = os.path.join(label_dir, file)

            frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            if len(frame_paths) == 0 or not os.path.exists(audio_path):
                continue

            records.append({
                "label": label,
                "clip_id": clip_id,
                "teacher_path": teacher_path,
                "frame_paths": frame_paths,
                "audio_path": audio_path,
            })

    print(f"Found {len(records)} {split} clips")
    return records


def load_gallery(records):
    """Gallery = the saved teacher embeddings. Returns G [N, D] plus the
    per-item labels and clip ids (index-aligned with the queries)."""

    G = []
    labels = []
    clip_ids = []
    for r in records:
        emb = torch.tensor(np.load(r["teacher_path"]), dtype=torch.float32)
        G.append(emb.unsqueeze(0))
        labels.append(r["label"])
        clip_ids.append(r["clip_id"])

    return torch.cat(G, dim=0), labels, clip_ids


# ---------------------------------------------------------------------------
# Query extraction (one function per model family)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_teacher_queries(records, imagebind_model_instance, device):
    """Re-encode each clip with ImageBind (vision + audio) -> [N, D] queries."""

    imagebind_model_instance.eval()

    queries = []
    for r in tqdm(records, desc="Teacher queries"):
        frame_paths = r["frame_paths"]
        middle_frame = frame_paths[len(frame_paths) // 2]

        query_inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([middle_frame], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([r["audio_path"]], device),
        }

        outputs = imagebind_model_instance(query_inputs)

        query_emb = torch.cat([
            outputs[ModalityType.VISION],
            outputs[ModalityType.AUDIO],
        ], dim=-1)

        queries.append(query_emb.cpu())

    return torch.cat(queries, dim=0)


@torch.no_grad()
def extract_mlp_student_queries(records, clip_encoder, ast_encoder, mlp, device, batch_size=16):
    """NaiveLateFusionMLP student: run the frozen CLIP/AST encoders on the raw
    frames/audio, then the trained MLP head, over each clip -> [N, 2048]."""

    clip_encoder.eval()
    ast_encoder.eval()
    mlp.eval()

    queries = []
    for start in tqdm(range(0, len(records), batch_size), desc="MLP student queries"):
        batch = records[start:start + batch_size]

        images = []
        waveforms = []
        for r in batch:
            images.append([Image.open(p).convert("RGB") for p in r["frame_paths"]])
            waveform, _ = torchaudio.load(r["audio_path"])
            waveforms.append(waveform)

        clip_emb = clip_encoder(images)
        ast_emb = ast_encoder(waveforms)
        pred = mlp(clip_emb, ast_emb)

        queries.append(pred.cpu())

    return torch.cat(queries, dim=0)


def _precomputed_feature_paths(teacher_path):
    """Map a clip's teacher_embeddings path to its precomputed student-feature
    paths (same <split>/<label>/<clip_id>.npy layout, different top dir)."""
    visual_path = teacher_path.replace("teacher_embeddings", "siglip2_embeddings")
    audio_path = teacher_path.replace("teacher_embeddings", "clap_embeddings")
    return visual_path, audio_path


@torch.no_grad()
def extract_transformer_student_queries(records, model, device, batch_size=64):
    """CrossAttentionStudent: load the PRECOMPUTED SigLIP 2 (768) + CLAP (512)
    feature vectors from disk and run the transformer head -> [N, 2048].
    Requires scripts/precompute_features.py to have been run for this split."""

    model.eval()

    queries = []
    for start in tqdm(range(0, len(records), batch_size), desc="Transformer student queries"):
        batch = records[start:start + batch_size]

        visual = []
        audio = []
        for r in batch:
            visual_path, audio_path = _precomputed_feature_paths(r["teacher_path"])
            visual.append(torch.from_numpy(np.load(visual_path)).float())
            audio.append(torch.from_numpy(np.load(audio_path)).float())

        visual = torch.stack(visual).to(device)
        audio = torch.stack(audio).to(device)
        pred = model(visual, audio)

        queries.append(pred.cpu())

    return torch.cat(queries, dim=0)
