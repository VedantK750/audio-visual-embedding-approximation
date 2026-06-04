import numpy as np
import torch
import glob
import os
import random

from PIL import Image
import torchaudio

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from tqdm import tqdm

# Student-side encoders + head. Reused exactly as in training so the student
# embeddings are computed consistently with how the MLP was trained.
from NaiveLateMLPFusion.model.naive_implementation import NaiveLateFusionMLP
from train_mlp import CLIPEncoder, ASTEncoder
from transformer_fusion import CrossAttentionStudent


# ---------------------------------------------------------------------------
# Data loading (shared by every model under evaluation)
# ---------------------------------------------------------------------------

def gather_test_records(root_dir, split="test"):
    """Walk the given split and return one record per clip that has all of
    frames + audio + teacher embedding. Records are the single source of truth
    for ordering, so every model's queries stay index-aligned with the gallery."""

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
    Requires precompute_student_embeddings.py to have been run for this split."""

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


# ---------------------------------------------------------------------------
# Metrics (model-agnostic, separate instance vs semantic)
# ---------------------------------------------------------------------------

def compute_instance_metrics(similarity):
    """Instance retrieval: the only correct match for query i is gallery item i
    (the exact same clip). Returns Recall@{1,5,10} and median rank."""

    _, indices = similarity.sort(dim=-1, descending=True)

    n = similarity.shape[0]
    ranks = []
    for i in range(n):
        rank = (indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": float((ranks <= 1).mean() * 100),
        "R@5": float((ranks <= 5).mean() * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "MedR": float(np.median(ranks)),
    }


def compute_semantic_metrics(similarity, query_labels, gallery_labels, exclude_self=True):
    """Semantic retrieval: a retrieved item counts as correct if it shares the
    query's class label. By default the exact same clip (the diagonal) is
    excluded so this measures class-level retrieval beyond the trivial self
    match. Rank = position of the first same-class item. Returns Recall@{1,5,10}
    and median rank of the first correct-class hit."""

    sim = similarity.clone()
    n = sim.shape[0]
    if exclude_self:
        sim[range(n), range(n)] = float("-inf")

    _, indices = sim.sort(dim=-1, descending=True)
    indices = indices.cpu().numpy()

    gallery_labels = np.asarray(gallery_labels)
    query_labels = np.asarray(query_labels)

    ranks = []
    for i in range(n):
        ranked = gallery_labels[indices[i]]
        same = np.where(ranked == query_labels[i])[0]
        ranks.append(int(same[0]) + 1 if len(same) else n + 1)

    ranks = np.array(ranks)
    return {
        "R@1": float((ranks <= 1).mean() * 100),
        "R@5": float((ranks <= 5).mean() * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "MedR": float(np.median(ranks)),
    }


def _print_metric_block(title, m):
    print(f"\n=== {title} ===")
    print(f"R@1={m['R@1']:.2f}  R@5={m['R@5']:.2f}  R@10={m['R@10']:.2f}  MedR={m['MedR']}")


def print_sample_retrievals(similarity, labels, clip_ids, top_k=6, n_samples=50, seed=42):
    rng = random.Random(seed)
    sampled = rng.sample(range(similarity.shape[0]), min(n_samples, similarity.shape[0]))

    for q in sampled:
        scores, idxs = torch.topk(similarity[q], k=top_k)

        print("=" * 80)
        print(f"Query Clip : {clip_ids[q]}")
        print(f"Query Label: {labels[q]}")
        print()

        for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
            print(
                f"{rank:2d}. "
                f"{labels[idx]:30s} "
                f"{score.item():.4f} "
                f"({clip_ids[idx]})"
            )
        print()


# ---------------------------------------------------------------------------
# Generic evaluation: feed any queries Q against gallery G
# ---------------------------------------------------------------------------

def evaluate_retrieval(Q, G, query_labels, gallery_labels, clip_ids, name, show_samples=True):
    """Normalize, build the cosine-similarity matrix, then report BOTH instance
    and semantic retrieval. Works for the teacher or any student because it only
    depends on the embeddings + labels."""

    Q = Q / Q.norm(dim=1, keepdim=True)
    G = G / G.norm(dim=1, keepdim=True)

    print(f"\n[{name}] Q={tuple(Q.shape)}  G={tuple(G.shape)}")
    similarity = torch.mm(Q, G.t())

    if show_samples:
        print_sample_retrievals(similarity, gallery_labels, clip_ids)

    instance = compute_instance_metrics(similarity)
    semantic = compute_semantic_metrics(similarity, query_labels, gallery_labels)

    _print_metric_block(f"{name} : INSTANCE RETRIEVAL", instance)
    _print_metric_block(f"{name} : SEMANTIC RETRIEVAL (same-class, self excluded)", semantic)

    return {"instance": instance, "semantic": semantic}


# ---------------------------------------------------------------------------
# Thin wrapper kept for backwards compatibility with the old entry point.
# ---------------------------------------------------------------------------

def evaluate_teacher_instance_level(root_dir, imagebind_model_instance, device):
    records = gather_test_records(root_dir)
    G, gallery_labels, clip_ids = load_gallery(records)
    Q = extract_teacher_queries(records, imagebind_model_instance, device)
    return evaluate_retrieval(Q, G, gallery_labels, gallery_labels, clip_ids, name="TEACHER")


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root_dir = "/home/uasdtu/audio_visual_embedding_approximation/processed_vggsound"
    mlp_ckpt = "best_mlp_epoch19.pth"
    transformer_ckpt = "best_transformer_epoch15.pth"

    # Shared data + gallery (built once, reused by every model).
    records = gather_test_records(root_dir)
    G, gallery_labels, clip_ids = load_gallery(records)

    results = {}

    # ---- Teacher baseline (upper bound) ----
    imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model_instance.eval()
    imagebind_model_instance.to(device)

    Q_teacher = extract_teacher_queries(records, imagebind_model_instance, device)
    results["teacher"] = evaluate_retrieval(
        Q_teacher, G, gallery_labels, gallery_labels, clip_ids, name="TEACHER"
    )

    # ---- MLP student (CLIP + AST -> NaiveLateFusionMLP) ----
    if os.path.exists(mlp_ckpt):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(mlp_ckpt, map_location=device))
        mlp.eval()

        Q_mlp = extract_mlp_student_queries(records, clip_encoder, ast_encoder, mlp, device)
        results["mlp_student"] = evaluate_retrieval(
            Q_mlp, G, gallery_labels, gallery_labels, clip_ids,
            name=f"MLP STUDENT ({mlp_ckpt})"
        )
    else:
        print(f"\n[skip] MLP student checkpoint not found: {mlp_ckpt}")

    # ---- Transformer student (precomputed SigLIP2 + CLAP -> CrossAttentionStudent) ----
    siglip_dir = os.path.join(root_dir, "test", "siglip2_embeddings")
    if not os.path.exists(transformer_ckpt):
        print(f"\n[skip] transformer checkpoint not found: {transformer_ckpt}")
    elif not os.path.isdir(siglip_dir):
        print(f"\n[skip] precomputed features missing: {siglip_dir} "
              f"(run precompute_student_embeddings.py)")
    else:
        transformer = CrossAttentionStudent().to(device)
        transformer.load_state_dict(torch.load(transformer_ckpt, map_location=device))
        transformer.eval()

        Q_tr = extract_transformer_student_queries(records, transformer, device)
        results["transformer_student"] = evaluate_retrieval(
            Q_tr, G, gallery_labels, gallery_labels, clip_ids,
            name=f"TRANSFORMER STUDENT ({transformer_ckpt})"
        )

    print("\n================ SUMMARY ================")
    for model_name, res in results.items():
        print(f"\n{model_name.upper()}")
        print(f"  instance: {res['instance']}")
        print(f"  semantic: {res['semantic']}")


if __name__ == "__main__":
    main()
