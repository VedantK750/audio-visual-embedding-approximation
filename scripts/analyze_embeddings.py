"""Embedding-space analysis for the teacher, MLP, and multi-token student spaces:
cluster quality (Silhouette + Davies-Bouldin), PCA/t-SNE plots, and per-clip
direct alignment to the teacher target.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from imagebind.models import imagebind_model

from avea.encoders import CLIPEncoder, ASTEncoder
from avea.models.mlp_fusion import NaiveLateFusionMLP
from avea.models.multitoken_fusion import MultiTokenFusionTransformer
from avea.eval.extraction import (
    gather_test_records,
    load_gallery,
    extract_teacher_queries,
    extract_mlp_student_queries,
    extract_multitoken_student_queries,
)
from avea.eval.cluster_alignment import analyze_embedding_space, analyze_direct_alignment

DATA_ROOT = "processed_vggsound"
MLP_CKPT = "checkpoints/mlp/best_mlp_epoch19.pth"
MULTITOKEN_CKPT = "checkpoints/multitoken/best_multitoken_epoch29_label_aware_infonce.pth"
OUT_DIR = "outputs"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Shared data + gallery (built once, reused by teacher and student).
    records = gather_test_records(DATA_ROOT)
    G, gallery_labels, clip_ids = load_gallery(records)

    results = {}

    # ---- Teacher ----
    imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model_instance.eval()
    imagebind_model_instance.to(device)

    Q_teacher = extract_teacher_queries(records, imagebind_model_instance, device)
    results["teacher"] = {
        "cluster": analyze_embedding_space(
            Q_teacher.numpy(), gallery_labels, "TEACHER", f"{OUT_DIR}/clusters_teacher"
        ),
        "alignment": analyze_direct_alignment(
            Q_teacher, G, "TEACHER", f"{OUT_DIR}/alignment_teacher_hist.png"
        ),
    }

    # ---- Student (trained MLP) ----
    if os.path.exists(MLP_CKPT):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
        mlp.eval()

        Q_student = extract_mlp_student_queries(records, clip_encoder, ast_encoder, mlp, device)
        results["student"] = {
            "cluster": analyze_embedding_space(
                Q_student.numpy(), gallery_labels, "STUDENT", f"{OUT_DIR}/clusters_student"
            ),
            "alignment": analyze_direct_alignment(
                Q_student, G, "STUDENT", f"{OUT_DIR}/alignment_student_hist.png"
            ),
        }
    else:
        print(f"\n[skip] MLP student checkpoint not found: {MLP_CKPT}")

    # ---- Multi-token student (precomputed SigLIP2/CLAP token seqs) ----
    tokens_dir = os.path.join(DATA_ROOT, "test", "siglip2_tokens")
    if not os.path.exists(MULTITOKEN_CKPT):
        print(f"\n[skip] multi-token checkpoint not found: {MULTITOKEN_CKPT}")
    elif not os.path.isdir(tokens_dir):
        print(f"\n[skip] precomputed token features missing: {tokens_dir} "
              f"(run scripts/precompute_tokens.py)")
    else:
        multitoken = MultiTokenFusionTransformer().to(device)
        multitoken.load_state_dict(torch.load(MULTITOKEN_CKPT, map_location=device))
        multitoken.eval()

        Q_mt = extract_multitoken_student_queries(records, multitoken, device)
        results["multitoken_student"] = {
            "cluster": analyze_embedding_space(
                Q_mt.numpy(), gallery_labels, "MULTITOKEN", f"{OUT_DIR}/clusters_multitoken"
            ),
            "alignment": analyze_direct_alignment(
                Q_mt, G, "MULTITOKEN", f"{OUT_DIR}/alignment_multitoken_hist.png"
            ),
        }

    print("\n================ SUMMARY ================")
    for model_name, res in results.items():
        print(f"\n{model_name.upper()}")
        if res.get("cluster"):
            print(f"  cluster : {res['cluster']}")
        if res.get("alignment"):
            print(f"  align   : {res['alignment']['stats']}")

    # Direct test of the hypothesis: is the student space more semantically organized?
    t = results.get("teacher", {}).get("cluster")
    s = results.get("student", {}).get("cluster")
    if t and s:
        print("\n--- Cluster quality: STUDENT vs TEACHER ---")
        print(f"Silhouette   : student {s['silhouette']:.4f}  vs  teacher {t['silhouette']:.4f}")
        print(f"Davies-Bouldin: student {s['davies_bouldin']:.4f}  vs  teacher {t['davies_bouldin']:.4f}")
        tighter = s["silhouette"] > t["silhouette"] and s["davies_bouldin"] < t["davies_bouldin"]
        if tighter:
            print("=> Student forms tighter same-class clusters: more semantically organized.")
        else:
            print("=> Student does NOT clearly beat the teacher on both metrics.")


if __name__ == "__main__":
    main()
