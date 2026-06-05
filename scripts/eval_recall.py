"""Retrieval benchmark: instance + semantic Recall@k / MedR for teacher and
both students (MLP and transformer), all against the teacher gallery."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from imagebind.models import imagebind_model

from avea.encoders import CLIPEncoder, ASTEncoder
from avea.models.mlp_fusion import NaiveLateFusionMLP
from avea.models.transformer_fusion import CrossAttentionStudent
from avea.models.multitoken_fusion import MultiTokenFusionTransformer
from avea.eval.extraction import (
    gather_test_records,
    load_gallery,
    extract_teacher_queries,
    extract_mlp_student_queries,
    extract_transformer_student_queries,
    extract_multitoken_student_queries,
)
from avea.eval.retrieval import evaluate_retrieval

DATA_ROOT = "processed_vggsound"
MLP_CKPT = "checkpoints/mlp/best_mlp_epoch19.pth"
TRANSFORMER_CKPT = "checkpoints/transformer/best_transformer_epoch15.pth"
MULTITOKEN_CKPT = "checkpoints/multitoken/best_multitoken_epoch11.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shared data + gallery (built once, reused by every model).
    records = gather_test_records(DATA_ROOT)
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
    if os.path.exists(MLP_CKPT):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
        mlp.eval()

        Q_mlp = extract_mlp_student_queries(records, clip_encoder, ast_encoder, mlp, device)
        results["mlp_student"] = evaluate_retrieval(
            Q_mlp, G, gallery_labels, gallery_labels, clip_ids,
            name=f"MLP STUDENT ({MLP_CKPT})"
        )
    else:
        print(f"\n[skip] MLP student checkpoint not found: {MLP_CKPT}")

    # ---- Transformer student (precomputed SigLIP2 + CLAP -> CrossAttentionStudent) ----
    siglip_dir = os.path.join(DATA_ROOT, "test", "siglip2_embeddings")
    if not os.path.exists(TRANSFORMER_CKPT):
        print(f"\n[skip] transformer checkpoint not found: {TRANSFORMER_CKPT}")
    elif not os.path.isdir(siglip_dir):
        print(f"\n[skip] precomputed features missing: {siglip_dir} "
              f"(run scripts/precompute_features.py)")
    else:
        transformer = CrossAttentionStudent().to(device)
        transformer.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location=device))
        transformer.eval()

        Q_tr = extract_transformer_student_queries(records, transformer, device)
        results["transformer_student"] = evaluate_retrieval(
            Q_tr, G, gallery_labels, gallery_labels, clip_ids,
            name=f"TRANSFORMER STUDENT ({TRANSFORMER_CKPT})"
        )

    # ---- Multi-token student (precomputed SigLIP2/CLAP token seqs -> MultiTokenFusionTransformer) ----
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
        results["multitoken_student"] = evaluate_retrieval(
            Q_mt, G, gallery_labels, gallery_labels, clip_ids,
            name=f"MULTITOKEN STUDENT ({MULTITOKEN_CKPT})"
        )

    print("\n================ SUMMARY ================")
    for model_name, res in results.items():
        print(f"\n{model_name.upper()}")
        print(f"  instance: {res['instance']}")
        print(f"  semantic: {res['semantic']}")


if __name__ == "__main__":
    main()
