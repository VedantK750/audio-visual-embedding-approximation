"""Downstream linear-probe classification.

Trains the SAME linear classifier on top of each embedding space and compares
test accuracy for predicting VGGSound class labels:

    1. Teacher query embeddings   (1 frame + audio, via ImageBind)
    2. MLP student embeddings     (CLIP middle frame + AST audio -> trained MLP)
    3. Transformer student        (precomputed SigLIP 2 + CLAP -> CrossAttentionStudent)
    4. Multi-token student        (precomputed SigLIP2/CLAP token seqs -> MultiTokenFusionTransformer)

Embeddings are computed in real time (the transformers read precomputed features).

NOTE: this is a *linear probe* (a classifier IS trained), not true zero-shot.
Choose the classifier with PROBE_CLF=logreg (default) or PROBE_CLF=svc.
"""

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
    extract_teacher_queries,
    extract_mlp_student_queries,
    extract_transformer_student_queries,
    extract_multitoken_student_queries,
)
from avea.eval.linear_probe import linear_probe

DATA_ROOT = "processed_vggsound"
MLP_CKPT = "checkpoints/mlp/best_mlp_epoch19.pth"
TRANSFORMER_CKPT = "checkpoints/transformer/best_transformer_epoch15.pth"
MULTITOKEN_CKPT = "checkpoints/multitoken/best_multitoken_epoch11.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_kind = os.environ.get("PROBE_CLF", "logreg")  # "logreg" or "svc"

    # Train split fits the classifier; test split scores it.
    train_records = gather_test_records(DATA_ROOT, split="train")
    test_records = gather_test_records(DATA_ROOT, split="test")

    train_y = [r["label"] for r in train_records]
    test_y = [r["label"] for r in test_records]

    results = {}

    # ---- Teacher query embeddings (1 frame + audio) ----
    imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model_instance.eval()
    imagebind_model_instance.to(device)

    teacher_train = extract_teacher_queries(train_records, imagebind_model_instance, device).numpy()
    teacher_test = extract_teacher_queries(test_records, imagebind_model_instance, device).numpy()
    results["teacher"] = linear_probe(
        teacher_train, train_y, teacher_test, test_y, "TEACHER QUERY", classifier_kind
    )

    # ---- MLP student embeddings ----
    if os.path.exists(MLP_CKPT):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
        mlp.eval()

        student_train = extract_mlp_student_queries(train_records, clip_encoder, ast_encoder, mlp, device).numpy()
        student_test = extract_mlp_student_queries(test_records, clip_encoder, ast_encoder, mlp, device).numpy()
        results["mlp_student"] = linear_probe(
            student_train, train_y, student_test, test_y, "MLP STUDENT", classifier_kind
        )
    else:
        print(f"\n[skip] MLP student checkpoint not found: {MLP_CKPT}")

    # ---- Transformer student embeddings (precomputed SigLIP2 + CLAP) ----
    siglip_dir = os.path.join(DATA_ROOT, "train", "siglip2_embeddings")
    if not os.path.exists(TRANSFORMER_CKPT):
        print(f"\n[skip] transformer checkpoint not found: {TRANSFORMER_CKPT}")
    elif not os.path.isdir(siglip_dir):
        print(f"\n[skip] precomputed features missing: {siglip_dir} "
              f"(run scripts/precompute_features.py)")
    else:
        transformer = CrossAttentionStudent().to(device)
        transformer.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location=device))
        transformer.eval()

        tr_train = extract_transformer_student_queries(train_records, transformer, device).numpy()
        tr_test = extract_transformer_student_queries(test_records, transformer, device).numpy()
        results["transformer_student"] = linear_probe(
            tr_train, train_y, tr_test, test_y, "TRANSFORMER STUDENT", classifier_kind
        )

    # ---- Multi-token student embeddings (precomputed SigLIP2/CLAP token seqs) ----
    tokens_dir = os.path.join(DATA_ROOT, "train", "siglip2_tokens")
    if not os.path.exists(MULTITOKEN_CKPT):
        print(f"\n[skip] multi-token checkpoint not found: {MULTITOKEN_CKPT}")
    elif not os.path.isdir(tokens_dir):
        print(f"\n[skip] precomputed token features missing: {tokens_dir} "
              f"(run scripts/precompute_tokens.py)")
    else:
        multitoken = MultiTokenFusionTransformer().to(device)
        multitoken.load_state_dict(torch.load(MULTITOKEN_CKPT, map_location=device))
        multitoken.eval()

        mt_train = extract_multitoken_student_queries(train_records, multitoken, device).numpy()
        mt_test = extract_multitoken_student_queries(test_records, multitoken, device).numpy()
        results["multitoken_student"] = linear_probe(
            mt_train, train_y, mt_test, test_y, "MULTITOKEN STUDENT", classifier_kind
        )

    # ---- Compare ----
    print("\n================ DOWNSTREAM SUMMARY ================")
    for model_name, res in results.items():
        print(f"{model_name.upper():22s} acc={res['accuracy'] * 100:.2f}%  macro-F1={res['macro_f1']:.4f}")

    ta = results.get("teacher", {}).get("accuracy")
    if ta is not None:
        for student_key in ("mlp_student", "transformer_student", "multitoken_student"):
            sa = results.get(student_key, {}).get("accuracy")
            if sa is None:
                continue
            verdict = "MORE" if sa > ta else "NOT more"
            print(f"\n{student_key}: {sa * 100:.2f}%  vs  Teacher {ta * 100:.2f}%  "
                  f"=> carries {verdict} class info than the teacher query space.")


if __name__ == "__main__":
    main()
