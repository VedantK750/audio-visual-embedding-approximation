"""Downstream linear-probe classification.

Trains the SAME linear classifier on top of two embedding spaces and compares
test accuracy for predicting VGGSound class labels:

    1. Teacher query embeddings  (1 frame + audio, via ImageBind)
    2. MLP student embeddings    (CLIP middle frame + AST audio -> trained MLP)

Embeddings are computed in real time (no precomputed cache needed).

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
from avea.eval.extraction import (
    gather_test_records,
    extract_teacher_queries,
    extract_mlp_student_queries,
)
from avea.eval.linear_probe import linear_probe

DATA_ROOT = "processed_vggsound"
MLP_CKPT = "checkpoints/mlp/best_mlp_epoch19.pth"


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
        results["student"] = linear_probe(
            student_train, train_y, student_test, test_y, "STUDENT", classifier_kind
        )
    else:
        print(f"\n[skip] MLP student checkpoint not found: {MLP_CKPT}")

    # ---- Compare ----
    print("\n================ DOWNSTREAM SUMMARY ================")
    for model_name, res in results.items():
        print(f"{model_name.upper():14s} acc={res['accuracy'] * 100:.2f}%  macro-F1={res['macro_f1']:.4f}")

    if "teacher" in results and "student" in results:
        ta = results["teacher"]["accuracy"]
        sa = results["student"]["accuracy"]
        print(f"\nStudent {sa * 100:.2f}%  vs  Teacher {ta * 100:.2f}%")
        if sa > ta:
            print("=> Student representation carries MORE class information (for this probe).")
        else:
            print("=> Student does NOT beat the teacher query space on this probe.")


if __name__ == "__main__":
    main()
