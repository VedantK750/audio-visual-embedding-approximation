"""Downstream linear-probe classification.

Trains the SAME linear classifier on top of two embedding spaces and compares
test accuracy for predicting VGGSound class labels:

    1. Teacher query embeddings  (1 frame + audio, via ImageBind)
    2. Student embeddings        (CLIP middle frame + AST audio -> trained MLP)

Embeddings are computed in real time (no precomputed cache needed), reusing the
extraction functions from evaluate_recall.py so encoding stays consistent.


Choose the classifier with the env var PROBE_CLF=logreg (default) or PROBE_CLF=svc.
"""

import os
import numpy as np
import torch

from imagebind.models import imagebind_model

from NaiveLateMLPFusion.model.naive_implementation import NaiveLateFusionMLP
from train_mlp import CLIPEncoder, ASTEncoder

from evaluate_recall import (
    gather_test_records,        # accepts split="train"/"test"
    extract_teacher_queries,
    extract_mlp_student_queries,
)


def _l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def make_classifier(kind="logreg", seed=42):
    """The SAME classifier is used for both spaces so the comparison is fair."""
    if kind == "svc":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=1.0, random_state=seed)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)


def linear_probe(train_X, train_y, test_X, test_y, name, kind="logreg"):
    """Fit the classifier on train embeddings, score on test embeddings.
    Embeddings are L2-normalized first (same geometry the model is trained in)."""

    from sklearn.metrics import accuracy_score, f1_score

    train_X = _l2_normalize(train_X)
    test_X = _l2_normalize(test_X)

    clf = make_classifier(kind)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    acc = float(accuracy_score(test_y, pred))
    macro_f1 = float(f1_score(test_y, pred, average="macro"))

    print(f"\n=== {name} : LINEAR PROBE ({kind}) ===")
    print(f"Top-1 accuracy = {acc * 100:.2f}%   macro-F1 = {macro_f1:.4f}")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root_dir = "/home/uasdtu/audio_visual_embedding_approximation/processed_vggsound"
    student_ckpt = "best_mlp_epoch19.pth"
    classifier_kind = os.environ.get("PROBE_CLF", "logreg")  # "logreg" or "svc"

    # Train split fits the classifier; test split scores it.
    train_records = gather_test_records(root_dir, split="train")
    test_records = gather_test_records(root_dir, split="test")

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

    # ---- Student embeddings ----
    if os.path.exists(student_ckpt):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(student_ckpt, map_location=device))
        mlp.eval()

        student_train = extract_mlp_student_queries(train_records, clip_encoder, ast_encoder, mlp, device).numpy()
        student_test = extract_mlp_student_queries(test_records, clip_encoder, ast_encoder, mlp, device).numpy()
        results["student"] = linear_probe(
            student_train, train_y, student_test, test_y, "STUDENT", classifier_kind
        )
    else:
        print(f"\n[skip] student checkpoint not found: {student_ckpt}")
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
