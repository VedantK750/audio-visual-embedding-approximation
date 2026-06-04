"""Embedding-space analysis (separate from retrieval evaluation).

Computes, for the teacher and student embedding spaces:
  - Cluster quality by class label: Silhouette + Davies-Bouldin
  - PCA and t-SNE 2-D visualizations (saved as PNGs)
  - Direct alignment: per-clip cosine(query_i, teacher_target_i)

Data loading / query extraction is reused from evaluate_recall.py so this file
stays purely about analysis and never duplicates the encoding logic.
"""

import os
import numpy as np
import torch

from imagebind.models import imagebind_model

from NaiveLateMLPFusion.model.naive_implementation import NaiveLateFusionMLP
from train_mlp import CLIPEncoder, ASTEncoder

from evaluate_recall import (
    gather_test_records,
    load_gallery,
    extract_teacher_queries,
    extract_mlp_student_queries,
)


# ---------------------------------------------------------------------------
# Cluster quality: how semantically organized is each embedding space?
# (model-agnostic; works on teacher queries, student queries, or the gallery)
# ---------------------------------------------------------------------------

def _l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def compute_cluster_metrics(embeddings, labels, name=""):
    """Silhouette (higher = tighter, better-separated same-class clusters) and
    Davies-Bouldin (lower = better). Embeddings are L2-normalized first so the
    Euclidean geometry both metrics use matches the cosine space retrieval uses."""

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    X = _l2_normalize(embeddings)
    y = np.asarray(labels)

    if len(set(y.tolist())) < 2:
        print(f"[cluster] {name}: need >=2 classes, skipping")
        return None

    sil = float(silhouette_score(X, y))
    db = float(davies_bouldin_score(X, y))

    print(f"\n=== {name} : CLUSTER QUALITY (by class label) ===")
    print(f"Silhouette = {sil:.4f}  (higher is better)")
    print(f"Davies-Bouldin = {db:.4f}  (lower is better)")
    return {"silhouette": sil, "davies_bouldin": db}


def plot_embedding_space(embeddings, labels, name, save_path, method="pca",
                         max_points=2000, seed=42):
    """Reduce embeddings to 2-D (PCA or t-SNE) and scatter, colored by class.
    Saved to save_path. Headless-safe (uses the Agg backend explicitly)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = _l2_normalize(embeddings)
    y = np.asarray(labels)

    # Subsample for legibility / t-SNE speed.
    if len(X) > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_points, replace=False)
        X, y = X[idx], y[idx]

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(X) // 4))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=seed)

    XY = reducer.fit_transform(X)

    classes = sorted(set(y.tolist()))
    cmap = plt.get_cmap("tab20", max(len(classes), 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(XY[m, 0], XY[m, 1], s=12, color=cmap(i), alpha=0.7, label=str(c))
    ax.set_title(f"{name} embeddings ({method.upper()})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if len(classes) <= 25:  # legend only when it stays readable
        ax.legend(markerscale=1.5, fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    save_path = os.path.abspath(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")
    return save_path


def analyze_embedding_space(embeddings, labels, name, prefix):
    """Cluster metrics + PCA and t-SNE plots for one embedding space.
    Wrapped defensively so a plotting/metric hiccup never kills the run."""

    out = {}
    try:
        out["metrics"] = compute_cluster_metrics(embeddings, labels, name=name)
    except Exception as e:
        print(f"[cluster] metrics failed for {name}: {e}")
        out["metrics"] = None

    for method in ("pca", "tsne"):
        try:
            plot_embedding_space(embeddings, labels, name, f"{prefix}_{method}.png", method=method)
        except Exception as e:
            print(f"[plot] {method} failed for {name}: {e}")

    return out["metrics"]


# ---------------------------------------------------------------------------
# Direct alignment: how close is each query to ITS OWN privileged target?
# No gallery, no ranking, no retrieval -- just the diagonal cosine.
# ---------------------------------------------------------------------------

def compute_direct_alignment(Q, G, name=""):
    """For each clip i, cosine_similarity(query_i, target_i). Q and G must be
    index-aligned (same records order), so row i of each is the same clip."""

    Qn = Q / Q.norm(dim=1, keepdim=True)
    Gn = G / G.norm(dim=1, keepdim=True)
    cos = (Qn * Gn).sum(dim=1)            # [N] diagonal cosine only
    cos_np = cos.cpu().numpy()

    stats = {
        "mean": float(cos_np.mean()),
        "median": float(np.median(cos_np)),
        "std": float(cos_np.std()),
        "min": float(cos_np.min()),
        "max": float(cos_np.max()),
    }

    print(f"\n=== {name} : DIRECT ALIGNMENT (cosine to own teacher target) ===")
    print(f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
          f"std={stats['std']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")

    return {"per_clip": cos_np, "stats": stats}


def plot_alignment_hist(cos_np, name, save_path, bins=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cos_np, bins=bins, color="tab:blue", alpha=0.8)
    ax.axvline(float(cos_np.mean()), color="red", linestyle="--",
               label=f"mean={cos_np.mean():.3f}")
    ax.set_xlabel("cosine(query_i, teacher_target_i)")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: direct alignment to privileged target")
    ax.legend()
    fig.tight_layout()
    save_path = os.path.abspath(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")
    return save_path


def analyze_direct_alignment(Q, G, name, save_path):
    """Diagonal cosine stats + a distribution histogram, wrapped defensively."""
    result = compute_direct_alignment(Q, G, name=name)
    try:
        plot_alignment_hist(result["per_clip"], name, save_path)
    except Exception as e:
        print(f"[plot] alignment histogram failed for {name}: {e}")
    return result


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root_dir = "/home/uasdtu/audio_visual_embedding_approximation/processed_vggsound"
    student_ckpt = "best_mlp_epoch19.pth"

    # Shared data + gallery (built once, reused by teacher and student).
    records = gather_test_records(root_dir)
    G, gallery_labels, clip_ids = load_gallery(records)

    results = {}

    # ---- Teacher ----
    imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_model_instance.eval()
    imagebind_model_instance.to(device)

    Q_teacher = extract_teacher_queries(records, imagebind_model_instance, device)
    results["teacher"] = {
        "cluster": analyze_embedding_space(
            Q_teacher.numpy(), gallery_labels, "TEACHER", "clusters_teacher"
        ),
        "alignment": analyze_direct_alignment(
            Q_teacher, G, "TEACHER", "alignment_teacher_hist.png"
        ),
    }

    # ---- Student (trained MLP) ----
    if os.path.exists(student_ckpt):
        clip_encoder = CLIPEncoder().to(device)
        ast_encoder = ASTEncoder().to(device)
        clip_encoder.eval()
        ast_encoder.eval()

        mlp = NaiveLateFusionMLP().to(device)
        mlp.load_state_dict(torch.load(student_ckpt, map_location=device))
        mlp.eval()

        Q_student = extract_mlp_student_queries(records, clip_encoder, ast_encoder, mlp, device)
        results["student"] = {
            "cluster": analyze_embedding_space(
                Q_student.numpy(), gallery_labels, "STUDENT", "clusters_student"
            ),
            "alignment": analyze_direct_alignment(
                Q_student, G, "STUDENT", "alignment_student_hist.png"
            ),
        }
    else:
        print(f"\n[skip] student checkpoint not found: {student_ckpt}")

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
