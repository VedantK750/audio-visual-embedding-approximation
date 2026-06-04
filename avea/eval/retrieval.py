"""Retrieval metrics (model-agnostic): instance + semantic Recall@k and MedR."""

import random

import numpy as np
import torch


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
    match. Rank = position of the first same-class item."""

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
