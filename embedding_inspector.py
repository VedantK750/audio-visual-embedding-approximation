import numpy as np
import os

embeddings = []
paths = []

for root, _, files in os.walk(
    "processed_vggsound/test/teacher_embeddings"
):
    for f in files:
        if f.endswith(".npy"):

            path = os.path.join(root, f)

            embeddings.append(
                np.load(path)
            )

            paths.append(path)

embeddings = np.stack(embeddings)

print(embeddings.shape)

embeddings = embeddings / (
    np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    ) + 1e-8
)

sim = embeddings @ embeddings.T

print(sim.shape)

query_idx = 3

scores = sim[query_idx]

topk = np.argsort(-scores)[:5]

print("\nQUERY:")
print(paths[query_idx])

print("\nTOP RETRIEVALS:")

for idx in topk:

    print(
        f"{scores[idx]:.3f}",
        paths[idx]
    )