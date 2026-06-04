"""Small shared helpers."""

import numpy as np


def l2_normalize(X):
    """Row-wise L2 normalization for a [N, D] array (safe on zero rows)."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
