"""Downstream linear-probe classification.

Trains the SAME linear classifier on top of an embedding space and scores it on
the test split, to measure how much linearly-decodable class information the
representation carries.
"""

from avea.utils import l2_normalize


def make_classifier(kind="logreg", seed=42):
    """The SAME classifier is used for every space so comparisons are fair."""
    if kind == "svc":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=1.0, random_state=seed)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)


def linear_probe(train_X, train_y, test_X, test_y, name, kind="logreg"):
    """Fit the classifier on train embeddings, score on test embeddings.
    Embeddings are L2-normalized first (same geometry the model is trained in)."""

    from sklearn.metrics import accuracy_score, f1_score

    train_X = l2_normalize(train_X)
    test_X = l2_normalize(test_X)

    clf = make_classifier(kind)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    acc = float(accuracy_score(test_y, pred))
    macro_f1 = float(f1_score(test_y, pred, average="macro"))

    print(f"\n=== {name} : LINEAR PROBE ({kind}) ===")
    print(f"Top-1 accuracy = {acc * 100:.2f}%   macro-F1 = {macro_f1:.4f}")
    return {"accuracy": acc, "macro_f1": macro_f1}
