"""Distillation losses for the student models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAlignmentLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        """
        pred:   [Batch, 2048] - Student output
        target: [Batch, 2048] - Teacher ImageBind target
        """
        # MSE component
        loss_mse = self.mse(pred, target)

        # Cosine-distance component
        # (1 - cosine_similarity turns it into a 0..2 minimization objective)
        loss_cos = 1.0 - self.cosine(pred, target).mean()

        # Combined weighted loss
        total_loss = (self.alpha * loss_mse) + (self.beta * loss_cos)

        return total_loss, loss_mse, loss_cos


class HybridContrastiveLoss(nn.Module):
    """MSE + cosine-distance + a symmetric InfoNCE term.

    The MSE/cosine terms pull each prediction toward its own teacher target;
    the InfoNCE term additionally pushes it AWAY from other targets in the
    batch, which directly attacks weak instance-level retrieval.

    Returns (total_loss, components_dict) where components has keys
    {"mse", "cos", "nce"} so the trainer can log each term.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target, labels=None):
        """
        pred:   [Batch, D] - Student output
        target: [Batch, D] - Teacher ImageBind target
        labels: optional list/seq of class labels (len Batch). When given, the
                InfoNCE term EXCLUDES same-class off-diagonal pairs from the
                negatives, so it stops pushing same-class clips apart (helps
                semantic retrieval). The diagonal stays the positive.
        """
        loss_mse = self.mse(pred, target)
        loss_cos = 1.0 - self.cosine(pred, target).mean()

        # Symmetric InfoNCE over the batch (diagonal = positive pairs).
        s = F.normalize(pred, dim=-1)
        t = F.normalize(target, dim=-1)
        logits = (s @ t.t()) / self.temperature          # [B, B]
        B = pred.size(0)
        idx = torch.arange(B, device=pred.device)

        if labels is not None:
            uniq = {lab: i for i, lab in enumerate(dict.fromkeys(labels))}
            lab_ids = torch.tensor([uniq[lab] for lab in labels], device=pred.device)
            same = lab_ids[:, None] == lab_ids[None, :]   # [B, B] same-class
            same[idx, idx] = False                        # keep the diagonal positive
            logits = logits.masked_fill(same, float("-inf"))

        loss_nce = 0.5 * (
            F.cross_entropy(logits, idx)
            + F.cross_entropy(logits.t(), idx)
        )

        total_loss = (
            self.alpha * loss_mse
            + self.beta * loss_cos
            + self.gamma * loss_nce
        )

        return total_loss, {"mse": loss_mse, "cos": loss_cos, "nce": loss_nce}
