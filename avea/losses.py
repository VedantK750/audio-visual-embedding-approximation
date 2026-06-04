"""Distillation losses for the student models."""

import torch.nn as nn


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
