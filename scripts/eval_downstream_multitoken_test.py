"""Quick downstream linear-probe eval for ONLY the multi-token student."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from avea.models.multitoken_fusion import MultiTokenFusionTransformer
from avea.eval.extraction import gather_test_records, extract_multitoken_student_queries
from avea.eval.linear_probe import linear_probe

DATA_ROOT = "processed_vggsound"
CKPT = "checkpoints/multitoken/best_multitoken_epoch29_label_aware_infonce.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_records = gather_test_records(DATA_ROOT, split="train")
test_records = gather_test_records(DATA_ROOT, split="test")
train_y = [r["label"] for r in train_records]
test_y = [r["label"] for r in test_records]

model = MultiTokenFusionTransformer().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

train_X = extract_multitoken_student_queries(train_records, model, device).numpy()
test_X = extract_multitoken_student_queries(test_records, model, device).numpy()

linear_probe(train_X, train_y, test_X, test_y, "MULTITOKEN", os.environ.get("PROBE_CLF", "logreg"))
