import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from avea.models.transformer_fusion import CrossAttentionStudent
from avea.eval.extraction import (
    gather_test_records,
    load_gallery,
    extract_teacher_queries,
    extract_transformer_student_queries,
)
from avea.eval.retrieval import evaluate_retrieval

DATA_ROOT = "processed_vggsound"
TRANSFORMER_CKPT = "checkpoints/transformer/best_transformer_epoch15.pth"

