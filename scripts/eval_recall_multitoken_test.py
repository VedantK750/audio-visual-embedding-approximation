import torch

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from avea.eval.extraction import gather_test_records, load_gallery, extract_multitoken_student_queries
from avea.eval.retrieval import evaluate_retrieval
from avea.models.multitoken_fusion import MultiTokenFusionTransformer

device='cuda' if torch.cuda.is_available() else 'cpu'
records = gather_test_records('processed_vggsound')
G, labels, clip_ids = load_gallery(records)
m = MultiTokenFusionTransformer().to(device)
m.load_state_dict(torch.load('checkpoints/multitoken/best_multitoken_epoch28.pth', map_location=device))
m.eval()
Q = extract_multitoken_student_queries(records, m, device)
evaluate_retrieval(Q, G, labels, labels, clip_ids, name='MULTITOKEN ep28', show_samples=False)