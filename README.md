# Audio–Visual Embedding Approximation

Distill **ImageBind** audio-visual embeddings into lightweight student models, then
benchmark how well the students approximate the teacher.

A frozen **ImageBind** teacher produces a 2048-d embedding per clip (vision + audio,
vision averaged over 5 frames). The goal is to reproduce that "privileged" embedding
from cheaper, frozen encoders + a small trainable head (the *student*), and to measure
how viable the approximation is on downstream semantic tasks.

Two students are implemented:

| Student | Visual encoder | Audio encoder | Head | Input dim |
|---|---|---|---|---|
| **MLP** (`NaiveLateFusionMLP`) | CLIP ViT-B/32 (middle frame, 512) | AST (768) | late-fusion MLP | 1280 |
| **Transformer** (`CrossAttentionStudent`) | SigLIP 2 base (middle frame, 768) | LAION-CLAP (512) | bi-directional cross-attention | — (precomputed) |

Both regress the 2048-d teacher target using `HybridAlignmentLoss = α·MSE + β·(1 − cosine)`.

---

## Repository layout

```
avea/                          # importable library (no entry points)
├── encoders.py                # CLIPEncoder, ASTEncoder (frozen)
├── losses.py                  # HybridAlignmentLoss
├── plotting.py                # LivePlot (live training curve)
├── utils.py                   # l2_normalize
├── data/
│   ├── dataset.py             # VGGSoundDataset (loads frames + audio + teacher)
│   └── precomputed_dataset.py # PrecomputedEmbeddingDataset (cached SigLIP2 + CLAP)
├── models/
│   ├── mlp_fusion.py          # NaiveLateFusionMLP
│   └── transformer_fusion.py  # CrossAttentionStudent
└── eval/
    ├── extraction.py          # gather records, load gallery, extract teacher/student queries
    ├── retrieval.py           # instance + semantic Recall@k / MedR
    ├── cluster_alignment.py   # Silhouette, Davies-Bouldin, PCA/t-SNE, direct alignment
    └── linear_probe.py        # LogisticRegression / LinearSVC probe

scripts/                       # runnable entry points (run from repo root)
├── preprocess_videos.py       # download clips + extract frames + build teacher embeddings
├── precompute_features.py     # cache SigLIP 2 + CLAP features to disk
├── train_mlp.py               # train the MLP student
├── train_transformer.py       # train the transformer student
├── eval_recall.py             # retrieval benchmark (teacher + both students)
├── eval_downstream.py         # linear-probe classification
└── analyze_embeddings.py      # cluster quality + direct alignment plots

tools/                         # ad-hoc inspectors (not part of the pipeline)
checkpoints/{mlp,transformer}/ # saved model weights (gitignored)
outputs/                       # plots / histograms
third_party/ImageBind/         # vendored ImageBind clone (gitignored)
processed_vggsound/            # dataset (gitignored)
```

### Data layout (`processed_vggsound/<split>/`)

```
frames/<label>/<clip_id>/*.jpg          # 5 frames per clip
audio/<label>/<clip_id>.wav             # 16 kHz mono
teacher_embeddings/<label>/<clip_id>.npy   # [2048] ImageBind target (5-frame mean ‖ audio)
siglip2_embeddings/<label>/<clip_id>.npy   # [768]  precomputed SigLIP 2 (middle frame)
clap_embeddings/<label>/<clip_id>.npy      # [512]  precomputed CLAP (audio)
```

---

## Setup

Requires Python 3.13 with PyTorch (CUDA), and:

```bash
pip install 'transformers>=4.49,<5' torchaudio scikit-learn matplotlib tqdm pillow numpy pandas
```

`imagebind` is expected to be importable (installed in your environment); the
`third_party/ImageBind/` clone is kept for reference only and is not on the import path.

The `scripts/` use a 2-line bootstrap so `import avea` resolves — **just run them from
the repository root** so the relative `processed_vggsound` path is found.

---

## Pipeline

```bash
# 1. (one-off) build the dataset + ImageBind teacher embeddings
python3 scripts/preprocess_videos.py

# 2. (one-off) cache SigLIP 2 + CLAP features for the transformer student
python3 scripts/precompute_features.py
#    options: FRAME_MODE=middle|mean   FORCE=1   SPLITS="train test"

# 3. train the students  (checkpoints -> checkpoints/{mlp,transformer}/)
python3 scripts/train_mlp.py
python3 scripts/train_transformer.py
#    disable the live plot with: LIVE_PLOT=0 python3 scripts/train_mlp.py

# 4. evaluate
python3 scripts/eval_recall.py        # instance + semantic Recall@k / MedR, vs teacher
python3 scripts/eval_downstream.py    # linear probe (PROBE_CLF=logreg|svc)
python3 scripts/analyze_embeddings.py # cluster quality + PCA/t-SNE + alignment plots
```

---

## Evaluation metrics

- **Instance retrieval** — does a query embedding retrieve its *own* teacher target (the exact clip)? Recall@{1,5,10} + median rank.
- **Semantic retrieval** — does it retrieve a *same-class* clip (self excluded)? Recall@{1,5,10} + median rank.
- **Cluster quality** — Silhouette (higher = better) and Davies-Bouldin (lower = better) of the embedding space by class label, plus PCA/t-SNE scatter plots.
- **Direct alignment** — per-clip cosine between the student embedding and its teacher target (no gallery, no ranking).
- **Downstream linear probe** — train `LogisticRegression`/`LinearSVC` on embeddings, compare test accuracy + macro-F1 across spaces.

---

## Notes

- Checkpoints are referenced by fixed names in the eval scripts
  (`checkpoints/mlp/best_mlp_epoch19.pth`, `checkpoints/transformer/best_transformer_epoch15.pth`) —
  update those constants if you train new bests.
- The teacher *target* averages 5 frames, while the students see a single (middle)
  frame, so there is an intentional information asymmetry the students must approximate.
