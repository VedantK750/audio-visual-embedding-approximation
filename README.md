# Audio-Visual Approximation of Video Semantic Space

**AIMS DTU Research Intern 2026**

Distill **ImageBind** audio-visual embeddings into lightweight student models, then
benchmark how well the students approximate the teacher.

A frozen **ImageBind** teacher produces a 2048-d embedding per clip (vision + audio,
vision averaged over 5 frames). The goal is to reproduce that "privileged" embedding
from cheaper, frozen encoders plus a small trainable head (the *student*), and to
measure how viable the approximation is on downstream semantic tasks.

Three students are implemented (full methodology, results, and figures in
**[REPORT.md](REPORT.md)**):

| Student | Visual encoder | Audio encoder | Head |
|---|---|---|---|
| **MLP** (`NaiveLateFusionMLP`) | CLIP ViT-B/32 (middle frame, 512) | AST (768) | late-fusion MLP over a concat vector |
| **Cross-attention** (`CrossAttentionStudent`) | SigLIP 2 base (768) | LAION-CLAP (512) | bi-directional cross-attention (one token per modality) |
| **Multi-token** (`MultiTokenFusionTransformer`) | SigLIP 2 base, 5 frame tokens | LAION-CLAP, 5 audio-window tokens | ViT-style self-attention over `[CLS, v1..v5, a1..a5]` |

Students regress the 2048-d teacher target with a hybrid loss (MSE + cosine
distance, plus InfoNCE for the multi-token model).

---

## Repository layout

```
avea/                          # importable library (no entry points)
├── encoders.py                # CLIPEncoder, ASTEncoder (frozen)
├── losses.py                  # HybridAlignmentLoss, HybridContrastiveLoss (InfoNCE)
├── plotting.py                # LivePlot (live training curve)
├── utils.py                   # l2_normalize
├── data/
│   ├── dataset.py             # VGGSoundDataset (frames + audio + teacher)
│   ├── precomputed_dataset.py # pooled SigLIP2 + CLAP features
│   └── token_dataset.py       # multi-token SigLIP2 / CLAP sequences
├── models/
│   ├── mlp_fusion.py          # NaiveLateFusionMLP
│   ├── transformer_fusion.py  # CrossAttentionStudent
│   └── multitoken_fusion.py   # MultiTokenFusionTransformer
└── eval/
    ├── extraction.py          # gather records, load gallery, extract teacher/student queries
    ├── retrieval.py           # instance + semantic Recall@k / MedR
    ├── cluster_alignment.py   # Silhouette, Davies-Bouldin, t-SNE/PCA, direct alignment
    └── linear_probe.py        # LogisticRegression / LinearSVC probe

scripts/                       # runnable entry points (run from repo root)
├── preprocess_videos.py       # build dataset + ImageBind teacher embeddings
├── precompute_features.py     # cache pooled SigLIP 2 + CLAP features
├── precompute_tokens.py       # cache multi-token SigLIP 2 + CLAP sequences
├── train_mlp.py               # train the MLP student
├── train_transformer.py       # train the cross-attention student
├── train_multitoken.py        # train the multi-token student
├── eval_recall.py             # retrieval benchmark (teacher + students)
├── eval_downstream.py         # linear-probe classification
└── analyze_embeddings.py      # cluster quality + alignment + t-SNE plots

tools/                         # ad-hoc inspectors + footprint.py (params / FLOPs)
checkpoints/{mlp,transformer,multitoken}/   # saved weights (gitignored)
outputs/                       # plots / histograms
third_party/ImageBind/         # vendored ImageBind clone (gitignored)
processed_vggsound/            # dataset (gitignored)
```

### Data layout (`processed_vggsound/<split>/`)

```
frames/<label>/<clip_id>/*.jpg              # 5 frames per clip
audio/<label>/<clip_id>.wav                 # 16 kHz mono
teacher_embeddings/<label>/<clip_id>.npy    # [2048] ImageBind target (5-frame mean + audio)
siglip2_embeddings/<label>/<clip_id>.npy    # [768]  pooled SigLIP 2 (middle frame)
clap_embeddings/<label>/<clip_id>.npy       # [512]  pooled CLAP (audio)
siglip2_tokens/<label>/<clip_id>.npy        # [5, 768] per-frame SigLIP 2 tokens
clap_tokens/<label>/<clip_id>.npy           # [5, 512] per-window CLAP tokens
```

---

## Setup

Requires Python 3.13 with PyTorch (CUDA), and:

```bash
pip install 'transformers>=4.49,<5' torchaudio scikit-learn matplotlib tqdm pillow numpy pandas
```

`imagebind` is expected to be importable (installed in your environment); the
`third_party/ImageBind/` clone is kept for reference only and is not on the import path.

The `scripts/` use a 2-line bootstrap so `import avea` resolves, so **run them from
the repository root** so the relative `processed_vggsound` path is found.

---

## Pipeline

```bash
# 1. (one-off) build the dataset + ImageBind teacher embeddings
python3 scripts/preprocess_videos.py

# 2. (one-off) cache student encoder features
python3 scripts/precompute_features.py   # pooled, for the MLP / cross-attention students
python3 scripts/precompute_tokens.py     # token sequences, for the multi-token student

# 3. train the students  (checkpoints -> checkpoints/{mlp,transformer,multitoken}/)
python3 scripts/train_mlp.py
python3 scripts/train_transformer.py
python3 scripts/train_multitoken.py
#    disable the live plot with: LIVE_PLOT=0 python3 scripts/train_mlp.py

# 4. evaluate
python3 scripts/eval_recall.py        # instance + semantic Recall@k / MedR, vs teacher
python3 scripts/eval_downstream.py    # linear probe (PROBE_CLF=logreg|svc)
python3 scripts/analyze_embeddings.py # cluster quality + t-SNE + alignment plots
python3 tools/footprint.py            # params + FLOPs per pipeline
```

Checkpoint paths are set as constants at the top of the eval scripts; update them
if you train new bests.

---

## Results and analysis

See **[REPORT.md](REPORT.md)** for the full technical report: methodology, loss
definitions, evaluation metrics, result tables, computational footprint, and
figures.

In short: the simple **MLP student is the strongest** (instance R@1 31.5, semantic
R@1 81.5, downstream linear probe within 0.3% of the teacher) at 4 to 10x fewer
parameters than ImageBind. The cross-attention transformer was a failed design
(a single token per modality makes its attention a no-op); the multi-token,
ViT-style transformer fixes that and produces the most class-organized space, but
does not beat the MLP overall.

---

## Notes

- The teacher target averages 5 frames while the students see a single (middle)
  frame, so there is an intentional information asymmetry the students must
  approximate. A single frame already reconstructs the 5-frame target almost
  perfectly, so the real task is the encoder-to-ImageBind mapping, not temporal
  modeling (see REPORT.md, section 2.1).
