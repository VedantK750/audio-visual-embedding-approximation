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

## Findings & analysis

A running lab-notebook of what I tried, why, and how I read the results. All
retrieval numbers are on the VGGSound **test** split (628 clips). Unless stated
otherwise, the gallery is the *privileged* teacher embedding — ImageBind run on
**5 frames (mean-pooled) ‖ audio**, 2048-d — and everything is compared with
cosine similarity.

### 1. First, is the teacher target even hard to hit?

Before training any student I wanted to know what I was aiming at. The teacher
target pools 5 frames, but encoding 5 frames per clip at train time is expensive,
so I asked: *how much do those extra frames actually buy?* I queried the gallery
with a much cheaper representation — **a single middle frame + audio** through
ImageBind — and the result was almost suspicious:

- Middle frame + audio → **R@1 ≈ 100%**
- Middle frame, **vision only (no audio)** → **R@1 ≈ 99.7%**

So a single representative frame already reconstructs the 5-frame teacher
embedding almost perfectly for this dataset. Temporal aggregation over 5 frames
adds very little discriminative information in ImageBind's space for VGGSound.
That reframed the whole problem: **the student's job is not to approximate a
multi-frame representation — it's to learn a good mapping from the CLIP/AST (and
later SigLIP/CLAP) feature spaces into the ImageBind latent space** while keeping
semantic structure intact. The single-frame choice for the students is therefore
fair, not a handicap.

### 2. The MLP student, and the asymmetry that makes the numbers make sense

It's worth being explicit about who plays which role, because the evaluation is
deliberately asymmetric:

```
Gallery (ground truth):   5 frames + audio  → ImageBind → 2048-d   (privileged target space)
Teacher query:            1 frame  + audio  → ImageBind → 2048-d
Student query:            CLIP + AST        → MLP       → 2048-d
```

Both queries are scored against the **same** ImageBind gallery. The student only
knows *where to land* because it was trained to mimic the 5-frame+audio teacher —
ImageBind is still doing the heavy lifting; the student is learning to point into
its space.

| Model (query) | Inst R@1 | Inst R@5 | Inst R@10 | Inst MedR | Sem R@1 | Sem R@5 | Sem R@10 | Sem MedR |
|---|---|---|---|---|---|---|---|---|
| Teacher (1 frame + audio) | **100.0** | 100.0 | 100.0 | 1 | 80.10 | 94.43 | 96.66 | 1 |
| MLP student (CLIP + AST)  | 31.53 | 63.85 | 81.05 | 3 | **81.53** | 93.47 | 96.97 | 1 |

The headline is the split: the student's **instance** retrieval collapses
(100 → 31.5 R@1), but its **semantic** retrieval actually *matches or slightly
beats* the teacher (80.1 → 81.5 R@1). My reading: the student is trained to
regress toward privileged targets across thousands of examples, so it learns a
**smoother** representation that throws away a lot of clip-specific detail (hence
weak exact-instance matching) while preserving — even sharpening — class-level
semantics. The qualitative dumps back this up: when the student misses the exact
clip, its top hits are almost always the *right class*.

Two more measurements support "sharpened, not just copied":

- **Cluster tightness (silhouette, by class label):** teacher query **0.093** →
  MLP student **0.179**. The student's same-class clusters are tighter / better
  separated than the teacher *query* space.
- **Direct alignment:** mean cosine between a student embedding and its own
  teacher target is only **0.71** — so the student is *not* faithfully
  reproducing each target vector, yet still lands close enough to the right
  semantic neighborhood to retrieve well.

**What I can claim:** the student learned a representation that is *more
class-consistent than the teacher's 1-frame+audio query representation* when both
are judged inside ImageBind's privileged gallery.

**What I cannot claim:** that the student "beat ImageBind." It wins *inside
ImageBind's world* — swap the gallery for a CLIP space or anything else and the
result could flip. The retrieval win is gallery-relative, not absolute.

### 3. Does that smoothing cost class information? (downstream linear probe)

Retrieval rankings depend on the gallery, so I wanted a gallery-free check:
freeze the embeddings, train a logistic-regression probe on the train split,
score on test.

| Space | Top-1 acc | macro-F1 |
|---|---|---|
| Teacher query (1 frame + audio) | 84.39% | 0.8448 |
| MLP student (CLIP + AST)        | 84.08% | 0.8446 |

Essentially **tied**. So despite only 0.71 cosine to the targets, the student
retained virtually all of the *linearly decodable* class information the teacher
query carries. The smoothing in §2 isn't destroying class content — it's
reorganizing it. (And note: a near-tie on the probe is the honest counterweight
to the retrieval "win" — it's further evidence the student didn't surpass the
teacher in any absolute sense.)

### 4. The SigLIP 2 + CLAP transformer student — the upgrade that hasn't paid off (yet)

The natural next step was stronger encoders (SigLIP 2 vision, LAION-CLAP audio)
and a real fusion mechanism (bi-directional cross-attention) instead of a plain
concat-MLP. I precomputed those features and trained `CrossAttentionStudent`.
Honestly, it did **worse**, not better:

| Model (query) | Inst R@1 | Inst R@5 | Inst R@10 | Inst MedR | Sem R@1 | Sem R@5 | Sem R@10 | Sem MedR |
|---|---|---|---|---|---|---|---|---|
| MLP student          | 31.53 | 63.85 | 81.05 | 3 | 81.53 | 93.47 | 96.97 | 1 |
| Transformer student  | 14.49 | 41.56 | 57.48 | 8 | 75.48 | 88.06 | 91.88 | 1 |

Instance R@1 roughly halved and semantic R@1 dropped ~6 points. The semantic
behavior is still "right class even when the instance is wrong," so it learned
*something* sensible — it just landed less precisely. I don't have a confirmed
cause yet, but my working hypotheses, roughly in order of suspicion:

1. **The cross-attention is barely attending.** Each modality is a *single*
   token, so the attention softmax is over one key — effectively the identity.
   The "bi-directional cross-attention" is closer to a learned linear fusion than
   real attention, and may be harder to optimize than the simple concat-MLP.
2. **Geometry mismatch.** SigLIP 2 and CLAP live in their own spaces that may be
   farther from ImageBind's than CLIP/AST were, making the regression harder.
3. **Under-tuned.** It early-stopped at epoch 15 with no real hyperparameter
   search (same `α=10, β=1`, lr `1e-3` as the MLP). Capacity, lr, and the
   loss weighting are all untouched knobs.

So for now the **plain MLP is the stronger student**, which is itself a useful
(if humbling) result: a richer architecture + newer encoders is not automatically
better at this distillation task. Cluster, alignment, and downstream-probe
numbers for the transformer are the obvious next measurements (the probe is
already wired into `eval_downstream.py`).

### Summary of the story

- The 5-frame teacher target is *easy* to hit from one frame → the real task is
  the CLIP/AST → ImageBind mapping, not temporal modeling.
- The MLP student trades **instance fidelity** (31.5 R@1) for a **smoother, more
  class-consistent** space (81.5 semantic R@1, silhouette 0.093 → 0.179) that is
  downstream-equivalent to the teacher query (84.1 vs 84.4 probe).
- Those wins are **relative to ImageBind's gallery**, not proof of beating
  ImageBind.
- The fancier SigLIP/CLAP cross-attention student currently *under*performs the
  MLP — an open thread, likely architecture/tuning rather than the encoders
  themselves.

---

## Notes

- Checkpoints are referenced by fixed names in the eval scripts
  (`checkpoints/mlp/best_mlp_epoch19.pth`, `checkpoints/transformer/best_transformer_epoch15.pth`) —
  update those constants if you train new bests.
- The teacher *target* averages 5 frames, while the students see a single (middle)
  frame, so there is an intentional information asymmetry the students must approximate.
