"""Train the multi-token fusion transformer student.

Improvements over train_transformer.py:
  * real self-attention over a token sequence (5 frame + 5 audio tokens + CLS)
  * modality dropout: randomly zero one modality per sample (robustness +
    makes single-modality / cross-modal queries meaningful)
  * HybridContrastiveLoss: MSE + cosine + InfoNCE (the contrastive term
    targets weak instance-level retrieval)

Precompute first with: python3 scripts/precompute_tokens.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from avea.data.token_dataset import TokenSequenceDataset
from avea.models.multitoken_fusion import MultiTokenFusionTransformer
from avea.losses import HybridContrastiveLoss
from avea.plotting import LivePlot
from avea.eval.retrieval import compute_instance_metrics, compute_semantic_metrics

NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 256
DATA_ROOT = "processed_vggsound"
CKPT_DIR = "checkpoints/multitoken"

MODALITY_DROPOUT = 0.0  # per-sample prob of zeroing ONE modality during training

# Live training-curve plot. Disable with: LIVE_PLOT=0 python3 scripts/train_multitoken.py
LIVE_PLOT = os.environ.get("LIVE_PLOT", "1") != "0"


def apply_modality_dropout(visual, audio, p):
    """For each sample, with prob p zero exactly one modality (50/50 which).
    visual: [B, Nv, Dv]   audio: [B, Na, Da]"""
    if p <= 0:
        return visual, audio
    B = visual.size(0)
    device = visual.device
    drop = torch.rand(B, device=device) < p           # which samples lose a modality
    drop_audio = torch.rand(B, device=device) < 0.5    # of those, which modality
    zero_v = drop & (~drop_audio)
    zero_a = drop & drop_audio
    visual = visual.clone()
    audio = audio.clone()
    visual[zero_v] = 0.0
    audio[zero_a] = 0.0
    return visual, audio


def _run_epoch(model, loader, criterion, device, optimizer=None, epoch=None, num_epochs=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    phase = "Train" if train else "Val"
    desc = phase if epoch is None else f"Epoch {epoch}/{num_epochs} [{phase}]"

    totals = {"total": 0.0, "mse": 0.0, "cos": 0.0, "nce": 0.0}
    pbar = tqdm(loader, desc=desc, leave=False)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in pbar:
            visual = batch["visual"].to(device)
            audio = batch["audio"].to(device)
            teacher = batch["teacher"].to(device)

            if train:
                visual, audio = apply_modality_dropout(visual, audio, MODALITY_DROPOUT)

            pred = model(visual, audio)
            loss, comps = criterion(pred, teacher, labels=batch["label"])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals["total"] += loss.item()
            for k in ("mse", "cos", "nce"):
                totals[k] += comps[k].item()
            n = pbar.n + 1
            pbar.set_postfix(
                loss=f"{totals['total'] / n:.4f}",
                mse=f"{totals['mse'] / n:.4f}",
                cos=f"{totals['cos'] / n:.4f}",
                nce=f"{totals['nce'] / n:.4f}",
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_retrieval_r1(model, val_loader, device):
    """Student val queries retrieved against the val teacher gallery.
    Returns (instance_R@1, semantic_R@1). Both modalities present (no dropout)."""
    model.eval()
    Qs, Gs, labels = [], [], []
    for batch in val_loader:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        Qs.append(model(visual, audio).cpu())
        Gs.append(batch["teacher"])
        labels.extend(batch["label"])

    Q = torch.cat(Qs)
    G = torch.cat(Gs)
    Q = Q / Q.norm(dim=1, keepdim=True)
    G = G / G.norm(dim=1, keepdim=True)
    similarity = torch.mm(Q, G.t())

    inst = compute_instance_metrics(similarity)["R@1"]
    sem = compute_semantic_metrics(similarity, labels, labels)["R@1"]
    return inst, sem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    full_train_dataset = TokenSequenceDataset(root_dir=DATA_ROOT, split="train")
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MultiTokenFusionTransformer().to(device)
    criterion = HybridContrastiveLoss(alpha=10.0, beta=1.0, gamma=5.0, temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score = -1.0  # selection metric = val (instance R@1 + semantic R@1)
    epochs_no_improve = 0
    live_plot = LivePlot(enabled=LIVE_PLOT, save_path="outputs/training_curve_multitoken.png")
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            epoch=epoch + 1, num_epochs=NUM_EPOCHS,
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, device, optimizer=None,
            epoch=epoch + 1, num_epochs=NUM_EPOCHS,
        )

        # Select checkpoints by what we actually care about: retrieval R@1.
        val_inst_r1, val_sem_r1 = val_retrieval_r1(model, val_loader, device)
        score = val_inst_r1 + val_sem_r1
        is_best = score > best_score

        epoch_bar.set_postfix(
            val=f"{val_metrics['total']:.4f}",
            inst=f"{val_inst_r1:.1f}",
            sem=f"{val_sem_r1:.1f}",
            best=f"{max(best_score, score):.1f}",
        )
        tqdm.write(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train loss {train_metrics['total']:.4f} - Val loss {val_metrics['total']:.4f} - "
            f"Val instance R@1 {val_inst_r1:.2f} / semantic R@1 {val_sem_r1:.2f} "
            f"(score {score:.2f})"
            + ("  *best*" if is_best else "")
        )

        live_plot.update(train_metrics, val_metrics)

        if is_best:
            best_score = score
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_multitoken_epoch{epoch+1}.pth"))
            tqdm.write("Saved best model")
        else:
            epochs_no_improve += 1
            tqdm.write(
                f"No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} "
                f"epoch(s) (best score {best_score:.2f})"
            )
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                tqdm.write(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                )
                break

    live_plot.close()


if __name__ == "__main__":
    main()
