"""Train the cross-attention transformer student on PRECOMPUTED embeddings.

Recycles LivePlot + HybridAlignmentLoss from train_mlp.py (single source of
truth) and mirrors its training loop / logging / early stopping. The key
difference: instead of running CLIP/AST encoders on the fly, this loads the
precomputed SigLIP 2 (768-d) and CLAP (512-d) feature vectors from disk, so each
epoch is just the transformer's matmuls -- dramatically faster.

Precompute first with: python3 precompute_student_embeddings.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from transformer_fusion import CrossAttentionStudent

# Recycled verbatim from the MLP trainer.
from train_mlp import LivePlot, HybridAlignmentLoss

NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 2
BATCH_SIZE = 64  # embeddings are tiny, so a large batch is cheap and faster

# Live training-curve plot. Disable with: LIVE_PLOT=0 python3 train_transformer.py
LIVE_PLOT = os.environ.get("LIVE_PLOT", "1") != "0"


class PrecomputedEmbeddingDataset(Dataset):
    """Loads the cached SigLIP 2 + CLAP feature vectors and the teacher target.
    Mirrors the teacher_embeddings/ layout used by the precompute script."""

    def __init__(self, root_dir, split):
        teacher_root = os.path.join(root_dir, split, "teacher_embeddings")
        siglip_root = os.path.join(root_dir, split, "siglip2_embeddings")
        clap_root = os.path.join(root_dir, split, "clap_embeddings")

        self.samples = []
        for label in sorted(os.listdir(teacher_root)):
            label_dir = os.path.join(teacher_root, label)
            if not os.path.isdir(label_dir):
                continue
            for file in sorted(os.listdir(label_dir)):
                if not file.endswith(".npy"):
                    continue
                visual_path = os.path.join(siglip_root, label, file)
                audio_path = os.path.join(clap_root, label, file)
                teacher_path = os.path.join(label_dir, file)
                if not (os.path.exists(visual_path) and os.path.exists(audio_path)):
                    continue
                self.samples.append({
                    "visual": visual_path,
                    "audio": audio_path,
                    "teacher": teacher_path,
                })

        print(f"Loaded {len(self.samples)} precomputed samples from {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "visual": torch.from_numpy(np.load(s["visual"])).float(),     # [768]
            "audio": torch.from_numpy(np.load(s["audio"])).float(),       # [512]
            "teacher": torch.from_numpy(np.load(s["teacher"])).float(),   # [2048]
        }


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=None, num_epochs=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Train" if epoch is None else f"Epoch {epoch}/{num_epochs} [Train]"
    pbar = tqdm(train_loader, desc=desc, leave=False)

    for batch in pbar:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        teacher_emb = batch["teacher"].to(device)

        pred = model(visual, audio)
        loss, loss_mse, loss_cos = criterion(pred, teacher_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cos += loss_cos.item()
        n = pbar.n + 1
        pbar.set_postfix(
            loss=f"{total_loss / n:.4f}",
            mse=f"{total_mse / n:.4f}",
            cos=f"{total_cos / n:.4f}",
        )

    n = len(train_loader)
    return {"total": total_loss / n, "mse": total_mse / n, "cos": total_cos / n}


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch=None, num_epochs=None):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Val" if epoch is None else f"Epoch {epoch}/{num_epochs} [Val]"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    for batch in pbar:
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        teacher_emb = batch["teacher"].to(device)

        pred = model(visual, audio)
        loss, loss_mse, loss_cos = criterion(pred, teacher_emb)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cos += loss_cos.item()
        n = pbar.n + 1
        pbar.set_postfix(
            loss=f"{total_loss / n:.4f}",
            mse=f"{total_mse / n:.4f}",
            cos=f"{total_cos / n:.4f}",
        )

    n = len(val_loader)
    return {"total": total_loss / n, "mse": total_mse / n, "cos": total_cos / n}


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_train_dataset = PrecomputedEmbeddingDataset(
        root_dir="processed_vggsound", split="train"
    )
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CrossAttentionStudent().to(device)

    # Same hybrid loss + weighting the MLP trainer settled on.
    criterion = HybridAlignmentLoss(alpha=10.0, beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    live_plot = LivePlot(enabled=LIVE_PLOT, save_path="training_curve_transformer.png")
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch + 1, num_epochs=NUM_EPOCHS,
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            epoch=epoch + 1, num_epochs=NUM_EPOCHS,
        )

        is_best = val_metrics["total"] < best_val_loss

        epoch_bar.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            best=f"{min(best_val_loss, val_metrics['total']):.4f}",
        )
        tqdm.write(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train: total {train_metrics['total']:.4f} "
            f"(mse {train_metrics['mse']:.4f}, cos {train_metrics['cos']:.4f}) - "
            f"Val: total {val_metrics['total']:.4f} "
            f"(mse {val_metrics['mse']:.4f}, cos {val_metrics['cos']:.4f})"
            + ("  *best*" if is_best else "")
        )

        live_plot.update(train_metrics, val_metrics)

        if is_best:
            best_val_loss = val_metrics["total"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_transformer_epoch{epoch+1}.pth")
            tqdm.write("Saved best model")
        else:
            epochs_no_improve += 1
            tqdm.write(
                f"No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} "
                f"epoch(s) (best val {best_val_loss:.4f})"
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
