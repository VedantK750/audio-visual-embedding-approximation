"""Train the NaiveLateFusionMLP student (CLIP + AST -> ImageBind target)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from avea.data.dataset import VGGSoundDataset
from avea.models.mlp_fusion import NaiveLateFusionMLP
from avea.encoders import CLIPEncoder, ASTEncoder
from avea.losses import HybridAlignmentLoss
from avea.plotting import LivePlot

NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 2
DATA_ROOT = "processed_vggsound"
CKPT_DIR = "checkpoints/mlp"

# Live training-curve plot. Disable with: LIVE_PLOT=0 python3 scripts/train_mlp.py
LIVE_PLOT = os.environ.get("LIVE_PLOT", "1") != "0"


def collate_fn(batch):
    return {
        "images": [x["images"] for x in batch],
        "waveforms": [x["waveform"] for x in batch],
        "teachers": torch.stack([x["teacher"] for x in batch]),
    }


def train_one_epoch(clip_encoder, ast_encoder, mlp, train_loader, optimizer, criterion, device, epoch=None, num_epochs=None):
    mlp.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Train" if epoch is None else f"Epoch {epoch}/{num_epochs} [Train]"
    pbar = tqdm(train_loader, desc=desc, leave=False)

    for batch in pbar:
        with torch.no_grad():
            clip_emb = clip_encoder(batch["images"])
            ast_emb = ast_encoder(batch["waveforms"])
        teacher_emb = batch["teachers"].to(device)
        pred = mlp(clip_emb, ast_emb)
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
def validate(clip_encoder, ast_encoder, mlp, val_loader, criterion, device, epoch=None, num_epochs=None):
    mlp.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Val" if epoch is None else f"Epoch {epoch}/{num_epochs} [Val]"
    pbar = tqdm(val_loader, desc=desc, leave=False)

    for batch in pbar:
        clip_emb = clip_encoder(batch["images"])
        ast_emb = ast_encoder(batch["waveforms"])
        teacher_emb = batch["teachers"].to(device)
        pred = mlp(clip_emb, ast_emb)
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
    os.makedirs(CKPT_DIR, exist_ok=True)

    full_train_dataset = VGGSoundDataset(root_dir=DATA_ROOT, split="train")
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    clip_encoder = CLIPEncoder().to(device)
    ast_encoder = ASTEncoder().to(device)
    clip_encoder.eval()
    ast_encoder.eval()
    for p in clip_encoder.parameters():
        p.requires_grad = False
    for p in ast_encoder.parameters():
        p.requires_grad = False

    mlp = NaiveLateFusionMLP().to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    criterion = HybridAlignmentLoss(alpha=10.0, beta=1.0)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    live_plot = LivePlot(enabled=LIVE_PLOT, save_path="outputs/training_curve_mlp.png")
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            clip_encoder, ast_encoder, mlp, train_loader, optimizer, criterion, device,
            epoch=epoch + 1, num_epochs=NUM_EPOCHS,
        )
        val_metrics = validate(
            clip_encoder, ast_encoder, mlp, val_loader, criterion, device,
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
            torch.save(mlp.state_dict(), os.path.join(CKPT_DIR, f"best_mlp_epoch{epoch+1}.pth"))
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
