from dataset_class import VGGSoundDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from NaiveLateMLPFusion.model.naive_implementation import NaiveLateFusionMLP
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPModel,
    ASTFeatureExtractor,
    ASTModel
)

NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 2

# Live training-curve plot. Disable with: LIVE_PLOT=0 python3 train_mlp.py
LIVE_PLOT = os.environ.get("LIVE_PLOT", "1") != "0"


class LivePlot:

    def __init__(self, enabled=True, save_path="training_curve.png"):
        self.enabled = enabled
        self.save_path = os.path.abspath(save_path)
        self.train_total = []
        self.val_total = []
        self.plt = None
        self.fig = None
        self.ax = None
        self.interactive = False
        if not self.enabled:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # An interactive backend needs a display; with Agg (headless) we
            # skip live drawing and rely entirely on savefig.
            self.interactive = "agg" not in matplotlib.get_backend().lower()
            self.plt = plt
            if self.interactive:
                plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
        except Exception as e:
            print(f"[LivePlot] matplotlib unavailable ({e}); plotting disabled")
            self.enabled = False

    def update(self, train_metrics, val_metrics):
        self.train_total.append(train_metrics["total"])
        self.val_total.append(val_metrics["total"])
        if not self.enabled:
            return
        self._render()
        # Write the PNG first, so it always exists regardless of the backend.
        try:
            self.fig.savefig(self.save_path)
            if len(self.train_total) == 1:
                print(f"[LivePlot] saving curve to {self.save_path}")
        except Exception as e:
            print(f"[LivePlot] could not save figure: {e}")
        # Best-effort live refresh; a display problem must never crash training.
        if self.interactive:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                self.plt.pause(0.01)
            except Exception:
                self.interactive = False

    def _render(self):
        epochs = range(1, len(self.train_total) + 1)
        self.ax.clear()
        self.ax.plot(epochs, self.train_total, "-o", label="Train", color="tab:blue")
        self.ax.plot(epochs, self.val_total, "-o", label="Val", color="tab:orange")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Total loss")
        self.ax.set_title("Training Progress")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

    def close(self):
        if not self.enabled:
            return
        try:
            self.fig.savefig(self.save_path)
        except Exception:
            pass
        if self.interactive:
            try:
                self.plt.ioff()
                self.plt.show()
            except Exception:
                pass

class CLIPEncoder(nn.Module):

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model = CLIPModel.from_pretrained(model_name)
        print(type(self.model))

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, images):

        """
        images:
            List[List[PIL.Image]]

        shape:
            batch_size x num_frames
        """

        batch_embeddings = []

        device = next(self.model.parameters()).device
        for frame_list in images:

            middle_frame = frame_list[len(frame_list) // 2]
            inputs = self.processor(images=middle_frame, return_tensors="pt")

            pixel_values = inputs["pixel_values"].to(device)

            image_features = self.model.get_image_features(
                pixel_values=pixel_values
            )

            batch_embeddings.append(
                image_features.squeeze(0)
            )

        return torch.stack(
            batch_embeddings,
            dim=0
        )
    

class ASTEncoder(nn.Module):

    def __init__(
        self,
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593"
    ):
        super().__init__()

        self.feature_extractor = (
            ASTFeatureExtractor.from_pretrained(
                model_name
            )
        )

        self.model = ASTModel.from_pretrained(
            model_name
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        waveforms
    ):
        """
        waveforms:
            List[Tensor]

        each tensor:
            [1, N]
        """

        batch_embeddings = []

        device = next(
            self.model.parameters()
        ).device

        for waveform in waveforms:

            waveform = (
                waveform.squeeze(0)
                .cpu()
                .numpy()
            )

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            outputs = self.model(
                **inputs
            )

            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
                .squeeze(0)
            )

            batch_embeddings.append(
                embedding
            )

        return torch.stack(
            batch_embeddings,
            dim=0
        )
    
class HybridAlignmentLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        """
        pred:   [Batch, 2048] - Student MLP output
        target: [Batch, 2048] - Teacher ImageBind target
        """
        # Calculate MSE component
        loss_mse = self.mse(pred, target)
        
        # Calculate Cosine Distance component 
        # (1 - cosine_similarity transforms it into a minimization objective from 0 to 2)
        loss_cos = 1.0 - self.cosine(pred, target).mean()
        
        # Combined weighted loss
        total_loss = (self.alpha * loss_mse) + (self.beta * loss_cos)
        
        return total_loss, loss_mse, loss_cos

def collate_fn(batch):

    return {
        "images": [x["images"] for x in batch],
        "waveforms": [x["waveform"] for x in batch],
        "teachers": torch.stack(
            [x["teacher"] for x in batch]
        )
    }

def train_one_epoch(clip_encoder, ast_encoder, mlp, train_loader, optimizer, criterion, device, epoch=None, num_epochs=None):
    mlp.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Train" if epoch is None else f"Epoch {epoch}/{num_epochs} [Train]"
    pbar = tqdm(
        train_loader,
        desc=desc,
        leave=False
    )

    for batch in pbar:
        with torch.no_grad():
            clip_emb = clip_encoder(batch["images"])
            ast_emb = ast_encoder(batch["waveforms"])
        teacher_emb = batch["teachers"].to(device)
        pred = mlp(clip_emb, ast_emb)
        target = torch.ones(pred.shape[0], dtype=torch.float32).to(device)
        loss, loss_mse, loss_cos = criterion(pred,teacher_emb)
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
            cos=f"{total_cos / n:.4f}"
        )
        optimizer.zero_grad()
    n = len(train_loader)
    return {
        "total": total_loss / n,
        "mse": total_mse / n,
        "cos": total_cos / n,
    }

@torch.no_grad()
def validate(clip_encoder, ast_encoder, mlp, val_loader, criterion, device, epoch=None, num_epochs=None):
    mlp.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    desc = "Val" if epoch is None else f"Epoch {epoch}/{num_epochs} [Val]"
    pbar = tqdm(
        val_loader,
        desc=desc,
        leave=False
    )
    for batch in pbar:
        clip_emb = clip_encoder(batch["images"])
        ast_emb = ast_encoder(batch["waveforms"])
        teacher_emb = batch["teachers"].to(device)
        pred = mlp(clip_emb, ast_emb)
        target = torch.ones(pred.shape[0], dtype=torch.float32).to(device)
        loss, loss_mse, loss_cos = criterion(pred,teacher_emb)
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cos += loss_cos.item()
        n = pbar.n + 1
        pbar.set_postfix(
            loss=f"{total_loss / n:.4f}",
            mse=f"{total_mse / n:.4f}",
            cos=f"{total_cos / n:.4f}"
        )
    n = len(val_loader)
    return {
        "total": total_loss / n,
        "mse": total_mse / n,
        "cos": total_cos / n,
    }

  
def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    full_train_dataset = VGGSoundDataset(
        root_dir="processed_vggsound",
        split="train"
    )
    train_size = int(
        0.8 * len(full_train_dataset)
    )

    val_size = (
        len(full_train_dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = VGGSoundDataset(
        root_dir="processed_vggsound",
        split="test")
    
    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Val: {len(val_dataset)}"
    )

    print(
        f"Test: {len(test_dataset)}"
    )

    # sample = dataset[0]

    clip_encoder = (CLIPEncoder().to(device))
    ast_encoder = (ASTEncoder().to(device))

    clip_encoder.eval()
    ast_encoder.eval()

    for p in clip_encoder.parameters():
        p.requires_grad = False

    for p in ast_encoder.parameters():
        p.requires_grad = False

    mlp = NaiveLateFusionMLP().to(device)

    # images = [
    #     sample["images"]
    # ]

    # waveforms = [
    #     sample["waveform"]
    # ]

    # clip_emb = clip_encoder(
    #     images
    # )

    # ast_emb = ast_encoder(
    #     waveforms
    # )

    # student_emb = torch.cat(
    #     [
    #         clip_emb,
    #         ast_emb
    #     ],
    #     dim=1
    # )

    # print(
    #     "CLIP:",
    #     clip_emb.shape
    # )

    # print(
    #     "AST:",
    #     ast_emb.shape
    # )

    # print(
    #     "Student:",
    #     student_emb.shape
    # )

    # print(
    #     "Teacher:",
    #     sample["teacher"].shape
    # )


    train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    # batch = next(iter(train_loader))

    # print(type(batch["images"]))
    # print(len(batch["images"]))

    # print(type(batch["images"][0]))
    # print(len(batch["images"][0]))

    # print(batch["waveforms"][0].shape)

    # print(batch["teachers"].shape)

    # criterion = nn.CosineEmbeddingLoss()
    criterion = HybridAlignmentLoss(alpha=10.0,beta=1.0)
    optimizer = torch.optim.AdamW(mlp.parameters(),lr=1e-3)


    best_val_loss = float("inf")
    epochs_no_improve = 0
    live_plot = LivePlot(enabled=LIVE_PLOT)
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            clip_encoder,
            ast_encoder,
            mlp,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch + 1,
            num_epochs=NUM_EPOCHS
        )

        val_metrics = validate(
            clip_encoder,
            ast_encoder,
            mlp,
            val_loader,
            criterion,
            device,
            epoch=epoch + 1,
            num_epochs=NUM_EPOCHS
        )

        is_best = val_metrics["total"] < best_val_loss

        epoch_bar.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            best=f"{min(best_val_loss, val_metrics['total']):.4f}"
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
            torch.save(mlp.state_dict(),f"best_mlp_epoch{epoch+1}.pth")
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