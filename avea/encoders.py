"""Frozen feature encoders for the MLP student (CLIP vision + AST audio).

Extracted from the old train_mlp.py so that both the trainer and the evaluation
scripts can import them without pulling in the whole training module.
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPProcessor,
    CLIPModel,
    ASTFeatureExtractor,
    ASTModel,
)


class CLIPEncoder(nn.Module):

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
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
            List[List[PIL.Image]]   (batch_size x num_frames)
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

        return torch.stack(batch_embeddings, dim=0)


class ASTEncoder(nn.Module):

    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, waveforms):
        """
        waveforms:
            List[Tensor]   (each tensor [1, N])
        """

        batch_embeddings = []

        device = next(self.model.parameters()).device

        for waveform in waveforms:

            waveform = waveform.squeeze(0).cpu().numpy()

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

            batch_embeddings.append(embedding)

        return torch.stack(batch_embeddings, dim=0)
