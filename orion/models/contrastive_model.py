import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from orion.models.pytorchvideo_models import get_pretrained_model


class VideoEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Get mVIT model and remove classification head
        self.backbone = get_pretrained_model("mvit_base_16x4", num_classes=1, task="regression")
        if isinstance(self.backbone, tuple):
            self.backbone = self.backbone[0]  # Extract model from tuple if needed

        # Get the input dimension of the last layer
        self.feature_dim = self.backbone.head.in_features

        # Replace classification head with projection head
        self.backbone.head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024), nn.ReLU(), nn.Linear(1024, output_dim)
        )

        # Normalize outputs
        self.normalize = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        # x shape: (batch_size, channels, frames, height, width)
        features = self.backbone(x)
        return self.normalize(features)


class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Load PubMedBERT
        self.bert = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024), nn.ReLU(), nn.Linear(1024, output_dim)
        )

        # Normalize outputs
        self.normalize = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, text):
        # Tokenize text if it's not already tokenized
        if isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], str)):
            inputs = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            inputs = {k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()}
        else:
            inputs = text

        # Get BERT outputs
        outputs = self.bert(**inputs)
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0]

        # Project to the same dimension as video features
        projected = self.projection(pooled_output)
        return self.normalize(projected)


class ContrastiveModel(nn.Module):
    def __init__(self, temperature=0.07, output_dim=512):
        super().__init__()
        self.video_encoder = VideoEncoder(output_dim=output_dim)
        self.text_encoder = TextEncoder(output_dim=output_dim)
        self.temperature = temperature

    def forward(self, videos, texts):
        # Encode videos and texts
        video_features = self.video_encoder(videos)
        text_features = self.text_encoder(texts)

        # Compute similarity matrix
        similarity = torch.matmul(video_features, text_features.T) / self.temperature

        # Labels for contrastive loss (diagonal is positive pairs)
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Compute loss in both directions (video->text and text->video)
        loss_v2t = F.cross_entropy(similarity, labels)
        loss_t2v = F.cross_entropy(similarity.T, labels)

        # Total loss is average of both directions
        total_loss = (loss_v2t + loss_t2v) / 2

        return {
            "loss": total_loss,
            "video_features": video_features,
            "text_features": text_features,
            "similarity": similarity,
        }

    def encode_video(self, video):
        return self.video_encoder(video)

    def encode_text(self, text):
        return self.text_encoder(text)

    def compute_similarity(self, video_features, text_features):
        return torch.matmul(video_features, text_features.T) / self.temperature
