import torch

# Disable TorchDynamo
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

torch._dynamo.config.suppress_errors = True


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(
            Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.feature_dim).float()), dim=-1
        )
        return attention_weights @ V


class VideoPairClassifier(nn.Module):
    def __init__(self, num_classes=2, feature_extractor_backbone="r3d_18", num_frames=64):
        super().__init__()

        self.feature_extractor_backbone = feature_extractor_backbone
        self.num_frames = num_frames
        self.feature_dim = self._get_feature_dim()

        self.video_feature_extractor = self._get_feature_extractor()

        if hasattr(self.video_feature_extractor, "fc"):
            self.video_feature_extractor.fc = nn.Identity()

        # Freeze early layers
        for name, param in self.video_feature_extractor.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        self.attention = nn.MultiheadAttention(self.feature_dim, num_heads=8)

        self.fc1 = nn.Linear(self.feature_dim * 2, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.dropout1 = nn.Dropout(p=0.6)
        self.dropout2 = nn.Dropout(p=0.6)
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(512)

    def _get_feature_extractor(self):
        if self.feature_extractor_backbone == "r3d_18":
            return models.video.r3d_18(pretrained=True)
        elif self.feature_extractor_backbone == "mc3_18":
            return models.video.mc3_18(pretrained=True)
        elif self.feature_extractor_backbone == "r2plus1d_18":
            return models.video.r2plus1d_18(pretrained=True)
        elif self.feature_extractor_backbone == "mvit_v1_b":
            return models.video.mvit_v1_b(pretrained=True)
        elif self.feature_extractor_backbone == "mvit_v2_s":
            return models.video.mvit_v2_s(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")

    def _get_feature_dim(self):
        if self.feature_extractor_backbone in ["r3d_18", "mc3_18", "r2plus1d_18"]:
            return 512
        elif self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            return 768
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")

    def forward(self, x):
        batch_size = x.shape[0]
        x1, x2 = x[:, 0], x[:, 1]

        if self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            # MViT models expect input shape (B, C, T, H, W)
            x1 = x1.transpose(1, 2)
            x2 = x2.transpose(1, 2)

        video1_features = self.video_feature_extractor(x1)
        video2_features = self.video_feature_extractor(x2)

        if self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            # For MViT, we need to pool over the temporal and spatial dimensions
            video1_features = video1_features.mean(dim=[2, 3, 4])  # Average pooling over T, H, W
            video2_features = video2_features.mean(dim=[2, 3, 4])
        else:
            # For other models, we might need to squeeze or reshape
            video1_features = video1_features.squeeze()
            video2_features = video2_features.squeeze()

        # Ensure the features are 2D tensors (batch_size, feature_dim)
        if video1_features.dim() == 1:
            video1_features = video1_features.unsqueeze(0)
            video2_features = video2_features.unsqueeze(0)

        # Apply attention
        video1_features = video1_features.unsqueeze(0)
        video2_features = video2_features.unsqueeze(0)
        video1_features, _ = self.attention(video1_features, video1_features, video1_features)
        video2_features, _ = self.attention(video2_features, video2_features, video2_features)
        video1_features = video1_features.squeeze(0)
        video2_features = video2_features.squeeze(0)

        combined_features = torch.cat((video1_features, video2_features), dim=1)

        out = self.fc1(combined_features)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)

        return out
