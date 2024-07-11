import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


import torch._dynamo


class VideoPairClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.video_feature_extractor = models.video.r3d_18(pretrained=True)
        self.video_feature_extractor.fc = nn.Identity()

        # Freeze the first few layers of the feature extractor
        for name, param in self.video_feature_extractor.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        self.attention = nn.MultiheadAttention(512, num_heads=8)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(512)

    @torch._dynamo.disable
    def forward(self, x):
        batch_size = x.shape[0]
        x1, x2 = x[:, 0], x[:, 1]  # Assume x has shape (batch_size, 2, C, T, H, W)

        video1_features = self.video_feature_extractor(x1)
        video2_features = self.video_feature_extractor(x2)

        # Reshape for attention: (seq_len, batch_size, feature_dim)
        video1_features = video1_features.unsqueeze(0)
        video2_features = video2_features.unsqueeze(0)

        # Apply attention
        video1_features, _ = self.attention(video1_features, video1_features, video1_features)
        video2_features, _ = self.attention(video2_features, video2_features, video2_features)

        # Reshape back: (batch_size, feature_dim)
        video1_features = video1_features.squeeze(0)
        video2_features = video2_features.squeeze(0)

        # Concatenate features
        combined_features = torch.cat((video1_features, video2_features), dim=1)

        # Fully connected layers
        out = self.fc1(combined_features)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
