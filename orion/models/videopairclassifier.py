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


class VideoPairClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.video_feature_extractor = models.video.r3d_18(pretrained=True)
        self.video_feature_extractor.fc = nn.Identity()

        for param in self.video_feature_extractor.parameters():
            param.requires_grad = True

        self.attention1 = SelfAttention(512)
        self.attention2 = SelfAttention(512)

        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc_artery = nn.Linear(11, 512)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, x_artery_label):
        num_pairs = x.shape[0]
        outputs = []

        for i in range(num_pairs):
            x1, x2 = x[i]

            x1 = x1.float()
            x2 = x2.float()

            video1_features = self.video_feature_extractor(x1.unsqueeze(0))
            video2_features = self.video_feature_extractor(x2.unsqueeze(0))

            video1_features = self.attention1(video1_features)
            video2_features = self.attention2(video2_features)

            artery_features = self.fc_artery(x_artery_label[i].float()).unsqueeze(0)

            pair_features = torch.cat((video1_features, video2_features, artery_features), dim=1)

            out = self.fc1(pair_features)
            out.relu_()  # In-place ReLU operation
            out = self.dropout(out)
            out = self.fc2(out)

            outputs.append(out.squeeze(0))

        outputs = torch.stack(outputs)

        return outputs
