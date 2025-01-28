import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        """
        query/key/value: [seq_len, batch_size, feature_dim]
        """
        attn_out, weights = self.attention(query, key, value)
        return attn_out, weights


class VideoPairClassifier(nn.Module):
    """
    A two-branch model that processes 2 videos, applies attention,
    and produces multi-head outputs.
    """

    def __init__(
        self,
        head_structure,
        feature_extractor_backbone="r3d_18",
        num_frames=64,
        checkpoint_path=None,
        freeze_ratio=0.0,  # 0.0 => fully trainable; 1.0 => fully frozen
        num_heads=2,
    ):
        """
        Args:
            head_structure (dict): e.g. {'y_true_cat_label': 2}
            feature_extractor_backbone (str): Which backbone architecture to use
            num_frames (int): Number of input frames
            checkpoint_path (str): Path to a pretrained checkpoint (x3d_m)
            freeze_ratio (float): Fraction of backbone parameters to freeze
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.head_structure = head_structure
        self.feature_extractor_backbone = feature_extractor_backbone
        self.num_frames = num_frames
        self.checkpoint_path = checkpoint_path
        self.freeze_ratio = freeze_ratio

        # 1. Initialize feature extractor
        self.video_feature_extractor = self._get_feature_extractor()

        # 2. Dynamically determine feature dimension
        self.feature_dim = self._get_feature_dim_dynamic()

        # 3. Replace final layer with identity if backbone has one
        if hasattr(self.video_feature_extractor, "fc"):
            self.video_feature_extractor.fc = nn.Identity()

        # 4. Partially freeze the backbone
        self._freeze_layers()

        # 5. Multi-Head Attention
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=num_heads)

        # 6. Shared layers after attention
        self.fc1 = nn.Linear(self.feature_dim * 2, 768)
        self.fc2 = nn.Linear(768, 512)

        # 7. Create output heads dynamically
        self.heads = nn.ModuleDict()
        for head_name, num_classes in self.head_structure.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        # 8. Dropouts / LayerNorm
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(512)

        # ----> Layer initialization <----
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

        for head_module in self.heads.values():
            # [Linear(512->256), ReLU, Dropout, Linear(256->num_classes)]
            nn.init.kaiming_normal_(head_module[0].weight, nonlinearity="relu")
            nn.init.constant_(head_module[0].bias, 0.0)
            nn.init.xavier_normal_(head_module[3].weight)
            nn.init.constant_(head_module[3].bias, 0.0)

    def forward(self, x):
        """
        Expects x of shape [B, 2, C, T, H, W].
        """
        assert x.ndim == 6, f"Input must be [B,2,C,T,H,W], got {x.shape}"
        assert x.shape[1] == 2, "Need exactly 2 videos per sample"

        # Split into two video streams
        x1 = x[:, 0]  # [batch, C, T, H, W]
        x2 = x[:, 1]  # [batch, C, T, H, W]

        # Extract features [batch, feature_dim]
        x1_feats = self._process_features(x1)
        x2_feats = self._process_features(x2)

        # Expand to [batch, seq_len=1, feature_dim] if needed
        if x1_feats.ndim == 2:
            x1_feats = x1_feats.unsqueeze(1)
        if x2_feats.ndim == 2:
            x2_feats = x2_feats.unsqueeze(1)

        # Concatenate sequences
        combined_seq = torch.cat([x1_feats, x2_feats], dim=1)

        # Apply attention with residual
        combined_attended = self._apply_attention(combined_seq)

        # Split back
        seq_len1 = x1_feats.shape[1]
        x1_attended = combined_attended[:, :seq_len1, :]
        x2_attended = combined_attended[:, seq_len1:, :]

        # Mean pool
        x1_feats = x1_attended.mean(dim=1)
        x2_feats = x2_attended.mean(dim=1)

        # Concatenate => [batch, 2 * feature_dim]
        combined = torch.cat((x1_feats, x2_feats), dim=1)

        # Shared layers
        out = self.fc1(combined)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Multi-head outputs
        return {head_name: head(out) for head_name, head in self.heads.items()}

    def _process_features(self, x):
        """
        Pass x through the backbone and forcibly reduce to [batch, c]
        with AdaptiveAvgPool3d, which avoids shape mismatch
        even if input is 256x256 or otherwise bigger than 112x112.
        """
        features = self.video_feature_extractor(x)
        # Force shape => [batch, c, 1, 1, 1] => [batch, c]
        if features.dim() == 5:
            features = F.adaptive_avg_pool3d(features, (1, 1, 1))
            features = features.view(features.size(0), -1)
        return features

    def _apply_attention(self, x):
        """
        x: [batch, seq_len, feature_dim]
        => returns same shape [batch, seq_len, feature_dim].
        """
        x_t = x.permute(1, 0, 2)  # => [seq_len, batch, feat]
        residual = x_t
        attn_out, _ = self.attention(x_t, x_t, x_t)
        attn_out += residual
        return attn_out.permute(1, 0, 2)

    def _get_feature_dim_dynamic(self):
        """
        Infers the actual output channel dimension
        by running a dummy forward pass.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        cpu_device = torch.device("cpu")
        self.video_feature_extractor.to(cpu_device)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.num_frames, 112, 112, device=cpu_device)
            feat = self.video_feature_extractor(dummy)
            if feat.dim() == 5:
                # Force pool => [1, c]
                feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))
                feat = feat.view(feat.size(0), -1)
            out_dim = feat.shape[1]

        self.video_feature_extractor.to(device)
        if was_training:
            self.train()

        return out_dim

    def _get_feature_extractor(self):
        """
        Returns requested backbone. If 'x3d_m', loads from PyTorchVideo.
        """
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
        elif self.feature_extractor_backbone == "x3d_m":
            # (Same logic as before, plus checkpoint loading if needed)
            import torch.hub

            try:
                model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
            except Exception as e:
                raise RuntimeError(f"Error loading x3d_m model: {str(e)}")

            # Force final pooling
            model.blocks[-1].pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            model.blocks[-1].proj = nn.Identity()
            return model
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")

    def _freeze_layers(self):
        """
        PARTIAL-FREEZE:
          - freeze_ratio=1.0 => all backbone params frozen
          - freeze_ratio=0.0 => none are frozen (fully trainable)
        """
        all_named_params = list(self.video_feature_extractor.named_parameters())
        total_params = len(all_named_params)

        # Number to freeze
        freeze_count = int(self.freeze_ratio * total_params)

        for i, (name, param) in enumerate(all_named_params):
            param.requires_grad = i >= freeze_count

        # Print stats (optional)
        total_num = sum(p.numel() for p in self.video_feature_extractor.parameters())
        frozen_num = sum(
            p.numel() for p in self.video_feature_extractor.parameters() if not p.requires_grad
        )
        train_num = total_num - frozen_num
        print(f"[VideoPairClassifier] freeze_ratio={self.freeze_ratio:.2f}")
        print(f"  Total backbone parameters:    {total_num:,}")
        print(f"  Frozen backbone parameters:   {frozen_num:,} ({frozen_num/total_num:.2%})")
        print(f"  Trainable backbone parameters:{train_num:,} ({train_num/total_num:.2%})")
