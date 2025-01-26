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
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # query/key/value: (batch_size, seq_len, feature_dim)
        # nn.MultiheadAttention expects: (seq_len, batch_size, feature_dim)
        # We'll do the permutation outside this wrapper for clarity.
        attn_out, weights = self.attention(query, key, value)
        return attn_out, weights


class VideoPairClassifier(nn.Module):
    """
    A two-branch model that processes 2 videos, applies attention, and produces multi-head outputs.
    """

    def __init__(
        self,
        head_structure,  # e.g., {"y_true_cat_label": 2}
        feature_extractor_backbone="r3d_18",
        num_frames=64,
        checkpoint_path=None,
    ):
        """
        Args:
            head_structure (dict): Dictionary specifying output heads (e.g., {'y_true_cat_label': 2})
            feature_extractor_backbone (str): Which backbone architecture to use
            num_frames (int): Number of input frames
            checkpoint_path (str): Path to a pretrained checkpoint if using x3d_m
        """
        super().__init__()
        self.head_structure = head_structure
        self.feature_extractor_backbone = feature_extractor_backbone
        self.num_frames = num_frames
        self.checkpoint_path = checkpoint_path
        self.feature_dim = self._get_feature_dim()

        # Initialize feature extractor
        self.video_feature_extractor = self._get_feature_extractor()

        # Replace final layer with identity if the backbone has an fc
        if hasattr(self.video_feature_extractor, "fc"):
            self.video_feature_extractor.fc = nn.Identity()

        # Freeze early layers
        for name, param in self.video_feature_extractor.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        # Multi-Head Attention
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=8)

        # Shared backbone layers after attention
        # We'll end up with shape [batch_size, (feature_dim * 2)] after concatenation
        self.fc1 = nn.Linear(self.feature_dim * 2, 768)
        self.fc2 = nn.Linear(768, 512)

        # Create output heads dynamically
        self.heads = nn.ModuleDict()
        for head_name, num_classes in self.head_structure.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
            )

        # Regularization
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = x[:, 0]  # [batch, C, T, H, W]
        x2 = x[:, 1]

        # Extract features for both videos
        x1_feats = self._process_features(x1)  # [batch, seq_len, feat] or [batch, feat]
        x2_feats = self._process_features(x2)

        # Ensure features are 3D (add dummy seq_len=1 if needed)
        if x1_feats.ndim == 2:
            x1_feats = x1_feats.unsqueeze(1)  # [batch, 1, feat]
        if x2_feats.ndim == 2:
            x2_feats = x2_feats.unsqueeze(1)

        # Concatenate sequences from both videos
        combined_seq = torch.cat([x1_feats, x2_feats], dim=1)  # [batch, seq_len1+seq_len2, feat]

        # Apply attention to the combined sequence
        combined_attended = self._apply_attention(
            combined_seq
        )  # [batch, seq_len1+seq_len2, feat]

        # Split back into individual video sequences
        split_idx = x1_feats.shape[1]  # Get original seq_len for video 1
        x1_attended = combined_attended[:, :split_idx, :]  # [batch, seq_len1, feat]
        x2_attended = combined_attended[:, split_idx:, :]  # [batch, seq_len2, feat]

        # Mean pool each attended sequence
        x1_feats = x1_attended.mean(dim=1)  # [batch, feat]
        x2_feats = x2_attended.mean(dim=1)

        # Concatenate pooled features
        combined = torch.cat((x1_feats, x2_feats), dim=1)  # [batch, feat_dim * 2]

        # Continue with shared layers
        x = self.fc1(combined)

        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)  # => [batch_size, 512]
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Multi-head outputs
        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(x)
        return outputs  # dict of {head_name: tensor}

    def _process_features(self, x):
        """
        Passes the input through the backbone, returning either:
          - [batch_size, feature_dim], or
          - [batch_size, T', feature_dim]
        depending on how we handle the intermediate steps.
        """
        # For mvit or r3d, the default forward is [batch, feat_dim, frames, h, w] => then we do some pooling.
        # For x3d_m, we often get [batch, channel, T', H', W'].
        if self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            # They expect input [batch, channels, T, H, W], but produce shape [batch, 768, t', h', w'].
            x = x.transpose(1, 2)  # if needed
        # Some backbones like x3d_m won't need this.

        features = self.video_feature_extractor(x)

        # For mvit we do mean across [2,3,4], giving [batch, 768].
        if self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            features = features.mean(dim=[2, 3, 4])  # => [batch, 768]
            return features

        elif self.feature_extractor_backbone == "x3d_m":
            if features.dim() == 5:
                # e.g. [batch, 432, t', 1, 1]
                features = features.squeeze(-1).squeeze(-1)  # => [batch, 432, t']
                # For multi-head attention, we want [batch, t', feature_dim].
                features = features.permute(0, 2, 1)  # => [batch, t', 432]
            return features

        else:
            # E.g. r3d_18 => [batch, 512, frames', H', W']
            # Usually we'd do an adaptive pool or something:
            if features.dim() == 5:
                # [batch, c, t', h', w']
                features = features.mean(dim=[2, 3, 4])  # => [batch, c]
            return features

    def _apply_attention(self, x):
        """Process sequence without pooling (returns [batch, seq_len, feature_dim])."""
        x = x.permute(1, 0, 2)  # [seq_len, batch, feat]
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.permute(1, 0, 2)  # [batch, seq_len, feat]
        return x

    def _get_feature_dim(self):
        """Return the final feature dimension for the given backbone."""
        if self.feature_extractor_backbone in ["r3d_18", "mc3_18", "r2plus1d_18"]:
            return 512
        elif self.feature_extractor_backbone in ["mvit_v1_b", "mvit_v2_s"]:
            return 768
        elif self.feature_extractor_backbone == "x3d_m":
            return 400
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")

    def _get_feature_extractor(self):
        """
        Returns a video feature extractor model.
        If x3d_m, tries to load from PyTorchVideo + checkpoint if specified.
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
            # Load the x3d_m model from PyTorchVideo
            try:
                model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=False)
                print("Loaded base x3d_m model from PyTorchVideo (pretrained=False).")
            except Exception as e:
                print(f"Error loading x3d_m model: {str(e)}")
                raise

            if self.checkpoint_path:
                # Load checkpoint
                try:
                    checkpoint = torch.load(
                        self.checkpoint_path,
                        map_location="cpu",
                    )
                    state_dict = checkpoint.get(
                        "model_state_dict", checkpoint.get("state_dict", checkpoint)
                    )
                    # Remove any 'module.' prefix
                    for prefix in ["module.", "_orig_mod.module."]:
                        if any(key.startswith(prefix) for key in state_dict.keys()):
                            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                                state_dict, prefix
                            )
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"Missing keys: {missing}")
                    if unexpected:
                        print(f"Unexpected keys: {unexpected}")
                    print(f"Successfully loaded checkpoint from {self.checkpoint_path}")
                except Exception as e:
                    print(f"Error loading checkpoint: {str(e)}")
                    raise
            else:
                print("No checkpoint path provided; using randomly initialized x3d_m model.")
            return model
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")
