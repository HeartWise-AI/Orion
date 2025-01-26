import torch
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
        # query/key/value: [batch_size, seq_len, feature_dim]
        # nn.MultiheadAttention expects: [seq_len, batch_size, feature_dim]
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
        freeze_ratio=0.8,  # NEW: ratio-based partial freeze
    ):
        """
        Args:
            head_structure (dict): Dictionary specifying output heads (e.g., {'y_true_cat_label': 2})
            feature_extractor_backbone (str): Which backbone architecture to use
            num_frames (int): Number of input frames
            checkpoint_path (str): Path to a pretrained checkpoint if using x3d_m
            freeze_ratio (float): Fraction of backbone parameters to keep trainable.
                                  e.g. 0.8 -> ~80% trainable, 20% frozen
        """
        super().__init__()
        self.head_structure = head_structure
        self.feature_extractor_backbone = feature_extractor_backbone
        self.num_frames = num_frames
        self.checkpoint_path = checkpoint_path
        self.freeze_ratio = freeze_ratio

        # 1. Initialize feature extractor
        self.video_feature_extractor = self._get_feature_extractor()

        # 2. Dynamically determine feature dimension by running a dummy forward pass
        self.feature_dim = self._get_feature_dim_dynamic()

        # 3. Replace final layer (fc) with identity if the backbone has one
        if hasattr(self.video_feature_extractor, "fc"):
            self.video_feature_extractor.fc = nn.Identity()

        # 4. Partially freeze the backbone using the freeze_ratio
        self._freeze_layers()

        # 5. Multi-Head Attention module
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=8)

        # 6. Shared backbone layers after attention (concatenate x1 & x2 => 2 * feature_dim)
        self.fc1 = nn.Linear(self.feature_dim * 2, 768)
        self.fc2 = nn.Linear(768, 512)

        # 7. Create output heads dynamically
        self.heads = nn.ModuleDict()
        for head_name, num_classes in self.head_structure.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        # 8. Regularization
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.layer_norm1 = nn.LayerNorm(768)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        """
        Expects x of shape [B, 2, C, T, H, W].
        """
        # Basic input checks
        assert x.ndim == 6, f"Input must be [B,2,C,T,H,W], got {x.shape}"
        assert x.shape[1] == 2, "Need exactly 2 videos per sample"

        # Split into two video streams
        x1 = x[:, 0]  # [batch, C, T, H, W]
        x2 = x[:, 1]  # [batch, C, T, H, W]

        # Extract features for both videos
        x1_feats = self._process_features(x1)  # => [batch, seq_len, feat] or [batch, feat]
        x2_feats = self._process_features(x2)

        # Ensure we have a sequence dimension
        if x1_feats.ndim == 2:
            x1_feats = x1_feats.unsqueeze(1)  # => [batch, 1, feat]
        if x2_feats.ndim == 2:
            x2_feats = x2_feats.unsqueeze(1)

        # Concatenate sequences: [batch, seq_len1 + seq_len2, feat]
        combined_seq = torch.cat([x1_feats, x2_feats], dim=1)

        # Apply attention to the combined sequence
        combined_attended = self._apply_attention(combined_seq)

        # Split back into individual video sequences
        split_idx = x1_feats.shape[1]  # seq_len1
        x1_attended = combined_attended[:, :split_idx, :]  # [batch, seq_len1, feat]
        x2_attended = combined_attended[:, split_idx:, :]  # [batch, seq_len2, feat]

        # Mean pool each attended sequence
        x1_feats = x1_attended.mean(dim=1)  # => [batch, feat]
        x2_feats = x2_attended.mean(dim=1)

        # Concatenate pooled features
        combined = torch.cat((x1_feats, x2_feats), dim=1)  # => [batch, 2 * feature_dim]

        # Shared layers
        out = self.fc1(combined)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)  # => [batch, 512]
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Multi-head outputs
        outputs = {}
        for head_name, head_module in self.heads.items():
            outputs[head_name] = head_module(out)

        return outputs

    def _process_features(self, x):
        """
        Pass the video tensor x through the backbone and optionally
        reduce spatiotemporal dimensions by mean pooling.
        """
        features = self.video_feature_extractor(x)

        if features.dim() == 5:
            # e.g. [batch, c, t', h', w']
            features = features.mean(dim=[3, 4])  # => [batch, c, t']
            features = features.mean(dim=2)  # => [batch, c]
        return features

    def _apply_attention(self, x):
        """
        Applies multi-head self-attention, returns [batch, seq_len, feature_dim].
        """
        x = x.permute(1, 0, 2)  # => [seq_len, batch, feat]
        attn_out, _ = self.attention(x, x, x)
        return attn_out.permute(1, 0, 2)

    def _get_feature_dim_dynamic(self):
        """
        Infers the actual output channel dimension by running a dummy forward pass.
        Adjust (H,W) if needed for your backbone's input size.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        cpu_device = torch.device("cpu")

        # Move feature extractor temporarily to CPU
        self.video_feature_extractor.to(cpu_device)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.num_frames, 112, 112, device=cpu_device)
            feat = self.video_feature_extractor(dummy)
            if feat.dim() == 5:  # [1, c, t', h', w']
                feat = feat.mean(dim=[3, 4])  # => [1, c, t']
            if feat.dim() == 3:  # => [1, c, t']
                feat = feat.mean(dim=2)  # => [1, c]
            out_dim = feat.shape[1]

        # Restore original device & training mode
        self.video_feature_extractor.to(device)
        if was_training:
            self.train()

        return out_dim

    def _get_feature_extractor(self):
        """
        Returns the requested video feature extractor backbone.
        If x3d_m, attempts to load from PyTorchVideo + checkpoint if provided.
        """
        if self.feature_extractor_backbone == "r3d_18":
            model = models.video.r3d_18(pretrained=True)
        elif self.feature_extractor_backbone == "mc3_18":
            model = models.video.mc3_18(pretrained=True)
        elif self.feature_extractor_backbone == "r2plus1d_18":
            model = models.video.r2plus1d_18(pretrained=True)
        elif self.feature_extractor_backbone == "mvit_v1_b":
            model = models.video.mvit_v1_b(pretrained=True)
        elif self.feature_extractor_backbone == "mvit_v2_s":
            model = models.video.mvit_v2_s(pretrained=True)
        elif self.feature_extractor_backbone == "x3d_m":
            try:
                model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
                print("Loaded base x3d_m from PyTorchVideo (pretrained=False).")
            except Exception as e:
                raise RuntimeError(f"Error loading x3d_m model: {str(e)}")

            if self.checkpoint_path:
                # Load checkpoint
                try:
                    device = torch.device("cpu")
                    checkpoint = torch.load(self.checkpoint_path, map_location=device)
                    state_dict = checkpoint.get(
                        "model_state_dict", checkpoint.get("state_dict", checkpoint)
                    )
                    # Strip known prefixes
                    for prefix in ["module.", "_orig_mod.module.", "backbone."]:
                        if any(k.startswith(prefix) for k in state_dict.keys()):
                            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                                state_dict, prefix
                            )
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"Missing keys when loading checkpoint: {missing}")
                    if unexpected:
                        print(f"Unexpected keys in checkpoint: {unexpected}")
                    print(f"Loaded checkpoint from {self.checkpoint_path}")
                except Exception as e:
                    raise RuntimeError(f"Error loading checkpoint: {str(e)}")
            else:
                print("No checkpoint path provided; using randomly initialized x3d_m model.")
            model.blocks[-1].pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            model.blocks[-1].proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {self.feature_extractor_backbone}")

        return model

    def _freeze_layers(self):
        """
        PARTIAL-FREEZE LOGIC:
        We gather all parameters in the backbone (video_feature_extractor)
        and freeze the first portion by index, leaving the rest trainable.

        If freeze_ratio=0.8, then ~80% of parameters remain trainable
        (i.e., the last 80% by index) and ~20% are frozen.
        """
        # Collect all parameters
        all_named_params = list(self.video_feature_extractor.named_parameters())
        total_count = len(all_named_params)

        # Number of params to keep trainable:
        trainable_count = int(self.freeze_ratio * total_count)
        # Number of params to freeze:
        freeze_count = total_count - trainable_count

        # Freeze the first freeze_count params by index
        for i, (name, param) in enumerate(all_named_params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Tally: count total vs. frozen vs. trainable
        total_params = sum(p.numel() for p in self.video_feature_extractor.parameters())
        frozen_params = sum(
            p.numel() for p in self.video_feature_extractor.parameters() if not p.requires_grad
        )
        train_params = total_params - frozen_params

        print(f"[VideoPairClassifier] freeze_ratio={self.freeze_ratio:.2f}")
        print(f"  Total backbone parameters:    {total_params:,}")
        print(
            f"  Frozen backbone parameters:   {frozen_params:,} ({frozen_params/total_params:.2%})"
        )
        print(
            f"  Trainable backbone parameters:{train_params:,} ({train_params/total_params:.2%})"
        )
