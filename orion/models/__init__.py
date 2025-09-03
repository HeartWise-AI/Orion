"""
Defining models used for orion
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from stam_pytorch import STAM
from timesformer_pytorch import TimeSformer

from orion.models.config import _C
from orion.models.masked_video_modeling import MaskedVideoModeling
from orion.models.rope_3d import Rope3D
from orion.models.videopairclassifier import VideoPairClassifier
from orion.models.vivit import ViViT
from orion.models.x3d_multi import Bottleneck as MultiBottleneck
from orion.models.x3d_multi import X3D_multi


def get_inplanes(version):
    if version == "S":
        return [(24, 24), (48, 48), (96, 96), (192, 192)]
    elif version == "M":
        return [(24, 24), (48, 48), (96, 96), (192, 192)]
    elif version == "L":
        return [(32, 32), (64, 64), (128, 128), (256, 256)]
    else:
        raise ValueError(f"Unsupported X3D version: {version}")


def get_blocks(version):
    if version == "S":
        return [3, 5, 11, 7]
    elif version == "M":
        return [3, 5, 11, 7]
    elif version == "L":
        return [5, 10, 25, 15]
    else:
        raise ValueError(f"Unsupported X3D version: {version}")


def x3d_multi(num_classes, **kwargs):
    model = X3D_multi(
        MultiBottleneck, get_blocks("M"), get_inplanes("M"), num_classes=num_classes, **kwargs
    )
    return model


def x3d_legacy(num_classes, **kwargs):
    model = X3D_legacy(
        LegacyBottleneck, get_blocks("M"), get_inplanes("M"), n_classes=num_classes, **kwargs
    )
    return model


def timesformer(num_classes, resize, **kwargs):
    """Constructs a timesformer model."""
    model = TimeSformer(
        dim=512,
        image_size=resize,
        patch_size=16,
        num_frames=64,
        num_classes=num_classes,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )
    return model


def vivit(num_classes, resize, num_frames, **kwargs):
    """Constructs a timesformer model."""
    model = ViViT(resize, num_classes, num_frames)
    return model


def movinet(num_classes, resize, **kwargs):
    """Constructs a timesformer model."""
    model = MoViNet(_C.MODEL.MoViNetA3, num_classes=num_classes)
    return model


def stam(num_classes, resize, **kwargs):
    model = STAM(
        dim=512,
        image_size=resize,  # size of image
        patch_size=32,  # patch size
        num_frames=64,  # number of image frames, selected out of video
        space_depth=12,  # depth of vision transformer
        space_heads=8,  # heads of vision transformer
        space_mlp_dim=2048,  # feedforward hidden dimension of vision transformer
        time_depth=6,  # depth of time transformer (in paper, it was shallower, 6)
        time_heads=8,  # heads of time transformer
        time_mlp_dim=2048,  # feedforward hidden dimension of time transformer
        num_classes=num_classes,  # number of output classes
        space_dim_head=64,  # space transformer head dimension
        time_dim_head=64,  # time transformer head dimension
        dropout=0.1,  # dropout
        emb_dropout=0.1,
    )  # embedding dropout
    return model


# New functions for loading and modifying models


class RegressionHead(nn.Module):
    def __init__(self, dim_in, num_classes=1):
        super().__init__()
        self.fc1 = nn.Conv3d(dim_in, 2048, bias=True, kernel_size=1, stride=1)
        self.regress = nn.Linear(2048, num_classes)
        
        # Initialize weights properly for regression
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.regress.weight)
        nn.init.constant_(self.regress.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # Add activation
        x = x.mean([2, 3, 4])
        x = self.regress(x)
        return x


class SimpleRegressionHead(nn.Module):
    """Simpler regression head for better stability."""
    def __init__(self, dim_in, num_classes=1):
        super().__init__()
        # Much smaller intermediate dimension
        self.fc1 = nn.Conv3d(dim_in, 512, bias=True, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.regress = nn.Linear(512, num_classes)
        
        # Proper initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.regress.weight, gain=0.01)  # Small gain for regression
        nn.init.constant_(self.regress.bias, 30.0)  # Initialize near mean LVEF

    def forward(self, x):
        x = self.fc1(x)
        x = x.mean([2, 3, 4])  # Global average pooling
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.regress(x)
        return x


class ClassificationHead(nn.Module):
    """Proper classification head for X3D models."""
    def __init__(self, dim_in, num_classes=1):
        super().__init__()
        # Smaller intermediate dimension for stability
        self.fc1 = nn.Conv3d(dim_in, 512, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.fc2 = nn.Conv3d(512, num_classes, kernel_size=1, stride=1, bias=True)
        
        # Better initialization for classification
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        # Conv3d expects [B, C, T, H, W]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Global average pooling
        x = x.mean([2, 3, 4])
        return x


class TransformerHead(nn.Module):
    def __init__(self, dim_in, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        # Global average pooling over the spatial and temporal dimensions
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MultiOutputTransformerHead(nn.Module):
    """Multi-output head for Transformer models like MViT."""
    def __init__(self, dim_in, head_structure: dict[str, int], dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleDict()
        for head_name, num_classes in head_structure.items():
            self.heads[head_name] = nn.Linear(dim_in, num_classes)
        self.head_structure = head_structure
    
    def forward(self, x):
        x = self.dropout(x)
        return {head_name: head_module(x) for head_name, head_module in self.heads.items()}


class MultiOutputHead(nn.Module):
    def __init__(self, dim_in, head_structure: dict[str, int], head_task: dict[str, str] = None):
        """
        Initialize a multi-output head with different number of classes per head.

        Args:
            dim_in (int): Input dimension
            head_structure (dict[str, int]): Dictionary mapping head names to their number of output classes
                e.g. {"contrast_agent": 1, "main_structure": 5, "stent_presence": 1}
            head_task (dict[str, str]): Optional dictionary mapping head names to task types ('classification' or 'regression')
        """
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.heads = nn.ModuleDict()
        self.head_task = head_task or {}
        
        for head_name, num_classes in head_structure.items():
            if not isinstance(num_classes, int) or num_classes < 1:
                raise ValueError(f"Invalid number of classes for head {head_name}: {num_classes}")
            
            # Check if this head is for regression or classification
            task_type = self.head_task.get(head_name, 'classification')
            
            if task_type == 'regression':
                # Use SimpleRegressionHead for regression tasks (more stable)
                self.heads[head_name] = SimpleRegressionHead(dim_in, num_classes)
            else:
                # For classification, use proper ClassificationHead
                self.heads[head_name] = ClassificationHead(dim_in, num_classes)

        self.head_structure = head_structure

    def forward(self, x):
        """
        Forward pass through all heads.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim_in]

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping head names to their outputs
        """
        x = self.dropout(x)
        return {head_name: head_module(x) for head_name, head_module in self.heads.items()}


def _patch_mvit_attention_for_rope(model, rope_modules):
    """Monkey-patch MViT's MultiScaleAttention to apply 3D RoPE.
    
    Args:
        model: MViT model to patch
        rope_modules: Dictionary of RoPE modules keyed by "{num_heads}_{head_dim}"
    """
    from types import MethodType
    import torch.nn.functional as F
    
    def patched_forward(self_attn, x, thw):
        """Patched forward for MultiScaleAttention with RoPE.
        
        Args:
            x: Input tensor [B, N, C]
            thw: Tuple of (T, H, W) dimensions
        """
        B, N, C = x.shape
        T, H, W = thw
        
        # Get actual dimensions from the attention module
        num_heads = self_attn.num_heads
        head_dim = self_attn.head_dim
        output_dim = self_attn.output_dim
        
        # Standard MViT attention computation
        # Project to Q, K, V  
        qkv = self_attn.qkv(x)
        # The output dimension is output_dim * 3 (for q, k, v)
        qkv = qkv.reshape(B, N, 3, num_heads, output_dim // num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply pooling if needed (MViT specific)
        if hasattr(self_attn, 'pool_q') and self_attn.pool_q is not None:
            q, thw_q = self_attn.pool_q(q, thw)
        else:
            thw_q = thw
            
        if hasattr(self_attn, 'pool_k') and self_attn.pool_k is not None:
            k, thw_k = self_attn.pool_k(k, thw)
        else:
            thw_k = thw
            
        if hasattr(self_attn, 'pool_v') and self_attn.pool_v is not None:
            v, thw_v = self_attn.pool_v(v, thw)
        else:
            thw_v = thw
        
        # Apply 3D RoPE to Q and K after pooling
        T_q, H_q, W_q = thw_q
        T_k, H_k, W_k = thw_k
        
        # Detect CLS tokens (N > T*H*W means we have special tokens)
        expected_q = T_q * H_q * W_q
        expected_k = T_k * H_k * W_k
        n_special_q = max(0, q.shape[2] - expected_q)  # Usually 0 or 1
        n_special_k = max(0, k.shape[2] - expected_k)
        
        # Apply RoPE if we have a matching module
        actual_head_dim = q.shape[-1]
        if rope_modules is not None and actual_head_dim % 6 == 0:
            rope_key = f"{num_heads}_{actual_head_dim}"
            if rope_key in rope_modules:
                try:
                    rope_module = rope_modules[rope_key]
                    # Handle Q and K separately if they have different dimensions
                    if q.shape[2] == k.shape[2] and (T_q, H_q, W_q) == (T_k, H_k, W_k):
                        # Same dimensions, apply RoPE to both
                        q, k = rope_module(q, k, T_q, H_q, W_q, n_special=n_special_q)
                    else:
                        # Different dimensions after pooling, apply separately
                        # Apply to Q
                        q_rot, _ = rope_module(q, q, T_q, H_q, W_q, n_special=n_special_q)
                        q = q_rot
                        # Apply to K with its dimensions
                        if (T_k * H_k * W_k + n_special_k) == k.shape[2]:
                            _, k_rot = rope_module(k, k, T_k, H_k, W_k, n_special=n_special_k)
                            k = k_rot
                    # Log only once per training to avoid spam
                    if not hasattr(self_attn, '_rope_applied_logged'):
                        cls_info = f" (with {n_special_q} CLS)" if n_special_q > 0 else ""
                        print(f"[RoPE] Applied to block with {num_heads} heads, head_dim={actual_head_dim}, grid=({T_q}x{H_q}x{W_q}){cls_info}")
                        self_attn._rope_applied_logged = True
                except Exception as e:
                    # Skip RoPE for this block if error
                    if not hasattr(self_attn, '_rope_error_logged'):
                        print(f"[RoPE] Error in block: {e}")
                        self_attn._rope_error_logged = True
        
        # Compute attention
        # MViT uses 'scaler' not 'scale'
        scale = 1.0 / math.sqrt(self_attn.head_dim) if hasattr(self_attn, 'head_dim') else 1.0 / math.sqrt(q.shape[-1])
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Apply relative position bias (fixes DDP unused parameters)
        if hasattr(self_attn, 'rel_pos_h') and hasattr(self_attn, 'rel_pos_w'):
            # Use MViT's relative position bias function
            from torchvision.models.video.mvit import _add_rel_pos
            attn = _add_rel_pos(
                attn,
                q,  # Need to pass q tensor
                thw_q,  # q dimensions  
                thw_k,  # k dimensions
                self_attn.rel_pos_h,
                self_attn.rel_pos_w,
                self_attn.rel_pos_t if hasattr(self_attn, 'rel_pos_t') else None
            )
        
        attn = attn.softmax(dim=-1)
        attn = self_attn.attn_drop(attn) if hasattr(self_attn, 'attn_drop') else attn
        
        # Apply to values
        x = attn @ v  # [B, num_heads, N_q, head_dim]
        x = x.transpose(1, 2)  # [B, N_q, num_heads, head_dim]
        
        # Reshape to [B, N_q, output_dim]
        # Note: output_dim may be different from C (input dim)
        x = x.reshape(B, x.shape[1], -1)
        x = self_attn.project(x)  # MViT uses 'project' not 'proj'
        
        return x, thw_q
    
    # Patch all attention blocks in MViT
    for block in model.blocks:
        if hasattr(block, 'attn'):
            block.attn.forward = MethodType(patched_forward, block.attn)


class MVMHead(nn.Module):
    """Head for Masked Video Modeling that wraps MVM functionality."""
    
    def __init__(self, dim_in, mvm_config):
        super().__init__()
        self.dim_in = dim_in
        self.mvm_config = mvm_config
        
        # Create MVM module
        self.mvm = MaskedVideoModeling(
            hidden_size=mvm_config.get('hidden_size', dim_in),
            decoder_hidden_size=mvm_config.get('decoder_hidden_size', 384),
            decoder_layers=mvm_config.get('decoder_layers', 4),
            decoder_heads=mvm_config.get('decoder_heads', 6),
            mask_ratio=mvm_config.get('mask_ratio', 0.75),
            mask_type=mvm_config.get('mask_type', 'random'),
            mask_token_learnable=mvm_config.get('mask_token_learnable', True),
            norm_predict_loss=mvm_config.get('norm_predict_loss', True),
            use_patch_projection=mvm_config.get('use_patch_projection', True),
            patch_size=mvm_config.get('patch_size', 16),
            num_frames=mvm_config.get('num_frames', 16),
            image_size=mvm_config.get('image_size', 224),
            target_type=mvm_config.get('target_type', 'pixel'),
            hog_config=mvm_config.get('hog_config', None),
        )
        
        # Store encoder reference (will be set by model)
        self.encoder = None
    
    def forward_with_encoder(self, x, encoder):
        """Forward pass for MVM head with encoder provided.
        
        Args:
            x: Video tensors [B, C, T, H, W]
            encoder: The encoder model to use for feature extraction
        """
        if x.dim() != 5:
            # Return a dummy loss for compatibility
            return {'loss': torch.tensor(0.0, device=x.device)}
            
        # Pass video through MVM with encoder
        outputs = self.mvm(x, encoder_model=encoder, mask_ratio=self.mvm_config.get('mask_ratio', 0.75))
        
        # The target is already set correctly inside MVM forward
        # For feature reconstruction, it's the encoded features
        # For patch reconstruction, it's the patchified video
        
        return outputs
    
    def forward(self, x):
        """Forward pass for MVM head (compatibility)."""
        # This should not be called directly when encoder is needed
        if x.dim() != 5:
            return {'loss': torch.tensor(0.0, device=x.device)}
        # Return dummy outputs for compatibility
        return {'loss': torch.tensor(0.0, device=x.device)}


def replace_final_layer(model, config):
    """
    Replace the final layer of the model based on the problem type.
    """
    if hasattr(model, "head"):
        if isinstance(model.head, nn.Sequential):
            in_features = model.head[-1].in_features
        elif isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
        else:
            in_features = model.head.fc1.in_channels if hasattr(model.head, "fc1") else None

        if in_features is None:
            raise AttributeError("Unable to determine input features for the final layer")

        if "mvit" in config["model_name"].lower():
            if config["task"] == "regression":
                # Check if using multi_head loss for regression
                if config.get("loss") == "multi_head" and "head_structure" in config:
                    # For multi-head regression with MViT, use MultiOutputTransformerHead
                    model.head = MultiOutputTransformerHead(
                        dim_in=in_features, 
                        head_structure=config["head_structure"],
                        dropout=0.5
                    )
                    print(f"Replaced final layer with MultiOutputTransformerHead(dim_in={in_features}, head_structure={config['head_structure']}) for multi-head regression")
                else:
                    # Single head regression
                    model.head = TransformerHead(dim_in=in_features, num_classes=1)
                    print(f"Replaced final layer with TransformerHead(dim_in={in_features}, num_classes=1) for regression")
            elif config["task"] == "classification":
                # Check if using multi_head loss for classification
                if config.get("loss") == "multi_head" and "head_structure" in config:
                    # For multi-head classification with MViT, use MultiOutputTransformerHead
                    model.head = MultiOutputTransformerHead(
                        dim_in=in_features, 
                        head_structure=config["head_structure"],
                        dropout=0.5
                    )
                    print(f"Replaced final layer with MultiOutputTransformerHead(dim_in={in_features}, head_structure={config['head_structure']}) for multi-head classification")
                else:
                    # Single head classification
                    num_classes = config["num_classes"]
                    model.head = TransformerHead(dim_in=in_features, num_classes=num_classes)
                    print(f"Replaced final layer with TransformerHead(dim_in={in_features}, num_classes={num_classes}) for classification")
            elif config["task"] == "masked_video_modeling":
                if "mvm_config" not in config:
                    # Default MVM configuration for MViT
                    # MViT uses different tokenization, so use feature reconstruction
                    config["mvm_config"] = {
                        "hidden_size": in_features,
                        "decoder_hidden_size": 384,
                        "decoder_layers": 4,
                        "decoder_heads": 6,
                        "mask_ratio": 0.75,
                        "mask_type": "random",
                        "use_patch_projection": False,  # MViT doesn't use standard patches
                        "target_type": "feature",  # Default to feature reconstruction
                        "patch_size": 16,
                        "num_frames": config.get("frames", 16),
                        "image_size": config.get("resize", 224),
                    }
                model.head = MVMHead(dim_in=in_features, mvm_config=config["mvm_config"])
                # Don't set encoder reference to avoid circular reference
                # Instead, we'll pass the encoder in the forward method
                print(f"Replaced final layer with MVMHead for masked video modeling pretraining")
                
                # Add get_intermediate_features method to MViT for MVM
                from types import MethodType
                def get_intermediate_features(self, x):
                    """Extract intermediate features without head for MVM."""
                    from torchvision.models.video.mvit import _unsqueeze
                    x = _unsqueeze(x, 5, 2)[0]
                    x = self.conv_proj(x)
                    x = x.flatten(2).transpose(1, 2)
                    x = self.pos_encoding(x)
                    thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
                    for block in self.blocks:
                        x, thw = block(x, thw)
                    x = self.norm(x)
                    return x  # Return features without head processing
                model.get_intermediate_features = MethodType(get_intermediate_features, model)
                
                # Monkey-patch MViT forward for MVM
                original_forward = model.forward
                def mvm_forward(self, x):
                    """MVM forward that returns MVM outputs directly."""
                    # Pass the encoder (self) to the head
                    return self.head.forward_with_encoder(x, self)
                model.forward = MethodType(mvm_forward, model)
                print("Patched MViT with get_intermediate_features and MVM forward")
            else:
                raise ValueError(f"Unsupported task for MViT: {config['task']}")
        elif "swin3d" in config["model_name"].lower():
            model.head = nn.Linear(
                in_features=in_features, out_features=config["num_classes"], bias=True
            )
            print(
                f"Replaced final layer with Linear(in_features={in_features}, out_features={config['num_classes']}, bias=True) for {config['task']}"
            )
        else:
            if config["task"] == "regression":
                # Check if using multi_head loss for regression
                if config.get("loss") == "multi_head" and "head_structure" in config:
                    # For multi-head regression with x3d, create MultiOutputHead
                    model.head = MultiOutputHead(
                        dim_in=in_features, 
                        head_structure=config["head_structure"],
                        head_task=config.get("head_task", {})
                    )
                    print(
                        f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']}) for multi-head regression"
                    )
                else:
                    # Single head regression
                    model.head = RegressionHead(dim_in=in_features, num_classes=1)
                    print(
                        f"Replaced final layer with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
                    )
            elif config["task"] == "classification":
                if "head_structure" not in config:
                    raise ValueError(
                        "head_structure must be defined in config for multi_head task"
                    )
                model.head = MultiOutputHead(
                    dim_in=in_features, 
                    head_structure=config["head_structure"],
                    head_task=config.get("head_task", {})
                )
                print(
                    f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
                )
            else:
                raise ValueError(f"Unsupported task: {config['task']}")
    elif hasattr(model, "blocks") and hasattr(model.blocks[-1], "proj"):
        # Check and print in_features or shape of the first block
        if hasattr(model.blocks[0], "proj"):
            print(f"in_features of model.blocks[0]: {model.blocks[0].proj.in_features}")
        else:
            print(f"model.blocks[0] does not have 'proj' attribute")

        # Print the in_features of the last block's projection layer
        in_features_last_block = model.blocks[-1].proj.in_features
        print(f"in_features of model.blocks[-1]: {in_features_last_block}")

        # Determine the input features for the new head based on the model type
        if "x3d" in config.get("model_name", ""):
            in_features = 192  # This value is specific for x3d models
        else:
            in_features = in_features_last_block

        # Replace the last block with the appropriate head based on the task
        if config["task"] == "regression":
            # Check if using multi_head loss for regression
            if config.get("loss") == "multi_head" and "head_structure" in config:
                # For multi-head regression, create MultiOutputHead
                model.blocks[-1] = MultiOutputHead(
                    dim_in=in_features, 
                    head_structure=config["head_structure"],
                    head_task=config.get("head_task", {})
                )
                print(
                    f"Replaced final block with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']}) for multi-head regression"
                )
            else:
                # Single head regression
                model.blocks[-1] = RegressionHead(dim_in=in_features, num_classes=1)
                print(
                    f"Replaced final block with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
                )
        elif config["task"] == "classification":
            if "head_structure" not in config:
                raise ValueError("head_structure must be defined in config for multi_head task")
            model.blocks[-1] = MultiOutputHead(
                dim_in=in_features, 
                head_structure=config["head_structure"],
                head_task=config.get("head_task", {})
            )
            print(
                f"Replaced final block with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
            )
        elif config["task"] == "masked_video_modeling":
            if "mvm_config" not in config:
                # Default MVM configuration
                config["mvm_config"] = {
                    "hidden_size": in_features,
                    "decoder_hidden_size": 384,
                    "decoder_layers": 4,
                    "decoder_heads": 6,
                    "mask_ratio": 0.75,
                    "mask_type": "random",
                    "use_patch_projection": True,
                    "patch_size": 16,
                    "num_frames": config.get("frames", 16),
                    "image_size": config.get("resize", 224),
                }
            model.blocks[-1] = MVMHead(dim_in=in_features, mvm_config=config["mvm_config"])
            print(
                f"Replaced final block with MVMHead for masked video modeling pretraining"
            )
        else:
            raise ValueError(f"Unsupported task: {config['task']}")
    elif hasattr(model, "norm") and isinstance(model.norm, nn.LayerNorm):
        in_features = model.head[1].in_features

        if config["task"] == "regression":
            # Check if using multi_head loss for regression
            if config.get("loss") == "multi_head" and "head_structure" in config:
                # For multi-head regression, create MultiOutputHead
                model.head = MultiOutputHead(
                    dim_in=in_features, 
                    head_structure=config["head_structure"],
                    head_task=config.get("head_task", {})
                )
                print(
                    f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']}) for multi-head regression"
                )
            else:
                # Single head regression
                model.head = RegressionHead(dim_in=in_features, num_classes=1)
                print(
                    f"Replaced final layer with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
                )
        elif config["task"] == "classification":
            if "head_structure" not in config:
                raise ValueError("head_structure must be defined in config for multi_head task")
            model.head = MultiOutputHead(
                dim_in=in_features, 
                head_structure=config["head_structure"],
                head_task=config.get("head_task", {})
            )
            print(
                f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
            )
        elif config["task"] == "masked_video_modeling":
            if "mvm_config" not in config:
                # Default MVM configuration
                config["mvm_config"] = {
                    "hidden_size": in_features,
                    "decoder_hidden_size": 384,
                    "decoder_layers": 4,
                    "decoder_heads": 6,
                    "mask_ratio": 0.75,
                    "mask_type": "random",
                    "use_patch_projection": True,
                    "patch_size": 16,
                    "num_frames": config.get("frames", 16),
                    "image_size": config.get("resize", 224),
                }
            model.head = MVMHead(dim_in=in_features, mvm_config=config["mvm_config"])
            print(
                f"Replaced final layer with MVMHead for masked video modeling pretraining"
            )
        else:
            raise ValueError(f"Unsupported task: {config['task']}")
    else:
        raise AttributeError("Unable to determine input features for the final layer")

    return model


def add_mvm_adapter_to_model(model, config):
    """Add MVM adapter to existing model for feature extraction.
    
    This allows the model to output intermediate features for MVM.
    """
    def get_intermediate_features(self, x):
        """Extract intermediate features for MVM."""
        # This is a simplified version - actual implementation depends on model architecture
        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)
        elif hasattr(self, 'stem'):
            x = self.stem(x)
        
        # Pass through most blocks but not the final head
        if hasattr(self, 'blocks'):
            for i, block in enumerate(self.blocks[:-1]):
                x = block(x)
        
        return x
    
    # Monkey-patch the method
    from types import MethodType
    model.get_intermediate_features = MethodType(get_intermediate_features, model)
    return model


def load_and_modify_model(config):
    """
    Load a model and modify its final layer based on the configuration.
    """
    if "num_classes" not in config and "head_structure" not in config and config.get("task") != "masked_video_modeling":
        raise ValueError("num_classes, head_structure, or masked_video_modeling task must be defined in config")
    num_classes = config["num_classes"] if "num_classes" in config else 1
    resize = config.get("resize", 256)
    num_frames = config.get("frames", 64)
    print(f"model_name: {config['model_name']}")

    if config["model_name"] == "x3d_multi":
        model = x3d_multi(num_classes, **config)
    elif config["model_name"] == "timesformer":
        model = timesformer(num_classes, resize, **config)
    elif config["model_name"] == "vivit":
        model = vivit(num_classes, resize, num_frames, **config)
    elif config["model_name"] == "stam":
        model = stam(num_classes, resize, **config)
    elif config["model_name"] in [
        "c2d_r50",
        "i3d_r50",
        "slow_r50",
        "slowfast_r50",
        "slowfast_r101",
        "slowfast_16x8_r101_50_50",
        "csn_r101",
        "r2plus1d_r50",
        "x3d_xs",
        "x3d_s",
        "x3d_m",
        "x3d_l",
        "efficient_x3d_xs",
        "efficient_x3d_s",
        "swin3d_s",
        "swin3d_b",
    ]:
        import torchvision

        if config["model_name"] in ["swin3d_s", "swin3d_b"]:
            model = getattr(torchvision.models.video, config["model_name"])(
                weights="KINETICS400_V1"
            )
        else:
            logging.basicConfig(level=logging.DEBUG)
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", config["model_name"], pretrained=True
            )

        # Replace the final layer only for these models
        model = replace_final_layer(model, config)
    elif config["model_name"] == "mvit_v1_b":
        from torchvision.models.video import mvit_v1_b

        model = mvit_v1_b(weights="KINETICS400_V1")
        model = replace_final_layer(model, config)
        
        # Apply RoPE if enabled
        if config.get("use_rope", False):
            rope_modules = {}
            # MViT v1 B typical configurations
            configs = [
                (1, 96),   # Early blocks
                (2, 96),   # Mid blocks
                (8, 96),   # Later blocks
            ]
            
            for num_heads, head_dim in configs:
                if head_dim % 6 == 0:  # Check 3D RoPE compatibility
                    key = f"{num_heads}_{head_dim}"
                    rope_module = Rope3D(
                        embed_dim=num_heads * head_dim,
                        num_heads=num_heads,
                        temporal_base=config.get("rope_base", 10000.0),
                        spatial_base=config.get("rope_base", 10000.0),
                        temporal_scale=config.get("rope_temporal_scale", 1.0),
                        normalize_mode=config.get("rope_normalize_mode", "separate")
                    )
                    # Keep RoPE in training mode to allow gradients to flow through
                    rope_modules[key] = rope_module
            
            if rope_modules:
                # Register RoPE modules as part of the model so they participate in DDP
                model.rope_modules = nn.ModuleDict(rope_modules)
                _patch_mvit_attention_for_rope(model, model.rope_modules)
                print(f"Applied 3D RoPE to MViT v1 B model with configurations: {list(rope_modules.keys())}")
    
    elif config["model_name"] == "mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        # Load base model
        if config.get("pretrained", True):
            model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        else:
            model = mvit_v2_s(weights=None)
        
        # Load from checkpoint if specified
        if config.get("checkpoint_path"):
            checkpoint_path = config["checkpoint_path"]
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle MVM pretraining checkpoint
            if config.get("load_mvm_encoder_only", False):
                # MVM checkpoint has 'model_state_dict' with encoder weights
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                
                # Filter out MVM-specific layers (decoder, etc.)
                encoder_state_dict = {}
                for k, v in state_dict.items():
                    # Skip MVM decoder and head layers
                    if not any(skip in k for skip in ["decoder", "mvm_head", "mask_token"]):
                        # Remove 'module.' prefix if present (from DDP)
                        key = k.replace("module.", "")
                        encoder_state_dict[key] = v
                
                # Load encoder weights
                missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
                print(f"Loaded MVM encoder weights from checkpoint")
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
            else:
                # Regular checkpoint loading
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded full model weights from checkpoint")
        
        model = replace_final_layer(model, config)
        
        # Apply RoPE if enabled
        if config.get("use_rope", False):
            rope_modules = {}
            # MViT v2 S typical configurations
            configs = [
                (1, 96),   # Blocks 0-2: 1 head, 96 dim/head
                (2, 96),   # Blocks 3-13: 2 heads, 96 dim/head
                (8, 96),   # Blocks 14-15: 8 heads, 96 dim/head
            ]
            
            for num_heads, head_dim in configs:
                if head_dim % 6 == 0:  # Check 3D RoPE compatibility
                    key = f"{num_heads}_{head_dim}"
                    rope_module = Rope3D(
                        embed_dim=num_heads * head_dim,
                        num_heads=num_heads,
                        temporal_base=config.get("rope_base", 10000.0),
                        spatial_base=config.get("rope_base", 10000.0),
                        temporal_scale=config.get("rope_temporal_scale", 1.0),
                        normalize_mode=config.get("rope_normalize_mode", "separate")
                    )
                    # Keep RoPE in training mode to allow gradients to flow through
                    rope_modules[key] = rope_module
            
            if rope_modules:
                # Register RoPE modules as part of the model so they participate in DDP
                model.rope_modules = nn.ModuleDict(rope_modules)
                _patch_mvit_attention_for_rope(model, model.rope_modules)
                print(f"Applied 3D RoPE to MViT v2 S model with configurations: {list(rope_modules.keys())}")
    elif config["model_name"] == "videopairclassifier":
        if "feature_extractor_backbone" not in config:
            raise ValueError("feature_extractor_backbone must be defined")

        # Get head structure from config
        head_structure = config.get("head_structure", None)

        model = VideoPairClassifier(
            head_structure=head_structure,  # Pass head structure
            feature_extractor_backbone=config["feature_extractor_backbone"],
            checkpoint_path=config.get("feature_extractor_checkpoint_path", None),
            num_frames=config.get("frames", 64),
            freeze_ratio=0,
            num_heads=2,
        )
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")
    
    # Add MVM adapter if using masked video modeling (skip for MViT, already handled)
    if config.get("task") == "masked_video_modeling" and "mvit" not in config["model_name"].lower():
        model = add_mvm_adapter_to_model(model, config)
    
    return model
