"""
Masked Video Modeling for self-supervised learning of video representations.
Supports both feature reconstruction and HOG-based MaskFeat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Literal
import numpy as np


class MaskedVideoModeling(nn.Module):
    """
    Masked Video Modeling for self-supervised learning.
    
    This module:
    - Randomly masks video tokens
    - Reconstructs masked tokens from visible ones
    - Uses a lightweight decoder for reconstruction
    - Supports both patch-level and frame-level masking
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        decoder_hidden_size: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
        mask_type: str = "random",  # "random", "frame", "tube", "cube"
        mask_token_learnable: bool = True,
        norm_predict_loss: bool = True,
        use_patch_projection: bool = True,
        patch_size: int = 16,
        num_frames: int = 16,
        image_size: int = 224,
        target_type: str = "pixel",  # "pixel", "hog", "feature"
        hog_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_token_learnable = mask_token_learnable
        self.norm_predict_loss = norm_predict_loss
        self.use_patch_projection = use_patch_projection
        self.target_type = target_type
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame
        
        # Learnable mask token
        self.use_mask_token = True  # Enable efficient single-pass approach
        if mask_token_learnable:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.register_buffer('mask_token', torch.zeros(1, 1, hidden_size))
        
        # Decoder embedding for masked tokens
        self.decoder_embed = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_hidden_size)  # +1 for CLS token
        )
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Decoder for reconstruction
        self.decoder = MaskedVideoDecoder(
            hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
        )
        
        # Setup target-specific components
        if target_type == "hog":
            # Use ultra-fast cached HOG for speed
            use_cached = hog_config.get('use_cached', False)
            if use_cached:
                from .cached_hog import CachedHOGExtractor
                self.hog_extractor = CachedHOGExtractor(
                    orientations=hog_config.get('orientations', 6),
                    pixels_per_cell=hog_config.get('pixels_per_cell', (16, 16)),
                    patch_size=patch_size,
                    image_size=image_size,
                )
            else:
                from .hog_features import HOGFeatureExtractor
                self.hog_extractor = HOGFeatureExtractor(
                    patch_size=patch_size,
                    image_size=image_size,
                    **hog_config
                )
            # Prediction head for HOG features
            self.predict_head = nn.Linear(decoder_hidden_size, self.hog_extractor.output_dim)
        elif use_patch_projection:
            # Project to pixel values (3 channels * patch_size^2)
            self.predict_head = nn.Linear(decoder_hidden_size, 3 * patch_size * patch_size)
        else:
            # Project back to hidden size for feature reconstruction
            self.predict_head = nn.Linear(decoder_hidden_size, hidden_size)
        
        # Note: predict_norm removed - normalization now done in forward_loss
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: [B, C, T, H, W]
            
        Returns:
            patches: [B, T*H*W, patch_size^2 * C]
        """
        B, C, T, H, W = imgs.shape
        p = self.patch_size
        assert H == W and H % p == 0
        h = w = H // p
        
        x = imgs.reshape(B, C, T, h, p, w, p)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)  # [B, T, h, w, C, p, p]
        x = x.reshape(B, T * h * w, C * p * p)
        return x
    
    def unpatchify(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Convert patches back to images.
        
        Args:
            x: [B, T*h*w, patch_size^2 * C]
            T, H, W: Original temporal and spatial dimensions
            
        Returns:
            imgs: [B, C, T, H, W]
        """
        B, N, _ = x.shape
        p = self.patch_size
        h = w = H // p
        C = 3  # RGB channels
        
        x = x.reshape(B, T, h, w, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6)  # [B, C, T, h, p, w, p]
        x = x.reshape(B, C, T, H, W)
        return x
    
    def random_masking(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on input tokens.
        
        Args:
            x: Input tokens [B, N, D] where N includes CLS token
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Tuple of (masked_tokens, mask, ids_restore)
        """
        B, N, D = x.shape
        device = x.device
        
        # Keep CLS token, mask only patch tokens
        has_cls = N == self.num_patches + 1
        if has_cls:
            cls_token = x[:, 0:1, :]
            x_patches = x[:, 1:, :]
            N_patches = N - 1
        else:
            x_patches = x
            N_patches = N
        
        len_keep = int(N_patches * (1 - mask_ratio))
        
        if self.mask_type == "random":
            # Random masking
            noise = torch.rand(B, N_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        elif self.mask_type == "frame":
            # Frame-level masking (mask entire frames)
            h = w = self.image_size // self.patch_size
            per_frame = h * w
            
            # Generate random frame ordering per batch
            frame_noise = torch.rand(B, self.num_frames, device=device)
            frame_order = torch.argsort(frame_noise, dim=1)  # [B, F]
            
            # Create patch indices for each frame position
            patch_idx = torch.arange(per_frame, device=device)[None, None, :].expand(B, self.num_frames, per_frame)
            
            # Compute shuffled indices based on frame order
            base = (frame_order * per_frame).unsqueeze(-1)  # [B, F, 1]
            ids_shuffle = (base + patch_idx).reshape(B, -1)  # [B, Np]
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
        elif self.mask_type == "tube":
            # Tube masking (mask same spatial locations across frames)
            h = w = self.image_size // self.patch_size
            spatial_positions = h * w
            
            # Generate random spatial position ordering per batch
            tube_noise = torch.rand(B, spatial_positions, device=device)
            tube_order = torch.argsort(tube_noise, dim=1)  # [B, spatial_positions]
            
            # Expand tube mask across all frames
            ids_shuffle_list = []
            for frame in range(self.num_frames):
                offset = frame * spatial_positions
                # Each batch element gets its own tube ordering
                ids_shuffle_list.append(tube_order + offset)
            
            ids_shuffle = torch.cat(ids_shuffle_list, dim=1)  # [B, N_patches]
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
        elif self.mask_type == "cube":
            # Cube masking (3D blocks in spatiotemporal volume)
            # For simplicity, use random masking but could implement structured cube masking
            # Generate random noise for all patches
            noise = torch.rand(B, N_patches, device=device)  # [B, N]
            
            # Sort noise to get shuffle indices
            ids_shuffle = torch.argsort(noise, dim=1)  # [B, N]
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
        else:
            # Default to random masking if unknown type
            noise = torch.rand(B, N_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x_patches, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N_patches], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Add CLS token back if present
        if has_cls:
            x_masked = torch.cat([cls_token, x_masked], dim=1)
            mask = torch.cat([torch.zeros(B, 1, device=device), mask], dim=1)
            # Adjust ids_restore for CLS token
            ids_restore = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device),
                                    ids_restore + 1], dim=1)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, 
        x: torch.Tensor,
        encoder_model: nn.Module,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder with masking.
        
        Args:
            x: Input video [B, C, T, H, W]
            encoder_model: The video encoder model (e.g., MViT)
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Tuple of (latent, mask, ids_restore)
        """
        # Get encoder embeddings
        # For MViT, we need to extract intermediate features
        latent = encoder_model.get_intermediate_features(x)
        
        # Apply random masking
        latent_masked, mask, ids_restore = self.random_masking(latent, mask_ratio)
        
        return latent_masked, mask, ids_restore
    
    def forward_encoder_with_mask_tokens(
        self,
        x: torch.Tensor,
        encoder_model: nn.Module,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mask tokens before encoder (like MaskFeat paper).
        This is more efficient as it requires only one forward pass.
        
        Args:
            x: Input video [B, C, T, H, W] 
            encoder_model: The video encoder model
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Tuple of (full_latent, mask, ids_restore, ids_keep)
        """
        B, C, T, H, W = x.shape
        
        # First, patchify the input
        if self.use_patch_projection:
            patches = self.patchify(x)  # [B, N, D]
        else:
            # For feature-based models, get initial patches
            # We need to use a patch embedding layer similar to the encoder's
            # This is model-specific - for MViT we need its patch embed
            if hasattr(encoder_model, 'patch_embed'):
                # Use encoder's patch embedding
                with torch.no_grad():
                    patches = encoder_model.patch_embed(x)  # Initial tokenization
            else:
                # Fallback to getting features (less efficient)
                return self.forward_encoder(x, encoder_model, mask_ratio)
        
        # Generate mask
        N = patches.shape[1]
        len_keep = int(N * (1 - mask_ratio))
        
        # Random masking
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Create binary mask: 0 is keep, 1 is remove  
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Apply mask tokens to patches
        D = patches.shape[-1]
        mask_tokens = self.mask_token.repeat(B, N, 1)
        
        # Mix patches with mask tokens
        w = mask.unsqueeze(-1).type_as(mask_tokens)
        patches_with_mask = patches * (1 - w) + mask_tokens * w
        
        # Now forward through encoder with masked patches
        # This is a single forward pass
        latent = encoder_model.forward_features(patches_with_mask)
        
        return latent, mask, ids_restore, ids_keep
    
    def forward_decoder(
        self, 
        latent: torch.Tensor, 
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the decoder to reconstruct full sequence.
        
        Args:
            latent: Latent representation [B, N_visible, D]
            ids_restore: Indices to restore full sequence
            
        Returns:
            Reconstructed tokens [B, N_full, decoder_hidden_size]
        """
        B, N_visible, D = latent.shape
        N_full = ids_restore.shape[1]
        device = latent.device
        
        # Embed tokens
        x = self.decoder_embed(latent)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(B, N_full - N_visible, 1)
        mask_tokens = self.decoder_embed(mask_tokens)
        
        # Concatenate visible and mask tokens
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to original order
        x = torch.gather(x_, dim=1, 
                        index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        
        # Add positional embedding
        x = x + self.decoder_pos_embed[:, :N_full, :]
        
        # Apply decoder
        x = self.decoder(x)
        
        # Predict
        x = self.predict_head(x)
        
        # Note: predict_norm removed - normalization now done in forward_loss
        
        return x
    
    def forward_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for masked tokens.
        
        Args:
            pred: Predicted tokens [B, N, D]
            target: Target tokens [B, N, D]
            mask: Binary mask [B, N], 1 for masked positions
            
        Returns:
            Reconstruction loss
        """
        # Simple MSE loss for pixel reconstruction (no normalization)
        # Following standard MAE: reconstruct raw pixels
        if self.use_patch_projection:
            # MSE loss for pixel reconstruction
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # Mean over channels/pixels
        else:
            # MSE loss for feature reconstruction
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss.mean(dim=-1)  # Mean over feature dimension
        
        # Apply mask to get loss only on masked tokens
        loss = (loss * mask).sum() / mask.sum()
        
        # Scale loss to prevent saturation with HOG features
        loss = loss * 0.01
        
        return loss
    
    def forward(
        self, 
        x: torch.Tensor,
        encoder_model: Optional[nn.Module] = None,
        mask_ratio: Optional[float] = None,
        cached_hog_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of masked video modeling.
        
        Args:
            x: Input video [B, C, T, H, W] or features [B, N, D]
            encoder_model: Optional encoder model for feature extraction
            mask_ratio: Ratio of tokens to mask (uses self.mask_ratio if None)
            cached_hog_features: Pre-computed HOG features [B, T, num_patches, hog_dim]
            
        Returns:
            dict: Contains loss and other outputs
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Handle different input types
        if x.dim() == 5:  # Video input [B, C, T, H, W]
            # Get encoder features and apply masking
            if encoder_model is not None:
                # Get encoder features ONCE
                with torch.no_grad():
                    encoder_features = encoder_model.get_intermediate_features(x)
                    num_tokens = encoder_features.shape[1]
                
                # Apply masking to the features
                latent_masked, mask, ids_restore = self.random_masking(encoder_features, mask_ratio)
                latent = latent_masked
                
                # Create target based on target_type
                if self.target_type == "hog":
                    # Use cached HOG features if available
                    if cached_hog_features is not None:
                        # cached_hog_features is [B, T, num_patches, hog_dim]
                        B, T_hog, N_patches, D_hog = cached_hog_features.shape
                        target = cached_hog_features.reshape(B, T_hog * N_patches, D_hog)
                        # Log first few batches to confirm
                        import random
                        if random.random() < 0.5:  # 50% chance to log
                            print(f"[MVM] ✅ USING CACHED HOG from disk: shape {cached_hog_features.shape}")
                        
                        # Truncate or pad to match encoder tokens if needed
                        if target.shape[1] > num_tokens:
                            target = target[:, :num_tokens]
                        elif target.shape[1] < num_tokens:
                            padding = torch.zeros(B, num_tokens - target.shape[1], D_hog, device=target.device)
                            target = torch.cat([target, padding], dim=1)
                    else:
                        # For HOG with encoder (like MViT), we need to match encoder output dimensions
                        import random
                        if random.random() < 0.5:  # 50% chance to log
                            print(f"[MVM] ⚠️ COMPUTING HOG on-the-fly (no cache) for batch size {x.shape[0]}")
                        with torch.no_grad():
                            B, C, T, H, W = x.shape
                            
                            # Extract HOG for center patch of each space-time cube
                            # This matches the MaskFeat paper approach
                            hog_list = []
                            for t in range(T):
                                frame_hog = self.hog_extractor(x[:, :, t, :, :])  # [B, num_patches, hog_dim]
                                hog_list.append(frame_hog)
                            
                            full_hog = torch.stack(hog_list, dim=1)  # [B, T, num_patches, hog_dim]
                            B, T_hog, N_patches, D_hog = full_hog.shape
                            
                            # Reshape for processing
                            full_hog = full_hog.reshape(B, T_hog * N_patches, D_hog)  # [B, T*num_patches, hog_dim]
                            
                            # Skip complex pooling - just use the flattened features
                            target = full_hog
                            
                            # Truncate or pad to match encoder tokens if needed
                            if target.shape[1] > num_tokens:
                                target = target[:, :num_tokens]
                            elif target.shape[1] < num_tokens:
                                padding = torch.zeros(B, num_tokens - target.shape[1], D_hog, device=target.device)
                                target = torch.cat([target, padding], dim=1)
                            
                elif self.use_patch_projection:
                    target = self.patchify(x)
                else:
                    # Use encoder features as target
                    with torch.no_grad():
                        target = encoder_model.get_intermediate_features(x)
            else:
                # If no encoder, work directly with patches
                target = self.patchify(x)
                latent, mask, ids_restore = self.random_masking(target, mask_ratio)
                
                if self.target_type == "hog":
                    # Extract HOG for all frames (fast vectorized)
                    with torch.no_grad():
                        B, C, T, H, W = x.shape
                        
                        # Reshape to batch all frames together for speed
                        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                        
                        # Process all frames at once
                        target_flat = self.hog_extractor(x_flat)  # [B*T, num_patches, hog_dim]
                        
                        # Reshape back
                        target = target_flat.reshape(B, T, -1, target_flat.shape[-1])  # [B, T, num_patches, hog_dim]
                        target = target.reshape(B, -1, target.shape[-1])  # [B, T*num_patches, hog_dim]
        else:  # Feature input [B, N, D]
            target = x
            latent, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Forward through decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute loss
        loss = self.forward_loss(pred, target, mask)
        
        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask": mask,
            "latent": latent,
            "ids_restore": ids_restore,
        }


class MaskedVideoDecoder(nn.Module):
    """
    Lightweight decoder for masked video modeling.
    """
    
    def __init__(
        self,
        hidden_size: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.mlp_ratio = mlp_ratio
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size=hidden_size,
                num_heads=decoder_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(decoder_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tokens [B, N, D]
            
        Returns:
            Decoded tokens [B, N, D]
        """
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block for masked video modeling.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=attn_dropout,
            batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.
        
        Args:
            x: Input tokens [B, N, D]
            
        Returns:
            Output tokens [B, N, D]
        """
        # Self-attention with residual
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        
        # MLP with residual
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x