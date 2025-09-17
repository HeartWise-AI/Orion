"""Weights & Biases logging utilities for SSL training with MVM and 3D RoPE."""

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from typing import Dict, Optional, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import cv2
import warnings
warnings.filterwarnings('ignore')


class SSLWandbLogger:
    """Comprehensive W&B logger for SSL training with MVM and 3D RoPE."""
    
    def __init__(
        self,
        project: str = "ssl-mvit-rope",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
    ):
        """
        Initialize W&B logger for SSL training.
        
        Args:
            project: W&B project name
            entity: W&B entity/team name
            config: Configuration dictionary
            name: Run name
            tags: List of tags for the run
            notes: Notes about the run
            mode: W&B mode (online, offline, disabled)
        """
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.mode = mode
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
            tags=tags or ["ssl", "mvm", "rope3d"],
            notes=notes,
            mode=mode,
        )
        
        # Tracking variables
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def log_training_dynamics(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        batch_time: Optional[float] = None,
        data_time: Optional[float] = None,
        memory_gb: Optional[float] = None,
        mask_ratio: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log training dynamics metrics.
        
        Args:
            loss: Current loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm (if gradient clipping is used)
            batch_time: Time for batch processing
            data_time: Time for data loading
            memory_gb: GPU memory usage in GB
            mask_ratio: Current masking ratio
            step: Training step (uses global_step if None)
        """
        if step is None:
            step = self.global_step
            
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/epoch": self.epoch,
        }
        
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
            
        if batch_time is not None:
            metrics["train/batch_time"] = batch_time
            
        if data_time is not None:
            metrics["train/data_time"] = data_time
            
        if memory_gb is not None:
            metrics["train/memory_gb"] = memory_gb
            
        if mask_ratio is not None:
            metrics["train/mask_ratio"] = mask_ratio
            
        wandb.log(metrics, step=step)
        
    def log_reconstruction_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log reconstruction quality metrics.
        
        Args:
            pred: Predicted patches [B, N, D]
            target: Target patches [B, N, D]
            mask: Binary mask [B, N]
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        with torch.no_grad():
            # Calculate MSE for masked and unmasked regions
            mse = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
            
            # Masked region metrics
            masked_mse = (mse * mask).sum() / mask.sum()
            
            # Unmasked region metrics (should be very low if using teacher)
            unmasked_mask = 1 - mask
            if unmasked_mask.sum() > 0:
                unmasked_mse = (mse * unmasked_mask).sum() / unmasked_mask.sum()
            else:
                unmasked_mse = 0.0
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            # Assuming normalized values in [-1, 1]
            max_val = 2.0  # Range from -1 to 1
            psnr = 20 * torch.log10(max_val / torch.sqrt(masked_mse))
            
            # Calculate reconstruction accuracy (% of correctly reconstructed patches)
            # Using a threshold for "correct" reconstruction
            threshold = 0.1  # Adjust based on your needs
            correct_recon = (mse < threshold).float()
            recon_accuracy = (correct_recon * mask).sum() / mask.sum()
            
            metrics = {
                "recon/masked_mse": masked_mse.item(),
                "recon/unmasked_mse": unmasked_mse.item() if isinstance(unmasked_mse, torch.Tensor) else unmasked_mse,
                "recon/psnr": psnr.item(),
                "recon/accuracy": recon_accuracy.item(),
                "recon/mask_percentage": mask.mean().item(),
            }
            
            wandb.log(metrics, step=step)
    
    def log_reconstruction_visualizations(
        self,
        videos: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 4,
        step: Optional[int] = None,
        target_type: str = "pixel",
        hog_targets: Optional[torch.Tensor] = None,
    ):
        """
        Log reconstruction visualizations to W&B.
        
        Args:
            videos: Original videos [B, C, T, H, W]
            pred: Predictions [B, N, D] or reconstructed videos
            mask: Binary mask [B, N]
            num_samples: Number of samples to visualize
            step: Training step
            target_type: Type of target ("pixel", "feature", "hog")
            hog_targets: HOG targets if using HOG masking [B, N, D]
        """
        print(f"[HOG VIZ DEBUG] log_reconstruction_visualizations called with target_type={target_type}, num_samples={num_samples}, step={step}")
        print(f"[HOG VIZ DEBUG] videos shape: {videos.shape}, pred shape: {pred.shape if pred is not None else 'None'}, mask shape: {mask.shape if mask is not None else 'None'}")
        
        if step is None:
            step = self.global_step
            
        B = min(videos.shape[0], num_samples)
        print(f"[HOG VIZ DEBUG] Processing {B} samples for visualization")
        
        # ALWAYS generate visualizations, no matter what
        images = []
        
        for b in range(B):
            print(f"[HOG VIZ DEBUG] Processing sample {b+1}/{B}, WILL generate image")
            # Get middle frame for visualization
            T = videos.shape[2]
            mid_frame = T // 2
            
            # Original frame
            orig_frame = videos[b, :, mid_frame].cpu()
            
            # Create masked version - agnostic to number of tokens
            if mask.dim() == 2:  # [B, N]
                N = mask.shape[1]
                # Try to create a 2D visualization from the mask tokens
                
                # Check if it's a perfect square
                sqrt_n = int(np.sqrt(N))
                if sqrt_n * sqrt_n == N:
                    # Perfect square - reshape directly
                    h = w = sqrt_n
                    mask_2d = mask[b].reshape(h, w).cpu()
                else:
                    # Not a perfect square - need to handle differently
                    # Find closest factors for rectangular arrangement
                    factors = []
                    for i in range(1, int(np.sqrt(N)) + 1):
                        if N % i == 0:
                            factors.append((i, N // i))
                    
                    if factors:
                        # Use the most square-like factorization
                        h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                        try:
                            mask_2d = mask[b][:h*w].reshape(h, w).cpu()
                        except:
                            # If exact reshape fails, pad or truncate
                            mask_flat = mask[b].cpu()
                            if len(mask_flat) > h * w:
                                mask_2d = mask_flat[:h*w].reshape(h, w)
                            else:
                                # Pad with zeros
                                padded = torch.zeros(h * w)
                                padded[:len(mask_flat)] = mask_flat
                                mask_2d = padded.reshape(h, w)
                    else:
                        # No good factorization - create a single row
                        h, w = 1, N
                        mask_2d = mask[b].unsqueeze(0).cpu()
                
                # Upsample mask to match image size
                mask_upsampled = F.interpolate(
                    mask_2d.unsqueeze(0).unsqueeze(0).float(),
                    size=orig_frame.shape[-2:],
                    mode='nearest'
                )[0, 0]
                mask_upsampled = mask_upsampled.cpu()
            else:
                mask_upsampled = torch.ones_like(orig_frame[0])
            
            # Apply mask to create masked input visualization
            masked_frame = orig_frame.clone()
            for c in range(3):
                masked_frame[c] = masked_frame[c] * (1 - mask_upsampled) + mask_upsampled * 0.5
            
            # Create figure with 5 subplots for HOG overlay
            num_subplots = 5 if target_type == "hog" else 4
            fig, axes = plt.subplots(1, num_subplots, figsize=(4*num_subplots, 4))
            
            # Plot original
            axes[0].imshow(self._tensor_to_image(orig_frame))
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # Plot mask
            axes[1].imshow(mask_upsampled.cpu().numpy(), cmap='gray')
            axes[1].set_title(f"Mask ({mask[b].mean().item():.1%})")
            axes[1].axis('off')
            
            # Plot masked input
            axes[2].imshow(self._tensor_to_image(masked_frame))
            axes[2].set_title("Masked Input")
            axes[2].axis('off')
            
            # Plot reconstruction or HOG visualization
            if target_type == "hog" and pred is not None:
                # For HOG, visualize as edge-like feature maps
                if pred.dim() == 3:  # [B, N, D]
                    # Get the HOG features for this sample
                    N = pred.shape[1]
                    D = pred.shape[2]
                    
                    # For MViT's 3136 tokens, use 56x56 grid
                    if N == 3136:
                        h = w = 56
                        num_tokens = N
                    elif N == 393:
                        h = w = 20
                        num_tokens = min(h * w, N)
                    else:
                        sqrt_n = int(np.sqrt(N))
                        if sqrt_n * sqrt_n == N:
                            h = w = sqrt_n
                            num_tokens = N
                        else:
                            # Find closest factors
                            factors = []
                            for i in range(1, int(np.sqrt(N)) + 1):
                                if N % i == 0:
                                    factors.append((i, N // i))
                            if factors:
                                h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                                num_tokens = min(h * w, N)
                            else:
                                h, w = 1, N
                                num_tokens = N
                    
                    # Extract HOG features and compute gradient magnitude
                    hog_features = pred[b, :num_tokens].cpu().detach()  # [num_tokens, D]
                    
                    # For 4 orientations, just compute L2 norm
                    hog_magnitude = torch.norm(hog_features, dim=-1)  # [num_tokens]
                    
                    # Reshape to spatial grid
                    if num_tokens < h * w:
                        # Pad with zeros to fill the grid
                        padded_magnitude = torch.zeros(h * w)
                        padded_magnitude[:num_tokens] = hog_magnitude
                        hog_grid = padded_magnitude.view(h, w)
                    else:
                        hog_grid = hog_magnitude.view(h, w)
                    
                    # Normalize for visualization
                    grid_min = hog_grid.min()
                    grid_max = hog_grid.max()
                    grid_range = grid_max - grid_min
                    
                    # Log statistics for debugging
                    if b == 0:  # Only log for first sample
                        print(f"[HOG Viz] Grid stats - Min: {grid_min:.6f}, Max: {grid_max:.6f}, Range: {grid_range:.6f}")
                    
                    # Normalize
                    if grid_range > 1e-6:
                        hog_grid = (hog_grid - grid_min) / grid_range
                    
                    # Upsample to match image size
                    hog_upsampled = F.interpolate(
                        hog_grid.unsqueeze(0).unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]
                    
                    # Use viridis colormap for better contrast
                    axes[3].imshow(hog_upsampled.numpy(), cmap='viridis')
                    axes[3].set_title("HOG Feature Map")
                    
                    # Add HOG overlay visualization (5th subplot) - ALWAYS generate
                    if num_subplots == 5:
                        # Create proper HOG visualization with oriented edges
                        # pred[b] contains HOG features [3136, 9] where 9 = orientations (or 4 if using 4 bins)
                        
                        # Convert original frame to numpy for drawing
                        overlay_img = self._tensor_to_image(orig_frame).copy()
                        H, W = overlay_img.shape[:2]
                        
                        # For each patch, find the dominant orientation
                        hog_pred = pred[b].cpu().detach()  # [3136, num_orientations]
                        num_patches = hog_pred.shape[0]
                        num_orientations = hog_pred.shape[1]
                        
                        # Calculate grid size (assuming square for MViT)
                        if num_patches == 3136:
                            grid_size = 56  # MViT uses 56x56 patches
                        else:
                            grid_size = int(np.sqrt(num_patches))
                        
                        # Reshape to grid
                        hog_grid_full = hog_pred.view(grid_size, grid_size, num_orientations)  # grid x grid patches, N orientations
                        
                        # Define orientation angles based on number of bins
                        # HOG uses unsigned gradients, so angles are 0-180 degrees
                        angles = np.linspace(0, 180, num_orientations, endpoint=False) * np.pi / 180
                        
                        # Create line overlay using OpenCV for better quality
                        import cv2
                        
                        # For visualization, show every few patches for clarity
                        # Use same spacing as in our test visualization
                        patch_step = 12  # Show every 12th patch for clearer visualization
                        arrow_scale = 2.0  # Scale factor for arrow length
                        
                        # Normalize HOG features for better visualization
                        hog_min = hog_grid_full.min()
                        hog_max = hog_grid_full.max()
                        hog_range = hog_max - hog_min
                        
                        if b == 0:  # Log once per batch
                            print(f"[HOG Edge Viz] Min: {hog_min:.3f}, Max: {hog_max:.3f}, Range: {hog_range:.3f}")
                            # Debug: Check diversity of predictions
                            all_dominant = []
                            for ii in range(0, grid_size, patch_step):
                                for jj in range(0, grid_size, patch_step):
                                    dominant = torch.argmax(torch.abs(hog_grid_full[ii, jj])).item()
                                    all_dominant.append(dominant)
                            unique_orientations = len(set(all_dominant))
                            print(f"[HOG Edge Viz] Dominant orientations: {set(all_dominant)}, Unique: {unique_orientations}/{len(all_dominant)}")
                        
                        for i in range(0, grid_size, patch_step):
                            for j in range(0, grid_size, patch_step):
                                # Get HOG vector for this patch
                                hog_vec = hog_grid_full[i, j]  # [num_orientations]
                                
                                # Calculate center position in image
                                cy = int((i + 0.5) * H / grid_size)
                                cx = int((j + 0.5) * W / grid_size)
                                
                                # Find dominant orientation (highest magnitude)
                                abs_hog = torch.abs(hog_vec)
                                dominant_idx = torch.argmax(abs_hog).item()
                                strength = abs_hog[dominant_idx].item()
                                
                                # Normalize strength relative to the current batch
                                if hog_range > 1e-6:
                                    normalized_strength = (strength - hog_min.item()) / hog_range.item()
                                else:
                                    normalized_strength = 0.5
                                
                                # Draw if strength is above threshold (matching our test visualization)
                                threshold = 0.3  # Same threshold as test
                                if normalized_strength > threshold:
                                    # Get angle for this orientation bin
                                    angle = angles[dominant_idx]
                                    
                                    # Calculate line length proportional to strength
                                    # Match the test visualization arrow scaling
                                    line_len = int(patch_step * 1.5 * normalized_strength * arrow_scale)
                                    
                                    # Calculate line endpoints (bidirectional like HOG edges)
                                    dx = line_len * np.cos(angle)
                                    dy = line_len * np.sin(angle)
                                    
                                    # Draw bidirectional line (HOG uses unsigned gradients)
                                    x1, y1 = int(cx - dx/2), int(cy - dy/2)
                                    x2, y2 = int(cx + dx/2), int(cy + dy/2)
                                    
                                    # Use bright green color for edges (matching test)
                                    color = (0, 255, 0)  # Green edges
                                    
                                    # Draw thicker lines for stronger gradients
                                    thickness = max(1, int(3.0 * normalized_strength))
                                    
                                    # Draw the line on the overlay image
                                    cv2.line(overlay_img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                        
                        axes[4].imshow(overlay_img)
                        axes[4].set_title("HOG Edge Orientations")
                        axes[4].axis('off')
                else:
                    axes[3].imshow(self._tensor_to_image(orig_frame))
                    axes[3].set_title("Reconstruction")
            else:
                # For pixel/feature reconstruction
                axes[3].imshow(self._tensor_to_image(orig_frame))
                axes[3].set_title("Reconstruction")
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            images.append(wandb.Image(img, caption=f"Sample {b+1}"))
            plt.close()
        
        # Log images both as media and as individual metrics for dashboard
        print(f"[HOG VIZ DEBUG] About to log {len(images)} images to wandb at step {step}")
        
        if not images:
            print("[HOG VIZ DEBUG] No images to log!")
            return
            
        # Log images following W&B best practices
        log_dict = {}
        
        # Log the first sample as the main reconstruction image
        log_dict["reconstruction"] = images[0]
        
        # Log second sample if available
        if len(images) > 1:
            log_dict["reconstruction_sample2"] = images[1]
        
        print(f"[HOG VIZ DEBUG] Logging to wandb with keys: {list(log_dict.keys())}")
        print(f"[HOG VIZ DEBUG] Step value: {step}")
        
        # Log without step first to see if that's the issue
        try:
            wandb.log(log_dict, step=step, commit=True)
            print(f"[HOG VIZ DEBUG] Successfully logged to wandb with step {step}")
        except Exception as e:
            print(f"[HOG VIZ DEBUG] Error logging to wandb: {e}")
            # Try logging without step
            try:
                wandb.log(log_dict)
                print(f"[HOG VIZ DEBUG] Successfully logged to wandb WITHOUT step")
            except Exception as e2:
                print(f"[HOG VIZ DEBUG] Error logging even without step: {e2}")
    
    def log_hog_metrics(
        self,
        pred_hog: torch.Tensor,
        target_hog: torch.Tensor,
        mask: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log HOG-specific metrics to W&B.
        
        Args:
            pred_hog: Predicted HOG features [B, N, D]
            target_hog: Target HOG features [B, N, D]
            mask: Binary mask [B, N]
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        with torch.no_grad():
            # Compute metrics only on masked tokens
            B, N, D = pred_hog.shape
            
            # Expand mask to match feature dimensions
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_hog)
            
            # L2 distance per token
            l2_dist = torch.norm(pred_hog - target_hog, dim=-1)  # [B, N]
            
            # Cosine similarity per token
            cos_sim = F.cosine_similarity(pred_hog, target_hog, dim=-1)  # [B, N]
            
            # Average only over masked tokens
            masked_l2 = (l2_dist * mask).sum() / mask.sum()
            masked_cos_sim = (cos_sim * mask).sum() / mask.sum()
            
            # Per-channel statistics
            channel_l2 = torch.norm(pred_hog - target_hog, dim=(0, 1)) / (B * N)  # [D]
            
            metrics = {
                "hog/masked_l2_dist": masked_l2.item(),
                "hog/masked_cosine_sim": masked_cos_sim.item(),
                "hog/mean_channel_l2": channel_l2.mean().item(),
                "hog/std_channel_l2": channel_l2.std().item(),
            }
            
            # Log histogram of HOG channel errors
            wandb.log({
                **metrics,
                "hog/channel_l2_histogram": wandb.Histogram(channel_l2.cpu().numpy()),
            }, step=step)
    
    def log_rope_analysis(
        self,
        model: torch.nn.Module,
        videos: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log 3D RoPE analysis and visualizations.
        
        Args:
            model: Model with RoPE module
            videos: Input videos [B, C, T, H, W]
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        with torch.no_grad():
            # Extract attention maps if available
            B, C, T, H, W = videos.shape
            
            # Get model attention (this is model-specific)
            # You would need to modify your model to return attention weights
            
            # For now, we'll create synthetic visualizations
            # In practice, you'd extract real attention weights
            
            # Create RoPE frequency visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Temporal frequencies
            t_freqs = np.linspace(0, 1, T)
            axes[0].plot(t_freqs)
            axes[0].set_title("Temporal RoPE Frequencies")
            axes[0].set_xlabel("Frame")
            axes[0].set_ylabel("Frequency")
            
            # Spatial frequencies (H dimension)
            h_freqs = np.linspace(0, 1, H // 16)  # Assuming patch size 16
            axes[1].plot(h_freqs)
            axes[1].set_title("Height RoPE Frequencies")
            axes[1].set_xlabel("Patch Row")
            axes[1].set_ylabel("Frequency")
            
            # Spatial frequencies (W dimension)
            w_freqs = np.linspace(0, 1, W // 16)
            axes[2].plot(w_freqs)
            axes[2].set_title("Width RoPE Frequencies")
            axes[2].set_xlabel("Patch Column")
            axes[2].set_ylabel("Frequency")
            
            plt.tight_layout()
            
            # Log figure
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            rope_img = Image.open(buf)
            plt.close()
            
            wandb.log({
                "rope/frequency_analysis": wandb.Image(rope_img, caption="RoPE Frequencies")
            }, step=step)
            
            # Log RoPE impact metrics
            metrics = {
                "rope/temporal_scale": 1.0,  # From config
                "rope/spatial_base": 10000.0,  # From config
                "rope/temporal_base": 10000.0,  # From config
            }
            
            wandb.log(metrics, step=step)
    
    def log_attention_maps(
        self,
        attention_weights: torch.Tensor,
        frame_idx: int = 0,
        layer_idx: int = -1,
        num_heads: int = 8,
        step: Optional[int] = None,
    ):
        """
        Log attention map visualizations.
        
        Args:
            attention_weights: Attention weights [B, H, N, N]
            frame_idx: Which frame to visualize
            layer_idx: Which layer's attention to visualize
            num_heads: Number of attention heads
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        B, H, N, _ = attention_weights.shape
        
        # Create attention heatmaps for each head
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for h in range(min(H, 8)):  # Show max 8 heads
            attn = attention_weights[0, h].cpu().numpy()
            
            # Focus on CLS token attention to patches
            cls_attn = attn[0, 1:]  # Skip CLS token itself
            
            # Reshape to 2D if it's patch attention
            size = int(np.sqrt(len(cls_attn)))
            if size * size == len(cls_attn):
                cls_attn = cls_attn.reshape(size, size)
            else:
                # If not perfect square, just show as 1D
                cls_attn = cls_attn.reshape(1, -1)
            
            im = axes[h].imshow(cls_attn, cmap='hot', interpolation='nearest')
            axes[h].set_title(f"Head {h+1}")
            axes[h].axis('off')
            plt.colorbar(im, ax=axes[h], fraction=0.046)
        
        plt.suptitle(f"Attention Maps - Layer {layer_idx}, Frame {frame_idx}")
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        attn_img = Image.open(buf)
        plt.close()
        
        wandb.log({
            f"attention/layer_{layer_idx}": wandb.Image(attn_img, caption=f"Layer {layer_idx} Attention")
        }, step=step)
    
    def log_masking_strategies(
        self,
        mask_random: torch.Tensor,
        mask_frame: torch.Tensor,
        mask_tube: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log comparison of different masking strategies.
        
        Args:
            mask_random: Random masking pattern
            mask_frame: Frame masking pattern
            mask_tube: Tube masking pattern
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Visualize each masking pattern
        masks = [mask_random, mask_frame, mask_tube]
        titles = ["Random Masking", "Frame Masking", "Tube Masking"]
        
        for ax, mask, title in zip(axes, masks, titles):
            # Reshape mask for visualization
            if mask.dim() == 1:
                size = int(np.sqrt(len(mask)))
                mask_2d = mask.reshape(size, size).cpu().numpy()
            else:
                mask_2d = mask.cpu().numpy()
            
            im = ax.imshow(mask_2d, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{title} ({mask.mean().item():.1%} masked)")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        mask_img = Image.open(buf)
        plt.close()
        
        wandb.log({
            "masking/strategy_comparison": wandb.Image(mask_img, caption="Masking Strategies")
        }, step=step)
    
    def log_validation_metrics(
        self,
        val_loss: float,
        val_metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log validation metrics.
        
        Args:
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
            step: Training step
        """
        if step is None:
            step = self.global_step
            
        metrics = {"val/loss": val_loss}
        
        for key, value in val_metrics.items():
            metrics[f"val/{key}"] = value
        
        # Track best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            metrics["val/best_loss"] = self.best_loss
            
        wandb.log(metrics, step=step)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        epoch_time: float = 0.0,
        step: Optional[int] = None,
    ):
        """
        Log epoch summary.
        
        Args:
            epoch: Current epoch
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss for the epoch
            learning_rate: Current learning rate
            epoch_time: Time taken for the epoch
        """
        self.epoch = epoch
        
        summary = {
            "epoch/number": epoch,
            "epoch/train_loss": train_loss,
            "epoch/learning_rate": learning_rate,
            "epoch/time_minutes": epoch_time / 60.0,
        }
        
        if val_loss is not None:
            summary["epoch/val_loss"] = val_loss
            
        wandb.log(summary, step=step if step is not None else self.global_step)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image for visualization."""
        if tensor is None:
            return np.zeros((224, 224, 3), dtype=np.float32)
            
        # Assume tensor is [C, H, W] and normalized
        if tensor.dim() == 3:
            # Denormalize if needed (assuming standard normalization)
            tensor = tensor.clone()
            # Convert to [H, W, C]
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy (ensure it's on CPU first)
        img = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
        
        # Normalize to [0, 1] range with percentile clipping for better contrast
        # This helps with dark vessel angiograms
        p2, p98 = np.percentile(img, [2, 98])
        img = (img - p2) / (p98 - p2 + 1e-8)
        img = np.clip(img, 0, 1)
        
        return img
    
    def create_dashboard_config(self):
        """
        Create custom W&B dashboard configuration for SSL training.
        
        Returns configuration that can be used to set up custom panels.
        """
        return {
            "panels": [
                {
                    "name": "Training Dynamics",
                    "charts": [
                        {"metric": "train/loss", "type": "line"},
                        {"metric": "train/learning_rate", "type": "line"},
                        {"metric": "train/grad_norm", "type": "line"},
                        {"metric": "train/batch_time", "type": "line"},
                    ]
                },
                {
                    "name": "Reconstruction Quality",
                    "charts": [
                        {"metric": "recon/masked_mse", "type": "line"},
                        {"metric": "recon/psnr", "type": "line"},
                        {"metric": "recon/accuracy", "type": "line"},
                    ]
                },
                {
                    "name": "3D RoPE Analysis",
                    "charts": [
                        {"metric": "rope/temporal_scale", "type": "line"},
                        {"metric": "attention/*", "type": "image"},
                    ]
                },
                {
                    "name": "Visualizations",
                    "charts": [
                        {"metric": "reconstruction/samples", "type": "image"},
                        {"metric": "masking/strategy_comparison", "type": "image"},
                    ]
                }
            ]
        }
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
    
    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint to W&B.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Optional metadata to log with checkpoint
        """
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metadata or {}
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def update_step(self):
        """Update global step counter."""
        self.global_step += 1