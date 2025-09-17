"""
HOG (Histogram of Oriented Gradients) feature extraction for MaskFeat.
Based on "Masked Feature Prediction for Self-Supervised Visual Pre-Training" 
by Wei et al. (2022) - https://arxiv.org/abs/2112.09133
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from einops import rearrange, repeat
try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. HOG features will use PyTorch implementation.")


class HOGFeatureExtractor(nn.Module):
    """
    HOG feature extractor for MaskFeat pre-training.
    
    HOG is particularly effective because:
    1. Local contrast normalization provides invariance to illumination changes
    2. Spatial binning provides geometric invariance
    3. Captures local patterns without high-frequency noise
    4. Computationally efficient (no external model needed)
    
    Args:
        orientations: Number of orientation bins (default: 9)
        pixels_per_cell: Size of cell in pixels (default: (8, 8))
        cells_per_block: Number of cells per block for normalization (default: (1, 1))
        block_norm: Block normalization method ('L1' or 'L2', default: 'L2')
        use_color: Whether to compute HOG on all color channels (default: True)
        patch_size: Size of patches for spatial grouping (default: 16)
        image_size: Input image size (default: 224)
    """
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (1, 1),
        block_norm: str = 'L2',
        use_color: bool = True,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.use_color = use_color
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Calculate output dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.cells_per_patch = (patch_size // pixels_per_cell[0], patch_size // pixels_per_cell[1])
        
        # HOG feature dimension per patch
        channels = 3 if use_color else 1
        self.hog_dim = (
            channels * orientations * 
            self.cells_per_patch[0] * self.cells_per_patch[1] * 
            cells_per_block[0] * cells_per_block[1]
        )
        
        # Always initialize PyTorch implementation for batch processing
        self._init_pytorch_hog()
    
    def _pytorch_hog_fast(self, images: torch.Tensor) -> torch.Tensor:
        """
        Fast PyTorch HOG - optimized Sobel gradients as per MaskFeat paper.
        
        Args:
            images: Tensor of shape [B, C, H, W]
        
        Returns:
            HOG features of shape [B, num_patches, hog_dim]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # Convert to grayscale if needed
        if not self.use_color and C == 3:
            images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            images = images.unsqueeze(1)
            C = 1
        
        # Compute Sobel gradients (proper edge detection as per paper)
        dx = F.conv2d(images.view(-1, 1, H, W), self.sobel_x[:1], padding=1)
        dy = F.conv2d(images.view(-1, 1, H, W), self.sobel_y[:1], padding=1)
        dx = dx.view(B, C, H, W)
        dy = dy.view(B, C, H, W)
        
        # Magnitude and orientation
        magnitude = torch.sqrt(dx**2 + dy**2 + 1e-6)
        orientation = torch.atan2(dy, dx)
        orientation = (orientation + np.pi) % np.pi  # [0, pi]
        
        # Extract all patches at once using unfold
        mag_patches = F.unfold(magnitude, kernel_size=self.patch_size, stride=self.patch_size)
        ori_patches = F.unfold(orientation, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Reshape: [B, patch_size^2, num_patches] -> [B, num_patches, patch_size, patch_size]
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        mag_patches = mag_patches.permute(0, 2, 1).reshape(B, num_patches, self.patch_size, self.patch_size)
        ori_patches = ori_patches.permute(0, 2, 1).reshape(B, num_patches, self.patch_size, self.patch_size)
        
        # Check if we're using simplified single-cell mode for speed
        if self.pixels_per_cell[0] == self.patch_size:
            # Fast path: single cell per patch
            # Flatten patches for histogram computation
            mag_flat = mag_patches.reshape(B, num_patches, -1)
            ori_flat = ori_patches.reshape(B, num_patches, -1)
            
            # Compute histogram with vectorized operations
            bin_width = np.pi / self.orientations
            bin_idx = (ori_flat / bin_width).long().clamp(0, self.orientations - 1)
            
            # Create histograms using scatter_add
            hog_features = torch.zeros(B, num_patches, self.orientations, device=device)
            for b in range(B):
                for p in range(num_patches):
                    hog_features[b, p].scatter_add_(0, bin_idx[b, p], mag_flat[b, p])
        else:
            # Full path: multiple cells per patch (slower but more accurate)
            cells_h = cells_w = self.patch_size // self.pixels_per_cell[0]
            cell_h = cell_w = self.pixels_per_cell[0]
            
            # Reshape into cells: [B, num_patches, cells_h, cell_h, cells_w, cell_w]
            mag_cells = mag_patches.reshape(B, num_patches, cells_h, cell_h, cells_w, cell_w)
            ori_cells = ori_patches.reshape(B, num_patches, cells_h, cell_h, cells_w, cell_w)
            
            # Permute and reshape for processing: [B, num_patches, cells_h*cells_w, cell_h*cell_w]
            mag_cells = mag_cells.permute(0, 1, 2, 4, 3, 5).reshape(B, num_patches, cells_h*cells_w, cell_h*cell_w)
            ori_cells = ori_cells.permute(0, 1, 2, 4, 3, 5).reshape(B, num_patches, cells_h*cells_w, cell_h*cell_w)
            
            # Compute histograms for all cells (vectorized)
            bin_width = np.pi / self.orientations
            hog_features = []
            
            for cell_idx in range(cells_h * cells_w):
                # Get magnitudes and orientations for this cell across all patches
                cell_mag = mag_cells[:, :, cell_idx, :].reshape(B, num_patches, -1)
                cell_ori = ori_cells[:, :, cell_idx, :].reshape(B, num_patches, -1)
                
                # Compute histogram with soft binning
                hist = torch.zeros(B, num_patches, self.orientations, device=device)
                
                # Soft binning computation
                bin_pos = cell_ori / bin_width
                bin_low = torch.floor(bin_pos).long() % self.orientations
                bin_high = (bin_low + 1) % self.orientations
                weight_high = bin_pos - bin_low.float()
                weight_low = 1.0 - weight_high
                
                # Accumulate using scatter_add (much faster than loops)
                for b in range(B):
                    for p in range(num_patches):
                        hist[b, p].scatter_add_(0, bin_low[b, p], cell_mag[b, p] * weight_low[b, p])
                        hist[b, p].scatter_add_(0, bin_high[b, p], cell_mag[b, p] * weight_high[b, p])
                
                hog_features.append(hist)
            
            # Concatenate all cell histograms: [B, num_patches, cells * orientations]
            hog_features = torch.cat(hog_features, dim=-1)
        
        # L2 normalization per patch (critical for contrast invariance)
        hog_features = F.normalize(hog_features + 1e-5, p=2, dim=-1)
        
        return hog_features
    
    def _init_pytorch_hog(self):
        """Initialize PyTorch-based HOG filters."""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        
        # Expand for all channels if using color
        channels = 3 if self.use_color else 1
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        
        # Orientation bins
        self.register_buffer('bin_centers', 
                           torch.linspace(0, np.pi, self.orientations, dtype=torch.float32))
    
    
    def _pytorch_hog(self, images: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of HOG feature extraction.
        
        Args:
            images: Tensor of shape [B, C, H, W]
        
        Returns:
            HOG features of shape [B, num_patches, hog_dim]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # Convert to grayscale if needed
        if not self.use_color and C == 3:
            images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            images = images.unsqueeze(1)
            C = 1
        
        # Compute gradients
        if self.use_color:
            # Process each channel separately
            dx = F.conv2d(images.view(-1, 1, H, W), self.sobel_x[:1], padding=1)
            dy = F.conv2d(images.view(-1, 1, H, W), self.sobel_y[:1], padding=1)
            dx = dx.view(B, C, H, W)
            dy = dy.view(B, C, H, W)
        else:
            dx = F.conv2d(images, self.sobel_x[:1], padding=1)
            dy = F.conv2d(images, self.sobel_y[:1], padding=1)
        
        # Compute magnitude and orientation
        magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        orientation = torch.atan2(dy, dx)  # Range: [-pi, pi]
        orientation = (orientation + np.pi) % np.pi  # Convert to [0, pi] for unsigned gradients
        
        # Create HOG features for each patch
        ph = pw = self.image_size // self.patch_size
        ch = cw = self.patch_size // self.pixels_per_cell[0]
        
        hog_features = []
        
        for i in range(ph):
            for j in range(pw):
                # Extract patch
                patch_mag = magnitude[
                    :, :,
                    i * self.patch_size:(i + 1) * self.patch_size,
                    j * self.patch_size:(j + 1) * self.patch_size
                ]
                patch_ori = orientation[
                    :, :,
                    i * self.patch_size:(i + 1) * self.patch_size,
                    j * self.patch_size:(j + 1) * self.patch_size
                ]
                
                # Compute HOG for each cell in the patch
                cell_hogs = []
                for ci in range(ch):
                    for cj in range(cw):
                        # Extract cell
                        cell_mag = patch_mag[
                            :, :,
                            ci * self.pixels_per_cell[0]:(ci + 1) * self.pixels_per_cell[0],
                            cj * self.pixels_per_cell[1]:(cj + 1) * self.pixels_per_cell[1]
                        ]
                        cell_ori = patch_ori[
                            :, :,
                            ci * self.pixels_per_cell[0]:(ci + 1) * self.pixels_per_cell[0],
                            cj * self.pixels_per_cell[1]:(cj + 1) * self.pixels_per_cell[1]
                        ]
                        
                        # Compute histogram
                        hist = self._compute_histogram(cell_mag, cell_ori)
                        cell_hogs.append(hist)
                
                # Concatenate cell HOGs and normalize
                patch_hog = torch.cat(cell_hogs, dim=-1)  # [B, C * orientations * cells]
                
                # L2 normalization (block normalization)
                if self.block_norm == 'L2':
                    patch_hog = F.normalize(patch_hog, p=2, dim=-1)
                elif self.block_norm == 'L1':
                    patch_hog = F.normalize(patch_hog, p=1, dim=-1)
                
                hog_features.append(patch_hog)
        
        # Stack all patches
        hog_features = torch.stack(hog_features, dim=1)  # [B, num_patches, hog_dim]
        
        return hog_features
    
    def _compute_histogram(self, magnitude: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation histogram for a cell.
        
        Args:
            magnitude: Tensor of gradient magnitudes [B, C, cell_h, cell_w]
            orientation: Tensor of gradient orientations [B, C, cell_h, cell_w]
        
        Returns:
            Histogram tensor [B, C * orientations]
        """
        B, C = magnitude.shape[:2]
        device = magnitude.device
        
        # Flatten spatial dimensions - use reshape for non-contiguous tensors
        mag_flat = magnitude.reshape(B, C, -1)  # [B, C, cell_size]
        ori_flat = orientation.reshape(B, C, -1)
        
        # Compute bin assignments and weights
        bin_width = np.pi / self.orientations
        bin_idx = (ori_flat / bin_width).long()
        bin_idx = torch.clamp(bin_idx, 0, self.orientations - 1)
        
        # Create histogram
        hist = torch.zeros(B, C, self.orientations, device=device)
        
        # Accumulate weighted votes
        for b in range(B):
            for c in range(C):
                for i in range(mag_flat.shape[2]):
                    hist[b, c, bin_idx[b, c, i]] += mag_flat[b, c, i]
        
        # Reshape and return
        return hist.view(B, -1)  # [B, C * orientations]
    
    def _skimage_hog(self, images: torch.Tensor) -> torch.Tensor:
        """
        Use scikit-image HOG implementation optimized for medical images.
        
        Args:
            images: Tensor of shape [B, C, H, W]
        
        Returns:
            HOG features of shape [B, num_patches, hog_dim]
        """
        B = images.shape[0]
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
        
        hog_features = []
        for img in images_np:
            # Normalize to 0-1 if needed
            if img.min() < 0 or img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            if self.use_color:
                channel_hogs = []
                for c in range(3):
                    h = hog(
                        img[:, :, c],
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        block_norm='L2',  # L2 normalization as in MaskFeat paper
                        feature_vector=False
                        # No transform_sqrt - paper doesn't use it
                    )
                    channel_hogs.append(h)
                
                # Concatenate channels
                h = np.concatenate(channel_hogs, axis=-1)
            else:
                # Convert to grayscale
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                
                h = hog(
                    gray,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm='L2',  # L2 normalization as in MaskFeat paper
                    feature_vector=False
                    # No transform_sqrt - paper doesn't use it
                )
            
            # Reshape to patches
            ph = pw = self.image_size // self.patch_size
            h = rearrange(h, '(ph dh) (pw dw) ... -> ph pw (dh dw ...)', ph=ph, pw=pw)
            h = h.reshape(self.num_patches, -1)
            hog_features.append(h)
        
        hog_features = np.stack(hog_features, axis=0)
        return torch.from_numpy(hog_features).float().to(images.device)
    
    def forward(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract HOG features from images.
        
        Args:
            images: Input images [B, C, H, W] or videos [B, C, T, H, W]
            mask: Optional mask indicating which patches to compute features for
        
        Returns:
            HOG features [B, num_patches, hog_dim] or [B, T, num_patches, hog_dim] for videos
        """
        # Handle video input - batch process all frames at once for efficiency
        if images.dim() == 5:
            B, C, T, H, W = images.shape
            
            # Reshape to process all frames in a single batch
            images_reshaped = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            
            # Always use PyTorch implementation for batch efficiency
            # Process all frames at once
            hog_features_flat = self._pytorch_hog(images_reshaped)
            
            # Reshape back to [B, T, num_patches, hog_dim]
            num_patches = hog_features_flat.shape[1]
            hog_dim = hog_features_flat.shape[2]
            hog_features = hog_features_flat.reshape(B, T, num_patches, hog_dim)
            
            return hog_features
        
        # Single image processing - use fast PyTorch implementation
        # This is the most common case for MaskFeat (center frame only)
        hog_features = self._pytorch_hog_fast(images)
        
        return hog_features
    
    @property
    def output_dim(self) -> int:
        """Get the output dimension of HOG features per patch."""
        return self.hog_dim


def visualize_hog_features(hog_features: torch.Tensor, patch_size: int = 16) -> np.ndarray:
    """
    Visualize HOG features as an image.
    
    Args:
        hog_features: HOG features [num_patches, hog_dim] or [H_patches, W_patches, hog_dim]
        patch_size: Size of each patch
    
    Returns:
        Visualization as numpy array
    """
    if hog_features.dim() == 2:
        # Reshape to 2D grid if needed
        num_patches = hog_features.shape[0]
        grid_size = int(np.sqrt(num_patches))
        hog_features = hog_features.view(grid_size, grid_size, -1)
    
    H, W, D = hog_features.shape
    
    # Create visualization
    viz = np.zeros((H * patch_size, W * patch_size))
    
    for i in range(H):
        for j in range(W):
            # Get HOG vector for this patch
            hog_vec = hog_features[i, j].cpu().numpy()
            
            # Simple visualization: show magnitude as intensity
            patch_viz = np.abs(hog_vec).mean() * np.ones((patch_size, patch_size))
            
            # Place in grid
            viz[i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size] = patch_viz
    
    return viz