"""
Optimized HOG feature extraction that uses Sobel gradients with vectorized operations.
Follows MaskFeat paper specifications but optimized for speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class OptimizedHOGExtractor(nn.Module):
    """
    Optimized HOG extractor with Sobel gradients and vectorized operations.
    Follows MaskFeat paper: 8x8 cells, 9 orientation bins, L2 normalization.
    """
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (1, 1),
        block_norm: str = 'L2',
        use_color: bool = False,
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
        
        # Calculate dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.cells_per_patch = (patch_size // pixels_per_cell[0], patch_size // pixels_per_cell[1])
        
        # HOG feature dimension per patch
        channels = 3 if use_color else 1
        self.hog_dim = (
            channels * orientations * 
            self.cells_per_patch[0] * self.cells_per_patch[1] * 
            cells_per_block[0] * cells_per_block[1]
        )
        
        # Initialize Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        
        # Register as buffers
        channels = 3 if use_color else 1
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        
        # Pre-compute bin edges for histogram
        self.register_buffer('bin_edges', 
                           torch.linspace(0, np.pi, self.orientations + 1, dtype=torch.float32))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Optimized HOG extraction using Sobel gradients and vectorized operations.
        
        Args:
            images: [B, C, H, W] or [B, C, T, H, W]
        
        Returns:
            HOG features [B, num_patches, hog_dim] or [B, T, num_patches, hog_dim]
        """
        # Handle video input
        if images.dim() == 5:
            B, C, T, H, W = images.shape
            # Reshape to process all frames at once
            images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            is_video = True
        else:
            B = images.shape[0]
            T = 1
            is_video = False
        
        device = images.device
        
        # Convert to grayscale if needed
        if not self.use_color and images.shape[1] == 3:
            images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            images = images.unsqueeze(1)
        
        # Compute Sobel gradients (proper edge detection)
        dx = F.conv2d(images.view(-1, 1, H, W), self.sobel_x[:1], padding=1)
        dy = F.conv2d(images.view(-1, 1, H, W), self.sobel_y[:1], padding=1)
        
        # Reshape back
        dx = dx.view(B * T, 1, H, W)
        dy = dy.view(B * T, 1, H, W)
        
        # Compute magnitude and orientation
        magnitude = torch.sqrt(dx**2 + dy**2 + 1e-6)
        orientation = torch.atan2(dy, dx)
        orientation = (orientation + np.pi) % np.pi  # Convert to [0, pi]
        
        # Extract patches efficiently using unfold
        # This extracts all patches at once
        mag_patches = F.unfold(magnitude, kernel_size=self.patch_size, stride=self.patch_size)
        ori_patches = F.unfold(orientation, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Reshape: [B*T, patch_size^2, num_patches] -> [B*T, num_patches, patch_size^2]
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        mag_patches = mag_patches.permute(0, 2, 1).reshape(B * T, num_patches, self.patch_size, self.patch_size)
        ori_patches = ori_patches.permute(0, 2, 1).reshape(B * T, num_patches, self.patch_size, self.patch_size)
        
        # Process each patch to compute HOG features
        cells_h, cells_w = self.cells_per_patch
        cell_h, cell_w = self.pixels_per_cell
        
        # Reshape patches into cells
        # [B*T, num_patches, patch_h, patch_w] -> [B*T, num_patches, cells_h, cells_w, cell_h, cell_w]
        mag_cells = mag_patches.reshape(B * T, num_patches, cells_h, cell_h, cells_w, cell_w)
        ori_cells = ori_patches.reshape(B * T, num_patches, cells_h, cell_h, cells_w, cell_w)
        
        # Permute to group cells: [B*T, num_patches, cells_h, cells_w, cell_h, cell_w]
        mag_cells = mag_cells.permute(0, 1, 2, 4, 3, 5)
        ori_cells = ori_cells.permute(0, 1, 2, 4, 3, 5)
        
        # Flatten cells: [B*T, num_patches, cells_h * cells_w, cell_h * cell_w]
        mag_cells = mag_cells.reshape(B * T, num_patches, cells_h * cells_w, cell_h * cell_w)
        ori_cells = ori_cells.reshape(B * T, num_patches, cells_h * cells_w, cell_h * cell_w)
        
        # Compute histograms for all cells at once using vectorized operations
        hog_features = []
        
        for cell_idx in range(cells_h * cells_w):
            cell_mag = mag_cells[:, :, cell_idx, :]  # [B*T, num_patches, cell_h * cell_w]
            cell_ori = ori_cells[:, :, cell_idx, :]
            
            # Compute histogram using soft binning (vectorized)
            hist = self._compute_histogram_vectorized(cell_mag, cell_ori)
            hog_features.append(hist)
        
        # Concatenate all cell histograms
        hog_features = torch.cat(hog_features, dim=-1)  # [B*T, num_patches, cells * orientations]
        
        # L2 normalization per patch (critical for contrast invariance)
        hog_features = F.normalize(hog_features + 1e-5, p=2, dim=-1)
        
        # Reshape back to video format if needed
        if is_video:
            hog_features = hog_features.reshape(B, T, num_patches, -1)
        
        return hog_features
    
    def _compute_histogram_vectorized(self, magnitudes: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation histogram using vectorized soft binning.
        
        Args:
            magnitudes: [B, num_patches, num_pixels]
            orientations: [B, num_patches, num_pixels]
        
        Returns:
            histogram: [B, num_patches, orientations]
        """
        B, P, N = magnitudes.shape
        device = magnitudes.device
        
        # Initialize histogram
        hist = torch.zeros(B, P, self.orientations, device=device)
        
        # Compute bin width
        bin_width = np.pi / self.orientations
        
        # Compute bin indices and weights for soft binning
        bin_pos = orientations / bin_width
        bin_low = torch.floor(bin_pos).long() % self.orientations
        bin_high = (bin_low + 1) % self.orientations
        
        # Compute weights
        weight_high = bin_pos - bin_low.float()
        weight_low = 1.0 - weight_high
        
        # Accumulate weighted votes (vectorized across batch and patches)
        # We need to use scatter_add for efficiency
        for b in range(B):
            for p in range(P):
                # Low bins
                hist[b, p].scatter_add_(0, bin_low[b, p], magnitudes[b, p] * weight_low[b, p])
                # High bins
                hist[b, p].scatter_add_(0, bin_high[b, p], magnitudes[b, p] * weight_high[b, p])
        
        return hist
    
    @property
    def output_dim(self) -> int:
        """Get the output dimension of HOG features per patch."""
        return self.hog_dim