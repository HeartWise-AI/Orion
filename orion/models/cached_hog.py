"""
Cached HOG feature extractor for ultra-fast MVM training.
Pre-computes and caches HOG features for all videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import hashlib


class CachedHOGExtractor(nn.Module):
    """
    Ultra-fast HOG extractor that caches features in memory.
    """
    
    def __init__(
        self,
        orientations: int = 6,  # Fewer bins for speed
        pixels_per_cell: tuple = (16, 16),  # Larger cells = fewer features
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Calculate dimensions
        self.num_patches = (image_size // patch_size) ** 2
        cells_per_patch = patch_size // pixels_per_cell[0]
        
        # Ultra-simple HOG: just orientations per patch (no cells)
        self.hog_dim = orientations  # Super simple!
        
        # Cache for storing computed HOG features
        self.cache = {}
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
    
    def _compute_simple_hog(self, image: torch.Tensor) -> torch.Tensor:
        """
        Ultra-simple HOG computation for a single image.
        
        Args:
            image: [C, H, W]
            
        Returns:
            HOG features [num_patches, orientations]
        """
        # Convert to grayscale if needed
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image[0]
        
        # Add batch dim
        gray = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Simple gradients (no padding for speed)
        dx = F.conv2d(gray, self.sobel_x)
        dy = F.conv2d(gray, self.sobel_y)
        
        # Magnitude and orientation
        mag = torch.sqrt(dx**2 + dy**2 + 1e-6)
        ori = torch.atan2(dy, dx)
        ori = (ori + np.pi) * (self.orientations / (2 * np.pi))  # [0, orientations]
        ori = ori.long().clamp(0, self.orientations - 1)
        
        # Pool into patches (super fast)
        ph = pw = self.image_size // self.patch_size
        
        # Use adaptive pooling for speed
        mag = F.adaptive_avg_pool2d(mag, (ph, pw))  # [1, 1, ph, pw]
        ori = F.adaptive_max_pool2d(ori.float(), (ph, pw)).long()  # [1, 1, ph, pw]
        
        mag = mag.squeeze(0).squeeze(0).flatten()  # [num_patches]
        ori = ori.squeeze(0).squeeze(0).flatten()  # [num_patches]
        
        # Create simple histogram per patch
        hog_features = torch.zeros(self.num_patches, self.orientations, device=image.device)
        for p in range(self.num_patches):
            # Just put magnitude in the dominant orientation bin
            hog_features[p, ori[p]] = mag[p]
        
        # Normalize
        hog_features = F.normalize(hog_features, p=2, dim=1)
        
        return hog_features
    
    def forward(self, images: torch.Tensor, video_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract HOG features with caching.
        
        Args:
            images: [B, C, H, W] or [B, C, T, H, W]
            video_ids: Optional tensor of video IDs for caching
            
        Returns:
            HOG features [B, num_patches, hog_dim]
        """
        if images.dim() == 5:
            # For video, just use center frame
            B, C, T, H, W = images.shape
            center_idx = T // 2
            images = images[:, :, center_idx, :, :]  # [B, C, H, W]
        
        B = images.shape[0]
        hog_list = []
        
        for b in range(B):
            # Try to use cache if video_id provided
            if video_ids is not None and video_ids[b].item() in self.cache:
                hog = self.cache[video_ids[b].item()]
            else:
                # Compute HOG
                hog = self._compute_simple_hog(images[b])
                
                # Cache if video_id provided
                if video_ids is not None:
                    self.cache[video_ids[b].item()] = hog
            
            hog_list.append(hog)
        
        return torch.stack(hog_list)
    
    @property
    def output_dim(self) -> int:
        return self.hog_dim