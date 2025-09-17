"""
Video dataset with cached HOG features for ultra-fast MVM training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import orion.utils
from typing import Optional, Dict, Any


class VideoCachedHOGDataset(Dataset):
    """Dataset that loads pre-computed HOG features instead of computing them."""
    
    def __init__(
        self,
        fnames,
        outcome=None,
        target_indices=None,
        frames=16,
        period=2,
        mean=None,
        std=None,
        resize=224,
        hog_cache_dir="hog_features_cache",
        transforms=None,
        **kwargs
    ):
        self.fnames = fnames
        self.outcome = outcome
        self.target_indices = target_indices
        self.frames = frames
        self.period = period
        self.resize = resize
        self.hog_cache_dir = hog_cache_dir
        self.transforms = transforms
        
        # Normalization
        if mean is None:
            mean = [99.63698, 99.63698, 99.63698]
        if std is None:
            std = [51.81118, 51.81118, 51.81118]
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        
        # Check cache exists
        if not os.path.exists(hog_cache_dir):
            raise ValueError(f"HOG cache directory not found: {hog_cache_dir}")
        
        # Verify all videos have cached features
        missing = []
        for fname in fnames[:10]:  # Check first 10
            video_name = Path(fname).stem
            cache_path = os.path.join(hog_cache_dir, f"{video_name}_hog.npz")
            if not os.path.exists(cache_path):
                missing.append(fname)
        
        if missing:
            print(f"Warning: {len(missing)} videos missing HOG cache. Will compute on-the-fly.")
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        video_fname = self.fnames[index]
        video_name = Path(video_fname).stem
        cache_path = os.path.join(self.hog_cache_dir, f"{video_name}_hog.npz")
        
        # Load pre-computed HOG features
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            hog_features = data['hog_features'].astype(np.float32)
            hog_features = torch.from_numpy(hog_features)  # [frames, num_patches, hog_dim]
        else:
            # Fallback: return dummy features (training will compute them)
            hog_features = None
        
        # Load video for reconstruction target
        video = orion.utils.loadvideo(video_fname).astype(np.float32)
        
        # Sample frames
        c, f, h, w = video.shape
        if f < self.frames * self.period:
            video = np.concatenate([video, np.zeros((c, self.frames * self.period - f, h, w))], axis=1)
        
        indices = np.arange(0, self.frames * self.period, self.period)
        video = video[:, indices]
        
        # Convert to tensor and normalize
        video = torch.from_numpy(video).float()  # [C, frames, H, W]
        
        # Resize if needed
        if h != self.resize or w != self.resize:
            video = video.permute(1, 0, 2, 3)  # [frames, C, H, W]
            video = torch.nn.functional.interpolate(
                video, size=(self.resize, self.resize), 
                mode='bilinear', align_corners=False
            )
            video = video.permute(1, 0, 2, 3)  # [C, frames, H, W]
        
        # Normalize
        video = (video - self.mean) / self.std
        
        # Apply transforms if any
        if self.transforms:
            video = video.permute(1, 0, 2, 3)  # [frames, C, H, W]
            for t in range(video.shape[0]):
                frame = video[t]
                for transform in self.transforms:
                    frame = transform(frame)
                video[t] = frame
            video = video.permute(1, 0, 2, 3)  # [C, frames, H, W]
        
        # Return format for MVM
        output = {
            "video": video,  # [C, frames, H, W]
            "labels": torch.tensor(0),  # Dummy label for MVM
            "fname": video_fname,
        }
        
        # Add pre-computed HOG if available
        if hog_features is not None:
            output["cached_hog_features"] = hog_features
        
        return output