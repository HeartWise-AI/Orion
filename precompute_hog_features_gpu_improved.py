#!/usr/bin/env python3
"""
Improved GPU-accelerated HOG feature pre-computation with better edge detection for medical images.
Uses proper Sobel filters instead of simple pixel differences.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import orion.utils
from orion.models.hog_features import HOGFeatureExtractor
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')


class ImprovedHOGFeatureExtractor(HOGFeatureExtractor):
    """
    Improved HOG extractor that uses proper Sobel filters for medical images.
    """
    
    def _pytorch_hog_fast(self, images: torch.Tensor) -> torch.Tensor:
        """
        Override to use proper Sobel gradients instead of simple differences.
        
        Args:
            images: Tensor of shape [B, C, H, W]
        
        Returns:
            HOG features of shape [B, num_patches, hog_dim]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # Convert to grayscale if needed
        if C == 3:
            # Use better weights for medical images (emphasize all channels equally)
            images_gray = images.mean(dim=1, keepdim=True)
        else:
            images_gray = images
        
        # Apply Sobel filters for proper edge detection
        # These are the standard Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Compute gradients using Sobel filters
        dx = F.conv2d(images_gray, sobel_x, padding=1)
        dy = F.conv2d(images_gray, sobel_y, padding=1)
        
        # Remove channel dimension
        dx = dx.squeeze(1)
        dy = dy.squeeze(1)
        
        # Magnitude and orientation
        mag = torch.sqrt(dx**2 + dy**2 + 1e-6)
        ori = torch.atan2(dy, dx)
        ori = (ori + np.pi) % np.pi  # Convert to [0, pi]
        
        # Extract patches using unfold
        mag_patches = F.unfold(mag.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        ori_patches = F.unfold(ori.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        
        # Reshape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        mag_patches = mag_patches.view(B, self.patch_size**2, num_patches).permute(0, 2, 1)
        ori_patches = ori_patches.view(B, self.patch_size**2, num_patches).permute(0, 2, 1)
        
        # Compute histograms with better binning
        bin_width = np.pi / self.orientations
        hog_features = torch.zeros(B, num_patches, self.orientations, device=device)
        
        # Use soft binning for smoother histograms
        for b in range(B):
            for p in range(num_patches):
                # Get patch data
                patch_mag = mag_patches[b, p]
                patch_ori = ori_patches[b, p]
                
                # Soft binning: distribute magnitude to neighboring bins
                for pixel_idx in range(patch_mag.shape[0]):
                    angle = patch_ori[pixel_idx].item()
                    magnitude = patch_mag[pixel_idx].item()
                    
                    # Find the two nearest bins
                    bin_pos = angle / bin_width
                    bin_low = int(np.floor(bin_pos)) % self.orientations
                    bin_high = (bin_low + 1) % self.orientations
                    
                    # Compute weights
                    weight_high = bin_pos - np.floor(bin_pos)
                    weight_low = 1.0 - weight_high
                    
                    # Distribute magnitude
                    hog_features[b, p, bin_low] += magnitude * weight_low
                    hog_features[b, p, bin_high] += magnitude * weight_high
        
        # L2 normalization with epsilon for stability
        hog_features = F.normalize(hog_features + 1e-5, p=2, dim=-1)
        
        # For compatibility with cells_per_patch, replicate features if needed
        if self.cells_per_patch[0] > 1:
            hog_features = hog_features.repeat(1, 1, self.cells_per_patch[0] * self.cells_per_patch[1])
        
        return hog_features


def process_video_batch_gpu_improved(
    video_paths, 
    output_dir, 
    frames, 
    resize, 
    period,
    hog_extractor,
    device='cuda:0',
    batch_size=32
):
    """Process a batch of videos on GPU with improved HOG extraction."""
    
    results = []
    batch_videos = []
    batch_paths = []
    batch_cache_paths = []
    
    # Load and preprocess videos
    for video_path in video_paths:
        video_name = Path(video_path).stem
        cache_path = os.path.join(output_dir, f"{video_name}_hog_improved.npz")
        
        # Skip if already cached
        if os.path.exists(cache_path):
            results.append(('skipped', 0))
            continue
        
        try:
            # Load video
            video = orion.utils.loadvideo(video_path).astype(np.float32)
            c, f, h, w = video.shape
            
            # Handle multi-channel videos
            if c > 3:
                video = video[:3]
            elif c == 1:
                video = np.repeat(video, 3, axis=0)
            
            # Sample frames
            if f < frames * period:
                indices = np.arange(0, min(f, frames * period), period)
                if len(indices) < frames:
                    last_idx = indices[-1] if len(indices) > 0 else 0
                    indices = np.pad(indices, (0, frames - len(indices)), constant_values=last_idx)
            else:
                indices = np.arange(0, frames * period, period)
            
            video = video[:, indices]  # [3, frames, H, W]
            
            # Store for batch processing
            batch_videos.append(video)
            batch_paths.append(video_path)
            batch_cache_paths.append(cache_path)
            
        except Exception as e:
            results.append(('error', str(e)))
    
    # Process batch on GPU
    if batch_videos:
        try:
            # Find max dimensions for padding
            max_h = max(v.shape[2] for v in batch_videos)
            max_w = max(v.shape[3] for v in batch_videos)
            
            # Pad videos to same size
            padded_videos = []
            for video in batch_videos:
                c, f, h, w = video.shape
                pad_h = max_h - h
                pad_w = max_w - w
                if pad_h > 0 or pad_w > 0:
                    video = np.pad(video, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
                padded_videos.append(video)
            
            # Stack into batch [batch, 3, frames, H, W]
            batch_tensor = torch.from_numpy(np.stack(padded_videos)).to(device)
            B, C, T, H, W = batch_tensor.shape
            
            # Resize all at once if needed
            if H != resize or W != resize:
                # Reshape for batch resize: [B*T, C, H, W]
                batch_tensor = batch_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                batch_tensor = F.interpolate(batch_tensor, size=(resize, resize), mode='bilinear', align_corners=False)
                batch_tensor = batch_tensor.reshape(B, T, C, resize, resize).permute(0, 2, 1, 3, 4)
            
            # Compute HOG for all videos and frames at once
            all_hog_features = []
            
            # Process all frames in batches to avoid OOM
            with torch.no_grad():
                for t in range(T):
                    # Get all videos at frame t: [B, C, H, W]
                    frame_batch = batch_tensor[:, :, t, :, :]
                    
                    # Compute HOG for entire batch on GPU
                    hog_feat = hog_extractor(frame_batch)  # [B, num_patches, hog_dim]
                    all_hog_features.append(hog_feat.cpu().numpy())
            
            # Stack: [B, T, num_patches, hog_dim]
            batch_hog = np.stack(all_hog_features, axis=1)
            
            # Save each video's features
            for i, (cache_path, hog_features) in enumerate(zip(batch_cache_paths, batch_hog)):
                np.savez_compressed(cache_path, hog_features=hog_features.astype(np.float16))
                results.append(('success', hog_features.nbytes))
                
        except Exception as e:
            for _ in batch_paths:
                results.append(('error', str(e)))
    
    return results


def compute_hog_features_gpu_improved(
    data_file: str,
    output_dir: str = "hog_features_cache_improved",
    frames: int = 16,
    resize: int = 224,
    period: int = 2,
    gpu_batch_size: int = 16,  # Reduced for improved method
    max_videos: int = None,
    skip_ratio: float = None,
    device: str = 'cuda:0'
):
    """Pre-compute improved HOG features using GPU acceleration."""
    
    # Load dataset
    df = pd.read_parquet(data_file)
    filenames = df['FileName'].values
    
    # Option to process subset
    if skip_ratio:
        step = int(1 / skip_ratio)
        filenames = filenames[::step]
        print(f"Processing subset: every {step}th video ({len(filenames)} total)")
    
    if max_videos:
        filenames = filenames[:max_videos]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(filenames)} videos on {device}...")
    print(f"GPU batch size: {gpu_batch_size}")
    print(f"Output directory: {output_dir}")
    print("Using IMPROVED HOG with Sobel gradients and soft binning")
    
    # Initialize improved HOG extractor on GPU
    hog_config = {
        'orientations': 9,
        'pixels_per_cell': (16, 16),
        'cells_per_block': (1, 1),
        'block_norm': 'L2',
        'use_color': False,  # Process as grayscale
        'patch_size': 16,
        'image_size': 224
    }
    
    hog_extractor = ImprovedHOGFeatureExtractor(**hog_config)
    hog_extractor = hog_extractor.to(device)
    hog_extractor.eval()
    
    print(f"Improved HOG extractor loaded on {device}")
    print(f"Expected output per frame: [{hog_extractor.num_patches}, {hog_extractor.hog_dim}]")
    
    # Filter already cached
    videos_to_process = []
    already_cached = 0
    for video_path in filenames:
        video_name = Path(video_path).stem
        cache_path = os.path.join(output_dir, f"{video_name}_hog_improved.npz")
        if os.path.exists(cache_path):
            already_cached += 1
        else:
            videos_to_process.append(video_path)
    
    print(f"Already cached: {already_cached}")
    print(f"Need to process: {len(videos_to_process)}")
    
    if not videos_to_process:
        print("All videos cached!")
        return
    
    # Process in GPU batches
    processed = 0
    errors = 0
    skipped = 0
    
    with tqdm(total=len(videos_to_process), desc="Computing Improved HOG on GPU") as pbar:
        for i in range(0, len(videos_to_process), gpu_batch_size):
            batch = videos_to_process[i:i+gpu_batch_size]
            
            # Process batch on GPU
            start_time = time.time()
            results = process_video_batch_gpu_improved(
                batch, output_dir, frames, resize, period,
                hog_extractor, device, gpu_batch_size
            )
            batch_time = time.time() - start_time
            
            # Update counters
            for status, data in results:
                if status == 'success':
                    processed += 1
                elif status == 'error':
                    errors += 1
                elif status == 'skipped':
                    skipped += 1
            
            # Update progress
            pbar.update(len(batch))
            pbar.set_postfix({
                'processed': processed,
                'errors': errors,
                'batch_time': f'{batch_time:.2f}s',
                'speed': f'{len(batch)/batch_time:.1f} vid/s'
            })
    
    # Final summary
    print("\n" + "="*50)
    print("Improved GPU HOG feature pre-computation complete!")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Skipped (cached): {already_cached}")
    
    # Calculate cache size
    cache_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    if cache_files:
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in cache_files)
        print(f"Cache size: {total_size / (1024**3):.2f} GB ({len(cache_files)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved GPU-accelerated HOG pre-computation")
    parser.add_argument("--data_file", default="data/deepcoro_split_dataset_clean.parquet")
    parser.add_argument("--output_dir", default="hog_features_cache_improved")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--gpu_batch_size", type=int, default=16,
                        help="Number of videos to process in parallel on GPU")
    parser.add_argument("--max_videos", type=int, default=10,
                        help="Max videos to process (for testing)")
    parser.add_argument("--skip_ratio", type=float, default=None,
                        help="Process fraction of dataset")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="GPU device to use")
    
    args = parser.parse_args()
    
    compute_hog_features_gpu_improved(
        args.data_file,
        output_dir=args.output_dir,
        frames=args.frames,
        resize=args.resize,
        period=args.period,
        gpu_batch_size=args.gpu_batch_size,
        max_videos=args.max_videos,
        skip_ratio=args.skip_ratio,
        device=args.device
    )