"""
HOG visualization utilities for overlaying gradient orientations on images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import math


def generate_orientation_lines(orientations: int = 4, line_length: int = 8):
    """
    Generate orientation line templates for HOG visualization.
    
    Args:
        orientations: Number of orientation bins
        line_length: Length of visualization lines in pixels
        
    Returns:
        Array of line templates [orientations, line_length, line_length]
    """
    angles = np.linspace(0, np.pi, orientations, endpoint=False)
    lines = np.zeros((orientations, line_length, line_length))
    
    center = line_length // 2
    for i, angle in enumerate(angles):
        # Create a line at the given angle
        for t in range(line_length):
            x = int(center + (t - center) * np.cos(angle))
            y = int(center + (t - center) * np.sin(angle))
            if 0 <= x < line_length and 0 <= y < line_length:
                lines[i, y, x] = 1.0
                
    return torch.from_numpy(lines).float()


def visualize_hog_overlay(
    hog_features: torch.Tensor,
    original_image: torch.Tensor,
    mask: torch.Tensor = None,
    orientations: int = 4,
    patch_size: int = 16,
    alpha: float = 0.7,
    color: str = 'green'
):
    """
    Create an overlay visualization of HOG features on the original image.
    
    Args:
        hog_features: HOG features [num_patches, hog_dim] or [H_patches, W_patches, hog_dim]
        original_image: Original image [C, H, W] 
        mask: Optional mask indicating which patches to visualize [num_patches]
        orientations: Number of orientation bins
        patch_size: Size of each patch
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        color: Color of HOG visualization lines
        
    Returns:
        Overlay image as numpy array [H, W, 3]
    """
    # Convert image to numpy and ensure it's in [0, 1] range
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.cpu().numpy()
        if img_np.shape[0] == 3:  # [C, H, W]
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = original_image
    
    # Normalize image to [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # Ensure we have at least 2D image
    if img_np.ndim == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    
    H, W = img_np.shape[:2]
    
    # Ensure HOG features are properly shaped
    if hog_features.dim() == 2:  # [num_patches, hog_dim]
        num_patches = hog_features.shape[0]
        # For MViT's 3136 patches, use 56x56 grid
        if num_patches == 3136:
            grid_size = 56
            hog_features = hog_features.view(grid_size, grid_size, -1)
        else:
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                # Find closest factorization
                factors = []
                for i in range(1, int(np.sqrt(num_patches)) + 1):
                    if num_patches % i == 0:
                        factors.append((i, num_patches // i))
                if factors:
                    h_grid, w_grid = min(factors, key=lambda x: abs(x[0] - x[1]))
                    hog_features = hog_features[:h_grid * w_grid].view(h_grid, w_grid, -1)
                else:
                    # Fallback to padding
                    grid_size = int(np.ceil(np.sqrt(num_patches)))
                    padded = torch.zeros(grid_size * grid_size, hog_features.shape[1])
                    padded[:num_patches] = hog_features
                    hog_features = padded.view(grid_size, grid_size, -1)
            else:
                hog_features = hog_features.view(grid_size, grid_size, -1)
    
    h_patches, w_patches, hog_dim = hog_features.shape
    
    # Create overlay image
    overlay = np.zeros((H, W, 3))
    
    # Color map for HOG visualization
    color_map = {
        'green': [0, 1, 0],
        'red': [1, 0, 0],
        'blue': [0, 0, 1],
        'cyan': [0, 1, 1],
        'yellow': [1, 1, 0],
        'white': [1, 1, 1]
    }
    line_color = np.array(color_map.get(color, [0, 1, 0]))
    
    # Calculate actual patch size based on grid dimensions
    # For MViT with 56x56 patches from 224x224 image, each patch is 4x4 pixels
    actual_patch_h = max(1, H // h_patches)
    actual_patch_w = max(1, W // w_patches)
    
    # Generate orientation templates with appropriate size
    template_size = max(2, min(actual_patch_h, actual_patch_w) // 2)
    line_templates = generate_orientation_lines(orientations, template_size)
    
    # Process each patch
    for i in range(h_patches):
        for j in range(w_patches):
            patch_idx = i * w_patches + j
            
            # Skip if masked out
            if mask is not None and patch_idx < len(mask) and mask[patch_idx] == 0:
                continue
            
            # Get HOG vector for this patch
            hog_vec = hog_features[i, j].detach().cpu().numpy()
            
            # Assuming hog_dim = orientations (or orientations * cells_per_patch)
            # For simplicity, we'll use the first 'orientations' values
            if len(hog_vec) >= orientations:
                orientation_strengths = hog_vec[:orientations]
            else:
                # Pad with zeros if needed
                orientation_strengths = np.pad(hog_vec, (0, orientations - len(hog_vec)))
            
            # Normalize strengths
            max_strength = np.max(np.abs(orientation_strengths))
            if max_strength > 0:
                orientation_strengths = orientation_strengths / max_strength
            
            # Calculate patch position in image using actual patch dimensions
            y_start = i * actual_patch_h
            y_end = min((i + 1) * actual_patch_h, H)
            x_start = j * actual_patch_w
            x_end = min((j + 1) * actual_patch_w, W)
            
            patch_h = y_end - y_start
            patch_w = x_end - x_start
            
            # Create patch visualization
            patch_viz = np.zeros((patch_h, patch_w, 3))
            
            # Overlay each orientation
            for k, strength in enumerate(orientation_strengths):
                if abs(strength) > 0.1:  # Threshold for visibility
                    # Get line template
                    if k < len(line_templates):
                        line = line_templates[k].numpy()
                        
                        # Resize line to match patch size - skip if dimensions are invalid
                        if patch_h > 0 and patch_w > 0 and line.shape[0] > 0 and line.shape[1] > 0:
                            try:
                                # Ensure line dimensions are valid
                                if line.shape[0] < patch_h or line.shape[1] < patch_w:
                                    # Upsample line if it's smaller than patch
                                    line_resized = F.interpolate(
                                        torch.from_numpy(line).unsqueeze(0).unsqueeze(0).float(),
                                        size=(patch_h, patch_w),
                                        mode='bilinear',
                                        align_corners=False
                                    )[0, 0].numpy()
                                else:
                                    # Downsample or use as-is if larger
                                    line_resized = line[:patch_h, :patch_w]
                                
                                # Apply line with strength-based intensity
                                for c in range(3):
                                    patch_viz[:, :, c] += line_resized * abs(strength) * line_color[c]
                            except Exception as e:
                                print(f"Warning: Could not resize line for patch ({i}, {j}): {e}")
                                continue
            
            # Add to overlay
            overlay[y_start:y_end, x_start:x_end] = patch_viz
    
    # Normalize overlay
    overlay = np.clip(overlay, 0, 1)
    
    # Combine with original image
    result = img_np * (1 - alpha) + overlay * alpha
    result = np.clip(result, 0, 1)
    
    return result


def create_hog_arrow_visualization(
    hog_features: torch.Tensor,
    original_image: torch.Tensor,
    mask: torch.Tensor = None,
    orientations: int = 4,
    patch_size: int = 16,
):
    """
    Create a matplotlib figure with HOG arrows overlaid on the image.
    
    Args:
        hog_features: HOG features [num_patches, hog_dim]
        original_image: Original image [C, H, W]
        mask: Optional mask [num_patches]
        orientations: Number of orientation bins
        patch_size: Size of each patch
        
    Returns:
        Matplotlib figure
    """
    # Convert image to numpy
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = original_image
    
    # Normalize to [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    H, W = img_np.shape[:2]
    
    # Reshape HOG features
    if hog_features.dim() == 2:
        num_patches = hog_features.shape[0]
        grid_size = int(np.sqrt(num_patches))
        hog_features = hog_features.view(grid_size, grid_size, -1)
    
    h_patches, w_patches, hog_dim = hog_features.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    
    # Define orientation angles
    angles = np.linspace(0, np.pi, orientations, endpoint=False)
    
    # Draw arrows for each patch
    for i in range(h_patches):
        for j in range(w_patches):
            patch_idx = i * w_patches + j
            
            # Skip if masked
            if mask is not None and patch_idx < len(mask) and mask[patch_idx] == 0:
                continue
            
            # Get HOG vector
            hog_vec = hog_features[i, j].cpu().numpy()
            
            # Get orientation strengths
            if len(hog_vec) >= orientations:
                strengths = hog_vec[:orientations]
            else:
                strengths = np.pad(hog_vec, (0, orientations - len(hog_vec)))
            
            # Normalize
            max_strength = np.max(np.abs(strengths))
            if max_strength > 0:
                strengths = strengths / max_strength
            
            # Patch center
            cx = j * patch_size + patch_size // 2
            cy = i * patch_size + patch_size // 2
            
            # Draw arrows for significant orientations
            for k, (angle, strength) in enumerate(zip(angles, strengths)):
                if abs(strength) > 0.2:  # Threshold
                    # Arrow length proportional to strength
                    arrow_len = patch_size * 0.4 * abs(strength)
                    
                    # Calculate arrow endpoints
                    dx = arrow_len * np.cos(angle)
                    dy = arrow_len * np.sin(angle)
                    
                    # Draw bidirectional arrow (HOG uses unsigned gradients)
                    ax.arrow(cx - dx/2, cy - dy/2, dx, dy,
                            head_width=2, head_length=1,
                            fc='lime', ec='lime', alpha=0.7,
                            linewidth=2*abs(strength))
    
    ax.set_title("HOG Feature Overlay")
    ax.axis('off')
    
    return fig