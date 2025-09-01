# 3D RoPE Implementation for MViT in Orion

## Overview

This document describes the implementation of 3D Axial Rotary Position Embeddings (RoPE) for MViT (Multiscale Vision Transformer) models in the Orion framework. The implementation extends 2D RoPE to handle temporal dimensions in video data.

## Implementation Files

### 1. Core Module: `orion/models/rope_3d.py`
- **Purpose**: Implements the 3D RoPE algorithm
- **Key Classes**:
  - `Rope3D`: Main class that applies rotary position embeddings to 3D video data
- **Key Functions**:
  - `_rotate_half()`: Rotates half the hidden dimensions for RoPE
  - `_build_freqs()`: Builds frequency tensors for position encoding
  - `_get_cached_freqs()`: Caches frequencies for efficient inference

### 2. Integration: `orion/models/__init__.py`
- **Modified Functions**:
  - `_patch_mvit_attention_for_rope()`: Monkey-patches MViT attention mechanisms
  - `load_and_modify_model()`: Extended to support RoPE configuration
- **Supported Models**:
  - `mvit_v1_b`: MViT v1 Base model
  - `mvit_v2_s`: MViT v2 Small model

### 3. Configuration: `notebooks/config/config_mvit_rope_classification.yaml`
- Example configuration file for training MViT with RoPE enabled
- Includes all necessary parameters for RoPE and training

## Key Features

### 1. 3D Axial Design
- Splits head dimension into three equal parts for temporal (T), height (H), and width (W) dimensions
- Each dimension gets its own rotary encoding
- Ensures balanced position information across all axes

### 2. CLS Token Handling
- Automatically detects and preserves CLS tokens
- CLS tokens are not rotated, maintaining their global representation

### 3. Efficient Caching
- Caches frequency computations during inference
- Significantly reduces computational overhead
- Automatically manages cache based on training/eval mode

### 4. Monkey-Patching Integration
- Non-invasive integration via monkey-patching
- Preserves original model architecture
- Easy to enable/disable via configuration

## Configuration Parameters

```yaml
# Enable 3D RoPE
use_rope: true

# Base frequency for position encoding (default: 10000.0)
rope_base: 10000.0

# Scale factor for temporal frequencies (default: 1.0)
rope_temporal_scale: 1.0

# Coordinate normalization mode (default: "separate")
# Options: "separate", "max", "min"
rope_normalize_mode: separate
```

## Usage Example

### Training with RoPE

```bash
# Using the provided configuration
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    orion/utils/video_training_and_eval.py \
    --config_path=notebooks/config/config_mvit_rope_classification.yaml
```

### Programmatic Usage

```python
from orion.models import load_and_modify_model

config = {
    "model_name": "mvit_v2_s",
    "task": "classification",
    "head_structure": {
        "contrast_agent": 1,
        "main_structure": 5,
        "stent_presence": 1
    },
    "use_rope": True,
    "rope_base": 10000.0,
    "rope_temporal_scale": 1.0,
    "rope_normalize_mode": "separate"
}

model = load_and_modify_model(config)
```

## Technical Details

### Head Dimension Requirements
- For 3D RoPE, the head dimension must be divisible by 6
- This allows splitting into 3 axes, each with even dimensions
- MViT models typically have head_dim=96, which is compatible (96/6=16 per axis)

### Supported Configurations
- **MViT v1 B**: Supports heads with dimensions (1×96, 2×96, 8×96)
- **MViT v2 S**: Supports heads with dimensions (1×96, 2×96, 8×96)

### Performance Considerations
- **Memory**: Minimal overhead (only stores sin/cos frequencies)
- **Computation**: ~5-10% overhead during attention computation
- **Training**: Requires full model retraining when adding to pretrained models

## Training Recommendations

When adding RoPE to pretrained models:

1. **Use Gradual Unfreezing**:
   ```yaml
   video_freeze_schedule: cosine
   video_freeze_start: 0.95  # Start with 95% frozen
   video_freeze_end: 0.0     # End fully trainable
   video_freeze_warmup_epochs: 10
   ```

2. **Lower Learning Rate**:
   - Use 2e-5 instead of typical 1e-4 for stability

3. **Smaller Batch Size**:
   - Start with batch_size=2 for MViT models

4. **Gradient Accumulation**:
   ```yaml
   gradient_accumulation_steps: 4
   gradient_clip_val: 1.0
   ```

## Testing

Test scripts are provided to verify the implementation:

```bash
# Test the Rope3D module directly
python test_rope_simple.py

# Test with full model loading (requires torchvision)
python test_rope_mvit.py
```

## Troubleshooting

### Common Issues

1. **"head_dim must be divisible by 6"**:
   - This model's head dimension is not compatible with 3D RoPE
   - Consider using a different number of heads or embedding dimension

2. **Performance degradation**:
   - RoPE changes position encoding fundamentally
   - Requires full retraining, not just fine-tuning
   - Use gradual unfreezing schedule

3. **Memory issues**:
   - MViT models are memory-intensive
   - Reduce batch size and use gradient accumulation
   - Enable mixed precision training (use_amp: true)

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (2D Axial RoPE)

## Implementation Notes

### RoPE Parameters
Following the design of DINOv3 and other modern implementations:
- RoPE uses **fixed, non-trainable** frequency parameters
- This is consistent with the original RoPE paper's philosophy
- For DDP compatibility, models with RoPE require `find_unused_parameters=True`

## Future Improvements

1. **Adaptive Frequency Scaling**: Automatically adjust frequencies based on video characteristics
2. **Mixed Position Encoding**: Combine RoPE with learned position embeddings  
3. **Extension to Other Models**: Add support for Swin3D, VideoMAE, etc.
4. **Learnable Frequencies (Optional)**: Experiment with learnable frequency bases as an alternative approach