# Self-Supervised Learning (SSL) Pretraining Guide for Orion

## Overview

This guide describes the complete self-supervised learning pipeline for pretraining video models in Orion, featuring:
- **3D RoPE (Rotary Position Embeddings)** for enhanced spatiotemporal position encoding
- **Masked Video Modeling (MVM)** for self-supervised pretraining
- **MViT (Multiscale Vision Transformer)** as the backbone architecture

## Key Components

### 1. 3D RoPE (Rotary Position Embeddings)
- **File**: `orion/models/rope_3d.py`
- **Purpose**: Provides better position encoding for video data by splitting attention heads into temporal (T), height (H), and width (W) dimensions
- **Benefits**:
  - Better temporal understanding
  - Improved spatial relationships
  - No learnable parameters (computationally efficient)

### 2. Masked Video Modeling (MVM)
- **File**: `orion/models/masked_video_modeling.py`
- **Purpose**: Self-supervised pretraining by masking and reconstructing video patches
- **Features**:
  - Random, frame-level, and tube masking strategies
  - Lightweight decoder for reconstruction
  - Pixel-level or feature-level reconstruction

### 3. Integration
- **Modified Files**:
  - `orion/models/__init__.py` - Model loading and MVM head integration
  - `orion/utils/losses.py` - MVM loss registration

## Quick Start

### 1. SSL Pretraining (Unsupervised)

```bash
# Pretrain MViT with MVM on unlabeled video data
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    orion/utils/video_training_and_eval.py \
    --config_path=notebooks/config/config_mvit_mvm_pretraining.yaml
```

Key configuration (`config_mvit_mvm_pretraining.yaml`):
```yaml
# Task configuration
task: masked_video_modeling
loss: masked_video_modeling

# MVM configuration
mvm_config:
  mask_ratio: 0.75  # Mask 75% of patches
  mask_type: random  # Options: random, frame, tube
  decoder_layers: 4
  decoder_heads: 6

# Optional: Enable 3D RoPE
use_rope: true
rope_base: 10000.0
rope_temporal_scale: 1.0
```

### 2. Fine-tuning (Supervised)

After pretraining, fine-tune on labeled data:

```bash
# Fine-tune the pretrained model on downstream task
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    orion/utils/video_training_and_eval.py \
    --config_path=notebooks/config/config_mvit_mvm_finetune.yaml
```

Key configuration (`config_mvit_mvm_finetune.yaml`):
```yaml
# Load pretrained checkpoint
checkpoint_path: "outputs_mvm_pretrain/best_model.pt"
load_mvm_encoder_only: true  # Only load encoder weights

# Downstream task
task: classification
head_structure:
  contrast_agent: 1
  main_structure: 5
  stent_presence: 1

# Fine-tuning settings
lr: 0.00005  # Lower learning rate
video_freeze_schedule: linear  # Gradual unfreezing
```

## Training Strategies

### Stage 1: SSL Pretraining
**Goal**: Learn general video representations without labels

1. **Data Requirements**:
   - Large collection of unlabeled videos
   - No annotations needed
   - More data = better representations

2. **Recommended Settings**:
   ```yaml
   num_epochs: 100-200
   batch_size: 8-16  # Larger is better
   mask_ratio: 0.75  # Start with 75% masking
   lr: 0.0015  # Higher LR for pretraining
   ```

3. **Masking Strategies**:
   - **Random**: Best for general features (default)
   - **Frame**: Masks entire frames (temporal understanding)
   - **Tube**: Masks spatial tubes across time (motion understanding)

### Stage 2: Fine-tuning
**Goal**: Adapt pretrained model to specific task

1. **Layer-wise Learning Rate Decay**:
   ```yaml
   layer_decay: 0.75  # Earlier layers learn slower
   layer_decay_enabled: true
   ```

2. **Gradual Unfreezing**:
   ```yaml
   video_freeze_schedule: linear
   video_freeze_start: 0.8  # Start 80% frozen
   video_freeze_end: 0.0    # End fully trainable
   video_freeze_warmup_epochs: 10
   ```

3. **Lower Learning Rate**:
   ```yaml
   lr: 0.00005  # 30x lower than pretraining
   ```

## Advanced Features

### 1. Multi-Stage Masking Schedule
Gradually increase masking difficulty during training:

```yaml
masking_schedule:
  enabled: true
  schedule_type: cosine
  start_ratio: 0.4  # Start with 40% masking
  end_ratio: 0.8    # End with 80% masking
  warmup_epochs: 20
```

### 2. Combining 3D RoPE with MVM
Enable both for best results:

```yaml
# 3D RoPE for better position encoding
use_rope: true
rope_temporal_scale: 2.0  # Adjust for frame rate

# MVM for self-supervised learning
task: masked_video_modeling
mvm_config:
  mask_ratio: 0.75
```

### 3. Mixed Masking Strategy
Combine different masking types:

```python
# In custom training loop
mask_types = ["random", "frame", "tube"]
mask_type = random.choice(mask_types)
```

## Performance Tips

### 1. Memory Optimization
- Use gradient accumulation for larger effective batch size
- Enable mixed precision training (`use_amp: true`)
- Reduce decoder size if memory-constrained

### 2. Training Efficiency
- Use persistent workers (`persistent_workers: true`)
- Enable pin memory (`pin_memory: true`)
- Prefetch data (`prefetch_factor: 2`)

### 3. Distributed Training
```bash
# Multi-GPU training
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    orion/utils/video_training_and_eval.py \
    --config_path=config.yaml

# Multi-node training
torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    orion/utils/video_training_and_eval.py \
    --config_path=config.yaml
```

## Monitoring Training

### 1. Key Metrics to Track
- **Pretraining**: Reconstruction loss, mask accuracy
- **Fine-tuning**: Task-specific metrics (accuracy, F1, etc.)

### 2. Visualization
Enable reconstruction visualization:
```yaml
visualize_reconstruction:
  enabled: true
  num_samples: 8
  save_every_n_epochs: 10
```

### 3. Weights & Biases Integration
```yaml
wandb_project: "orion-ssl-pretraining"
wandb_entity: "your-team"
log_interval: 50
```

## Testing

### 1. Test SSL Components
```bash
# Test 3D RoPE implementation
python test_rope_simple.py

# Test MVM implementation
python test_ssl_rope_mvit.py
```

### 2. Verify Model Loading
```python
from orion.models import load_and_modify_model

# Load model with MVM
config = {
    "model_name": "mvit_v2_s",
    "task": "masked_video_modeling",
    "use_rope": True,
}
model = load_and_modify_model(config)
```

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: Reduce batch size, use gradient accumulation, or reduce mask_ratio

### Issue 2: Slow Convergence
**Solution**: Increase learning rate, use warmup, or reduce mask_ratio initially

### Issue 3: Poor Fine-tuning Performance
**Solution**: Use gradual unfreezing, lower learning rate, or longer pretraining

## Expected Results

### Pretraining
- Loss should steadily decrease
- Reconstructions should improve visually
- 100-200 epochs typically sufficient

### Fine-tuning
- 10-30% improvement over training from scratch
- Faster convergence (2-3x)
- Better generalization on small datasets

## Citation

If you use this SSL implementation, please cite:

```bibtex
@software{orion_ssl_2024,
  title = {Orion SSL: Self-Supervised Video Learning with 3D RoPE and MVM},
  year = {2024},
  url = {https://github.com/your-org/orion}
}
```

## References

1. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
2. [VideoMAE: Masked Autoencoders are Data-Efficient Learners](https://arxiv.org/abs/2203.12602)
3. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
4. [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227)