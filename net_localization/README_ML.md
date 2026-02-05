# Net Localization ML Pipeline

Complete pipeline for training a semantic segmentation model to detect nets in sonar imagery.

## Overview

This pipeline uses:
- **Ground Truth**: Extracted from simulation material IDs
- **Model**: U-Net with ResNet34 encoder (pretrained on ImageNet)
- **Loss**: Dice Loss (handles class imbalance)
- **Post-processing**: Skeleton extraction + line fitting

## Quick Start

### 1. Export Training Data

Open `net_localization_cartesian.ipynb` and run:

```python
# Export 1000 frames (800 train, 200 val)
export_training_data(num_frames=1000, train_split=0.8)
```

This creates:
```
training_data/
├── train/
│   ├── images/  # Sonar images (PNG)
│   └── masks/   # Binary masks (PNG)
└── val/
    ├── images/
    └── masks/
```

### 2. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 3. Train Model

```bash
python train_segmentation.py --data_dir ../training_data --epochs 50 --batch_size 8
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8, reduce if OOM)
- `--lr`: Learning rate (default: 1e-4)

**Expected Training Time:**
- CPU: ~30-60 min/epoch (not recommended)
- GPU (RTX 3080): ~2-5 min/epoch

### 4. Run Inference

```bash
python inference.py --model best_net_segmentation.pth --image_dir test_images/ --output_dir results/
```

## Data Augmentation

Training uses:
- Horizontal flip (50%)
- Rotation (±15°, 50%)
- Brightness/contrast (30%)
- Gaussian noise (20%)

## Model Architecture

**U-Net with ResNet34 Encoder:**
- Input: 400×400 RGB image (grayscale converted to 3-channel)
- Encoder: ResNet34 pretrained on ImageNet
- Decoder: Upsampling + skip connections
- Output: 400×400 binary mask

**Why U-Net?**
- Excellent for medical/scientific imaging
- Skip connections preserve spatial detail
- Handles class imbalance well with Dice Loss

## Post-Processing

After prediction, extract line segments:

```python
from inference import extract_lines_from_mask

mask = predict_mask(model, image)
lines = extract_lines_from_mask(mask, min_length=20)
# lines: [(x1, y1, x2, y2), ...] in pixel coordinates
```

**Steps:**
1. Thin mask to 1-pixel skeleton (morphological)
2. Find contours in skeleton
3. Fit lines using least-squares (cv2.fitLine)
4. Filter by minimum length

## Evaluation Metrics

Key metrics for segmentation:
- **Dice Score (F1)**: Harmonic mean of precision/recall
- **IoU**: Intersection over Union
- **Pixel Accuracy**: Correctly classified pixels

Add evaluation code in `train_segmentation.py` if needed.

## Troubleshooting

**OOM (Out of Memory):**
- Reduce `--batch_size` to 4 or 2
- Reduce image size in export (modify polar_to_cartesian output_size)

**Poor Performance:**
- Check material ID: Use slider in notebook to verify ID=1 shows nets
- Increase thickness: `export_training_data(thickness=3)`
- More data: Export more frames (>1000)
- Longer training: Increase epochs to 100+

**Slow Training:**
- Use GPU (CUDA)
- Reduce `num_workers` in DataLoader if CPU bottleneck
- Use mixed precision training (add `torch.cuda.amp`)

## Integration with Tracking

Use predictions for robot control:

```python
# Get net position from lines
def get_net_center(lines):
    if len(lines) == 0:
        return None
    points = np.concatenate([lines[:, :2], lines[:, 2:]], axis=0)
    return points.mean(axis=0)

center_x, center_y = get_net_center(lines)
# Use for path planning/following
```

## Next Steps

1. **Export data** (notebook cell)
2. **Train model** (50 epochs, ~2-4 hours on GPU)
3. **Validate** (check predictions in notebook)
4. **Deploy** (integrate with tracking system)

See `net_localization_cartesian.ipynb` for interactive visualization and testing.
