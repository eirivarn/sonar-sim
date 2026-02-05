# Temporal Smoothing for Sonar Segmentation

## Overview

Temporal smoothing reduces prediction flickering in video sequences by leveraging temporal consistency across consecutive frames. This is a **post-processing technique** that improves results **without retraining** the model.

## Why Temporal Smoothing?

**Sonar characteristics:**
- High noise (speckle, acoustic artifacts)
- Frame-to-frame variation in same scene
- Stationary/slow-moving structures (nets)

**Benefits:**
- ✅ Reduces prediction flickering
- ✅ Smoother tracking in videos
- ✅ Better temporal consistency
- ✅ No model retraining required
- ✅ Works with existing trained models

## Quick Start

### 1. Basic Usage (Image Sequences)

```bash
# Without smoothing (baseline)
python inference.py --model best_net_segmentation.pth --image_dir test_frames/

# With temporal smoothing (recommended)
python inference.py --model best_net_segmentation.pth --image_dir test_frames/ \
    --temporal_smoothing --smooth_method median --window_size 5
```

### 2. Video Processing

```bash
# Process simulation frames with temporal smoothing
python inference.py --model best_net_segmentation.pth \
    --image_dir ../simulation/data/runs/net_following_fish/frames/ \
    --video_mode --output_video output_smoothed.mp4 \
    --temporal_smoothing --smooth_method median --window_size 5
```

### 3. Python API

```python
from temporal_smoothing import ProbabilityTemporalSmoother
import numpy as np

# Initialize smoother
smoother = ProbabilityTemporalSmoother(
    window_size=5,          # 5 frames
    method='median',        # Robust to outliers
    threshold=0.5           # Binary threshold
)

# Process video frames
for frame in video_frames:
    # Get model prediction (probability map 0-1)
    prob_map = model.predict(frame)
    
    # Apply temporal smoothing
    smoothed_mask = smoother.update_and_threshold(prob_map)
    
    # Use smoothed mask for tracking/visualization
    process_mask(smoothed_mask)
```

## Methods

### 1. Median Filter (Recommended for Sonar)

**Best for:** Noisy sonar data with outliers

```python
smoother = TemporalSmoother(window_size=5, method='median')
```

**Pros:**
- Robust to outliers and noise spikes
- Preserves sharp edges
- No tuning parameters

**Cons:**
- Slightly higher computation (sorting)

### 2. Mean Filter

**Best for:** Low noise, smooth motion

```python
smoother = TemporalSmoother(window_size=5, method='mean')
```

**Pros:**
- Fast computation
- Simple averaging

**Cons:**
- Sensitive to outliers
- Can blur edges

### 3. Exponential Smoothing

**Best for:** Real-time streaming with minimal memory

```python
smoother = TemporalSmoother(window_size=5, method='exponential', alpha=0.3)
```

**Pros:**
- Memory efficient (no history buffer)
- Continuous adaptation
- Tunable responsiveness (alpha)

**Cons:**
- Delayed response to changes
- Requires tuning alpha

**Alpha guidelines:**
- `α = 0.1-0.2`: Heavy smoothing, slow adaptation
- `α = 0.3-0.4`: Balanced (recommended)
- `α = 0.5-0.7`: Light smoothing, fast adaptation

## Window Size Selection

**Frame rate: 30 FPS**

| Window Size | Time Span | Use Case |
|-------------|-----------|----------|
| 3 | 0.1 sec | Fast motion, minimal lag |
| 5 | 0.17 sec | **Recommended default** |
| 7 | 0.23 sec | Heavy noise, slow motion |
| 9 | 0.30 sec | Maximum smoothing |

**Rule of thumb:** Start with 5, increase if still flickering

## Performance Impact

| Method | Overhead per Frame | Memory |
|--------|-------------------|--------|
| No smoothing | 0 ms | - |
| Median (5 frames) | ~2-3 ms | 5 × frame_size |
| Mean (5 frames) | ~1 ms | 5 × frame_size |
| Exponential | <1 ms | 1 × frame_size |

For 400×400 images @ 30 FPS: **negligible impact**

## Advanced: Batch Smoothing

For offline processing of saved predictions:

```python
from temporal_smoothing import smooth_batch
import numpy as np

# Load pre-computed predictions
predictions = np.load('predictions.npy')  # Shape: (T, H, W)

# Apply temporal smoothing
smoothed = smooth_batch(predictions, method='median', window_size=5)

# Save smoothed results
np.save('predictions_smoothed.npy', smoothed)
```

## Examples

### Example 1: Compare Smoothing Methods

```bash
python demo_temporal_smoothing.py
```

Output:
- `temporal_smoothing_comparison.png` - Visual comparison
- `online_smoothing_demo.png` - Signal smoothing plot

### Example 2: Process Simulation Data

```python
from pathlib import Path
import torch
from inference import load_model, process_video_frames

# Load model
model = load_model('best_net_segmentation.pth', device='cpu')

# Process with temporal smoothing
process_video_frames(
    model,
    frame_dir='../simulation/data/runs/net_following_fish/frames/',
    output_video='output_smoothed.mp4',
    device='cpu',
    temporal_smoothing=True,
    smooth_method='median',
    window_size=5,
    fps=30
)
```

## Integration with Tracker

Add temporal smoothing to your tracking pipeline:

```python
from temporal_smoothing import ProbabilityTemporalSmoother

class NetTrackerWithSmoothing:
    def __init__(self, model, tracker_config):
        self.model = model
        self.tracker = NetTracker(tracker_config)
        self.smoother = ProbabilityTemporalSmoother(
            window_size=5,
            method='median',
            threshold=0.5
        )
    
    def process_frame(self, frame):
        # Predict with model
        prob_map = self.model.predict(frame)
        
        # Apply temporal smoothing
        mask = self.smoother.update_and_threshold(prob_map)
        
        # Track using smoothed mask
        contour = self.tracker.find_and_update(mask, frame.shape)
        
        return mask, contour
```

## Tips & Best Practices

1. **Always sort frames**: Temporal smoothing requires sequential order
   ```python
   frame_files = sorted(frame_dir.glob('frame_*.npz'))
   ```

2. **Reset between sequences**: Call `smoother.reset()` when video breaks
   ```python
   if new_sequence:
       smoother.reset()
   ```

3. **Use probability smoothing**: Smooth before thresholding for best results
   ```python
   # Good: Smooth probabilities
   smoother = ProbabilityTemporalSmoother(...)
   mask = smoother.update_and_threshold(prob_map)
   
   # OK: Smooth binary masks
   smoother = TemporalSmoother(...)
   mask = smoother.update(binary_mask)
   ```

4. **Warm-up period**: First N frames less smooth (filling buffer)
   ```python
   if smoother.is_warmed_up():
       # Buffer full, fully smoothed
       pass
   ```

## Troubleshooting

**Problem: Still flickering**
- Increase window_size (5 → 7)
- Try median if using mean
- Check frame ordering (must be sequential)

**Problem: Too much lag/blur**
- Decrease window_size (7 → 5 → 3)
- Use exponential with higher alpha
- Reduce smoothing strength

**Problem: Performance issues**
- Use exponential method (fastest)
- Reduce window_size
- Process at lower resolution

## Future Enhancements

**Phase 2: Learned Temporal Features**
- ConvLSTM layer in model
- True temporal modeling
- Better motion understanding

**Phase 3: Optical Flow**
- Motion compensation before smoothing
- Account for robot movement
- Adaptive smoothing based on motion

## References

- [Temporal Median Filtering](https://en.wikipedia.org/wiki/Median_filter)
- [Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)
- Sonar Image Processing: Coiras et al., IEEE OCEANS 2009
