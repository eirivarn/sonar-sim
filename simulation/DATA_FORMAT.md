# Simulation Data Format

## Overview
The simulation now saves data in the exact same format as real sonar data from the SOLAQUA system, enabling seamless integration with all existing analysis tools.

## File Structure

### During Data Collection
The simulation saves individual NPZ files for each frame:
```
data/runs/run_name/
  ├── frames/
  │   ├── frame_000000.npz
  │   ├── frame_000001.npz
  │   └── ...
  └── path_config.json
```

### After Data Collection
A consolidated NPZ file is automatically created:
```
data/runs/run_name/
  ├── frames/              # Individual frames (kept for reference)
  ├── path_config.json     # Path configuration
  └── run_name_cones.npz  # ← CONSOLIDATED FILE (use this!)
```

## Consolidated NPZ Format

The consolidated file matches the real data format exactly:

```python
with np.load('run_name_cones.npz', allow_pickle=True) as data:
    cones = data['cones']           # (T, H, W) float32 - all frames stacked
    extent = data['extent']         # (4,) float64 - spatial bounds
    ts_unix_ns = data['ts_unix_ns'] # (T,) int64 - nanosecond timestamps
    meta_json = data['meta_json']   # JSON string - metadata
```

### Field Details

**`cones`**: Shape (T, H, W) where:
- T = number of frames (time)
- H = 1024 (range bins)
- W = 256 (beams)
- dtype = float32
- Values represent sonar echo intensity

**`extent`**: Shape (4,) tuple:
- `[x_min, x_max, y_min, y_max]` in meters
- Defines the spatial bounds of the sonar cone
- Used for pixel-to-meter conversion

**`ts_unix_ns`**: Shape (T,) array:
- Nanoseconds since Unix epoch
- One timestamp per frame
- dtype = int64

**`meta_json`**: JSON string containing:
```json
{
  "num_frames": 100,
  "range_m": 20.0,
  "fov_deg": 90.0,
  "rmin": 0.0,
  "rmax": 20.0,
  "display_range_max_m": 20.0,
  "collection_mode": "circular",
  "frames": [
    {
      "frame": 0,
      "t": 1234567890.123,
      "sonar_position": [25.0, 25.0],
      "fish_positions": [[...], ...],
      ...
    },
    ...
  ]
}
```

## Usage with SOLAQUA Tools

### Load Data
```python
from utils.sonar_utils import load_cone_run_npz

# Load simulation data (same as real data)
cones, timestamps, extent, meta = load_cone_run_npz(
    "/path/to/run_name_cones.npz"
)

print(f"Loaded {cones.shape[0]} frames")
print(f"Shape per frame: {cones.shape[1:]} (H x W)")
print(f"Spatial extent: {extent}")
```

### Run Analysis
```python
from utils.sonar_analysis import analyze_npz_sequence

# Analyze simulation data (same as real data)
df = analyze_npz_sequence(
    npz_file_path="/path/to/run_name_cones.npz",
    frame_start=0,
    frame_count=100,
    save_outputs=True
)

print(f"Detection rate: {df['detection_success'].mean() * 100:.1f}%")
```

### Generate Videos
```python
from utils.video_generation import create_enhanced_contour_detection_video

# Create tracking video (same as real data)
video_path = create_enhanced_contour_detection_video(
    npz_file="/path/to/run_name_cones.npz",
    frame_count=300,
    output_path="tracking.mp4"
)
```

## Testing Your Simulation Data

Use the provided notebook:
```bash
jupyter notebook 07_simulation_testing.ipynb
```

The notebook provides:
1. Data loading and validation
2. Frame visualization
3. Net tracking analysis
4. Video generation

## Key Benefits

✅ **100% Compatible**: Works with all existing SOLAQUA tools  
✅ **Same Format**: No special handling needed  
✅ **Efficient**: Single file instead of thousands  
✅ **Complete**: Includes all metadata and timestamps  
✅ **Flexible**: Individual frames still available if needed

## Migration from Old Format

If you have old simulation data (individual NPZ files), you can consolidate them:

```python
import numpy as np
import json
from pathlib import Path

def consolidate_frames(frames_dir, output_path):
    """Consolidate individual frame NPZ files into single file."""
    frames_dir = Path(frames_dir)
    npz_files = sorted(frames_dir.glob('frame_*.npz'))
    
    all_frames = []
    all_timestamps = []
    first_extent = None
    
    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as data:
            all_frames.append(data['sonar_image'])
            all_timestamps.append(data['ts_unix_ns'][0])
            if first_extent is None:
                first_extent = data['extent']
    
    # Stack and save
    cones = np.stack(all_frames, axis=0).astype(np.float32)
    ts = np.array(all_timestamps, dtype=np.int64)
    
    meta = {'num_frames': len(all_frames)}
    
    np.savez_compressed(
        output_path,
        cones=cones,
        extent=first_extent,
        ts_unix_ns=ts,
        meta_json=json.dumps(meta)
    )
    
    print(f"✓ Consolidated {len(all_frames)} frames to {output_path}")

# Usage
consolidate_frames('data/runs/old_run/frames', 'old_run_cones.npz')
```
