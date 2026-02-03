# Minimal Net Tracker (SOLAQUA Edition)

A self-contained implementation of the full SOLAQUA NetTracker pipeline, including **adaptive momentum merging**, corridor splitting, and advanced linearity scoring.

## Features

- **Adaptive Momentum Merging**: Structure tensor-based orientation detection with anisotropic filtering
- **Corridor Splitting**: Extended search regions along major axis for far-field tracking
- **Linearity Scoring**: PCA-based contour straightness analysis
- **Aspect Ratio Filtering**: Preference for wide rectangular shapes (net edges)
- **Smoothed Tracking**: Separate alpha values for center, size, and angle
- **4-Panel Debug View**: Raw → Binary → Momentum Merged → Tracking

## What's Different from Basic Version

This version includes the full SOLAQUA pipeline:

1. **Binary Thresholding**: Signal-strength independent processing
2. **Adaptive Momentum Merge**: Advanced edge enhancement using:
   - Structure tensor computation (orientation + coherency)
   - Quantized angle bins (36 steps)
   - Top-K bin selection by linearity
   - Anisotropic Gaussian filtering per orientation
3. **Enhanced Edge Detection**: High-pass filter on momentum-merged frame
4. **Advanced Tracking**:
   - Ellipse + corridor search mask
   - Linearity scoring (PCA on contour points)
   - Aspect ratio scoring (prefer 2:1 to 5:1 rectangles)
   - Multi-parameter smoothing (center, size, angle)

## Requirements

```bash
pip install numpy pandas opencv-python rosbags tqdm
```

## Usage

### Basic Usage

```bash
python run_tracker.py /path/to/bag/file.bag output_video.mp4
```

### Process Specific Frame Range

```bash
# Process frames 0-1000
python run_tracker.py video.bag output.mp4 --start 0 --end 1000
/Volumes/LaCie/SOLAQUA_data/raw_data/2024-08-20_13-39-34_video.bag

# Process every 5th frame
python run_tracker.py video.bag output.mp4 --stride 5

# Combine: frames 100-500, every 2nd frame
python run_tracker.py video.bag output.mp4 --start 100 --end 500 --stride 2
```

## What It Does

1. **Loads Sonar Data**: Extracts raw polar sonar data from ROS bag file (Sonoptix format)
2. **Binary Conversion**: Threshold-based signal detection (removes intensity variations)
3. **Adaptive Momentum Merging**:
   - Computes structure tensors (dominant orientations)
   - Quantizes to 36 angle bins
   - Filters by linearity (coherency measure)
   - Applies anisotropic enhancement per orientation
4. **Edge Extraction**: High-pass filtering on enhanced frame
5. **Advanced Tracking**:
   - Creates ellipse + corridor search mask
   - Finds contours with linearity + aspect ratio scoring
   - Fits ellipse with multi-parameter smoothing
   - Calculates distance and angle
6. **Generates 4-Panel Debug Video**:
   - Panel 1 (top-left): Raw sonar image
   - Panel 2 (top-right): Binary threshold
   - Panel 3 (bottom-left): Momentum merged (key enhancement step)
   - Panel 4 (bottom-right): Tracking with overlays

## Configuration

All parameters are in the `config` dictionary in `generate_tracking_video()`:

### Tracking Parameters
```python
'center_smoothing_alpha': 0.4,          # Center position smoothing (0-1)
'ellipse_size_smoothing_alpha': 0.01,   # Size smoothing (very slow)
'ellipse_orientation_smoothing_alpha': 0.2,  # Angle smoothing
'ellipse_max_movement_pixels': 30.0,    # Max center movement per frame
'max_distance_change_pixels': 20,       # Max distance change per frame
'ellipse_expansion_factor': 0.5,        # Search region expansion
'use_corridor_splitting': True,         # Enable far-field corridor search
'corridor_band_k': 2.0,                 # Corridor width multiplier
'corridor_length_factor': 2.0,          # Corridor length multiplier
'min_contour_area': 200,                # Minimum contour size
'max_frames_without_detection': 30,     # Reset after this many lost frames
'linearity_score_weight': 1.0,          # Weight for linearity scoring
'aspect_ratio_score_weight': 1.0,       # Weight for aspect ratio scoring
```

### Image Processing
```python
'binary_threshold': 128,                # Binary threshold (0-255)
'use_advanced_momentum': True,          # Enable adaptive momentum merge

# Momentum Merging (advanced)
'angle_steps': 36,                      # Number of angle bins
'base_radius': 3,                       # Base kernel radius
'momentum_boost': 0.8,                  # Enhancement strength
'linearity_threshold': 0.15,            # Min linearity for processing
'momentum_downscale': 2,                # Downscale factor for speed
'top_k_bins': 8,                        # Max angle bins to process
'min_coverage_percent': 0.5,            # Min % coverage for bin

# Basic fallback (if advanced disabled)
'basic_open_kernel_size': 3,            # Morphological opening kernel

# Edge post-processing
'morph_close_kernel': 0,                # Morphological closing (0=disabled)
'edge_dilation_iterations': 0,          # Edge dilation (0=disabled)
```

## Output

The generated video shows a 2×2 grid:

- **Top-left (Raw)**: Original sonar intensity data
- **Top-right (Binary)**: Binary threshold (128)
- **Bottom-left (Momentum Merged)**: After adaptive enhancement - this is the key step that improves net detection
- **Bottom-right (Tracking)**: Final tracking with overlays:
  - Green: Detected contour
  - Blue: Fitted ellipse
  - Red: Ellipse center
  - Pink/Blue overlay: Search region (ellipse + corridor)
  - Cyan: Distance line (horizontal)
  - Magenta: Angle indicator
  - White text: Frame number, status, distance, angle

## Technical Details

### Adaptive Momentum Merging

The key innovation in SOLAQUA is adaptive momentum merging:

1. **Structure Tensor Analysis**: Computes local orientation and coherency at each pixel
2. **Angle Quantization**: Maps continuous orientations to 36 discrete bins
3. **Linearity Filtering**: Only processes regions with high coherency (straight edges)
4. **Top-K Selection**: Picks the K most linear angle bins to reduce computation
5. **Anisotropic Enhancement**: Applies oriented Gaussian filtering along each dominant direction
6. **Weighted Blending**: Combines enhancements weighted by local linearity

This produces much cleaner net edges compared to simple morphological operations.

### Corridor Splitting

When tracking is established, the search region becomes:
- **Ellipse**: Expanded around the last known position
- **Corridor**: Extended rectangular region along the major axis

This helps maintain tracking when the net moves far between frames.

## Example

```bash
# Process a full bag file
python run_tracker.py /Volumes/LaCie/bags/2024-08-20_13-39-34_video.bag tracking_output.mp4

# Quick preview: first 300 frames, every 3rd frame
python run_tracker.py /Volumes/LaCie/bags/2024-08-20_13-39-34_video.bag preview.mp4 --end 300 --stride 3
```

## Troubleshooting

### "No Sonoptix topics found"
- Verify the bag file contains sonar data
- Check topic names with: `ros2 bag info your_file.bag`

### Tracking lost frequently
- Check the "Momentum Merged" panel - edges should be clear
- Increase `ellipse_expansion_factor` (try 0.7 or 1.0)
- Decrease `linearity_threshold` (try 0.10) to enhance more regions
- Enable corridor: `use_corridor_splitting: True`

### Momentum merge not enhancing enough
- Increase `momentum_boost` (try 1.0 or 1.2)
- Decrease `min_coverage_percent` (try 0.3) to process more angle bins
- Increase `top_k_bins` (try 12 or 16)

### Too much noise in edges
- Increase `linearity_threshold` (try 0.20 or 0.25)
- Increase `binary_threshold` (try 140 or 150)
- Enable edge post-processing: `morph_close_kernel: 3`

### Processing too slow
- Increase `momentum_downscale` (try 3 or 4)
- Decrease `angle_steps` (try 24 or 18)
- Decrease `top_k_bins` (try 4 or 6)
- Disable advanced mode: `use_advanced_momentum: False`

## File Structure

```
minimal_net_tracker/
├── run_tracker.py          # Main script (everything in one file)
└── README.md              # This file
```

## Performance

- **Loading**: ~1-2 seconds per 1000 frames
- **Processing**: ~5-10 FPS depending on resolution
- **Memory**: ~100-200 MB for typical datasets

## Limitations

- Only supports Sonoptix sonar format
- Edge detection parameters may need tuning per dataset
- No multi-system synchronization (DVL, FFT)
- Fixed CLAHE and Canny parameters (edit code to change)

## Next Steps

For more advanced features:
- Use the full SOLAQUA pipeline for multi-system synchronization
- See `utils/video_generation.py` for advanced video rendering
- See `utils/sonar_tracking.py` for the full tracker implementation
