#!/usr/bin/env python3
"""Enhanced video generation with 6-panel visualization."""

import json
import cv2
import numpy as np
from pathlib import Path

from .config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from .coordinate_transform import polar_to_db_normalized, rasterize_cone, to_uint8_gray
from .image_processing import preprocess_edges, adaptive_linear_momentum_merge_fast
from .data_loading import load_or_extract_sonar_data
from .tracker import NetTracker


def create_enhanced_contour_detection_video(
    npz_path: Path,
    output_path: Path,
    frame_start: int = 0, 
    frame_count: int = 100,
    frame_step: int = 5,
    fps: int = 15
):
    """
    Create 2x3 grid video showing complete contour detection pipeline with NetTracker.
    
    Grid Layout:
        Row 1: Raw Frame | Momentum-Merged | Edges
        Row 2: Search Mask | Best Contour | Distance Measurement
    
    Args:
        npz_path: Path to NPZ file or ROS bag (will be converted to NPZ)
        output_path: Output video file path
        frame_start: Starting frame index
        frame_count: Maximum number of frames to process
        frame_step: Process every Nth frame (stride)
        fps: Output video frame rate
        
    Returns:
        Path to generated video or None on error
    """
    print("=== ENHANCED CONTOUR DETECTION VIDEO (2x3 Grid with NetTracker) ===")
    print(f"Input: {npz_path}")
    print(f"Output: {output_path}")
    
    # Convert bag files to NPZ first
    npz_path = Path(npz_path)
    if npz_path.suffix == '.bag':
        print("\n⚠️  Input is a ROS bag file. Converting to NPZ format...")
        df, metadata = load_or_extract_sonar_data(npz_path, use_cache=True)
        
        # Get the cached NPZ path (in cache subdirectory)
        cache_dir = npz_path.parent / "cache"
        cache_path = cache_dir / f"{npz_path.stem}_raw_polar.npz"
        
        if not cache_path.exists():
            print(f"ERROR: Failed to create NPZ cache at {cache_path}")
            return None
        
        npz_path = cache_path
        print(f"✓ Using cached NPZ: {npz_path}\n")
    
    # Load data - handle both polar and cone-view formats
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # Check which key is available
            if 'cones' in data:
                cones_raw = data['cones']  # May be polar or cone-view
            elif 'raw_polar' in data:
                cones_raw = data['raw_polar']  # Polar format from bag extraction
            else:
                print(f"ERROR: NPZ file must contain 'cones' or 'raw_polar' key")
                print(f"Available keys: {list(data.keys())}")
                return None
            
            timestamps = data.get('ts_unix_ns', data.get('timestamps', np.arange(len(cones_raw))))
            extent = data.get('extent', None)
            
            # Check if data is polar format (tall/narrow) and needs conversion
            aspect_ratio = cones_raw.shape[1] / cones_raw.shape[2]  # H/W
            is_polar = aspect_ratio > 2.0  # Polar is typically ~4+ aspect ratio
            
            if is_polar:
                print(f"⚠️  Data is in POLAR format (aspect={aspect_ratio:.1f}), converting to cone-view...")
                # Apply polar-to-cone transformation for each frame
                cones = []
                for i in range(len(cones_raw)):
                    # Normalize polar data
                    polar_norm = polar_to_db_normalized(cones_raw[i])
                    # Convert to cone-view
                    cone_frame, frame_extent = rasterize_cone(polar_norm)
                    cones.append(cone_frame)
                    if i == 0:
                        extent = frame_extent  # Use extent from first frame
                cones = np.array(cones)
                print(f"  ✓ Converted to cone-view: {cones.shape}")
            else:
                print(f"✓ Data is already in cone-view format (aspect={aspect_ratio:.1f})")
                cones = cones_raw
            
            # Extract metadata
            meta_str = data.get('meta_json', None)
            if meta_str is not None:
                metadata = json.loads(meta_str.item() if hasattr(meta_str, 'item') else str(meta_str))
            else:
                metadata = {}
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return None
    
    T = len(cones)
    actual = int(min(frame_count, max(0, (T - frame_start) // max(1, frame_step))))
    
    if actual <= 0:
        print("Error: Not enough frames to process")
        return None
    
    first_u8 = to_uint8_gray(cones[frame_start])
    H, W = first_u8.shape
    
    # Grid is 2 rows x 3 columns
    grid_h = H * 2
    grid_w = W * 3
    
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(outp), fourcc, fps, (grid_w, grid_h))
    
    if not vw.isOpened():
        print("Error: Could not initialize video writer")
        return None
    
    print(f"Processing {actual} frames...")
    print(f"Grid layout (2x3):")
    print(f"  Row 1: Raw | Momentum-Merged | Edges")
    print(f"  Row 2: Search Mask | Best Contour | Distance")
    print(f"Output grid size: {grid_w}x{grid_h}")
    
    # Initialize NetTracker (same as analysis)
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    print(f"Tracker config:")
    print(f"  expansion: {config['ellipse_expansion_factor']}")
    print(f"  center_alpha: {config['center_smoothing_alpha']}")
    print(f"  size_alpha: {config['ellipse_size_smoothing_alpha']}")
    print(f"  angle_alpha: {config['ellipse_orientation_smoothing_alpha']}")
    
    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        
        # 1. Binary conversion
        binary = (frame_u8 > config['binary_threshold']).astype(np.uint8) * 255
        
        # Just for showcasing the momentum merge step
        try:
            use_advanced = config.get('use_advanced_momentum_merging', True)
            if use_advanced:
                momentum = adaptive_linear_momentum_merge_fast(binary,
                    angle_steps=config['adaptive_angle_steps'],
                    base_radius=config['adaptive_base_radius'],
                    max_elongation=config['adaptive_max_elongation'],
                    momentum_boost=config['momentum_boost'],
                    linearity_threshold=config['adaptive_linearity_threshold'],
                    downscale_factor=config['downscale_factor'],
                    top_k_bins=config['top_k_bins'],
                    min_coverage_percent=config['min_coverage_percent'],
                    gaussian_sigma=config['gaussian_sigma']
                )
            else:
                # Use basic enhancement methods (morphological dilation or Gaussian blur)
                use_dilation = config.get('basic_use_dilation', True)
                if use_dilation:
                    kernel_size = config.get('basic_dilation_kernel_size', 3)
                    iterations = config.get('basic_dilation_iterations', 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    momentum = cv2.dilate(binary, kernel, iterations=iterations)
                else:
                    kernel_size = config.get('basic_gaussian_kernel_size', 3)
                    gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
                    momentum = cv2.GaussianBlur(binary, (kernel_size, kernel_size), gaussian_sigma)
            momentum_display = cv2.cvtColor(momentum, cv2.COLOR_GRAY2BGR)
        except:
            momentum = binary
            momentum_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 2. Image processing
        try:
            _, edges = preprocess_edges(binary, config)
            edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        except:
            edges = binary
            edges_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 4. Search mask (from tracker)
        search_mask_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        search_mask = tracker._get_search_mask((H, W))
        if search_mask is not None:
            overlay = search_mask_display.copy()
            overlay[search_mask > 0] = [0, 255, 0]  # Green
            search_mask_display = cv2.addWeighted(search_mask_display, 0.7, overlay, 0.3, 0)
            
            # Draw ellipse outline
            if tracker.center and tracker.size and tracker.angle is not None:
                try:
                    ell = ((int(tracker.center[0]), int(tracker.center[1])),
                           (int(tracker.size[0] * (1 + config['ellipse_expansion_factor'])),
                            int(tracker.size[1] * (1 + config['ellipse_expansion_factor']))),
                           tracker.angle)
                    cv2.ellipse(search_mask_display, ell, (255, 0, 255), 2)
                except:
                    pass
        
        # 5. Track using NetTracker
        contour = tracker.find_and_update(edges, (H, W))
        
        best_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        if contour is not None:
            cv2.drawContours(best_display, [contour], -1, (0, 255, 0), 2)
            if len(contour) >= 5:
                try:
                    ell = cv2.fitEllipse(contour)
                    cv2.ellipse(best_display, ell, (255, 0, 255), 1)
                except:
                    pass
        
        # 6. Distance visualization
        distance_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        distance_result = tracker.calculate_distance(W, H)
        
        if distance_result is not None:
            distance_px, angle_deg = distance_result
        else:
            distance_px, angle_deg = None, None
        
        if distance_px is not None:
            # Draw center line
            cv2.line(distance_display, (W//2, 0), (W//2, H), (128, 128, 128), 1)
            
            # Draw distance point
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (0, 0, 255), -1)
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (255, 255, 255), 2)
            
            # Draw the red line
            if tracker.center and tracker.size and tracker.angle is not None:
                ang_r = np.radians(angle_deg)
                half_len = max(tracker.size) / 2
                
                p1x = int(tracker.center[0] + half_len * np.cos(ang_r))
                p1y = int(tracker.center[1] + half_len * np.sin(ang_r))
                p2x = int(tracker.center[0] - half_len * np.cos(ang_r))
                p2y = int(tracker.center[1] - half_len * np.sin(ang_r))
                
                cv2.line(distance_display, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)
            
            dist_m = None
            if extent is not None:
                px2m = (extent[3] - extent[2]) / H
                dist_m = extent[2] + distance_px * px2m
                cv2.putText(distance_display, f"{dist_m:.2f}m", (W//2 + 15, int(distance_px)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Ensure all arrays have correct format
        if binary.ndim == 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if momentum_display.ndim == 2:
            momentum_display = cv2.cvtColor(momentum_display, cv2.COLOR_GRAY2BGR)
        if edges_display.ndim == 2:
            edges_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)

        # Add text labels
        cv2.putText(binary, "1. Raw - Binary Frame", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(momentum_display, "2. Momentum Merged", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(edges_display, "3. Edges", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(search_mask_display, "4. Search Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(best_display, "5. Best Contour", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(distance_display, "6. Distance", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Stack the images
        row0 = np.hstack([binary, momentum_display, edges_display])
        row1 = np.hstack([search_mask_display, best_display, distance_display])
        grid_frame = np.vstack([row0, row1])
        
        # Ensure uint8
        if grid_frame.dtype != np.uint8:
            grid_frame = np.clip(grid_frame, 0, 255).astype(np.uint8)
        
        # Frame info
        frame_info = f'Frame: {idx} | {tracker.get_status()}'
        cv2.putText(grid_frame, frame_info, (10, grid_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        vw.write(grid_frame)
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{actual} frames")
    
    vw.release()
    print(f"\n✓ Video saved to: {output_path}")
    print(f"Grid layout: Raw | Momentum | Edges")
    print(f"             Search Mask | Best Contour | Distance")
    
    return output_path
