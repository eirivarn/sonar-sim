#!/usr/bin/env python3
"""Video generation functions."""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from .coordinate_transform import polar_to_db_normalized, rasterize_cone, to_uint8_gray
from .image_processing import preprocess_edges, adaptive_linear_momentum_merge_fast
from .data_loading import load_or_extract_sonar_data, get_sonar_frame_polar, load_raw_polar_npz
from .tracker import NetTracker


def generate_tracking_video(
    df: pd.DataFrame, 
    output_path: Path, 
    start_frame: int = 0, 
    end_frame: int = None,
    stride: int = 1,
    fov_deg: float = 120.0,
    rmin: float = 0.5,
    rmax: float = 20.0,
    img_h: int = 700,
    img_w: int = 900
):
    """
    Generate tracking debug video with full processing pipeline.
    
    Args:
        df: DataFrame with sonar data (from load_or_extract_sonar_data)
        output_path: Output video file path
        start_frame: First frame to process
        end_frame: Last frame to process (None = all)
        stride: Process every Nth frame
        fov_deg: Field of view in degrees
        rmin: Minimum range in meters
        rmax: Maximum range in meters
        img_h: Output image height
        img_w: Output image width
    """
    # Initialize tracker
    tracker = NetTracker(TRACKING_CONFIG)
    
    # Frame range
    N = len(df)
    start_frame = max(0, start_frame)
    end_frame = min(N, end_frame if end_frame is not None else N)
    frame_indices = list(range(start_frame, end_frame, stride))
    
    print(f"\nProcessing {len(frame_indices)} frames...")
    print(f"  FOV: {fov_deg}° | Range: {rmin}-{rmax}m | Size: {img_w}x{img_h}")
    
    # Initialize video writer (H.264 codec for VSCode compatibility)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = 15.0
    
    # Process frames
    for idx in tqdm(frame_indices, desc="Tracking"):
        try:
            # 1. Get raw polar data
            polar_img = get_sonar_frame_polar(df, idx)
            
            # 2. Convert to dB and normalize to [0,1]
            polar_db_normalized = polar_to_db_normalized(polar_img, db_norm=60.0)
            
            # 3. Convert to Cartesian cone-view
            sonar_img, extent = rasterize_cone(polar_db_normalized, fov_deg, rmin, rmax, img_h, img_w)
            
            # 4. Convert to uint8
            sonar_img_u8 = to_uint8_gray(sonar_img)
            H, W = sonar_img_u8.shape
            
            # 5. Preprocess and extract edges
            binary, edges = preprocess_edges(sonar_img_u8, IMAGE_PROCESSING_CONFIG)
            
            # 6. Track
            contour = tracker.find_and_update(edges, (H, W))
            distance, angle = tracker.calculate_distance(W, H)
            
            # 7. Create debug frame (4-panel)
            debug_frame = _create_debug_frame(sonar_img_u8, binary, binary, edges,
                                            contour, tracker, distance, angle, idx)
            
            # Initialize writer on first frame
            if writer is None:
                frame_h, frame_w = debug_frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                        (frame_w, frame_h), True)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {output_path}")
                print(f"✓ Video writer initialized: {frame_w}x{frame_h} @ {fps} FPS")
            
            # Write frame
            writer.write(debug_frame)
            
        except Exception as e:
            print(f"\nError processing frame {idx}: {e}")
            continue
    
    # Cleanup
    if writer is not None:
        writer.release()
    
    print(f"\n✓ Video saved to: {output_path}")


def _create_debug_frame(
    sonar_img_u8: np.ndarray,
    binary: np.ndarray,
    momentum: np.ndarray,
    edges: np.ndarray,
    contour: np.ndarray,
    tracker: NetTracker,
    distance: float,
    angle: float,
    frame_idx: int
) -> np.ndarray:
    """Create 4-panel debug visualization."""
    H, W = sonar_img_u8.shape
    
    # Convert all to color
    sonar_color = cv2.cvtColor(sonar_img_u8, cv2.COLOR_GRAY2BGR)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    momentum_color = cv2.cvtColor(momentum, cv2.COLOR_GRAY2BGR)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Draw tracking on edges panel
    if tracker.center is not None and tracker.size is not None:
        cx, cy = int(tracker.center[0]), int(tracker.center[1])
        w, h = int(tracker.size[0]), int(tracker.size[1])
        ang = tracker.angle
        
        # Draw ellipse
        cv2.ellipse(edges_color, (cx, cy), (w//2, h//2), ang, 0, 360, (0, 255, 0), 2)
        cv2.circle(edges_color, (cx, cy), 5, (0, 0, 255), -1)
        
        # Draw perpendicular line (distance indicator)
        if distance is not None and angle is not None:
            red_angle_rad = np.radians(angle)
            length = 100
            dx = int(length * np.cos(red_angle_rad))
            dy = int(length * np.sin(red_angle_rad))
            p1x, p1y = cx - dx, cy - dy
            p2x, p2y = cx + dx, cy + dy
            cv2.line(edges_color, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)
    
    # Add labels
    cv2.putText(sonar_color, "1. Cone-View", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(binary_color, "2. Binary", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(momentum_color, "3. Momentum Merged", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(edges_color, "4. Tracking", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add status
    status = f"Frame: {frame_idx} | {tracker.get_status()}"
    if distance is not None:
        status += f" | Dist: {distance:.1f}px | Angle: {angle:.1f}deg"
    
    cv2.putText(edges_color, status, (10, H - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Stack: 2x2 grid
    top_row = np.hstack([sonar_color, binary_color])
    bottom_row = np.hstack([momentum_color, edges_color])
    debug_frame = np.vstack([top_row, bottom_row])
    
    return debug_frame


def generate_raw_video(npz_path, output_path, frame_start=0, frame_count=None, frame_step=1, fps=30):
    """
    Generate a simple video showing just the raw sonar data (cone-view).
    
    Args:
        npz_path: Path to NPZ file with sonar data (or DataFrame)
        output_path: Output video path (will use raw_video_ prefix)
        frame_start: Starting frame index
        frame_count: Number of frames to process (None = all)
        frame_step: Step between frames
        fps: Video framerate
    """
    import os
    import re
    import cv2
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    
    # Extract timestamp or filename from input file path
    source_identifier = None
    if isinstance(npz_path, str) or hasattr(npz_path, '__fspath__'):
        path = Path(npz_path)
        # Look for date pattern in filename (YYYY-MM-DD or YYYYMMDD)
        filename = path.stem
        date_match = re.search(r'(\d{4})[_-]?(\d{2})[_-]?(\d{2})', filename)
        if date_match:
            source_identifier = f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
        else:
            # No timestamp found - use filename (e.g., simulation data)
            # Remove common suffixes like _cones, _raw_polar
            clean_name = filename.replace('_cones', '').replace('_raw_polar', '')
            source_identifier = clean_name
    
    # Load data - handle ROS bag, simulation NPZ, or cached NPZ
    df = None
    metadata = {}
    
    if isinstance(npz_path, str) or hasattr(npz_path, '__fspath__'):
        path = Path(npz_path)
        if path.suffix == '.bag':
            # ROS bag file - extract with caching
            print(f"Loading ROS bag: {path.name}")
            df, metadata = load_or_extract_sonar_data(path)
        else:
            # NPZ file - check structure
            with np.load(path, allow_pickle=True) as data:
                if 'cones' in data.keys():
                    # Simulation NPZ - load all cone data before context closes
                    cones = data['cones']
                    print(f"Loading simulation data: {len(cones)} frames")
                    # Copy all cone data to list before context exits
                    cone_list = [cones[i].copy() for i in range(len(cones))]
                    # Create DataFrame with cone data directly accessible
                    df = pd.DataFrame({
                        'frame_idx': range(len(cone_list)),
                        'cone_data': cone_list
                    })
                    metadata = {}
                elif 'raw_polar' in data.keys():
                    # Cached NPZ from ROS bag - will load after context
                    pass
                else:
                    raise ValueError(f"NPZ file must contain 'raw_polar' or 'cones' key. Found: {list(data.keys())}")
            
            # If df not set (raw_polar case), load it properly
            if df is None:
                df, metadata = load_raw_polar_npz(npz_path)
    else:
        df = npz_path  # Already a DataFrame
    
    num_frames = len(df)
    
    if frame_count is None:
        frame_count = num_frames - frame_start
    
    frame_end = min(frame_start + frame_count * frame_step, num_frames)
    
    # Create output path with identifier prefix
    base_name = os.path.splitext(os.path.basename(str(output_path)))[0]
    output_dir = os.path.dirname(str(output_path))
    if source_identifier:
        raw_output_path = os.path.join(output_dir, f"raw_video_{source_identifier}.mp4")
    else:
        raw_output_path = os.path.join(output_dir, f"raw_video_{base_name}.mp4")
    
    # Get first frame to determine size - handle both DataFrame types
    if 'cone_data' in df.columns:
        # Simulation data - check if polar format and convert to Cartesian
        first_frame = df.iloc[frame_start]['cone_data']
        # Check aspect ratio to determine if polar format
        aspect_ratio = first_frame.shape[0] / first_frame.shape[1]
        if aspect_ratio > 2.0:
            # Polar format - convert to Cartesian cone-view
            first_cone, _ = rasterize_cone(polar_to_db_normalized(first_frame))
        else:
            # Already in cone format - just use it
            first_cone = first_frame
        first_cone = to_uint8_gray(first_cone)
    else:
        # ROS bag data - needs polar to cone conversion
        first_polar = get_sonar_frame_polar(df, frame_start)
        first_cone, _ = rasterize_cone(polar_to_db_normalized(first_polar))
        first_cone = to_uint8_gray(first_cone)
    
    H, W = first_cone.shape
    
    # Setup video writer (H.264 codec for VSCode compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (W, H))
    
    print(f"\nGenerating raw video: {raw_output_path}")
    print(f"Frames: {frame_start} to {frame_end} (step={frame_step})")
    
    # Generate frames
    for frame_idx in tqdm(range(frame_start, frame_end, frame_step), desc="Raw video"):
        if 'cone_data' in df.columns:
            # Simulation data - check if polar format and convert to Cartesian
            frame = df.iloc[frame_idx]['cone_data']
            # Check aspect ratio to determine if polar format
            aspect_ratio = frame.shape[0] / frame.shape[1]
            if aspect_ratio > 2.0:
                # Polar format - convert to Cartesian cone-view
                cone, _ = rasterize_cone(polar_to_db_normalized(frame))
                cone_u8 = to_uint8_gray(cone)
            else:
                # Already in cone format
                cone_u8 = to_uint8_gray(frame)
        else:
            # ROS bag data - convert from polar
            polar = get_sonar_frame_polar(df, frame_idx)
            cone, _ = rasterize_cone(polar_to_db_normalized(polar))
            cone_u8 = to_uint8_gray(cone)
        
        cone_color = cv2.cvtColor(cone_u8, cv2.COLOR_GRAY2BGR)
        
        # Add frame number
        cv2.putText(cone_color, f"Frame: {frame_idx}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(cone_color)
    
    out.release()
    print(f"Raw video saved to: {raw_output_path}")
    return raw_output_path
