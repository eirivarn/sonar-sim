#!/usr/bin/env python3
"""Analysis functions for sonar data."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional

from .config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from .coordinate_transform import polar_to_db_normalized, rasterize_cone, to_uint8_gray
from .image_processing import preprocess_edges
from .tracker import NetTracker


def analyze_npz_sequence(
    npz_path: Union[str, Path],
    frame_start: int = 0,
    frame_count: int = 100,
    frame_step: int = 1,
    save_outputs: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Analyze a sequence of sonar frames for net detection and distance measurement.
    
    Processing Pipeline:
    1. Load cone-view sonar frames from NPZ file
    2. Convert to binary images (eliminates intensity variations)
    3. Apply edge enhancement using morphological operations
    4. Track net structures using NetTracker system
    5. Convert pixel distances to meters using spatial extent
    6. Return time series of measurements
    
    Args:
        npz_path: Path to NPZ file containing cone-view sonar data
        frame_start: First frame to process
        frame_count: Number of frames to process
        frame_step: Process every Nth frame
        save_outputs: Save results to CSV file
        output_dir: Directory for output CSV (defaults to same as npz_path)
        
    Returns:
        DataFrame with columns: frame_index, timestamp, distance_pixels, 
        distance_meters, angle_degrees, detection_success, tracking_status, area
    """
    npz_path = Path(npz_path)
    
    # Load NPZ data and handle polar format
    data = np.load(npz_path)
    cones_raw = data['cones']
    timestamps = data.get('ts_unix_ns', np.arange(len(cones_raw)))
    extent = data.get('extent', None)
    
    # Check if data is polar format and needs conversion
    aspect_ratio = cones_raw.shape[1] / cones_raw.shape[2]
    is_polar = aspect_ratio > 2.0
    
    if is_polar:
        print(f"Loaded NPZ: {npz_path.name}")
        print(f"  Frames: {len(cones_raw)}")
        print(f"  Raw shape: {cones_raw[0].shape} (POLAR FORMAT, aspect={aspect_ratio:.1f})")
        print(f"  Converting to cone-view...")
        
        # Convert all frames from polar to cone-view
        cones = []
        for i in range(len(cones_raw)):
            polar_norm = polar_to_db_normalized(cones_raw[i])
            cone_frame, frame_extent = rasterize_cone(polar_norm)
            cones.append(cone_frame)
            if i == 0:
                extent = frame_extent
        cones = np.array(cones)
        print(f"  ✓ Converted shape: {cones[0].shape}")
    else:
        cones = cones_raw
        print(f"Loaded NPZ: {npz_path.name}")
        print(f"  Frames: {len(cones)}")
        print(f"  Shape: {cones[0].shape} (cone-view format)")
    
    if extent is not None:
        print(f"  Extent: {extent}")
    
    # Extract filename for ID
    file_id = npz_path.stem
    
    # Simple timestamp handling - convert to datetime if needed
    try:
        timestamps = pd.to_datetime(timestamps, unit='ns')
    except:
        # If timestamps are already datetime or invalid, use frame indices
        timestamps = pd.to_datetime(np.arange(len(cones)), unit='s')
    
    frame_indices = list(range(
        max(0, frame_start),
        min(len(cones), frame_start + frame_count),
        max(1, frame_step)
    ))
    
    print(f"Analyzing {len(frame_indices)} frames from {npz_path.name}")
    print(f"File ID: {file_id}")
    print(f"Using NetTracker system with binary processing and ellipse fitting")
    
    # Create NetTracker with combined configuration
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    results = []
    
    for i, frame_idx in enumerate(frame_indices):
        frame_u8 = to_uint8_gray(cones[frame_idx])
        H, W = frame_u8.shape[:2]
        
        # 1. Binary conversion (eliminates signal strength dependency)
        binary = (frame_u8 > config['binary_threshold']).astype(np.uint8) * 255

        # 2. Edge enhancement using morphological operations
        try:
            _, edges = preprocess_edges(binary, config)
        except:
            edges = binary  # Fallback to binary if enhancement fails
        
        # 3. Track net using NetTracker system
        contour = tracker.find_and_update(edges, (H, W))
        distance_px, angle_deg = tracker.calculate_distance(W, H)
        
        # 4. Convert pixel distance to meters using spatial extent
        distance_m = None
        if distance_px is not None and extent is not None:
            # Meters per pixel in Y (range) direction
            px2m = (extent[3] - extent[2]) / H
            # Convert: y_min + pixels * scaling_factor  
            distance_m = extent[2] + distance_px * px2m
        
        # 5. Store comprehensive result
        results.append({
            'frame_index': frame_idx,
            'timestamp': pd.Timestamp(timestamps[frame_idx]),
            'distance_pixels': distance_px,
            'distance_meters': distance_m,
            'angle_degrees': angle_deg,
            'detection_success': (contour is not None),
            'tracking_status': tracker.get_status(),
            'area': float(cv2.contourArea(contour)) if contour is not None else 0.0
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frame_indices)} | Status: {tracker.get_status()}")

    df = pd.DataFrame(results)
    
    # Store file_id as metadata (accessible for plotting)
    df.attrs['file_id'] = file_id

    # Print analysis summary
    detection_rate = df['detection_success'].mean() * 100
    print(f"\nAnalysis Summary:")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print(f"  Valid Distance Measurements: {df['distance_meters'].notna().sum()}")
    if df['distance_meters'].notna().any():
        print(f"  Distance Range: {df['distance_meters'].min():.2f} - {df['distance_meters'].max():.2f} m")
    
    if save_outputs:
        if output_dir is None:
            output_dir = npz_path.parent / 'analysis_output'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_id}_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    return df
