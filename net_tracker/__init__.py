#!/usr/bin/env python3
"""
Net Tracker Package - Modular sonar net tracking system.

This package provides a complete pipeline for tracking fishing nets in sonar imagery:
- Configuration management
- Coordinate transformations (polar ↔ cone-view)
- Image enhancement and preprocessing
- ROS bag loading with NPZ caching
- Net tracking with elliptical AOI
- Video generation with debug visualizations
- Analysis and distance measurement

Usage Examples:
--------------

1. Load data from ROS bag (with caching):
    from net_tracker import load_or_extract_sonar_data
    df, metadata = load_or_extract_sonar_data(Path("data.bag"))

2. Track nets and generate video:
    from net_tracker import generate_tracking_video
    generate_tracking_video(df, Path("output.mp4"))

3. Analyze NPZ sequence:
    from net_tracker import analyze_npz_sequence
    results_df = analyze_npz_sequence("data.npz", frame_count=500)

4. Generate enhanced 6-panel video:
    from net_tracker import create_enhanced_contour_detection_video
    create_enhanced_contour_detection_video("data.npz", "output.mp4")

5. Manual tracking:
    from net_tracker import NetTracker, preprocess_edges, IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
    
    tracker = NetTracker(TRACKING_CONFIG)
    _, edges = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
    contour = tracker.find_and_update(edges, (H, W))
    distance, angle = tracker.calculate_distance(W, H)
"""

# Configuration
from .config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG

# Coordinate transformations
from .coordinate_transform import (
    rasterize_cone,
    polar_to_db_normalized,
    to_uint8_gray,
)

# Image processing
from .image_processing import (
    preprocess_edges,
    adaptive_linear_momentum_merge_fast,
    compute_structure_tensor_field_fast,
    create_elliptical_kernel_fast,
)

# Data loading
from .data_loading import (
    load_or_extract_sonar_data,
    load_raw_polar_npz,
    extract_sonar_from_bag,
    get_sonar_frame_polar,
    get_sonar_frame_cone,
    save_raw_polar_to_npz,
)

# Tracker
from .tracker import NetTracker

# Video generation
from .video import generate_tracking_video, generate_raw_video
from .video_enhanced import create_enhanced_contour_detection_video

# Analysis
from .analysis import analyze_npz_sequence

__version__ = "2.0.0"

__all__ = [
    # Configuration
    "IMAGE_PROCESSING_CONFIG",
    "TRACKING_CONFIG",
    
    # Coordinate transformations
    "rasterize_cone",
    "polar_to_db_normalized",
    "to_uint8_gray",
    
    # Image processing
    "preprocess_edges",
    "adaptive_linear_momentum_merge_fast",
    "compute_structure_tensor_field_fast",
    "create_elliptical_kernel_fast",
    
    # Data loading
    "load_or_extract_sonar_data",
    "load_raw_polar_npz",
    "extract_sonar_from_bag",
    "get_sonar_frame_polar",
    "get_sonar_frame_cone",
    "save_raw_polar_to_npz",
    
    # Tracker
    "NetTracker",
    
    # Video generation
    "generate_tracking_video",
    "generate_raw_video",
    "create_enhanced_contour_detection_video",
    
    # Analysis
    "analyze_npz_sequence",
]
