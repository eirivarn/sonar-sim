#!/usr/bin/env python3
"""Configuration dictionaries for net tracker."""

from typing import Dict

IMAGE_PROCESSING_CONFIG: Dict = {
    # === BINARY CONVERSION ===
    'binary_threshold': 128,
    
    # === ADAPTIVE LINEAR MOMENTUM MERGING ===
    'use_advanced_momentum_merging': True,
    
    # Advanced momentum parameters
    'adaptive_angle_steps': 20,
    'adaptive_base_radius': 3,
    'adaptive_max_elongation': 1.0,
    'momentum_boost': 10.0,
    'adaptive_linearity_threshold': 0.75,
    'downscale_factor': 2,
    'top_k_bins': 8,
    'min_coverage_percent': 0.3,
    'gaussian_sigma': 5.0,
    
    # Basic Gaussian parameters (when use_advanced_momentum_merging=False)
    'basic_gaussian_kernel_size': 3,
    'basic_gaussian_sigma': 1.0,
    'basic_use_dilation': True,
    'basic_dilation_kernel_size': 3,
    'basic_dilation_iterations': 3,
    'basic_use_opening': True,
    'basic_open_kernel_size': 3,
    'basic_open_iterations': 1,
    
    # === MORPHOLOGICAL POST-PROCESSING ===
    'morph_close_kernel': 0,
    'edge_dilation_iterations': 0,
    
    # === CONTOUR FILTERING ===
    'min_contour_area': 200,
    'aoi_boost_factor': 10.0,
    
    # === DISTANCE TRACKING STABILITY ===
    'max_distance_change_pixels': 20,
}

TRACKING_CONFIG: Dict = {
    # === ELLIPTICAL AOI SETTINGS ===
    'use_elliptical_aoi': True,
    'ellipse_expansion_factor': 0.5,
    
    # === CONTOUR SCORING WEIGHTS ===
    'aoi_boost_factor': 10.0,
    'score_area_weight': 1.0,
    'score_linearity_weight': 2.0,
    'score_aspect_ratio_weight': 1.5,
    
    # === FALLBACK RECTANGULAR AOI ===
    'aoi_center_x_percent': 50,
    'aoi_center_y_percent': 60,
    'aoi_width_percent': 60,
    'aoi_height_percent': 70,
    
    # === SMOOTHING PARAMETERS ===
    'center_smoothing_alpha': 0.8,
    'ellipse_size_smoothing_alpha': 0.01,
    'ellipse_orientation_smoothing_alpha': 0.1,
    'ellipse_max_movement_pixels': 30.0,
    
    # === CORRIDOR EXTENSION ===
    'corridor_band_k': 1.0,
    'corridor_length_factor': 2.0,
    'use_corridor_splitting': True,
    
    # === PERSISTENCE ===
    'max_frames_without_detection': 30,
    'aoi_decay_factor': 0.98,
    
    # === SCORING WEIGHTS (for backward compatibility) ===
    'linearity_score_weight': 1.0,
    'aspect_ratio_score_weight': 1.0,
}
