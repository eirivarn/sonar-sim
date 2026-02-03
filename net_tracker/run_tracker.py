#!/usr/bin/env python3
# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Standalone Net Tracker - Complete Pipeline
- All dependencies included (no external utils imports)
- ROS bag loading with NPZ caching
- Polar to cone-view conversion
- Full tracking pipeline
- Video generation with 4-panel debug view

USAGE:
------
From command line (ROS bag or NPZ):
    python run_tracker.py input.bag output_video.mp4
    python run_tracker.py cached_polar.npz output_video.mp4
    python run_tracker.py --start 100 --end 500 --stride 2 input.bag output.mp4

From Python:
    from run_tracker import NetTracker, preprocess_edges, load_or_extract_sonar_data
    
    # Load data
    df, metadata = load_or_extract_sonar_data(Path("data.bag"))
    
    # Process frames...
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Union

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# ============================================================================
# IMAGE ENHANCEMENT FUNCTIONS
# ============================================================================

# Global kernel cache
_KERNEL_CACHE = {}
_CACHE_MAX_SIZE = 200

def _get_cache_key(kernel_type, *params):
    """Generate cache key for kernel caching."""
    return (kernel_type,) + tuple(float(p) for p in params)

def _cache_kernel(cache_key, kernel):
    """Add kernel to cache with size management."""
    global _KERNEL_CACHE
    if len(_KERNEL_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_KERNEL_CACHE))
        del _KERNEL_CACHE[oldest_key]
    _KERNEL_CACHE[cache_key] = kernel

def create_elliptical_kernel_fast(base_radius, elongation_factor, angle_degrees):
    """Fast elliptical kernel creation."""
    elongated_radius = int(base_radius * elongation_factor)
    size = 2 * elongated_radius + 1
    center = size // 2
    
    y, x = np.ogrid[:size, :size]
    y_c, x_c = y - center, x - center
    
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_rot = x_c * cos_a + y_c * sin_a
    y_rot = -x_c * sin_a + y_c * cos_a
    
    ellipse_dist = (x_rot / elongated_radius) ** 2 + (y_rot / base_radius) ** 2
    kernel = (ellipse_dist <= 1.0).astype(np.float32)
    
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

def compute_structure_tensor_field_fast(grad_x: np.ndarray, grad_y: np.ndarray, 
                                       sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized structure tensor computation for entire image.
    
    Args:
        grad_x, grad_y: Gradient components
        sigma: Gaussian smoothing parameter
        
    Returns:
        orientation_map: Dominant orientation at each pixel (0-180)
        coherency_map: Coherency (linearity measure) at each pixel (0-1)
    """
    # Structure tensor components
    Jxx = grad_x * grad_x
    Jyy = grad_y * grad_y  
    Jxy = grad_x * grad_y
    
    # Apply Gaussian smoothing (vectorized)
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_gauss = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel_gauss @ kernel_gauss.T
    
    Jxx_smooth = cv2.filter2D(Jxx, -1, kernel_2d)
    Jyy_smooth = cv2.filter2D(Jyy, -1, kernel_2d)
    Jxy_smooth = cv2.filter2D(Jxy, -1, kernel_2d)
    
    # Vectorized eigenvalue analysis
    trace = Jxx_smooth + Jyy_smooth
    det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
    
    # Compute orientation (vectorized)
    orientation_map = np.zeros_like(Jxx_smooth)
    coherency_map = np.zeros_like(Jxx_smooth)
    
    # Mask for valid regions (non-zero gradients)
    valid_mask = np.abs(Jxy_smooth) > 1e-6
    
    # Vectorized orientation computation
    orientation_map[valid_mask] = 0.5 * np.arctan2(2 * Jxy_smooth[valid_mask], 
                                                   Jxx_smooth[valid_mask] - Jyy_smooth[valid_mask])
    orientation_map = (orientation_map * 180 / np.pi + 180) % 180
    
    # Handle near-zero cases
    horizontal_mask = (~valid_mask) & (Jxx_smooth > Jyy_smooth)
    vertical_mask = (~valid_mask) & (Jxx_smooth <= Jyy_smooth)
    orientation_map[horizontal_mask] = 0    # Horizontal
    orientation_map[vertical_mask] = 90     # Vertical
    
    # Vectorized coherency computation (safely)
    valid_coherency_mask = (trace > 1e-6) & (det >= 0)
    coherency_map[valid_coherency_mask] = ((trace[valid_coherency_mask] - 
                                          2 * np.sqrt(det[valid_coherency_mask])) / 
                                         trace[valid_coherency_mask])
    coherency_map = np.clip(coherency_map, 0, 1)
    
    return orientation_map, coherency_map

def adaptive_linear_momentum_merge_fast(
    frame: np.ndarray,
    angle_steps: int = 36,
    base_radius: int = 3,
    max_elongation: float = 3.0,
    momentum_boost: float = 0.8,
    linearity_threshold: float = 0.15,
    downscale_factor: int = 2,
    top_k_bins: int = 8,
    min_coverage_percent: float = 0.5,
    gaussian_sigma: float = 1.0
) -> np.ndarray:
    """
    ADVANCED OPTIMIZED version using structure tensors and sophisticated filtering.
    """
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = frame.astype(np.float32)
    h, w = result.shape
    
    # Early exit for low contrast images
    frame_std = np.std(result)
    if frame_std < 5.0:
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.3 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Downsampled dimensions for structure tensor analysis
    h_small = max(h // downscale_factor, 32)
    w_small = max(w // downscale_factor, 32)
    frame_small = cv2.resize(result, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # Structure tensor-based orientation detection
    grad_x = cv2.Sobel(frame_small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_small, cv2.CV_64F, 0, 1, ksize=3)
    
    orientations, linearity_map_small = compute_structure_tensor_field_fast(
        grad_x, grad_y, sigma=1.5
    )
    
    # Quantize orientations to integer angle bins
    orientations_normalized = orientations / 180.0
    direction_bin_map_small = np.round(orientations_normalized * (angle_steps - 1)).astype(np.int32)
    direction_bin_map_small = np.clip(direction_bin_map_small, 0, angle_steps - 1)
    
    # Normalize linearity map
    max_linearity = np.max(linearity_map_small)
    if max_linearity > 0:
        linearity_map_small = linearity_map_small / max_linearity
    else:
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Upsample maps to full resolution
    linearity_map = cv2.resize(linearity_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
    direction_bin_map = cv2.resize(direction_bin_map_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    # Create binary mask for linear regions
    linear_mask = linearity_map > linearity_threshold
    
    if np.sum(linear_mask) == 0:
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Aggressive bin filtering and Top-K selection
    unique_bins, bin_counts = np.unique(direction_bin_map[linear_mask], return_counts=True)
    
    # Calculate coverage percentages and total linearity per bin
    total_linear_pixels = np.sum(linear_mask)
    bin_coverage_percent = (bin_counts / total_linear_pixels) * 100.0
    
    # Calculate total linearity strength per bin
    bin_linearity_totals = []
    for bin_idx in unique_bins:
        bin_mask = (direction_bin_map == bin_idx) & linear_mask
        total_linearity = np.sum(linearity_map[bin_mask])
        bin_linearity_totals.append(total_linearity)
    bin_linearity_totals = np.array(bin_linearity_totals)
    
    # Filter bins by minimum coverage
    coverage_filter = bin_coverage_percent >= min_coverage_percent
    filtered_bins = unique_bins[coverage_filter]
    filtered_linearity = bin_linearity_totals[coverage_filter]
    
    if len(filtered_bins) == 0:
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Select top-K bins by linearity strength
    if len(filtered_bins) > top_k_bins:
        top_k_indices = np.argsort(filtered_linearity)[-top_k_bins:]
        significant_bins = filtered_bins[top_k_indices]
    else:
        significant_bins = filtered_bins
    
    # Base enhancement with separable Gaussian blur
    enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
    
    # ROI-based convolution processing
    for angle_bin in significant_bins:
        angle_degrees = float(angle_bin * 180.0 / angle_steps)
        
        bin_mask = (direction_bin_map == angle_bin) & linear_mask
        
        if np.sum(bin_mask) == 0:
            continue
            
        # Find bounding box of the mask
        rows, cols = np.where(bin_mask)
        if len(rows) == 0:
            continue
            
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Expand ROI by kernel radius
        kernel_margin = base_radius + 2
        roi_row_start = max(0, row_min - kernel_margin)
        roi_row_end = min(h, row_max + kernel_margin + 1)
        roi_col_start = max(0, col_min - kernel_margin)
        roi_col_end = min(w, col_max + kernel_margin + 1)
        
        # Extract ROI
        roi_frame = result[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_mask = bin_mask[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_linearity = linearity_map[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        
        if roi_frame.size == 0:
            continue
        
        # Calculate average elongation for this bin
        avg_elongation = 1 + (max_elongation - 1) * np.mean(roi_linearity[roi_mask])
        
        # Cached elliptical kernel
        cache_key = _get_cache_key('ellipse', base_radius, avg_elongation, angle_degrees)
        if cache_key not in _KERNEL_CACHE:
            _cache_kernel(cache_key, create_elliptical_kernel_fast(base_radius, avg_elongation, angle_degrees))
        ellipse_kernel = _KERNEL_CACHE[cache_key]
        
        # Apply elliptical convolution to ROI
        roi_enhanced = cv2.filter2D(roi_frame, -1, ellipse_kernel)
        
        # Masked blending
        blend_weights = roi_linearity * roi_mask.astype(np.float32)
        
        roi_current = enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_blended = roi_current * (1.0 - blend_weights) + roi_enhanced * blend_weights
        enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end] = roi_blended
    
    # Final enhancement combination
    final_result = result + momentum_boost * enhanced
    
    # Adaptive soft clipping
    clip_upper = 255.0 * (1.0 + momentum_boost * 0.2)
    final_result = np.clip(final_result, 0.0, clip_upper)
    final_result = 255.0 * np.tanh(final_result / 255.0)
    
    return np.clip(final_result, 0.0, 255.0).astype(np.uint8)

def preprocess_edges(frame_u8: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess sonar frame for edge detection with optional advanced enhancement.
    
    Args:
        frame_u8: Input sonar frame (uint8)
        config: IMAGE_PROCESSING_CONFIG dictionary
        
    Returns:
        Tuple of (raw_edges, processed_edges)
    """
    
    # STEP 1: Convert to binary frame
    binary_threshold = config.get('binary_threshold', 128)
    binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255
    
    # STEP 2: Apply enhancement (advanced or basic)
    use_advanced = config.get('use_advanced_momentum_merging', True)
    
    if use_advanced:
        enhanced_binary = adaptive_linear_momentum_merge_fast(
            binary_frame,
            angle_steps=config.get('adaptive_angle_steps', 36),
            base_radius=config.get('adaptive_base_radius', 3),
            max_elongation=config.get('adaptive_max_elongation', 3.0),
            momentum_boost=config.get('momentum_boost', 0.8),
            linearity_threshold=config.get('adaptive_linearity_threshold', 0.15),
        )
    else:
        # Basic path
        enhanced = binary_frame

        use_opening = config.get('basic_use_opening', True)
        if use_opening:
            kernel_size = config.get('basic_open_kernel_size', config.get('basic_dilation_kernel_size', 3))
            iterations = config.get('basic_open_iterations', 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            kernel_size = config.get('basic_dilation_kernel_size', 3)
            iterations = config.get('basic_dilation_iterations', 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            enhanced = cv2.dilate(enhanced, kernel, iterations=iterations)

        # Light Gaussian blur
        kernel_size = config.get('basic_gaussian_kernel_size', 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
        enhanced = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), gaussian_sigma)

        enhanced_binary = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # STEP 3: Extract edges
    kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    
    # Raw edges
    raw_edges = cv2.filter2D(frame_u8, cv2.CV_32F, kernel_edge)
    raw_edges = np.clip(raw_edges, 0, 255).astype(np.uint8)
    raw_edges = (raw_edges > 0).astype(np.uint8) * 255
    
    # Enhanced edges
    enhanced_edges = cv2.filter2D(enhanced_binary, cv2.CV_32F, kernel_edge)
    enhanced_edges = np.clip(enhanced_edges, 0, 255).astype(np.uint8)
    enhanced_edges = (enhanced_edges > 0).astype(np.uint8) * 255
    
    # Post-process edges
    mks = int(config.get('morph_close_kernel', 0))
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        enhanced_edges = cv2.morphologyEx(enhanced_edges, cv2.MORPH_CLOSE, kernel)
    
    dil = int(config.get('edge_dilation_iterations', 0))
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced_edges = cv2.dilate(enhanced_edges, kernel2, iterations=dil)
    
    return raw_edges, enhanced_edges

# ============================================================================
# NET TRACKER CLASS
# ============================================================================

class NetTracker:
    """
    Simple net tracker. One class, clear logic.
    
    Smoothing formula (same for all parameters):
        new = old * (1 - alpha) + measured * alpha
        
    Alpha interpretation:
        alpha = 0.0 → 100% old (infinite smoothing)
        alpha = 0.5 → 50/50 blend
        alpha = 1.0 → 100% new (no smoothing)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Current ellipse state 
        self.center: Optional[Tuple[float, float]] = None
        self.size: Optional[Tuple[float, float]] = None  # (width, height)
        self.angle: Optional[float] = None  # degrees
        
        # Tracking
        self.last_distance: Optional[float] = None
        self.frames_lost: int = 0
        
    def find_and_update(self, edges: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find net in edges, update tracking state.
        Returns best contour or None.
        """
        H, W = image_shape
        
        # Get search mask
        search_mask = self._get_search_mask((H, W))
        
        # Apply mask to edges
        if search_mask is not None:
            masked_edges = cv2.bitwise_and(edges, search_mask)
        else:
            masked_edges = edges
        
        # Find contours
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best
        best = self._find_best_contour(contours)
        
        # Update state
        if best is not None and len(best) >= 5:
            self._update_from_contour(best)
            self.frames_lost = 0
        else:
            self.frames_lost += 1
            if self.frames_lost > self.config.get('max_frames_without_detection', 30):
                self._reset()
        
        return best
    
    def calculate_distance(self, image_width: int, image_height: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculate distance and angle from contour."""
        if self.center is None or self.angle is None:
            return self.last_distance, None
        
        try:
            cx, cy = self.center
            w, h = self.size if self.size else (1, 1)
            
            # Return the raw angle directly - this is the major axis angle
            major_axis_angle = self.angle
            
            # CRITICAL FIX: The red line is perpendicular to major axis
            # So the red line angle is major_axis_angle + 90°
            red_line_angle = (major_axis_angle + 90.0) % 360.0
            
            # Calculate intersection with center line using RED LINE angle
            center_x = image_width / 2
            ang_r = np.radians(red_line_angle)
            cos_ang = np.cos(ang_r)
            
            if abs(cos_ang) > 1e-6:
                t = (center_x - cx) / cos_ang
                intersect_y = cy + t * np.sin(ang_r)
                distance = intersect_y
            else:
                distance = cy
            
            distance = np.clip(distance, 0, image_height - 1)
            
            # Smooth distance
            if self.last_distance is not None:
                max_change = self.config.get('max_distance_change_pixels', 20)
                change = abs(distance - self.last_distance)
                if change > max_change:
                    direction = 1 if distance > self.last_distance else -1
                    distance = self.last_distance + (direction * max_change)
            
            self.last_distance = distance
            
            return float(distance), float(red_line_angle)
        except:
            return self.last_distance, None
    
    def _get_search_mask(self, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get search mask (ellipse + corridor)."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = image_shape
        
        # Get expansion
        expansion = self.config.get('ellipse_expansion_factor', 0.5)
        
        # Expand if losing track
        if self.frames_lost > 0:
            decay = self.config.get('aoi_decay_factor', 0.98)
            growth = 1.0 + (1.0 - decay) * self.frames_lost
            expansion *= growth
        
        # Create ellipse mask
        w, h = self.size
        expanded_size = (w * (1 + expansion), h * (1 + expansion))
        
        mask = np.zeros((H, W), dtype=np.uint8)
        
        try:
            ellipse = (
                (int(self.center[0]), int(self.center[1])),
                (int(expanded_size[0]), int(expanded_size[1])),
                self.angle
            )
            cv2.ellipse(mask, ellipse, 255, -1)
        except:
            return None
        
        # Add corridor
        if self.config.get('use_corridor_splitting', True):
            try:
                corridor = self._make_corridor_mask((H, W))
                if corridor is not None:
                    mask = cv2.bitwise_or(mask, corridor)
            except:
                pass
        
        return mask
    
    def _make_corridor_mask(self, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Make corridor mask along major axis."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = image_shape
        cx, cy = self.center
        w, h = self.size
        
        # Corridor parameters
        band_k = self.config.get('corridor_band_k', 2.0)
        length_factor = self.config.get('corridor_length_factor', 2.0)
        
        # Dimensions
        half_width = band_k * min(w, h) / 2.0
        half_length = length_factor * max(w, h) / 2.0
        
        # Rectangle in local coordinates
        local_pts = np.array([
            [-half_length, -half_width],
            [+half_length, -half_width],
            [+half_length, +half_width],
            [-half_length, +half_width],
        ], dtype=np.float32)
        
        # Rotate by major axis angle
        major_angle = self.angle
        if h > w:
            major_angle = (self.angle + 90.0) % 360.0
        
        ang_r = np.radians(major_angle)
        cos_a, sin_a = np.cos(ang_r), np.sin(ang_r)
        
        # Rotation matrix
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = local_pts @ R.T
        
        # Translate to center
        world_pts = rotated + np.array([cx, cy])
        
        # Draw
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [world_pts.astype(np.int32)], 255)
        
        return mask
    
    def _find_best_contour(self, contours) -> Optional[np.ndarray]:
        """Find best contour by area."""
        min_area = self.config.get('min_contour_area', 200)
        if self.center is not None:
            min_area *= 0.3
        
        best = None
        best_score = 0.0
        
        for c in contours:
            if c is None or len(c) < 5:
                continue
            
            try:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                
                score = area
                
                # Proximity bonus
                if self.center is not None:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        dist = np.sqrt((cx - self.center[0])**2 + (cy - self.center[1])**2)
                        proximity = max(0.1, 1.0 - dist / 100.0)
                        score *= proximity
                
                # Linearity and aspect ratio scores
                linearity_score = self._calculate_contour_linearity(c)
                aspect_ratio_score = self._calculate_aspect_ratio_score(c)
                
                # Combine scores
                score += linearity_score * self.config.get('linearity_score_weight', 1.0)
                score += aspect_ratio_score * self.config.get('aspect_ratio_score_weight', 1.0)
                
                if score > best_score:
                    best = c
                    best_score = score
            except:
                continue
        
        return best
    
    def _update_from_contour(self, contour: np.ndarray):
        """Update ellipse state from detected contour with smoothing."""
        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        except:
            return
        
        # Get alphas from config
        alpha_center = self.config.get('center_smoothing_alpha', 0.4)
        alpha_size = self.config.get('ellipse_size_smoothing_alpha', 0.01)
        alpha_angle = self.config.get('ellipse_orientation_smoothing_alpha', 0.2)
        
        # Smooth center
        if self.center is None:
            self.center = (cx, cy)
        else:
            old_cx, old_cy = self.center
            new_cx = old_cx * (1 - alpha_center) + cx * alpha_center
            new_cy = old_cy * (1 - alpha_center) + cy * alpha_center
            
            # Limit movement
            max_move = self.config.get('ellipse_max_movement_pixels', 30.0)
            dx = new_cx - old_cx
            dy = new_cy - old_cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > max_move:
                scale = max_move / dist
                new_cx = old_cx + dx * scale
                new_cy = old_cy + dy * scale
            
            self.center = (new_cx, new_cy)
        
        # Smooth size
        if self.size is None:
            self.size = (w, h)
        else:
            old_w, old_h = self.size
            new_w = old_w * (1 - alpha_size) + w * alpha_size
            new_h = old_h * (1 - alpha_size) + h * alpha_size
            self.size = (new_w, new_h)
        
        # Smooth angle
        if self.angle is None:
            self.angle = angle
        else:
            angle_diff = angle - self.angle
            if angle_diff > 90:
                angle_diff -= 180
            elif angle_diff < -90:
                angle_diff += 180
            self.angle = self.angle + angle_diff * alpha_angle
    
    def _reset(self):
        """Reset to initial state."""
        self.center = None
        self.size = None
        self.angle = None
        self.last_distance = None
        self.frames_lost = 0
    
    def get_status(self) -> str:
        """Get tracking status."""
        if self.center is None:
            return "LOST"
        elif self.frames_lost == 0:
            return "TRACKED"
        else:
            max_frames = self.config.get('max_frames_without_detection', 30)
            return f"SEARCHING ({self.frames_lost}/{max_frames})"
    
    def _calculate_contour_linearity(self, contour):
        """Calculate how linear/straight a contour is."""
        if len(contour) < 2:
            return 0.0
        
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        mean = np.mean(contour_points, axis=0)
        centered = contour_points - mean
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        
        if eigenvalues[1] < 1e-6:
            return 1.0
        
        linearity_ratio = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1])
        return float(linearity_ratio)
    
    def _calculate_aspect_ratio_score(self, contour):
        """Calculate aspect ratio preference score."""
        if len(contour) < 5:
            return 0.5
        
        try:
            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), angle = rect
            
            if height > width:
                width, height = height, width
            
            if height < 1e-6:
                return 0.0
            
            aspect_ratio = width / height
            
            ideal_ratio = 3.0
            
            if aspect_ratio < 1.0:
                return 0.3
            elif aspect_ratio <= ideal_ratio:
                return 0.5 + 0.5 * (aspect_ratio - 1.0) / (ideal_ratio - 1.0)
            elif aspect_ratio <= ideal_ratio * 2:
                return 1.0 - 0.3 * (aspect_ratio - ideal_ratio) / ideal_ratio
            else:
                return 0.4
                
        except:
            return 0.5


# ============================================================================
# DATA LOADING - BAG FILES AND NPZ CACHING
# ============================================================================

def load_or_extract_sonar_data(
    bag_path: Path, 
    cache_dir: Path | None = None,
    use_cache: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Load sonar data from cached NPZ if available, otherwise extract from bag.
    
    Args:
        bag_path: Path to ROS bag file
        cache_dir: Directory for NPZ cache files (default: bag_path.parent / "cache")
        use_cache: If True, try to load from cache and save to cache
        
    Returns:
        Tuple of (DataFrame with raw polar data, metadata dict)
    """
    if cache_dir is None:
        cache_dir = bag_path.parent / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate cache filename
    cache_file = cache_dir / f"{bag_path.stem}_raw_polar.npz"
    
    # Try loading from cache
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        return load_raw_polar_npz(cache_file)
    
    # Extract from bag
    df, metadata = extract_sonar_from_bag(bag_path)
    
    # Save to cache if requested
    if use_cache:
        print(f"Saving to cache: {cache_file.name}")
        save_raw_polar_to_npz(df, cache_file, metadata)
    
    return df, metadata


def save_raw_polar_to_npz(df: pd.DataFrame, npz_path: Path, metadata: dict):
    """Save raw polar sonar DataFrame to NPZ file for fast loading."""
    # Extract all frames
    frames = []
    timestamps = []
    dim_info = None
    
    for idx in range(len(df)):
        frame = get_sonar_frame_polar(df, idx)
        frames.append(frame)
        timestamps.append(df.iloc[idx]["t"])
        if dim_info is None:
            dim_info = {
                "dim_labels": df.iloc[idx].get("dim_labels", ["range_bins", "beams"]),
                "dim_sizes": list(frame.shape)
            }
    
    # Stack into 3D array
    raw_polar = np.stack(frames, axis=0).astype(np.float32)
    ts_array = np.array(timestamps, dtype=np.float64)
    
    # Save compressed
    np.savez_compressed(
        npz_path,
        raw_polar=raw_polar,
        timestamps=ts_array,
        dim_labels=dim_info["dim_labels"],
        dim_sizes=dim_info["dim_sizes"],
        metadata=json.dumps(metadata)
    )
    print(f"  ✓ Saved {len(frames)} frames to NPZ cache")


def load_raw_polar_npz(npz_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load raw polar sonar data from NPZ cache file."""
    with np.load(npz_path, allow_pickle=True) as data:
        raw_polar = data["raw_polar"]
        timestamps = data["timestamps"]
        dim_labels = data["dim_labels"].tolist() if "dim_labels" in data else ["range_bins", "beams"]
        dim_sizes = data["dim_sizes"].tolist() if "dim_sizes" in data else list(raw_polar[0].shape)
        metadata = json.loads(str(data["metadata"]))
    
    # Reconstruct DataFrame
    rows = []
    for idx in range(len(raw_polar)):
        rows.append({
            "t": timestamps[idx],
            "topic": metadata.get("topic", ""),
            "data": raw_polar[idx].ravel().tolist(),
            "dim_labels": dim_labels,
            "dim_sizes": dim_sizes,
        })
    
    df = pd.DataFrame(rows)
    print(f"  ✓ Loaded {len(df)} frames from NPZ cache")
    return df, metadata


def extract_sonar_from_bag(bag_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Extract Sonoptix sonar data from ROS bag file.
    
    Returns:
        (DataFrame with sonar data, metadata dict)
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError(
            "rosbags is required.\nInstall with: pip install rosbags"
        ) from exc

    print(f"Loading sonar data from: {bag_path.name}")
    rows = []
    
    with AnyReader([bag_path]) as reader:
        # Find Sonoptix topics
        sonar_conns = [c for c in reader.connections if "sonoptix" in c.topic.lower()]
        
        if not sonar_conns:
            raise RuntimeError(f"No Sonoptix topics found in bag")
        
        for conn in sonar_conns:
            for idx, (connection, timestamp, rawdata) in enumerate(reader.messages([conn]), start=1):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # Extract timestamp
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                ts = pd.Timestamp(sec, unit="s", tz="UTC") + pd.to_timedelta(nsec, unit="ns")
                
                # Extract dimensions
                dims = msg.array_data.layout.dim
                dim_labels = [d.label or f"dim_{i}" for i, d in enumerate(dims)] or ["range", "beam"]
                dim_sizes = [int(d.size) for d in dims]
                
                # Extract raw data
                data_vals = [float(v) for v in msg.array_data.data]
                
                rows.append({
                    "topic": conn.topic,
                    "ts_utc": ts.isoformat(),
                    "t": sec + nsec / 1e9,
                    "dim_labels": dim_labels,
                    "dim_sizes": dim_sizes,
                    "data": data_vals,
                })
                
                if idx % 100 == 0:
                    print(f"  Extracted {idx} frames...", end="\r")
    
    print(f"\n✓ Extracted {len(rows)} sonar frames")
    
    df = pd.DataFrame(rows)
    
    # Create metadata
    metadata = {
        "bag_name": bag_path.name,
        "num_frames": len(df),
        "topic": df.iloc[0]["topic"] if len(df) > 0 else None,
    }
    
    return df, metadata


def get_sonar_frame_polar(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Extract a single sonar frame as numpy array in polar format (range_bins, beams)."""
    row = df.iloc[idx]
    data = row["data"]
    dim_sizes = row["dim_sizes"]
    dim_labels = row.get("dim_labels", [])
    
    # Infer H, W from dimensions
    # Sonoptix format: typically range_bins (rows) × beams (cols)
    if len(dim_sizes) >= 2:
        # Check labels to determine correct orientation
        labels_str = str(dim_labels).lower()
        if "beam" in labels_str and "range" in labels_str:
            # Find which dimension is which
            beam_idx = next((i for i, l in enumerate(dim_labels) if "beam" in str(l).lower()), 0)
            range_idx = next((i for i, l in enumerate(dim_labels) if "range" in str(l).lower() or "bin" in str(l).lower()), 1)
            
            # Range bins should be rows (H), beams should be columns (W)
            H = dim_sizes[range_idx]
            W = dim_sizes[beam_idx]
        else:
            # Default: assume first dim is range, second is beams
            # But check if dimensions seem swapped (beams usually < range_bins)
            if dim_sizes[0] < dim_sizes[1]:
                # Likely beams × range_bins, need to transpose
                W, H = dim_sizes[0], dim_sizes[1]
                frame = np.array(data, dtype=np.float32).reshape(W, H).T
                return frame
            else:
                H, W = dim_sizes[0], dim_sizes[1]
    else:
        H, W = dim_sizes[0], dim_sizes[1] if len(dim_sizes) > 1 else 1
    
    # Reshape to (H, W) = (range_bins, beams)
    frame = np.array(data, dtype=np.float32).reshape(H, W)
    return frame


# ============================================================================
# POLAR TO CONE-VIEW CONVERSION
# ============================================================================

def rasterize_cone(
    polar_frame_normalized: np.ndarray,
    fov_deg: float = 120.0,
    rmin: float = 0.5,
    rmax: float = 20.0,
    img_h: int = 700,
    img_w: int = 900,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Convert NORMALIZED polar sonar image to Cartesian cone-view.
    
    IMPORTANT: Input should be pre-normalized to [0,1] range (after dB scaling).
    
    Args:
        polar_frame_normalized: Normalized polar image [0,1] (range_bins, beams)
        fov_deg: Field of view in degrees
        rmin: Minimum range in meters
        rmax: Maximum range in meters
        img_h: Output image height
        img_w: Output image width
        
    Returns:
        Tuple of (cone_image, extent=(xmin, xmax, ymin, ymax))
    """
    H, W = polar_frame_normalized.shape  # H=range_bins, W=beams
    
    half_fov = np.deg2rad(0.5 * fov_deg)
    y_min = max(0.0, rmin)
    y_max = rmax
    x_max = np.sin(half_fov) * y_max
    x_min = -x_max
    
    # Create Cartesian grid
    x = np.linspace(x_min, x_max, img_w)
    y = np.linspace(y_min, y_max, img_h)
    Xg, Yg = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    theta = np.arctan2(Xg, Yg)  # angle from +Y axis
    r = np.hypot(Xg, Yg)
    
    # Mask valid region
    mask = (r >= rmin) & (r <= y_max) & (theta >= -half_fov) & (theta <= half_fov)
    
    # Map (r, theta) to (row, col) in polar image
    rowf = (r - rmin) / max((rmax - rmin), 1e-12) * (H - 1)
    colf = (theta + half_fov) / max((2 * half_fov), 1e-12) * (W - 1)
    rows = np.rint(np.clip(rowf, 0, H - 1)).astype(np.int32)
    cols = np.rint(np.clip(colf, 0, W - 1)).astype(np.int32)
    
    # Sample from polar image
    cone = np.full((img_h, img_w), np.nan, dtype=np.float32)
    mflat = mask.ravel()
    cone.ravel()[mflat] = polar_frame_normalized[rows.ravel()[mflat], cols.ravel()[mflat]]
    
    extent = (x_min, x_max, y_min, y_max)
    return cone, extent


def polar_to_db_normalized(raw_polar: np.ndarray, db_norm: float = 60.0) -> np.ndarray:
    """
    Convert raw polar sonar to dB scale and normalize to [0,1].
    
    Args:
        raw_polar: Raw polar image (any scale)
        db_norm: dB normalization constant
        
    Returns:
        Normalized image in [0,1] range
    """
    image_db = 10 * np.log10(np.maximum(raw_polar, 1e-10))
    display_frame = np.clip((image_db + db_norm) / db_norm, 0, 1)
    return display_frame


def to_uint8_gray(frame01: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] frame to uint8, handling NaN values."""
    safe = np.nan_to_num(frame01, nan=0.0, posinf=1.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    return (safe * 255.0).astype(np.uint8)


def get_sonar_frame_cone(
    df: pd.DataFrame, 
    idx: int,
    fov_deg: float = 120.0,
    rmin: float = 0.5,
    rmax: float = 20.0,
    img_h: int = 700,
    img_w: int = 900,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Extract sonar frame and convert to cone-view (Cartesian coordinates).
    Full pipeline: polar -> dB normalized -> cone-view.
    """
    polar_frame = get_sonar_frame_polar(df, idx)
    polar_normalized = polar_to_db_normalized(polar_frame)
    cone_frame, extent = rasterize_cone(polar_normalized, fov_deg, rmin, rmax, img_h, img_w)
    return cone_frame, extent


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def create_debug_frame(
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
    
    # Initialize video writer
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            debug_frame = create_debug_frame(sonar_img_u8, binary, binary, edges,
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run NetTracker on ROS bag/NPZ and save debug video"
    )
    parser.add_argument("input_file", type=Path, help="Path to ROS bag or NPZ file")
    parser.add_argument("output_video", type=Path, help="Output video path (e.g., output.mp4)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (None for all)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (1 for every frame)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Load data
    print("\n" + "=" * 60)
    print("NET TRACKER PIPELINE")
    print("=" * 60)
    
    if args.input_file.suffix == '.bag':
        df, metadata = load_or_extract_sonar_data(args.input_file, use_cache=True)
    elif args.input_file.suffix == '.npz':
        df, metadata = load_raw_polar_npz(args.input_file)
    else:
        print(f"Error: Unsupported file type: {args.input_file.suffix}")
        print("Supported types: .bag, .npz")
        return 1
    
    # Generate video
    generate_tracking_video(
        df, 
        args.output_video,
        start_frame=args.start,
        end_frame=args.end,
        stride=args.stride
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return 0

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
        npz_path: Path to simulated data NPZ file (with 'cones' key)
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
    
    # Load data - handle both polar and cone-view formats
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            cones_raw = data['cones']  # May be polar or cone-view
            timestamps = data.get('ts_unix_ns', np.arange(len(cones_raw)))
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
        # (not used in tracking for the rest of the pipeline)
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
                    # Use morphological dilation to grow non-zero pixels into nearby zero pixels
                    kernel_size = config.get('basic_dilation_kernel_size', 3)
                    iterations = config.get('basic_dilation_iterations', 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    momentum = cv2.dilate(binary, kernel, iterations=iterations)
                else:
                    # Fallback to Gaussian blur
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
            
            # Draw ellipse outline (if tracker has established tracking)
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
            
            # Draw distance point at the intersection of red line with center line
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (0, 0, 255), -1)
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (255, 255, 255), 2)
            
            # Draw the red line itself (perpendicular to major axis)
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
        
        # Ensure all arrays have the same number of dimensions for np.hstack
        if binary.ndim == 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if momentum_display.ndim == 2:
            momentum_display = cv2.cvtColor(momentum_display, cv2.COLOR_GRAY2BGR)
        if edges_display.ndim == 2:
            edges_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

        # Add text labels to each panel (single text only)
        cv2.putText(binary, "1. Raw - Binary Frame", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(momentum_display, "2. Momentum Merged", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(edges_display, "3. Edges", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(search_mask_display, "4. Search Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(best_display, "5. Best Contour", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(distance_display, "6. Distance", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Stack the images horizontally
        row0 = np.hstack([binary, momentum_display, edges_display])
        row1 = np.hstack([search_mask_display, best_display, distance_display])
        grid_frame = np.vstack([row0, row1])
        
        # Ensure uint8 before writing
        if grid_frame.dtype != np.uint8:
            grid_frame = np.clip(grid_frame, 0, 255).astype(np.uint8)
        
        # Frame info (single text only)
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
if __name__ == "__main__":
    import sys
    sys.exit(main())
