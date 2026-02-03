#!/usr/bin/env python3
"""Coordinate transformations: polar to cone-view and utilities."""

import numpy as np


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
