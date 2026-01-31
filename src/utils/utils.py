"""Utility functions for sonar simulation."""
import numpy as np


def polar_to_cartesian(polar_image, range_m=20.0, fov_deg=120.0, output_size=(512, 512)):
    """
    Convert polar sonar image to Cartesian representation.
    
    Args:
        polar_image: (batch, range_bins, num_beams) or (range_bins, num_beams) polar sonar image
        range_m: Maximum range in meters
        fov_deg: Field of view in degrees
        output_size: (height, width) of output Cartesian image
        
    Returns:
        Multi-channel image (batch, 4, height, width) or (4, height, width):
            [0] intensity: sonar intensity
            [1] valid_mask: binary mask (1 where valid, 0 outside cone)
            [2] y_map: y-coordinate in meters for each pixel
            [3] z_map: z-coordinate in meters for each pixel
    """
    # Handle batch dimension
    if len(polar_image.shape) == 3:
        batch_size = polar_image.shape[0]
        results = []
        for i in range(batch_size):
            result = polar_to_cartesian(polar_image[i], range_m, fov_deg, output_size)
            results.append(result)
        return np.stack(results, axis=0)
    
    num_range_bins, num_beams = polar_image.shape
    height, width = output_size
    
    # Calculate extent
    x_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    y_extent = range_m
    
    # Create coordinate grids
    j_grid, i_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixels to meters in sonar frame
    z_m = (j_grid - width/2) * (2 * x_extent / width)
    y_m = i_grid * (y_extent / height)
    
    # Convert to polar coordinates
    r_m = np.sqrt(z_m**2 + y_m**2)
    theta_rad = np.arctan2(z_m, y_m)
    
    # Create mask for points within sonar cone
    valid_mask = (r_m <= range_m) & (np.abs(theta_rad) <= np.deg2rad(fov_deg/2))
    
    # Find corresponding indices in polar image
    r_idx = ((r_m / range_m) * (num_range_bins - 1)).astype(np.int32)
    theta_idx = ((theta_rad + np.deg2rad(fov_deg/2)) / np.deg2rad(fov_deg) * (num_beams - 1)).astype(np.int32)
    
    # Clamp indices
    r_idx = np.clip(r_idx, 0, num_range_bins - 1)
    theta_idx = np.clip(theta_idx, 0, num_beams - 1)
    
    # Create multi-channel output
    intensity = np.zeros((height, width), dtype=np.float32)
    intensity[valid_mask] = polar_image[r_idx[valid_mask], theta_idx[valid_mask]]
    
    # Stack all channels: [intensity, valid_mask, y_map, z_map]
    multi_channel = np.stack([
        intensity,
        valid_mask.astype(np.float32),
        y_m,
        z_m
    ], axis=0)
    
    return multi_channel
