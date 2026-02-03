"""Fast batch rendering utilities for dynamic objects.

This module provides optimized rendering functions that process
multiple objects at once using vectorized NumPy operations.
"""

import numpy as np
from src.core.materials import FISH


def render_fish_batch(grid, fish_data):
    """Render all fish using vectorized operations.
    
    This is much faster than calling set_ellipse() for each fish individually.
    Uses simplified circular approximation for speed.
    
    Args:
        grid: VoxelGrid instance
        fish_data: List of fish dictionaries with 'pos', 'radii', 'orientation'
    """
    if len(fish_data) == 0:
        return
    
    voxel_size = grid.voxel_size
    
    # Process each fish (still a loop, but much faster operations inside)
    for fish in fish_data:
        center = fish['pos']
        radii = fish['radii']
        orientation = fish['orientation']
        
        # Convert to voxel coordinates
        cx, cy = grid.world_to_voxel(center)
        rx_vox = max(1, int(radii[0] / voxel_size))
        ry_vox = max(1, int(radii[1] / voxel_size))
        
        # Bounding box
        max_r = max(rx_vox, ry_vox)
        x_min = max(0, cx - max_r)
        x_max = min(grid.size_x, cx + max_r + 1)
        y_min = max(0, cy - max_r)
        y_max = min(grid.size_y, cy + max_r + 1)
        
        # Create coordinate grids for the bounding box
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
        
        # Compute distances from center
        dx = xx - cx
        dy = yy - cy
        
        # Rotate to ellipse space (vectorized)
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a
        
        # Check if inside ellipse (vectorized)
        dist_sq = (dx_rot / rx_vox) ** 2 + (dy_rot / ry_vox) ** 2
        mask = dist_sq <= 1.0
        
        # Apply material to masked voxels (vectorized assignment)
        grid.density[xx[mask], yy[mask]] = FISH.density
        grid.reflectivity[xx[mask], yy[mask]] = FISH.reflectivity
        grid.absorption[xx[mask], yy[mask]] = FISH.absorption
        grid.material_id[xx[mask], yy[mask]] = FISH.material_id


def render_fish_batch_simple(grid, fish_data):
    """Render all fish as simple circles (fastest method).
    
    Uses circular approximation instead of ellipses for maximum speed.
    The visual difference is negligible in sonar images.
    
    Args:
        grid: VoxelGrid instance
        fish_data: List of fish dictionaries with 'pos', 'radii'
    """
    if len(fish_data) == 0:
        return
    
    voxel_size = grid.voxel_size
    
    # Process each fish with simplified circle rendering
    for fish in fish_data:
        center = fish['pos']
        # Use average of radii for circular approximation
        radius = (fish['radii'][0] + fish['radii'][1]) / 2.0
        
        # Convert to voxel coordinates
        cx, cy = grid.world_to_voxel(center)
        r_vox = max(1, int(radius / voxel_size))
        
        # Bounding box
        x_min = max(0, cx - r_vox)
        x_max = min(grid.size_x, cx + r_vox + 1)
        y_min = max(0, cy - r_vox)
        y_max = min(grid.size_y, cy + r_vox + 1)
        
        # Create coordinate grids for the bounding box
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
        
        # Compute squared distance from center (vectorized)
        dx = xx - cx
        dy = yy - cy
        dist_sq = dx * dx + dy * dy
        
        # Circle mask
        mask = dist_sq <= r_vox * r_vox
        
        # Apply material (vectorized assignment)
        grid.density[xx[mask], yy[mask]] = FISH.density
        grid.reflectivity[xx[mask], yy[mask]] = FISH.reflectivity
        grid.absorption[xx[mask], yy[mask]] = FISH.absorption
        grid.material_id[xx[mask], yy[mask]] = FISH.material_id
