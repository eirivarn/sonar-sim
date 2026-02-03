"""Numba-accelerated ray marching for sonar simulation.

This module provides JIT-compiled versions of the ray marching algorithm
for significantly faster performance (10-100x speedup).
"""

import numpy as np
from numba import jit, prange
import math


@jit(nopython=True, cache=True, fastmath=True)
def march_ray_numba(grid_density, grid_reflectivity, grid_absorption, grid_material_id,
                    grid_size_x, grid_size_y, voxel_size,
                    origin_x, origin_y, direction_x, direction_y,
                    range_m, range_bins, beam_strength,
                    step_size_factor, energy_threshold,
                    speckle_shape, aspect_std, aspect_range_min, aspect_range_max,
                    spreading_loss_min, water_absorption,
                    jitter_probability, jitter_std_base, jitter_range_factor, jitter_max_offset,
                    spread_probability, spread_bin_options, spread_bin_probs,
                    absorption_factor, scattering_loss_factor):
    """JIT-compiled ray marching through voxel grid.
    
    Returns:
        output_bins: Array of accumulated returns for each range bin
    """
    output_bins = np.zeros(range_bins, dtype=np.float32)
    
    step_size = voxel_size * step_size_factor
    num_steps = int(range_m / step_size)
    
    current_x = origin_x
    current_y = origin_y
    energy = 1.0
    
    for step in range(num_steps):
        distance = step * step_size
        if distance >= range_m or energy < energy_threshold:
            break
        
        # Get voxel indices
        x = int(current_x / voxel_size)
        y = int(current_y / voxel_size)
        
        # Check bounds
        if x < 0 or x >= grid_size_x or y < 0 or y >= grid_size_y:
            current_x += direction_x * step_size
            current_y += direction_y * step_size
            continue
        
        density = grid_density[x, y]
        reflectivity = grid_reflectivity[x, y]
        absorption = grid_absorption[x, y]
        
        # VOLUME SCATTERING
        if density > 0.01:
            # ACOUSTIC SPECKLE (Gamma distribution)
            speckle = np.random.gamma(speckle_shape, 1.0/speckle_shape)
            
            # ASPECT ANGLE VARIATION
            aspect_variation = 0.5 + aspect_std * np.random.randn()
            aspect_variation = max(aspect_range_min, min(aspect_range_max, aspect_variation))
            
            # GEOMETRIC SHADOWING
            scatter = energy * density * reflectivity * step_size * speckle * aspect_variation
            
            # Two-way propagation loss
            spreading_loss = 1.0 / (distance**2 + spreading_loss_min)
            water_abs = math.exp(-0.05 * distance * 2 * water_absorption)
            
            return_energy = scatter * spreading_loss * water_abs
            
            # SPATIAL JITTER
            bin_idx = int((distance / range_m) * (range_bins - 1))
            
            if np.random.rand() < jitter_probability:
                range_factor = 1.0 + (distance / range_m) * jitter_range_factor
                jitter_offset = int(round(np.random.randn() * jitter_std_base * range_factor))
                jitter_offset = max(-jitter_max_offset, min(jitter_max_offset, jitter_offset))
                bin_jitter = bin_idx + jitter_offset
                bin_jitter = max(0, min(range_bins - 1, bin_jitter))
            else:
                bin_jitter = bin_idx
            
            # MULTI-BIN SPREADING
            if np.random.rand() < spread_probability:
                # Choose number of bins to spread across
                rand_val = np.random.rand()
                cumsum = 0.0
                num_spread_bins = spread_bin_options[0]
                for i in range(len(spread_bin_probs)):
                    cumsum += spread_bin_probs[i]
                    if rand_val < cumsum:
                        num_spread_bins = spread_bin_options[i]
                        break
                
                spread_center = bin_jitter
                half_spread = num_spread_bins // 2
                
                for offset in range(-half_spread, half_spread + 1):
                    spread_bin = spread_center + offset
                    if 0 <= spread_bin < range_bins:
                        spread_weight = math.exp(-0.5 * (offset / (num_spread_bins/3))**2)
                        range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                        output_bins[spread_bin] += return_energy * beam_strength * range_quality * spread_weight / num_spread_bins
            else:
                # Single bin deposit
                if 0 <= bin_jitter < range_bins:
                    range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                    output_bins[bin_jitter] += return_energy * beam_strength * range_quality
        
        # ABSORPTION
        if density > 0.01:
            energy *= math.exp(-absorption * step_size * absorption_factor)
            energy *= (1.0 - density * reflectivity * step_size * scattering_loss_factor)
            energy = max(0.0, energy)
        
        # Move forward
        current_x += direction_x * step_size
        current_y += direction_y * step_size
    
    return output_bins


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def scan_parallel_numba(grid_density, grid_reflectivity, grid_absorption, grid_material_id,
                        grid_size_x, grid_size_y, voxel_size,
                        position_x, position_y, direction_x, direction_y,
                        range_m, range_bins, num_beams, fov_rad,
                        beam_pattern_falloff, step_size_factor, energy_threshold,
                        speckle_shape, aspect_std, aspect_range_min, aspect_range_max,
                        spreading_loss_min, water_absorption,
                        jitter_probability, jitter_std_base, jitter_range_factor, jitter_max_offset,
                        spread_probability, spread_bin_options, spread_bin_probs,
                        absorption_factor, scattering_loss_factor):
    """JIT-compiled parallel sonar scan with ray marching.
    
    Returns:
        image: (range_bins, num_beams) array of sonar returns
    """
    image = np.zeros((range_bins, num_beams), dtype=np.float32)
    
    # Direction angle
    dir_angle = math.atan2(direction_y, direction_x)
    
    # Process beams in parallel
    for beam_idx in prange(num_beams):
        # Beam direction
        t = beam_idx / (num_beams - 1) if num_beams > 1 else 0.5
        angle = (-fov_rad / 2) + t * fov_rad
        
        # BEAM PATTERN: Gaussian falloff toward edges
        beam_pattern = math.exp(-((angle / (fov_rad/2))**2) * beam_pattern_falloff)
        
        # Beam angle
        beam_angle = dir_angle + angle
        beam_dir_x = math.cos(beam_angle)
        beam_dir_y = math.sin(beam_angle)
        
        # March ray
        output_bins = march_ray_numba(
            grid_density, grid_reflectivity, grid_absorption, grid_material_id,
            grid_size_x, grid_size_y, voxel_size,
            position_x, position_y, beam_dir_x, beam_dir_y,
            range_m, range_bins, beam_pattern,
            step_size_factor, energy_threshold,
            speckle_shape, aspect_std, aspect_range_min, aspect_range_max,
            spreading_loss_min, water_absorption,
            jitter_probability, jitter_std_base, jitter_range_factor, jitter_max_offset,
            spread_probability, spread_bin_options, spread_bin_probs,
            absorption_factor, scattering_loss_factor
        )
        
        image[:, beam_idx] = output_bins
    
    return image


@jit(nopython=True, cache=True)
def compute_ground_truth_numba(grid_material_id, grid_size_x, grid_size_y, voxel_size,
                               position_x, position_y, direction_x, direction_y,
                               range_m, range_bins, num_beams, fov_rad):
    """JIT-compiled ground truth generation.
    
    Returns:
        ground_truth: (range_bins, num_beams) array of material IDs
    """
    ground_truth = np.zeros((range_bins, num_beams), dtype=np.uint8)
    
    dir_angle = math.atan2(direction_y, direction_x)
    
    for beam_idx in range(num_beams):
        # Beam direction
        t = beam_idx / (num_beams - 1) if num_beams > 1 else 0.5
        angle = (-fov_rad / 2) + t * fov_rad
        
        beam_angle = dir_angle + angle
        beam_dir_x = math.cos(beam_angle)
        beam_dir_y = math.sin(beam_angle)
        
        # Sample material at each range
        for range_idx in range(range_bins):
            distance = (range_idx / range_bins) * range_m
            world_x = position_x + beam_dir_x * distance
            world_y = position_y + beam_dir_y * distance
            
            vx = int(world_x / voxel_size)
            vy = int(world_y / voxel_size)
            
            if 0 <= vx < grid_size_x and 0 <= vy < grid_size_y:
                ground_truth[range_idx, beam_idx] = grid_material_id[vx, vy]
    
    return ground_truth
