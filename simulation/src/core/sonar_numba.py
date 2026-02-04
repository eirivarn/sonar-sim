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
                    range_m, range_bins, beam_strength, beam_angle_normalized,
                    step_size_factor, energy_threshold,
                    speckle_shape, aspect_std, aspect_range_min, aspect_range_max,
                    spreading_loss_min, water_absorption,
                    jitter_probability, jitter_std_base, jitter_range_factor, jitter_max_offset,
                    spread_probability, spread_bin_options, spread_bin_probs,
                    absorption_factor, scattering_loss_factor,
                    angle_scatter_strength, angle_scatter_power,
                    density_scatter_threshold, density_scatter_strength, density_noise_boost,
                    proximity_shadow_strength, proximity_shadow_max_distance):
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
    
    # Track closest significant obstacle for proximity shadowing
    closest_obstacle_distance = range_m  # Start at max range
    proximity_shadow_active = False
    
    # Calculate angle-dependent scatter multiplier (stronger at edges)
    # beam_angle_normalized is in range [-1, 1] where ±1 are the FOV edges
    angle_factor = 1.0 + angle_scatter_strength * (abs(beam_angle_normalized) ** angle_scatter_power)
    
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
            # Calculate density-dependent scatter boost
            density_factor = 1.0
            if density > density_scatter_threshold:
                # High density areas scatter more
                density_excess = (density - density_scatter_threshold) / (1.0 - density_scatter_threshold)
                density_factor = 1.0 + density_scatter_strength * density_excess
            
            # ACOUSTIC SPECKLE (Gamma distribution) - enhanced by angle and density
            speckle = np.random.gamma(speckle_shape, 1.0/speckle_shape)
            speckle *= angle_factor * density_factor
            
            # ASPECT ANGLE VARIATION - enhanced in high-density areas
            aspect_std_effective = aspect_std * (1.0 + 0.5 * (density - density_scatter_threshold) if density > density_scatter_threshold else 1.0)
            aspect_variation = 0.5 + aspect_std_effective * np.random.randn()
            aspect_variation = max(aspect_range_min, min(aspect_range_max, aspect_variation))
            
            # GEOMETRIC SHADOWING
            scatter = energy * density * reflectivity * step_size * speckle * aspect_variation
            
            # Two-way propagation loss
            spreading_loss = 1.0 / (distance**2 + spreading_loss_min)
            water_abs = math.exp(-0.05 * distance * 2 * water_absorption)
            
            return_energy = scatter * spreading_loss * water_abs
            
            # SPATIAL JITTER - enhanced at angles and in dense areas
            bin_idx = int((distance / range_m) * (range_bins - 1))
            
            # Increase jitter probability in dense areas
            jitter_prob_effective = jitter_probability
            if density > density_scatter_threshold:
                jitter_prob_effective = min(0.95, jitter_probability + density_noise_boost * (density - density_scatter_threshold))
            
            if np.random.rand() < jitter_prob_effective:
                range_factor = 1.0 + (distance / range_m) * jitter_range_factor
                # Apply angle-dependent jitter
                jitter_strength = jitter_std_base * range_factor * angle_factor * density_factor
                jitter_offset = int(round(np.random.randn() * jitter_strength))
                jitter_offset = max(-jitter_max_offset, min(jitter_max_offset, jitter_offset))
                bin_jitter = bin_idx + jitter_offset
                bin_jitter = max(0, min(range_bins - 1, bin_jitter))
            else:
                bin_jitter = bin_idx
            
            # MULTI-BIN SPREADING - more likely in dense areas and at angles
            spread_prob_effective = spread_probability * angle_factor * density_factor
            spread_prob_effective = min(0.9, spread_prob_effective)
            
            if np.random.rand() < spread_prob_effective:
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
                
                # Determine scatter mode: disappear, additive, or normal
                mode_rand = np.random.rand()
                if mode_rand < 0.15:  # 15% disappear
                    # Don't deposit - energy disappears
                    pass
                elif mode_rand < 0.40:  # 25% additive (creates bright spots)
                    for offset in range(-half_spread, half_spread + 1):
                        spread_bin = spread_center + offset
                        if 0 <= spread_bin < range_bins:
                            spread_weight = math.exp(-0.5 * (offset / (num_spread_bins/3))**2)
                            range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                            # Additive: boost energy and ADD to existing
                            boosted_energy = return_energy * 1.8 * beam_strength * range_quality * spread_weight / num_spread_bins
                            output_bins[spread_bin] += boosted_energy
                else:  # 60% normal
                    for offset in range(-half_spread, half_spread + 1):
                        spread_bin = spread_center + offset
                        if 0 <= spread_bin < range_bins:
                            spread_weight = math.exp(-0.5 * (offset / (num_spread_bins/3))**2)
                            range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                            output_bins[spread_bin] += return_energy * beam_strength * range_quality * spread_weight / num_spread_bins
            else:
                # Single bin deposit - also with modes
                mode_rand = np.random.rand()
                if mode_rand < 0.15:  # 15% disappear
                    # Don't deposit
                    pass
                elif mode_rand < 0.40:  # 25% additive boost
                    if 0 <= bin_jitter < range_bins:
                        range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                        boosted_energy = return_energy * 1.8 * beam_strength * range_quality
                        output_bins[bin_jitter] += boosted_energy
                else:  # 60% normal
                    if 0 <= bin_jitter < range_bins:
                        range_quality = 1.0 / (1.0 + (distance / range_m) * 0.8)
                        output_bins[bin_jitter] += return_energy * beam_strength * range_quality
        
        # ABSORPTION
        if density > 0.01:
            # Track closest obstacle for proximity shadowing
            if not proximity_shadow_active and density > 0.3:
                closest_obstacle_distance = distance
                proximity_shadow_active = True
            
            # Calculate proximity shadow multiplier (closer objects cast stronger shadows)
            # Shadow strength decreases linearly with distance up to max_distance (10m)
            proximity_multiplier = 1.0
            if proximity_shadow_active and distance > closest_obstacle_distance:
                # We're behind an obstacle - apply proximity shadow
                # Shadow strength: 1.0 (no effect) to shadow_strength (max effect)
                # Inversely proportional to obstacle distance (closer = stronger shadow)
                distance_factor = 1.0 - min(1.0, closest_obstacle_distance / proximity_shadow_max_distance)
                proximity_multiplier = 1.0 + proximity_shadow_strength * distance_factor
            
            # Apply absorption with proximity enhancement
            energy *= math.exp(-absorption * step_size * absorption_factor * proximity_multiplier)
            energy *= (1.0 - density * reflectivity * step_size * scattering_loss_factor * proximity_multiplier)
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
                        absorption_factor, scattering_loss_factor,
                        angle_scatter_strength, angle_scatter_power,
                        density_scatter_threshold, density_scatter_strength, density_noise_boost,
                        proximity_shadow_strength, proximity_shadow_max_distance):
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
        
        # Calculate normalized angle for scatter effects: -1 (left edge) to +1 (right edge)
        beam_angle_normalized = angle / (fov_rad / 2)
        
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
            range_m, range_bins, beam_pattern, beam_angle_normalized,
            step_size_factor, energy_threshold,
            speckle_shape, aspect_std, aspect_range_min, aspect_range_max,
            spreading_loss_min, water_absorption,
            jitter_probability, jitter_std_base, jitter_range_factor, jitter_max_offset,
            spread_probability, spread_bin_options, spread_bin_probs,
            absorption_factor, scattering_loss_factor,
            angle_scatter_strength, angle_scatter_power,
            density_scatter_threshold, density_scatter_strength, density_noise_boost,
            proximity_shadow_strength, proximity_shadow_max_distance
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
