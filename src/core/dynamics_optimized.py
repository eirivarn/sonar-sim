"""Optimized dynamic object updates using spatial partitioning.

This module provides faster versions of the dynamics functions using
spatial hashing to avoid O(N²) neighbor searches.
"""

import numpy as np
from src.core.voxel_grid import VoxelGrid
from src.core.materials import FISH, EMPTY


class SpatialHash:
    """Simple spatial hash for fast neighbor queries."""
    
    def __init__(self, cell_size=3.0):
        """Initialize spatial hash.
        
        Args:
            cell_size: Size of hash cells (should match interaction range)
        """
        self.cell_size = cell_size
        self.hash_table = {}
    
    def _hash(self, pos):
        """Convert position to cell coordinates."""
        return (int(pos[0] / self.cell_size), int(pos[1] / self.cell_size))
    
    def insert(self, idx, pos):
        """Insert object at position."""
        cell = self._hash(pos)
        if cell not in self.hash_table:
            self.hash_table[cell] = []
        self.hash_table[cell].append(idx)
    
    def query(self, pos):
        """Get all objects in neighboring cells (includes current cell)."""
        cx, cy = self._hash(pos)
        neighbors = []
        
        # Check 3x3 grid of cells around query position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (cx + dx, cy + dy)
                if cell in self.hash_table:
                    neighbors.extend(self.hash_table[cell])
        
        return neighbors


def update_fish_optimized(grid: VoxelGrid, fish_data: list, cage_center: np.ndarray, 
                         cage_radius: float, sonar_pos: np.ndarray, dt: float = 0.1):
    """Optimized fish update using spatial hashing.
    
    Reduces complexity from O(N²) to O(N) for fish interactions.
    
    Args:
        grid: VoxelGrid to update
        fish_data: List of fish dictionaries
        cage_center: Center of cage
        cage_radius: Radius of cage
        sonar_pos: Sonar position for avoidance behavior
        dt: Time step in seconds
    """
    # Clear existing fish
    grid.clear_fish()
    
    if len(fish_data) == 0:
        return
    
    # Species behavior parameters
    behavior = {
        'A': {'same_attract': 2.0, 'other_attract': 0.1, 'avoid': 0.8, 'sonar_avoid': 1.0},
        'B': {'same_attract': 0.0, 'other_attract': 0.0, 'avoid': 2.0, 'sonar_avoid': 2.5},
        'C': {'same_attract': 0.8, 'other_attract': 0.6, 'avoid': 1.0, 'sonar_avoid': 1.8}
    }
    
    # Build spatial hash for current positions
    spatial_hash = SpatialHash(cell_size=3.0)
    for i, fish in enumerate(fish_data):
        spatial_hash.insert(i, fish['pos'])
    
    # Update each fish using spatial hash
    for i, fish in enumerate(fish_data):
        # Update position
        fish['pos'] += fish['vel'] * dt
        
        # Flocking behavior using spatial hash
        species = fish['species']
        avoid_vec = np.zeros(2)
        same_attract_vec = np.zeros(2)
        other_attract_vec = np.zeros(2)
        same_count = 0
        other_count = 0
        
        # Query only nearby fish using spatial hash
        nearby_indices = spatial_hash.query(fish['pos'])
        
        for j in nearby_indices:
            if i == j:
                continue
            
            other = fish_data[j]
            diff = other['pos'] - fish['pos']
            dist_sq = diff[0]*diff[0] + diff[1]*diff[1]  # Avoid sqrt
            
            if dist_sq < 9.0 and dist_sq > 0.0001:  # 3.0m range
                dist = np.sqrt(dist_sq)
                
                # Avoidance (short range)
                if dist_sq < 0.64:  # 0.8m
                    avoid_vec -= diff / (dist_sq + 0.1)
                
                # Attraction (medium range)
                if other['species'] == species:
                    same_attract_vec += diff / (dist + 0.1)
                    same_count += 1
                else:
                    other_attract_vec += diff / (dist + 0.1)
                    other_count += 1
        
        # Apply behaviors based on species
        params = behavior[species]
        steer = np.zeros(2)
        
        if same_count > 0:
            steer += params['same_attract'] * same_attract_vec / same_count
        if other_count > 0:
            steer += params['other_attract'] * other_attract_vec / other_count
        steer += params['avoid'] * avoid_vec
        
        # Sonar avoidance
        diff_from_sonar = fish['pos'] - sonar_pos
        dist_sq_sonar = diff_from_sonar[0]**2 + diff_from_sonar[1]**2
        
        if dist_sq_sonar < 25.0 and dist_sq_sonar > 0.0001:  # 5m range
            dist_sonar = np.sqrt(dist_sq_sonar)
            flee_strength = params['sonar_avoid'] / (dist_sq_sonar + 0.5)
            steer += flee_strength * diff_from_sonar / dist_sonar
        
        # Perimeter preference
        dx = fish['pos'][0] - cage_center[0]
        dy = fish['pos'][1] - cage_center[1]
        dist_from_center_sq = dx*dx + dy*dy
        
        if dist_from_center_sq > 0.0001:
            dist_from_center = np.sqrt(dist_from_center_sq)
            target_radius = cage_radius * 0.85
            
            if dist_from_center < target_radius * 0.7:
                to_perimeter = np.array([dx, dy]) / dist_from_center
                steer += to_perimeter * 0.4
        
        # Random component
        steer += (np.random.rand(2) - 0.5) * 0.1 * (dt * 10)
        
        # Apply steering
        if np.linalg.norm(steer) > 0.01:
            fish['vel'] += steer * 0.02 * (dt * 10)
            
            # Limit speed
            speed_sq = fish['vel'][0]**2 + fish['vel'][1]**2
            max_speed = 0.15
            min_speed = 0.03
            
            if speed_sq > max_speed * max_speed:
                speed = np.sqrt(speed_sq)
                fish['vel'] = fish['vel'] / speed * max_speed
            elif speed_sq < min_speed * min_speed and speed_sq > 0.0001:
                speed = np.sqrt(speed_sq)
                fish['vel'] = fish['vel'] / speed * min_speed
        
        # Update orientation
        if np.linalg.norm(fish['vel']) > 0.01:
            fish['orientation'] = np.arctan2(fish['vel'][1], fish['vel'][0])
        
        # Boundary collision (cage perimeter)
        from src.config import SCENE_CONFIG
        current_strength = SCENE_CONFIG['current_strength']
        current_direction = np.array(SCENE_CONFIG['current_direction'])
        
        dx = fish['pos'][0] - cage_center[0]
        dy = fish['pos'][1] - cage_center[1]
        
        # Calculate deflection based on vertical position
        deflection_factor = max(0, dy / cage_radius) if cage_radius > 0 else 0
        deflection = current_strength * deflection_factor * current_direction
        lateral_deflection = np.sin(np.arctan2(dy, dx)) * 0.4 * current_strength * 0.3
        deflection += np.array([lateral_deflection, 0])
        
        # Actual boundary position
        actual_boundary = cage_radius
        distance_to_center = np.sqrt(dx*dx + dy*dy)
        
        if distance_to_center > actual_boundary - 0.3:
            normal = np.array([dx, dy]) / (distance_to_center + 0.0001)
            fish['vel'] -= 2 * np.dot(fish['vel'], normal) * normal
            fish['pos'] = cage_center + normal * (actual_boundary - 0.35)
    
    # Redraw all fish
    for fish in fish_data:
        grid.set_ellipse(
            fish['pos'],
            fish['radii'],
            fish['orientation'],
            FISH
        )


def update_debris_optimized(grid: VoxelGrid, debris_data: list, cage_center: np.ndarray, 
                           cage_radius: float, dt: float = 0.1):
    """Optimized debris update (already O(N), just cleaned up).
    
    Args:
        grid: VoxelGrid to update
        debris_data: List of debris dictionaries
        cage_center: Center of cage
        cage_radius: Radius of cage
        dt: Time step in seconds
    """
    grid.clear_debris()
    
    if len(debris_data) == 0:
        return
    
    max_speed = 0.05
    
    for debris in debris_data:
        # Random turbulence/drift
        debris['vel'] += (np.random.rand(2) - 0.5) * 0.008 * (dt * 10)
        
        # Update position
        debris['pos'] += debris['vel'] * dt
        
        # Limit speed
        speed = np.linalg.norm(debris['vel'])
        if speed > max_speed:
            debris['vel'] = debris['vel'] / speed * max_speed
        
        # Boundary collision (simple reflection)
        dx = debris['pos'][0] - cage_center[0]
        dy = debris['pos'][1] - cage_center[1]
        distance_to_center = np.sqrt(dx*dx + dy*dy)
        
        if distance_to_center > cage_radius - debris['size']:
            normal = np.array([dx, dy]) / (distance_to_center + 0.0001)
            debris['vel'] -= 2 * np.dot(debris['vel'], normal) * normal
            debris['pos'] = cage_center + normal * (cage_radius - debris['size'] - 0.1)
        
        # Redraw
        grid.set_circle(debris['pos'], debris['size'], debris['material'])
