"""Data collection paths for automated sonar data gathering.

This module provides various path generators for collecting sonar data without GUI.
Paths can be predefined (circular, grid, spiral) or random sampling.

USAGE:
------
From command line:
    python simulation.py --collect circular --save my_dataset --num-samples 100

From code:
    path_gen = CircularPath(center=[15, 15], radius=10, num_samples=50)
    for pos, direction in path_gen:
        # Use pos and direction for data collection
        pass

ADDING CUSTOM PATHS:
-------------------
Create a new class that inherits from PathGenerator:
    
    class MyCustomPath(PathGenerator):
        def __init__(self, ...):
            self.positions = [...]  # List of (pos, dir) tuples
            self.index = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.index >= len(self.positions):
                raise StopIteration
            pos, dir = self.positions[self.index]
            self.index += 1
            return np.array(pos), np.array(dir)
        
        def __len__(self):
            return len(self.positions)
"""

import numpy as np
from abc import ABC, abstractmethod


class PathGenerator(ABC):
    """Base class for data collection path generators."""
    
    @abstractmethod
    def __iter__(self):
        """Return iterator."""
        pass
    
    @abstractmethod
    def __next__(self):
        """Return next (position, direction) tuple."""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return total number of samples."""
        pass


class CircularPath(PathGenerator):
    """Circular path inside a cage with smooth continuous motion.
    
    Generates positions along a circle using incremental steps instead of
    pre-generating a fixed path. Each call to next() advances the angle
    and calculates position/orientation on-the-fly.
    
    Args:
        center: [x, y] center of circle
        radius: Mean radius of circular path
        num_samples: Number of positions to sample (used to calculate angle step)
        radius_variation: Amplitude of radius variation (meters)
        orientation_mode: 'inward' (default), 'tangent', 'outward', or 'mixed'
        orientation_noise_deg: Amplitude of orientation variation (degrees)
        seed: Random seed for reproducibility (not used in smooth mode)
    """
    
    def __init__(self, center, radius, num_samples=100, 
                 radius_variation=1.0, orientation_mode='inward',
                 orientation_noise_deg=15.0, seed=None, dt=0.033):
        from src.config import DATA_COLLECTION_CONFIG, VISUALIZATION_CONFIG
        
        self.center = np.array(center)
        self.radius = radius
        self.num_samples = num_samples
        self.radius_variation = radius_variation
        self.orientation_mode = orientation_mode
        self.orientation_noise_deg = orientation_noise_deg
        
        # Use dt from config if not provided
        self.dt = dt if dt != 0.033 else VISUALIZATION_CONFIG.get('dt', 0.033)
        
        # Calculate angular velocity based on desired duration for full circle
        duration = DATA_COLLECTION_CONFIG['circular_path_duration_seconds']
        self.angular_velocity = (2 * np.pi) / duration  # radians per second
        
        # Angle step per frame based on angular velocity and time step
        self.angle_step = self.angular_velocity * self.dt
        
        # State: current angle and sample index
        self.current_angle = 0.0
        self.current_index = 0
    
    def _calculate_position(self, angle, index):
        """Calculate position and direction for given angle and index."""
        from src.config import DATA_COLLECTION_CONFIG
        
        # Smooth radius variation using multiple sine waves from config
        radius_wave = 0.0
        for wave in DATA_COLLECTION_CONFIG['radius_sine_waves']:
            radius_wave += np.sin(angle * wave['frequency']) * wave['amplitude']
        
        r = self.radius + radius_wave * self.radius_variation
        r = np.clip(r, 2.0, self.radius * 1.5)  # Keep reasonable bounds
        
        # Position on circle
        x = self.center[0] + r * np.cos(angle)
        y = self.center[1] + r * np.sin(angle)
        pos = np.array([x, y])
        
        # Direction based on mode
        if self.orientation_mode == 'inward':
            base_dir = self.center - pos
        elif self.orientation_mode == 'tangent':
            base_dir = np.array([-np.sin(angle), np.cos(angle)])
        elif self.orientation_mode == 'outward':
            base_dir = pos - self.center
        elif self.orientation_mode == 'mixed':
            # Use angle to determine mode smoothly
            mode_val = np.sin(angle * 3)
            if mode_val < -0.33:
                base_dir = self.center - pos
            elif mode_val > 0.33:
                base_dir = pos - self.center
            else:
                base_dir = np.array([-np.sin(angle), np.cos(angle)])
        else:
            raise ValueError(f"Unknown orientation_mode: {self.orientation_mode}")
        
        # Normalize base direction
        base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-9)
        
        # Smooth orientation variation using multiple sine waves from config
        smooth_noise = 0.0
        for wave in DATA_COLLECTION_CONFIG['orientation_sine_waves']:
            phase = wave.get('phase_offset', 0.0) * index
            smooth_noise += (np.sin(angle * wave['frequency'] + phase) * 
                           wave['amplitude'])
        smooth_noise *= np.deg2rad(self.orientation_noise_deg)
        
        cos_n, sin_n = np.cos(smooth_noise), np.sin(smooth_noise)
        direction = np.array([
            base_dir[0] * cos_n - base_dir[1] * sin_n,
            base_dir[0] * sin_n + base_dir[1] * cos_n
        ])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        
        return pos, direction
    
    def __iter__(self):
        self.current_angle = 0.0
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        # Calculate position on-the-fly using current state
        pos, direction = self._calculate_position(self.current_angle, self.current_index)
        
        # Advance state for next iteration
        self.current_angle += self.angle_step
        self.current_index += 1
        
        return pos, direction
    
    def __len__(self):
        return self.num_samples


class GridPath(PathGenerator):
    """Regular grid sampling across world space.
    
    Args:
        min_pos: [x_min, y_min] lower left corner
        max_pos: [x_max, y_max] upper right corner
        grid_size: [nx, ny] number of grid points
        direction_mode: 'fixed' (all same), 'random', or 'sweep' (systematic rotation)
        fixed_direction: Used if direction_mode='fixed'
        seed: Random seed
    """
    
    def __init__(self, min_pos, max_pos, grid_size, 
                 direction_mode='fixed', fixed_direction=[0, 1], seed=None):
        self.min_pos = np.array(min_pos)
        self.max_pos = np.array(max_pos)
        self.grid_size = grid_size
        self.direction_mode = direction_mode
        self.fixed_direction = np.array(fixed_direction)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate grid positions
        x_coords = np.linspace(min_pos[0], max_pos[0], grid_size[0])
        y_coords = np.linspace(min_pos[1], max_pos[1], grid_size[1])
        
        self.positions = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                pos = np.array([x, y])
                
                if direction_mode == 'fixed':
                    direction = self.fixed_direction / (np.linalg.norm(self.fixed_direction) + 1e-9)
                elif direction_mode == 'random':
                    angle = np.random.rand() * 2 * np.pi
                    direction = np.array([np.cos(angle), np.sin(angle)])
                elif direction_mode == 'sweep':
                    # Rotate systematically
                    total_points = grid_size[0] * grid_size[1]
                    angle = 2 * np.pi * (i * grid_size[1] + j) / total_points
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    raise ValueError(f"Unknown direction_mode: {direction_mode}")
                
                self.positions.append((pos, direction))
        
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.positions):
            raise StopIteration
        pos, direction = self.positions[self.index]
        self.index += 1
        return pos, direction
    
    def __len__(self):
        return len(self.positions)


class RandomPath(PathGenerator):
    """Random sampling within a region.
    
    Args:
        min_pos: [x_min, y_min] bounds
        max_pos: [x_max, y_max] bounds
        num_samples: Number of random positions
        direction_mode: 'random' or 'toward_center'
        seed: Random seed
    """
    
    def __init__(self, min_pos, max_pos, num_samples=100,
                 direction_mode='random', seed=None):
        self.min_pos = np.array(min_pos)
        self.max_pos = np.array(max_pos)
        self.num_samples = num_samples
        self.direction_mode = direction_mode
        
        if seed is not None:
            np.random.seed(seed)
        
        center = (self.min_pos + self.max_pos) / 2
        
        self.positions = []
        for _ in range(num_samples):
            # Random position
            pos = self.min_pos + np.random.rand(2) * (self.max_pos - self.min_pos)
            
            if direction_mode == 'random':
                angle = np.random.rand() * 2 * np.pi
                direction = np.array([np.cos(angle), np.sin(angle)])
            elif direction_mode == 'toward_center':
                direction = center - pos
                direction = direction / (np.linalg.norm(direction) + 1e-9)
            else:
                raise ValueError(f"Unknown direction_mode: {direction_mode}")
            
            self.positions.append((pos, direction))
        
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.positions):
            raise StopIteration
        pos, direction = self.positions[self.index]
        self.index += 1
        return pos, direction
    
    def __len__(self):
        return len(self.positions)


class SpiralPath(PathGenerator):
    """Spiral path from center outward or inward.
    
    Args:
        center: [x, y] center of spiral
        start_radius: Starting radius
        end_radius: Ending radius
        num_turns: Number of complete rotations
        num_samples: Number of positions along spiral
        direction_mode: 'tangent', 'inward', or 'outward'
        seed: Random seed
    """
    
    def __init__(self, center, start_radius, end_radius, 
                 num_turns=3, num_samples=100, direction_mode='tangent', seed=None):
        self.center = np.array(center)
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.num_turns = num_turns
        self.num_samples = num_samples
        self.direction_mode = direction_mode
        
        if seed is not None:
            np.random.seed(seed)
        
        self.positions = []
        total_angle = num_turns * 2 * np.pi
        
        for i in range(num_samples):
            t = i / (num_samples - 1)  # 0 to 1
            angle = t * total_angle
            r = start_radius + t * (end_radius - start_radius)
            
            # Position on spiral
            x = self.center[0] + r * np.cos(angle)
            y = self.center[1] + r * np.sin(angle)
            pos = np.array([x, y])
            
            # Direction
            if direction_mode == 'tangent':
                # Tangent to spiral
                direction = np.array([-np.sin(angle), np.cos(angle)])
            elif direction_mode == 'inward':
                direction = self.center - pos
            elif direction_mode == 'outward':
                direction = pos - self.center
            else:
                raise ValueError(f"Unknown direction_mode: {direction_mode}")
            
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            self.positions.append((pos, direction))
        
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.positions):
            raise StopIteration
        pos, direction = self.positions[self.index]
        self.index += 1
        return pos, direction
    
    def __len__(self):
        return len(self.positions)


def get_path_generator(path_type, scene_config, num_samples=100, seed=None, **kwargs):
    """Factory function to create path generator based on scene and type.
    
    Args:
        path_type: 'circular', 'grid', 'random', or 'spiral'
        scene_config: Scene configuration dict from create_scene()
        num_samples: Number of samples to collect
        seed: Random seed
        **kwargs: Additional arguments for specific path types
    
    Returns:
        PathGenerator instance
    """
    world_size = scene_config['world_size']
    scene_type = scene_config['scene_type']
    
    if path_type == 'circular':
        # For fish cage: circle inside the net
        if scene_type == 'fish_cage':
            from src.config import SCENE_CONFIG
            center = SCENE_CONFIG['cage_center']
            radius = SCENE_CONFIG['cage_radius'] * 0.7  # Inside the cage
        else:
            # Default: center of world
            center = [world_size / 2, world_size / 2]
            radius = world_size * 0.3
        
        return CircularPath(
            center=center,
            radius=radius,
            num_samples=num_samples,
            seed=seed,
            **kwargs
        )
    
    elif path_type == 'grid':
        margin = world_size * 0.1
        return GridPath(
            min_pos=[margin, margin],
            max_pos=[world_size - margin, world_size - margin],
            grid_size=[int(np.sqrt(num_samples)), int(np.sqrt(num_samples))],
            seed=seed,
            **kwargs
        )
    
    elif path_type == 'random':
        margin = world_size * 0.1
        return RandomPath(
            min_pos=[margin, margin],
            max_pos=[world_size - margin, world_size - margin],
            num_samples=num_samples,
            seed=seed,
            **kwargs
        )
    
    elif path_type == 'spiral':
        center = [world_size / 2, world_size / 2]
        return SpiralPath(
            center=center,
            start_radius=2.0,
            end_radius=world_size * 0.4,
            num_samples=num_samples,
            seed=seed,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown path_type: {path_type}")
