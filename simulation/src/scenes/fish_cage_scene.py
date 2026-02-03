"""Fish cage scene with layered rendering for better performance.

This version uses separate static and dynamic layers to avoid rebuilding
the entire scene every frame. The static net structure is cached and only
the fish positions are updated.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.voxel_grid import VoxelGrid
from src.core.materials import FISH, NET, DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY, EMPTY
from src.core.dynamics import update_fish_optimized as update_fish
from src.core.dynamics import update_debris_optimized as update_debris
from src.core.fast_render import render_fish_batch
from src.config import SCENE_CONFIG


def create_scene():
    """Create fish farm net cage with layered rendering.
    
    Returns:
        dict with scene configuration including static layer cache
    """
    # Create 30m x 30m world at 10cm resolution (300×300 voxels)
    grid = VoxelGrid(300, 300, voxel_size=0.1)
    
    # Net cage parameters
    cage_center = np.array(SCENE_CONFIG['cage_center'])
    cage_radius = SCENE_CONFIG['cage_radius']
    num_sides = SCENE_CONFIG['num_sides']
    
    # Current effect
    current_direction = np.array(SCENE_CONFIG['current_direction'])
    current_strength = SCENE_CONFIG['current_strength']
    
    # Create circular net panels with current deflection
    for i in range(num_sides):
        angle1 = (i / num_sides) * 2 * np.pi
        angle2 = ((i + 1) / num_sides) * 2 * np.pi
        
        # Panel corners (base positions)
        x1_base = cage_center[0] + cage_radius * np.cos(angle1)
        y1_base = cage_center[1] + cage_radius * np.sin(angle1)
        x2_base = cage_center[0] + cage_radius * np.cos(angle2)
        y2_base = cage_center[1] + cage_radius * np.sin(angle2)
        
        # Apply current deflection
        deflection1 = current_strength * max(0, (y1_base - cage_center[1]) / cage_radius) * current_direction
        deflection2 = current_strength * max(0, (y2_base - cage_center[1]) / cage_radius) * current_direction
        
        lateral_factor1 = np.sin(angle1) * 0.4
        lateral_factor2 = np.sin(angle2) * 0.4
        deflection1 += np.array([lateral_factor1 * current_strength * 0.3, 0])
        deflection2 += np.array([lateral_factor2 * current_strength * 0.3, 0])
        
        x1 = x1_base + deflection1[0]
        y1 = y1_base + deflection1[1]
        x2 = x2_base + deflection2[0]
        y2 = y2_base + deflection2[1]
        
        # Create net line for this panel
        for t in np.linspace(0, 1, 100):
            x_linear = x1 + t * (x2 - x1)
            y_linear = y1 + t * (y2 - y1)
            
            # Add catenary-like sag
            sag = 0.25 * (1 - (2*t - 1)**2)
            
            panel_dx = x2 - x1
            panel_dy = y2 - y1
            panel_length = np.sqrt(panel_dx**2 + panel_dy**2)
            if panel_length > 0:
                perp_x = -panel_dy / panel_length
                perp_y = panel_dx / panel_length
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                to_center_x = cage_center[0] - mid_x
                to_center_y = cage_center[1] - mid_y
                to_center_length = np.sqrt(to_center_x**2 + to_center_y**2)
                
                if to_center_length > 0:
                    sag_x = sag * to_center_x / to_center_length
                    sag_y = sag * to_center_y / to_center_length
                else:
                    sag_x = sag_y = 0
                
                x = x_linear + sag_x
                y = y_linear + sag_y
            else:
                x = x_linear
                y = y_linear
            
            grid.set_circle(
                np.array([x, y]),
                0.08,
                NET
            )
            
            # Occasional rope - DISABLED for cleaner net-only simulation
            # if t % (1.0 / 7) < 0.025:
            #     grid.set_box(
            #         np.array([x - 0.12, y - 0.12]),
            #         np.array([x + 0.12, y + 0.12]),
            #         ROPE
            #     )
    
    # Feed pipe - DISABLED for cleaner net-only simulation
    # grid.set_box(
    #     np.array([cage_center[0] - 0.15, cage_center[1] - 0.15]),
    #     np.array([cage_center[0] + 0.15, cage_center[1] + 0.15]),
    #     ROPE
    # )
    
    # Cache the static environment (net only, no feed pipe)
    static_density = grid.density.copy()
    static_reflectivity = grid.reflectivity.copy()
    static_absorption = grid.absorption.copy()
    static_material_id = grid.material_id.copy()
    
    # Add fish
    np.random.seed(42)
    num_fish = SCENE_CONFIG['num_fish']
    fish_data = []
    
    for _ in range(num_fish):
        angle = np.random.rand() * 2 * np.pi
        r_fraction = 0.65 + 0.27 * (np.random.rand() ** 0.5)
        r = cage_radius * r_fraction
        
        x_base = cage_center[0] + r * np.cos(angle)
        y_base = cage_center[1] + r * np.sin(angle)
        
        deflection = current_strength * max(0, (y_base - cage_center[1]) / cage_radius) * current_direction
        lateral_factor = np.sin(angle) * 0.4
        deflection += np.array([lateral_factor * current_strength * 0.3, 0])
        deflection *= r_fraction
        
        x = x_base + deflection[0]
        y = y_base + deflection[1]
        
        swim_angle = np.random.rand() * 2 * np.pi
        swim_speed = 0.08 + np.random.rand() * 0.12
        vx = swim_speed * np.cos(swim_angle)
        vy = swim_speed * np.sin(swim_angle)
        
        fish_length = 0.4 + np.random.rand() * 0.2
        fish_width = fish_length * 0.20
        
        species = ['A', 'B', 'C'][_ % 3]
        
        fish_data.append({
            'pos': np.array([x, y]),
            'vel': np.array([vx, vy]),
            'orientation': swim_angle,
            'radii': np.array([fish_length, fish_width]),
            'turn_timer': np.random.rand() * 100,
            'species': species
        })
    
    # Render fish initially
    for fish in fish_data:
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)
    
    # Cache the static environment in dynamic_objects for fast restore
    # This allows update_scene to restore the static layer quickly
    return {
        'grid': grid,
        'world_size': SCENE_CONFIG['world_size_m'],
        'scene_type': 'fish_cage',
        'sonar_start_pos': np.array([15.0, 2.0]),
        'sonar_start_dir': np.array([0.0, 1.0]),
        'dynamic_objects': {
            'fish_data': fish_data,
            'debris_data': [],
            # Cached static environment for fast restore
            'static_cache': {
                'density': static_density,
                'reflectivity': static_reflectivity,
                'absorption': static_absorption,
                'material_id': static_material_id,
            }
        },
    }


def update_scene(grid, dynamic_objects, sonar_pos, dt=0.1):
    """Update fish positions using layered rendering.
    
    This optimized version:
    1. Restores static environment from cache (fast array copy)
    2. Updates only fish positions
    3. Renders fish on top of static layer
    
    Args:
        grid: VoxelGrid to update
        dynamic_objects: Dynamic objects dict with fish_data and static_cache
        sonar_pos: Sonar position for avoidance
        dt: Time step in seconds
    """
    fish_data = dynamic_objects['fish_data']
    debris_data = dynamic_objects['debris_data']
    cage_center = np.array(SCENE_CONFIG['cage_center'])
    cage_radius = SCENE_CONFIG['cage_radius']
    
    # Update fish physics (positions and velocities)
    update_fish(grid, fish_data, cage_center, cage_radius, sonar_pos, dt)
    update_debris(grid, debris_data, cage_center, cage_radius, dt)
    
    # Fast rendering using cached static layer
    if 'static_cache' in dynamic_objects:
        cache = dynamic_objects['static_cache']
        
        # Restore static environment (single array copy operation)
        grid.density[:] = cache['density']
        grid.reflectivity[:] = cache['reflectivity']
        grid.absorption[:] = cache['absorption']
        grid.material_id[:] = cache['material_id']
        
        # Fast batch rendering of all fish using vectorized operations
        render_fish_batch(grid, fish_data)
    else:
        # Fallback to old method if cache not available
        grid.clear_fish()
        for fish in fish_data:
            grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)
        for fish in fish_data:
            grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)


def render_map(ax, scene_data, sonar):
    """Render the fish cage map view."""
    import matplotlib.pyplot as plt
    
    cage_center = np.array(SCENE_CONFIG['cage_center'])
    cage_radius = SCENE_CONFIG['cage_radius']
    num_sides = SCENE_CONFIG['num_sides']
    current_strength = SCENE_CONFIG['current_strength']
    current_direction = np.array(SCENE_CONFIG['current_direction'])
    fish_data = scene_data['fish_data']
    
    # Draw bent cage outline
    cage_x = []
    cage_y = []
    
    for i in range(num_sides):
        angle1 = (i / num_sides) * 2 * np.pi
        angle2 = ((i + 1) / num_sides) * 2 * np.pi
        
        x1_base = cage_center[0] + cage_radius * np.cos(angle1)
        y1_base = cage_center[1] + cage_radius * np.sin(angle1)
        x2_base = cage_center[0] + cage_radius * np.cos(angle2)
        y2_base = cage_center[1] + cage_radius * np.sin(angle2)
        
        deflection1 = current_strength * max(0, (y1_base - cage_center[1]) / cage_radius) * current_direction
        lateral_factor1 = np.sin(angle1) * 0.4
        deflection1 += np.array([lateral_factor1 * current_strength * 0.3, 0])
        
        deflection2 = current_strength * max(0, (y2_base - cage_center[1]) / cage_radius) * current_direction
        lateral_factor2 = np.sin(angle2) * 0.4
        deflection2 += np.array([lateral_factor2 * current_strength * 0.3, 0])
        
        x1 = x1_base + deflection1[0]
        y1 = y1_base + deflection1[1]
        x2 = x2_base + deflection2[0]
        y2 = y2_base + deflection2[1]
        
        segment_x = []
        segment_y = []
        for t in np.linspace(0, 1, 20):
            x_linear = x1 + t * (x2 - x1)
            y_linear = y1 + t * (y2 - y1)
            
            sag = 0.25 * (1 - (2*t - 1)**2)
            panel_dx = x2 - x1
            panel_dy = y2 - y1
            panel_length = np.sqrt(panel_dx**2 + panel_dy**2)
            
            if panel_length > 0:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                to_center_x = cage_center[0] - mid_x
                to_center_y = cage_center[1] - mid_y
                to_center_length = np.sqrt(to_center_x**2 + to_center_y**2)
                
                if to_center_length > 0:
                    sag_x = sag * to_center_x / to_center_length
                    sag_y = sag * to_center_y / to_center_length
                else:
                    sag_x = sag_y = 0
                
                x = x_linear + sag_x
                y = y_linear + sag_y
            else:
                x = x_linear
                y = y_linear
            
            segment_x.append(x)
            segment_y.append(y)
        
        cage_x.extend(segment_x)
        cage_y.extend(segment_y)
    
    ax.plot(cage_x, cage_y, 'c-', linewidth=0.5, alpha=0.7, label='Net cage')
    
    # Draw fish
    for fish in fish_data:
        color = {'A': 'orange', 'B': 'green', 'C': 'purple'}[fish['species']]
        ax.plot(fish['pos'][0], fish['pos'][1], 'o', color=color, markersize=3)
    
    ax.set_xlim(0, SCENE_CONFIG['world_size_m'])
    ax.set_ylim(0, SCENE_CONFIG['world_size_m'])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
