"""
Interactive voxel-based sonar simulator with realistic acoustic effects.

OVERVIEW:
---------
This is the main entry point for the simulation. It orchestrates all modules
to create an interactive sonar visualization with realistic acoustic effects.

ARCHITECTURE:
------------
The simulation is modular with clear separation of concerns:

- materials.py: Material definitions and acoustic properties
- voxel_grid.py: 2D spatial grid storing material properties  
- sonar.py: Volumetric ray marching sonar simulation
- dynamics.py: Dynamic object behavior (fish, cars, debris)
- visualization.py: Display and user interaction
- scenes/*.py: Scene definitions (geometry + behavior)
- config.py: All tunable parameters

This file (simulation.py) is the thin orchestration layer that:
1. Loads the requested scene module
2. Initializes the sonar
3. Sets up visualization
4. Connects keyboard controls
5. Starts the animation loop

USAGE:
------
Run with default scene (fish cage):
    python simulation.py

Run with specific scene:
    python simulation.py --scene scenes.street_scene

Run with custom scene:
    python simulation.py --scene scenes.my_custom_scene

FEATURES:
---------
- Voxel-based scene representation with material properties
- Volumetric ray marching (no surface raycasting)
- Multiple acoustic noise effects for realistic sonar appearance:
  * Acoustic speckle (coherent interference)
  * Spatial jitter (range uncertainty)
  * Multi-bin spreading (volume backscatter)
  * Temporal decorrelation (frame-to-frame variability)
  * Aspect angle variation (micro-scale roughness)
  * Beam pattern effects (Gaussian falloff)
- Three-panel visualization: sonar view, world map, ground truth
- Interactive controls (WASD movement, arrow rotation)
- Real-time flickering to simulate temporal decorrelation
- Modular scene system for easy extension

HOW TO CREATE A NEW SCENE:
--------------------------
See detailed guide below in the docstring, or refer to:
- scenes/fish_cage_scene.py for underwater example
- scenes/street_scene.py for urban example
- scenes/README.md for step-by-step instructions

A scene file must provide three functions:
1. create_scene() - Initialize world and return config dict
2. update_scene() - Update dynamic objects each frame
3. render_map() - Draw world map view

SCENE CREATION METHODOLOGY:
---------------------------

STEP 1: Create scenes/my_scene.py
    
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from voxel_grid import VoxelGrid
    from materials import FISH, WALL, CONCRETE  # etc.
    from dynamics import update_fish  # or custom
    from config import MY_SCENE_CONFIG  # optional

STEP 2: Implement create_scene()

    def create_scene():
        # Choose world size and resolution
        world_size = 50.0  # meters
        grid_size = 500    # voxels (50m / 0.1m)
        grid = VoxelGrid(grid_size, grid_size, voxel_size=0.1)
        
        # Add static geometry (buildings, terrain)
        grid.set_box([0, 20], [50, 22], CONCRETE)
        
        # Initialize dynamic objects
        fish_data = [...]  # List of dicts with pos, vel, etc.
        
        # Draw initial positions
        for fish in fish_data:
            grid.set_ellipse(fish['pos'], fish['radii'], ...)
        
        # Set sonar start
        sonar_start_pos = np.array([25.0, 45.0])
        sonar_start_dir = np.array([0.0, -1.0])  # normalized
        
        return {
            'grid': grid,
            'world_size': world_size,
            'scene_type': 'my_scene',
            'sonar_start_pos': sonar_start_pos,
            'sonar_start_dir': sonar_start_dir,
            'sonar_range': 20.0,
            'dynamic_objects': {'fish_data': fish_data, ...}
        }

STEP 3: Implement update_scene()

    def update_scene(grid, dynamic_objects, sonar_pos):
        fish_data = dynamic_objects['fish_data']
        update_fish(grid, fish_data, ...)  # Or custom logic

STEP 4: Implement render_map()

    def render_map(ax, dynamic_objects, sonar):
        # Draw static structures
        ax.add_patch(plt.Rectangle(...))
        # Draw dynamic objects
        ax.scatter(positions[:, 0], positions[:, 1], ...)

STEP 5: Run your scene

    python simulation.py --scene scenes.my_scene

SCENE DESIGN TIPS:
-----------------
Material Selection:
- High reflectivity (0.7-0.9): Metal, concrete (strong returns)
- Medium reflectivity (0.3-0.6): Wood, biomass (moderate returns)
- Low reflectivity (0.1-0.3): Nets, foliage (weak returns)
- High density: Volumetric scattering (biomass, foliage)
- Low density: Surface returns (nets, walls)

Spatial Layout:
- World size: 2-3x sonar range (room to explore)
- Voxel size: 0.1m typical (balance resolution/performance)
- Objects: >0.3m wide to be visible (>3 voxels)
- Sonar: Start with clear view of interesting features

Dynamic Behavior:
- Use existing update functions (update_fish, update_cars, update_debris)
- Or implement custom: grid.clear_*() then update physics then grid.set_*()
- Velocities: 0.05-0.2 m/s typical for natural motion

TECHNICAL DETAILS:
-----------------
Coordinate Systems:
- World: Origin at (0,0), meters, continuous
- Voxel: Grid indices, discrete, converted by grid.world_to_voxel()
- Sonar: Polar (range, beam), converted to Cartesian for display

Update Loop:
1. User presses key → keyboard handler updates sonar pose
2. Animation timer fires → update_display() called
3. update_display() calls scene.update_scene() → dynamic objects move
4. update_display() calls sonar.scan() → generates images
5. update_display() redraws all three panels
6. Repeat at animation_interval (100ms default)

Performance:
- Typical: 10 FPS for 300×300 grid, 512×256 sonar, 50 fish
- Bottleneck: Ray marching in sonar.scan()
- Optimization: Reduce range_bins or num_beams in config

Dependencies: numpy, matplotlib
"""

import matplotlib.pyplot as plt
from sonar import VoxelSonar
from visualization import (setup_figure, update_display, create_keyboard_handler, 
                           setup_animation, print_controls)


def main(scene_path='scenes.fish_cage_scene'):
    """Run interactive voxel sonar viewer.
    
    Args:
        scene_path: Module path to scene file (e.g., 'scenes.fish_cage_scene')
    """
    # Dynamically import the scene module
    import importlib
    try:
        scene_module = importlib.import_module(scene_path)
    except ImportError as e:
        print(f"Error: Could not import scene module '{scene_path}'")
        print(f"Details: {e}")
        print("\nAvailable scenes:")
        print("  - scenes.fish_cage_scene")
        print("  - scenes.street_scene")
        return
    
    # Create the scene
    print(f"Loading scene: {scene_path}")
    scene_config = scene_module.create_scene()
    
    grid = scene_config['grid']
    world_size = scene_config['world_size']
    scene_type = scene_config['scene_type']
    dynamic_objects = scene_config['dynamic_objects']
    
    print("Initializing sonar...")
    sonar = VoxelSonar(
        position=scene_config['sonar_start_pos'],
        direction=scene_config['sonar_start_dir']
        # range_m uses SONAR_CONFIG['range_m'] by default
    )
    
    # Setup visualization
    fig, ax_sonar, ax_map, ax_gt = setup_figure(scene_type)
    
    # Create keyboard handler
    on_key = create_keyboard_handler(sonar, grid, dynamic_objects, scene_module, 
                                     ax_sonar, ax_map, ax_gt, world_size)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size)
    
    # Setup continuous animation
    anim = setup_animation(fig, sonar, grid, dynamic_objects, scene_module, 
                          ax_sonar, ax_map, ax_gt, world_size)
    
    # Print controls
    print_controls()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sonar simulation with different scenes')
    parser.add_argument('--scene', type=str, default='scenes.fish_cage_scene',
                       help='Scene module path (e.g., scenes.fish_cage_scene or scenes.street_scene)')
    
    args = parser.parse_args()
    main(scene_path=args.scene)
