"""
Interactive voxel-based sonar simulator with realistic acoustic effects.

This simulator uses volumetric ray marching through a voxel grid to model sonar
returns. Each voxel stores material properties (density, reflectivity, absorption),
and the sonar accumulates returns as rays march through the volume.

Features:
- Voxel-based scene representation with material properties
- Volumetric ray marching (no surface raycasting)
- Multiple acoustic noise effects for realistic sonar appearance:
  * Acoustic speckle (coherent interference)
  * Spatial jitter (range uncertainty)
  * Multi-bin spreading (volume backscatter)
  * Temporal decorrelation (frame-to-frame variability)
  * Aspect angle variation (micro-scale roughness)
  * Beam pattern effects (Gaussian falloff)
- 2D top-down visualization showing net cage, fish, and sonar platform
- Interactive controls (WASD movement, arrow rotation)
- Real-time flickering to simulate temporal decorrelation

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
        direction=scene_config['sonar_start_dir'],
        range_m=scene_config['sonar_range']
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
