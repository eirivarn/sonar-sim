"""Visualization and interactive display for sonar simulation.

OVERVIEW:
---------
This module handles all display and user interaction for the simulation.
It creates a three-panel view and manages real-time updates and controls.

THREE-PANEL LAYOUT:
------------------
The display shows three synchronized views:

1. LEFT PANEL - Sonar View (Polar):
   - Raw sonar image in polar coordinates (range × beams)
   - Converted to dB scale for display: 10*log10(intensity)
   - Normalized to [0, 1] for colormap
   - Colormap: 'viridis' (dark = weak, bright = strong returns)
   - Shows what the sonar "sees" with all noise effects
   - Updates continuously with temporal decorrelation (flicker)

2. MIDDLE PANEL - World Map (Cartesian):
   - Top-down bird's eye view of the scene
   - Shows ground truth positions of objects
   - Scene-specific rendering (cage, streets, objects)
   - Sonar position marked with red triangle
   - Sonar direction shown as red arrow
   - FOV cone drawn as dashed lines
   - Y-axis flipped (sonar at bottom looking up)

3. RIGHT PANEL - Ground Truth Segmentation:
   - Material ID map in polar coordinates (range × beams)
   - Each material has unique color from config
   - Perfect labels (no noise) for training ML models
   - Legend shows material types present in scene
   - Same dimensions as sonar view for direct comparison

DISPLAY PIPELINE:
----------------
Each frame:
1. Scene updates dynamic objects (fish, cars, debris)
2. Sonar performs scan → returns (sonar_image, ground_truth)
3. Sonar image converted to dB and normalized
4. All three panels cleared and redrawn
5. Canvas updated

Processing steps:
    raw_image → dB scale → normalization → colormap → display
    raw_image: float32, range [0, ∞]
    dB: float, range [-∞, 0] typically [-60, 0]
    normalized: float, range [0, 1]
    colored: RGB, range [0, 255]

KEYBOARD CONTROLS:
-----------------
Movement (WASD):
- W: Move forward (along direction vector)
- S: Move backward (opposite direction)
- A: Strafe left (perpendicular to direction)
- D: Strafe right (perpendicular to direction)
- Speed: VISUALIZATION_CONFIG['move_speed'] (default 1.0 m/step)

Rotation (Arrow Keys):
- Left Arrow: Rotate counterclockwise
- Right Arrow: Rotate clockwise  
- Speed: VISUALIZATION_CONFIG['rotate_speed'] (default 15°/step)

Implementation:
    def on_key(event):
        if event.key == 'w':
            sonar.move(sonar.direction * move_speed)
        elif event.key == 'left':
            sonar.rotate(-rotate_speed)
        update_display()  # Redraw after movement

ANIMATION LOOP:
--------------
Continuous updates even without user input:
- FuncAnimation calls update_display() at regular intervals
- Interval: VISUALIZATION_CONFIG['animation_interval'] (default 100ms)
- Creates flickering effect from temporal decorrelation
- Simulates dynamic sonar behavior (moving water, swaying nets)

Purpose:
- Realistic sonar continuously updates (not static)
- Training data includes temporal variability
- Visual feedback that simulation is running

GROUND TRUTH RENDERING:
----------------------
Converts material IDs to RGB image:

    gt_rgb = np.zeros((height, width, 3), dtype=uint8)
    for material_id, color in material_colors.items():
        mask = ground_truth_map == material_id
        gt_rgb[mask] = color  # RGB tuple

Colors defined in VISUALIZATION_CONFIG['material_colors']:
- 0 (EMPTY): Black [0, 0, 0]
- 1 (NET): Blue [0, 100, 255]
- 3 (FISH): Orange [255, 140, 0]
- etc.

MAP RENDERING:
-------------
Scene-specific via scene_module.render_map():
- Each scene draws its own map representation
- Fish cage: Circle for cage, dots for fish, patches for debris
- Street: Rectangles for buildings, lines for roads, boxes for cars
- Common: Sonar position, direction arrow, FOV cone

This delegation allows scenes to customize visualization while
keeping visualization.py scene-agnostic.

COORDINATE SYSTEMS:
------------------
Sonar View (Polar):
- X-axis: Beam index [0, num_beams-1] → angle across FOV
- Y-axis: Range bin [0, range_bins-1] → distance from sonar
- Origin: Bottom-left (near sonar, left edge of FOV)

World Map (Cartesian):
- X-axis: World X coordinate [0, world_size] meters
- Y-axis: World Y coordinate [world_size, 0] meters (inverted!)
- Origin: Bottom-left (sonar typically near bottom)

Ground Truth (Polar):
- Same as sonar view (for easy comparison)
- Material IDs instead of intensity values

MATLOTLIB CONFIGURATION:
-----------------------
Disables default keybindings to avoid conflicts:
- Quit, save, fullscreen, navigation, etc. all disabled
- Only our custom key handler is active
- Prevents accidental interference with WASD controls

DEBUG OUTPUT:
------------
Prints diagnostic info each frame:
- Sonar position
- Materials present in ground truth (with percentages)
- Materials in voxel grid (with percentages)
- Sonar image shape

Useful for:
- Verifying scene setup
- Checking material coverage
- Debugging dimension mismatches

USAGE:
------
Called from simulation.py main loop:

    # Setup
    fig, ax_sonar, ax_map, ax_gt = setup_figure(scene_type)
    on_key = create_keyboard_handler(...)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(...)
    
    # Animation
    anim = setup_animation(...)
    plt.show()

RELATIONSHIP TO OTHER MODULES:
-----------------------------
- sonar.py: Calls scan() to get sonar image and ground truth
- voxel_grid.py: Scene renders grid contents in map view  
- scenes/*.py: Delegates to scene.update_scene() and scene.render_map()
- config.py: VISUALIZATION_CONFIG controls all display settings
- simulation.py: Orchestrates setup and event loop

CUSTOMIZATION:
-------------
All visual settings in config.py:
- figure_size: Window dimensions
- sonar_colormap: Color scheme for sonar display
- material_colors: RGB colors for each material
- db_normalization: dB range for display scaling
- animation_interval: Update frequency

Change these to adjust appearance without modifying code.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from config import VISUALIZATION_CONFIG
from materials import (MATERIAL_ID_EMPTY, MATERIAL_ID_NET, MATERIAL_ID_ROPE, 
                       MATERIAL_ID_FISH, MATERIAL_ID_DEBRIS_LIGHT)


def setup_figure(scene_type: str):
    """Setup matplotlib figure and subplots.
    
    Args:
        scene_type: Name of scene for title
        
    Returns:
        Tuple of (fig, ax_sonar, ax_map, ax_gt)
    """
    fig_size = VISUALIZATION_CONFIG['figure_size']
    layout = VISUALIZATION_CONFIG['subplot_layout']
    fig, (ax_sonar, ax_map, ax_gt) = plt.subplots(layout[0], layout[1], figsize=fig_size)
    fig.suptitle(f'Interactive Voxel Sonar Simulation - {scene_type.replace("_", " ").title()}', 
                 fontsize=14, weight='bold')
    
    # Disable matplotlib default key bindings to avoid conflicts
    plt.rcParams['keymap.quit'] = []
    plt.rcParams['keymap.save'] = []
    plt.rcParams['keymap.fullscreen'] = []
    plt.rcParams['keymap.home'] = []
    plt.rcParams['keymap.back'] = []
    plt.rcParams['keymap.forward'] = []
    plt.rcParams['keymap.pan'] = []
    plt.rcParams['keymap.zoom'] = []
    plt.rcParams['keymap.grid'] = []
    
    return fig, ax_sonar, ax_map, ax_gt


def update_display(sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir=None, frame_counter=None):
    """Update all display panels.
    
    Args:
        sonar: VoxelSonar instance
        grid: VoxelGrid instance
        dynamic_objects: Dynamic objects dict from scene
        scene_module: Scene module with update_scene and render_map functions
        ax_sonar: Sonar display axis
        ax_map: Map display axis
        ax_gt: Ground truth display axis
        world_size: Size of world for map display
        save_dir: Optional directory to save frames
        frame_counter: Optional dict with 'count' key to track frame numbers
    """
    # Update dynamic objects using scene's update function
    scene_module.update_scene(grid, dynamic_objects, sonar.position)
    
    print(f"Scanning from position {sonar.position}...")
    sonar_image_polar, ground_truth_map = sonar.scan(grid, return_ground_truth=True)
    
    # DEBUG: Check what materials are in the ground truth
    unique_materials, counts = np.unique(ground_truth_map, return_counts=True)
    print(f"DEBUG: Ground truth materials found:")
    material_names = {
        0: "Empty", 1: "Net", 2: "Rope", 3: "Fish", 
        4: "Wall", 5: "Biomass", 6: "Debris_Light", 
        7: "Debris_Medium", 8: "Debris_Heavy", 9: "Concrete",
        10: "Wood", 11: "Foliage", 12: "Metal", 13: "Glass"
    }
    for mat_id, count in zip(unique_materials, counts):
        pct = 100.0 * count / ground_truth_map.size
        print(f"  Material {mat_id} ({material_names.get(mat_id, 'Unknown')}): {count} pixels ({pct:.2f}%)")
    
    # DEBUG: Check what's actually in the voxel grid
    grid_unique, grid_counts = np.unique(grid.material_id, return_counts=True)
    print(f"DEBUG: Materials in voxel grid:")
    for mat_id, count in zip(grid_unique, grid_counts):
        pct = 100.0 * count / grid.material_id.size
        print(f"  Material {mat_id} ({material_names.get(mat_id, 'Unknown')}): {count} voxels ({pct:.2f}%)")
    
    # Verify dimensions
    print(f"Sonar image shape: {sonar_image_polar.shape}")
    
    # Convert polar to dB and normalize for display
    image_db = 10 * np.log10(np.maximum(sonar_image_polar, 1e-10))
    db_norm = VISUALIZATION_CONFIG['db_normalization']
    image_normalized = np.clip((image_db + db_norm) / db_norm, 0, 1)
    
    # === SONAR DISPLAY (Polar) ===
    ax_sonar.clear()
    cmap = VISUALIZATION_CONFIG['sonar_colormap']
    ax_sonar.imshow(image_normalized, cmap=cmap, aspect='auto', origin='lower')
    ax_sonar.set_title(f'Sonar View - Polar\nPos: [{sonar.position[0]:.1f}, {sonar.position[1]:.1f}]')
    ax_sonar.set_xlabel('Beams')
    ax_sonar.set_ylabel('Range Bins')
    ax_sonar.grid(False)
    
    # === MAP DISPLAY ===
    ax_map.clear()
    
    # Use scene's custom map renderer
    scene_module.render_map(ax_map, dynamic_objects, sonar)
    
    # Draw sonar position and direction (common to all scenes)
    ax_map.scatter(sonar.position[0], sonar.position[1], c='red', s=100, marker='^', label='Sonar', zorder=5)
    
    # Draw sonar direction vector
    arrow_length = 3.0
    ax_map.arrow(sonar.position[0], sonar.position[1], 
                 sonar.direction[0] * arrow_length, sonar.direction[1] * arrow_length,
                 head_width=0.8, head_length=0.5, fc='red', ec='red', zorder=5)
    
    # Draw FOV cone
    fov_rad = np.deg2rad(sonar.fov_deg)
    dir_angle = np.arctan2(sonar.direction[1], sonar.direction[0])
    left_angle = dir_angle + fov_rad / 2
    right_angle = dir_angle - fov_rad / 2
    
    cone_length = sonar.range_m * 0.5
    left_x = sonar.position[0] + cone_length * np.cos(left_angle)
    left_y = sonar.position[1] + cone_length * np.sin(left_angle)
    right_x = sonar.position[0] + cone_length * np.cos(right_angle)
    right_y = sonar.position[1] + cone_length * np.sin(right_angle)
    
    ax_map.plot([sonar.position[0], left_x], [sonar.position[1], left_y], 'r--', alpha=0.4, linewidth=1)
    ax_map.plot([sonar.position[0], right_x], [sonar.position[1], right_y], 'r--', alpha=0.4, linewidth=1)
    
    # Formatting (with flipped Y-axis)
    ax_map.set_xlim(0, world_size)
    ax_map.set_ylim(world_size, 0)  # Flipped: sonar at bottom looking up
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    ax_map.set_title('World Map')
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(loc='upper right', fontsize=8)
    
    # === GROUND TRUTH DISPLAY ===
    ax_gt.clear()
    
    # Get colors from config
    material_colors = VISUALIZATION_CONFIG['material_colors']
    
    # Convert ground truth map to RGB
    gt_rgb = np.zeros((ground_truth_map.shape[0], ground_truth_map.shape[1], 3), dtype=np.uint8)
    for mat_id, color in material_colors.items():
        mask = ground_truth_map == mat_id
        gt_rgb[mask] = color
    
    ax_gt.imshow(gt_rgb, aspect='auto', origin='lower')
    ax_gt.set_title('Ground Truth - Material Segmentation')
    ax_gt.set_xlabel('Beams')
    ax_gt.set_ylabel('Range Bins')
    ax_gt.grid(False)
    
    # Add legend for material types
    legend_elements = [
        Patch(facecolor=np.array(material_colors[MATERIAL_ID_EMPTY])/255, label='Empty'),
        Patch(facecolor=np.array(material_colors[MATERIAL_ID_NET])/255, label='Net'),
        Patch(facecolor=np.array(material_colors[MATERIAL_ID_ROPE])/255, label='Rope'),
        Patch(facecolor=np.array(material_colors[MATERIAL_ID_FISH])/255, label='Fish'),
        Patch(facecolor=np.array(material_colors[MATERIAL_ID_DEBRIS_LIGHT])/255, label='Debris'),
    ]
    ax_gt.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    # Save frames if save_dir is provided
    if save_dir is not None and frame_counter is not None:
        frame_num = frame_counter['count']
        
        # Save sonar image (raw float values)
        sonar_path = save_dir / 'sonar' / f'frame_{frame_num:06d}.npy'
        np.save(sonar_path, sonar_image_polar)
        
        # Save ground truth (uint8 material IDs)
        gt_path = save_dir / 'ground_truth' / f'frame_{frame_num:06d}.npy'
        np.save(gt_path, ground_truth_map)
        
        # Save sonar metadata (position, direction) as JSON
        import json
        metadata = {
            'frame': frame_num,
            'sonar_position': sonar.position.tolist(),
            'sonar_direction': sonar.direction.tolist(),
            'range_m': sonar.range_m,
            'fov_deg': sonar.fov_deg,
        }
        meta_path = save_dir / 'metadata' / f'frame_{frame_num:06d}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        frame_counter['count'] += 1
        
        if frame_num % 10 == 0:
            print(f"Saved frame {frame_num}")


def create_keyboard_handler(sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir=None, frame_counter=None):
    """Create keyboard event handler function.
    
    Args:
        save_dir: Optional directory to save frames
        frame_counter: Optional dict with 'count' key
    
    Returns:
        Function that handles keyboard events
    """
    def on_key(event):
        """Handle keyboard input."""
        move_speed = VISUALIZATION_CONFIG['move_speed']
        rotate_speed = VISUALIZATION_CONFIG['rotate_speed']
        
        if event.key == 'w':
            sonar.move(sonar.direction * move_speed)
        elif event.key == 's':
            sonar.move(-sonar.direction * move_speed)
        elif event.key == 'a':
            # Move perpendicular (left)
            perp = np.array([-sonar.direction[1], sonar.direction[0]])
            sonar.move(-perp * move_speed)
        elif event.key == 'd':
            # Move perpendicular (right)
            perp = np.array([-sonar.direction[1], sonar.direction[0]])
            sonar.move(perp * move_speed)
        elif event.key == 'left':
            sonar.rotate(-rotate_speed)
        elif event.key == 'right':
            sonar.rotate(rotate_speed)
        else:
            return
        
        update_display(sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir, frame_counter)
    
    return on_key


def setup_animation(fig, sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir=None, frame_counter=None):
    """Setup matplotlib animation for continuous updates.
    
    Args:
        save_dir: Optional directory to save frames
        frame_counter: Optional dict with 'count' key
    
    Returns:
        FuncAnimation object
    """
    def anim_update(frame):
        update_display(sonar, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir, frame_counter)
        return []
    
    anim_interval = VISUALIZATION_CONFIG['animation_interval']
    anim_cache = VISUALIZATION_CONFIG['animation_cache']
    return FuncAnimation(fig, anim_update, interval=anim_interval, cache_frame_data=anim_cache)


def print_controls():
    """Print control instructions."""
    print("\nControls:")
    print("  W/S: Move forward/back")
    print("  A/D: Move left/right")
    print("  Left/Right arrows: Rotate")
    print("\nNote: Image flickers continuously due to:")
    print("  - Acoustic speckle (coherent interference)")
    print("  - Aspect angle variations (micro-scale roughness)")
    print("  - Temporal decorrelation (net sway, water movement)")
    print("  - Receiver noise (electronic/thermal)")
