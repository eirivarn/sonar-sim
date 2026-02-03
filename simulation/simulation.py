"""
Interactive voxel-based sonar simulator with realistic acoustic effects.

OVERVIEW:
---------
This is the main entry point for the simulation. It orchestrates all modules
to create an interactive sonar visualization with realistic acoustic effects.

ARCHITECTURE:
------------
The simulation is modular with clear separation of concerns:

- src/core/materials.py: Material definitions and acoustic properties
- src/core/voxel_grid.py: 2D spatial grid storing material properties  
- src/core/sonar.py: Volumetric ray marching sonar simulation
- src/core/dynamics.py: Dynamic object behavior (fish, cars, debris)
- src/visualization/visualization.py: Display and user interaction
- src/scenes/*.py: Scene definitions (geometry + behavior)
- src/config.py: All tunable parameters

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
    from src.core.dynamics import update_fish  # or custom
    from src.config import MY_SCENE_CONFIG  # optional

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

"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.sonar import VoxelSonar
from src.core.robot import Robot
from src.scripts.visualization import (setup_figure, update_display, create_keyboard_handler, 
                           setup_animation, print_controls)


def main(scene_path='src.scenes.fish_cage_scene', save_run=None, collect_mode=None, num_samples=100, path_kwargs=None):
    """Run interactive voxel sonar viewer or headless data collection.
    
    Args:
        scene_path: Module path to scene file (e.g., 'scenes.fish_cage_scene')
        save_run: Optional run name for saving data. If True, uses timestamp.
        collect_mode: If provided, run headless data collection with this path type
                     ('circular', 'grid', 'random', 'spiral')
        num_samples: Number of samples to collect in headless mode
        path_kwargs: Additional arguments for path generator
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
    scene_config = scene_module.create_scene()
    
    grid = scene_config['grid']
    world_size = scene_config['world_size']
    scene_type = scene_config['scene_type']
    dynamic_objects = scene_config['dynamic_objects']
    
    # Create robot at starting position
    start_pos = scene_config['sonar_start_pos']
    start_dir = scene_config['sonar_start_dir']
    start_yaw = np.arctan2(start_dir[1], start_dir[0])  # Convert direction to yaw angle
    
    robot = Robot(initial_x=start_pos[0], initial_y=start_pos[1], initial_yaw=start_yaw)
    robot_state = robot.get_state()
    sonar = VoxelSonar(
        position=robot_state['position'],
        direction=robot_state['direction']
        # range_m uses SONAR_CONFIG['range_m'] by default
    )
    
    # Setup save directory if requested (needed for both GUI and headless modes)
    save_dir = None
    frame_counter = None
    if save_run:
        from pathlib import Path
        from datetime import datetime
        import json
        
        # Use timestamp if save_run is True, otherwise use provided name
        if save_run is True:
            run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            run_name = save_run
        
        # Create directory structure
        save_dir = Path('data') / 'runs' / run_name
        (save_dir / 'frames').mkdir(parents=True, exist_ok=True)  # All data in .npz files
        
        # Save run configuration
        from src.config import SONAR_CONFIG, VISUALIZATION_CONFIG
        run_config = {
            'run_name': run_name,
            'scene_path': scene_path,
            'scene_type': scene_type,
            'timestamp': datetime.now().isoformat(),
            'sonar_config': SONAR_CONFIG,
            'world_size': world_size,
        }
        with open(save_dir / 'run_config.json', 'w') as f:
            json.dump(run_config, f, indent=2)
        
        # Save scene snapshot for visualization reconstruction
        scene_snapshot = {
            'scene_path': scene_path,
            'scene_type': scene_type,
            'world_size': world_size,
            'dynamic_objects_initial': {}
        }
        
        # Serialize dynamic objects (convert numpy arrays to lists)
        for key, value in dynamic_objects.items():
            # Skip static_cache (large arrays, not needed for snapshot)
            if key == 'static_cache':
                continue
                
            if isinstance(value, list):
                # List of objects (fish, cars, etc.)
                serialized = []
                for obj in value:
                    obj_dict = {}
                    for k, v in obj.items():
                        if isinstance(v, np.ndarray):
                            obj_dict[k] = v.tolist()
                        else:
                            obj_dict[k] = v
                    serialized.append(obj_dict)
                scene_snapshot['dynamic_objects_initial'][key] = serialized
            elif isinstance(value, np.ndarray):
                scene_snapshot['dynamic_objects_initial'][key] = value.tolist()
            else:
                scene_snapshot['dynamic_objects_initial'][key] = value
        
        with open(save_dir / 'scene_snapshot.json', 'w') as f:
            json.dump(scene_snapshot, f, indent=2)
        
        frame_counter = {'count': 0}
        print(f"\nSaving run data to: {save_dir}")
        print("Data will be saved in:")
        print(f"  - sonar/         (raw sonar images as .npy)")
        print(f"  - ground_truth/  (material ID maps as .npy)")
        print(f"  - metadata/      (frame metadata as .json)")
        print()
    
    # HEADLESS DATA COLLECTION MODE
    if collect_mode is not None:
        if not save_run:
            print("Error: --save is required when using --collect mode")
            return
        
        from src.config import VISUALIZATION_CONFIG
        import json
        
        dt = VISUALIZATION_CONFIG['dt']
        
        print(f"\n{'='*60}")
        print(f"HEADLESS DATA COLLECTION MODE")
        print(f"{'='*60}")
        print(f"Collection type: {collect_mode}")
        print(f"Target samples: {num_samples}")
        print(f"Scene: {scene_type}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*60}\n")
        
        # Determine collection strategy
        if collect_mode == 'random':
            # Random exploration mode: Robot wanders randomly, capturing frames periodically
            print("Mode: Random exploration with robot physics")
            print("Robot will move randomly through the world, capturing frames as it goes\n")
            
            frame_counter = {'count': 0}
            # More robot updates per capture for better movement coverage
            robot_updates_per_capture = 15
            
            # Get cage boundaries from SCENE_CONFIG (for fish_cage_scene)
            from src.config import SCENE_CONFIG
            cage_center = np.array(SCENE_CONFIG['cage_center'])
            cage_radius = SCENE_CONFIG['cage_radius']
            safe_radius = cage_radius - 2.0  # Stay 2m away from nets
            
            # Wandering state
            current_forward = 0.6
            current_yaw_rate = 0.0
            wander_timer = 0
            
            # Random movement loop
            for sample_idx in range(num_samples):
                # Run several robot physics updates to get meaningful movement
                for update_idx in range(robot_updates_per_capture):
                    # Update wander behavior every few steps
                    if wander_timer <= 0:
                        # Change direction slightly
                        current_yaw_rate += np.random.uniform(-0.4, 0.4)
                        current_yaw_rate = np.clip(current_yaw_rate, -0.8, 0.8)
                        # Vary speed slightly
                        current_forward += np.random.uniform(-0.2, 0.2)
                        current_forward = np.clip(current_forward, 0.4, 0.9)
                        # Reset timer
                        wander_timer = np.random.randint(3, 8)
                    
                    wander_timer -= 1
                    
                    # Check distance from center
                    robot_state = robot.get_state()
                    robot_pos = robot_state['position']
                    dist_from_center = np.linalg.norm(robot_pos - cage_center)
                    
                    # If getting close to boundary, turn toward center
                    if dist_from_center > safe_radius * 0.8:
                        # Calculate angle to center
                        to_center = cage_center - robot_pos
                        angle_to_center = np.arctan2(to_center[1], to_center[0])
                        angle_diff = angle_to_center - robot.yaw
                        # Normalize angle difference to [-pi, pi]
                        while angle_diff > np.pi:
                            angle_diff -= 2 * np.pi
                        while angle_diff < -np.pi:
                            angle_diff += 2 * np.pi
                        
                        # Steer toward center
                        current_yaw_rate = np.clip(angle_diff * 0.8, -1.0, 1.0)
                    
                    # Apply thrust commands
                    robot.set_thrust(
                        forward=current_forward,
                        lateral=np.random.uniform(-0.15, 0.15),  # Small random drift
                        yaw_rate=current_yaw_rate
                    )
                    
                    robot.update(dt)
                
                # Update sonar to follow robot
                robot_state = robot.get_state()
                sonar.position = robot_state['position'].copy()
                sonar.direction = robot_state['direction'].copy()
                
                # Update scene once per capture (fish move forward in time)
                # Use larger time step for scene to speed up: robot_updates * dt
                scene_dt = robot_updates_per_capture * dt
                scene_module.update_scene(grid, dynamic_objects, sonar.position, scene_dt)
                
                # Perform scan
                sonar_image, ground_truth = sonar.scan(grid, return_ground_truth=True)
                
                # Scale to [0-255] range to match real sonar data
                # Simulation returns [0-1] normalized values, real data is in [0-255]
                sonar_image = np.clip(sonar_image * 255.0, 0, 255).astype(np.float32)
                
                # Save data
                frame_num = frame_counter['count']
                
                # ROS-compatible timestamp (seconds since epoch)
                import time
                timestamp_s = time.time()
                timestamp_ns = int(timestamp_s * 1e9)  # Convert to nanoseconds
                
                # Verify sonar image is float32 and shape is (1024, 256)
                assert sonar_image.shape == (1024, 256), f"Expected (1024, 256), got {sonar_image.shape}"
                assert sonar_image.dtype == np.float32, f"Expected float32, got {sonar_image.dtype}"
                
                # Prepare metadata as JSON string (SOLAQUA format)
                metadata = {
                    'frame': frame_num,
                    't': timestamp_s,  # UTC timestamp in seconds (ROS format)
                    'ts_unix_ns': timestamp_ns,  # Nanosecond timestamp
                    'dim_sizes': [1024, 256],  # [range_bins, beams] matching ROS
                    'dim_labels': ['range', 'beam'],  # ROS dimension labels
                    'sonar_position': sonar.position.tolist(),
                    'sonar_direction': sonar.direction.tolist(),
                    'range_m': float(sonar.range_m),
                    'fov_deg': float(sonar.fov_deg),
                    'data_shape': list(sonar_image.shape),
                    'data_dtype': str(sonar_image.dtype),
                    'intensity_scale': [0.0, 255.0],  # Data range matches real sonar
                    'fish_positions': [fish['pos'].tolist() for fish in dynamic_objects['fish_data']],
                    'fish_species': [fish['species'] for fish in dynamic_objects['fish_data']],
                }
                
                # Calculate spatial extent (for SOLAQUA compatibility)
                # Sonar FOV creates a cone - calculate approximate bounds
                fov_rad = np.deg2rad(sonar.fov_deg)
                max_range = sonar.range_m
                half_width = max_range * np.tan(fov_rad / 2)
                extent = np.array([
                    sonar.position[0] - half_width,  # x_min
                    sonar.position[0] + half_width,  # x_max
                    sonar.position[1],                # y_min (sonar position)
                    sonar.position[1] + max_range    # y_max (max range ahead)
                ], dtype=np.float32)
                
                # Save as NPZ file (SOLAQUA-compatible format)
                npz_path = save_dir / 'frames' / f'frame_{frame_num:06d}.npz'
                np.savez_compressed(
                    npz_path,
                    sonar_image=sonar_image,           # (1024, 256) float32 array
                    ground_truth=ground_truth,         # (1024, 256) uint8 array
                    ts_unix_ns=np.array([timestamp_ns], dtype=np.int64),  # Nanosecond timestamp
                    extent=extent,                     # (4,) spatial bounds
                    meta_json=json.dumps(metadata)     # JSON string with all metadata
                )
                
                frame_counter['count'] += 1
                
                # Progress update
                if (frame_num + 1) % max(1, num_samples // 10) == 0:
                    progress = 100 * (frame_num + 1) / num_samples
                    print(f"Progress: {frame_num+1}/{num_samples} ({progress:.1f}%) - Position: {sonar.position}")
        
        else:
            # Path-based collection (circular, spiral, grid)
            from src.scripts.data_collection import get_path_generator
            
            print(f"Mode: Path-based collection ({collect_mode})")
            print("Using pre-defined path with robot teleportation\n")
            
            # Generate path
            if path_kwargs is None:
                path_kwargs = {}
            path_gen = get_path_generator(collect_mode, scene_config, num_samples=num_samples, **path_kwargs)
            
            # Convert to list to allow multiple iterations
            path_list = [(pos, direction) for pos, direction in path_gen]
            
            # Save path configuration
            path_config = {
                'path_type': collect_mode,
                'num_samples': num_samples,
                'path_kwargs': path_kwargs,
                'positions': [(pos.tolist(), direction.tolist()) for pos, direction in path_list]
            }
            with open(save_dir / 'path_config.json', 'w') as f:
                json.dump(path_config, f, indent=2)
            
            # Collect data at each position
            frame_counter = {'count': 0}
            
            for i, (pos, direction) in enumerate(path_list):
                # Teleport sonar to path position
                sonar.position = pos.copy()
                sonar.direction = direction.copy()
                
                # Update scene (fish continue moving in the same world)
                from src.utils.profiler import get_profiler
                profiler = get_profiler()
                
                with profiler.measure('scene_update'):
                    scene_module.update_scene(grid, dynamic_objects, sonar.position, dt)
                
                # Perform scan
                with profiler.measure('sonar_scan'):
                    sonar_image, ground_truth = sonar.scan(grid, return_ground_truth=True)
                
                # Scale to [0-255] range to match real sonar data
                # Simulation returns [0-1] normalized values, real data is in [0-255]
                sonar_image = np.clip(sonar_image * 255.0, 0, 255).astype(np.float32)
                
                # Save data
                frame_num = frame_counter['count']
                
                # ROS-compatible timestamp (seconds since epoch)
                import time
                timestamp_s = time.time()
                timestamp_ns = int(timestamp_s * 1e9)
                
                # Verify sonar image is float32 and shape is (1024, 256)
                assert sonar_image.shape == (1024, 256), f"Expected (1024, 256), got {sonar_image.shape}"
                assert sonar_image.dtype == np.float32, f"Expected float32, got {sonar_image.dtype}"
                
                # Prepare metadata as JSON string (SOLAQUA format)
                metadata = {
                    'frame': frame_num,
                    't': timestamp_s,
                    'ts_unix_ns': timestamp_ns,
                    'dim_sizes': [1024, 256],
                    'dim_labels': ['range', 'beam'],
                    'sonar_position': sonar.position.tolist(),
                    'sonar_direction': sonar.direction.tolist(),
                    'range_m': float(sonar.range_m),
                    'fov_deg': float(sonar.fov_deg),
                    'data_shape': list(sonar_image.shape),
                    'data_dtype': str(sonar_image.dtype),
                    'intensity_scale': [0.0, 255.0],  # Data range matches real sonar
                    'fish_positions': [fish['pos'].tolist() for fish in dynamic_objects['fish_data']],
                    'fish_species': [fish['species'] for fish in dynamic_objects['fish_data']],
                }
                
                # Calculate spatial extent
                fov_rad = np.deg2rad(sonar.fov_deg)
                max_range = sonar.range_m
                half_width = max_range * np.tan(fov_rad / 2)
                extent = np.array([
                    sonar.position[0] - half_width,
                    sonar.position[0] + half_width,
                    sonar.position[1],
                    sonar.position[1] + max_range
                ], dtype=np.float32)
                
                # Save as NPZ file (SOLAQUA-compatible format)
                npz_path = save_dir / 'frames' / f'frame_{frame_num:06d}.npz'
                with profiler.measure('save_npz'):
                    np.savez_compressed(
                        npz_path,
                        sonar_image=sonar_image,
                        ground_truth=ground_truth,
                        ts_unix_ns=np.array([timestamp_ns], dtype=np.int64),
                        extent=extent,
                        meta_json=json.dumps(metadata)
                    )
                
                frame_counter['count'] += 1
                
                # Progress update
                if (i + 1) % max(1, num_samples // 10) == 0:
                    progress = 100 * (i + 1) / num_samples
                    print(f"Progress: {i+1}/{num_samples} ({progress:.1f}%) - Position: {pos}")
        
        print(f"\n{'='*60}")
        print(f"Data collection complete!")
        print(f"Collected {frame_counter['count']} frames")
        print(f"{'='*60}\n")
        
        # CONSOLIDATE NPZ FILES INTO SINGLE FILE (SOLAQUA FORMAT)
        print("Consolidating frames into single NPZ file (SOLAQUA format)...")
        from src.utils.profiler import get_profiler
        profiler = get_profiler()
        
        with profiler.measure('consolidation_total'):
            frames_dir = save_dir / 'frames'
            individual_npz_files = sorted(frames_dir.glob('frame_*.npz'))
            
            if individual_npz_files:
                # Load all frames
                all_frames = []
                all_timestamps = []
                all_metadata = []
                first_extent = None
                
                with profiler.measure('consolidation_load_frames'):
                    for npz_file in individual_npz_files:
                        with np.load(npz_file, allow_pickle=True) as data:
                            all_frames.append(data['sonar_image'])
                            all_timestamps.append(data['ts_unix_ns'][0])  # Extract single timestamp
                            
                            if first_extent is None:
                                first_extent = data['extent']
                            
                            # Parse metadata
                            meta_str = data['meta_json']
                            meta_dict = json.loads(meta_str.item() if hasattr(meta_str, 'item') else str(meta_str))
                            all_metadata.append(meta_dict)
                
                # Stack frames into (T, H, W) array
                with profiler.measure('consolidation_stack'):
                    cones_stacked = np.stack(all_frames, axis=0).astype(np.float32)
                    raw_polar_stacked = np.stack(all_frames, axis=0).astype(np.float32)  # Same as cones for simulation
                    ts_stacked = np.array(all_timestamps, dtype=np.int64)
                
                # Create consolidated metadata
                consolidated_meta = {
                    'num_frames': len(all_frames),
                    'range_m': all_metadata[0]['range_m'],
                    'fov_deg': all_metadata[0]['fov_deg'],
                    'rmin': 0.0,  # Add SOLAQUA-expected fields
                    'rmax': all_metadata[0]['range_m'],
                    'display_range_max_m': all_metadata[0]['range_m'],
                    'collection_mode': collect_mode,
                    'frames': all_metadata,  # Include all individual frame metadata
                    'dim_labels': ['time', 'range_bins', 'beams'],
                    'raw_polar_shape': list(raw_polar_stacked.shape),
                }
                
                # Save consolidated NPZ file (matching real data format)
                consolidated_path = save_dir / f'{save_dir.name}_cones.npz'
                with profiler.measure('consolidation_save'):
                    np.savez_compressed(
                        consolidated_path,
                        cones=cones_stacked,           # (T, H, W) float32 array - PROCESSED
                        raw_polar=raw_polar_stacked,   # (T, range_bins, beams) float32 - RAW
                        extent=first_extent,            # (4,) float64 tuple
                        ts_unix_ns=ts_stacked,          # (T,) int64 array
                        meta_json=json.dumps(consolidated_meta)  # JSON string
                    )
                
                print(f"✓ Consolidated NPZ saved: {consolidated_path.name}")
                print(f"  Cones shape: {cones_stacked.shape}")
                print(f"  Raw polar shape: {raw_polar_stacked.shape}")
                print(f"  File size: {consolidated_path.stat().st_size / 1024 / 1024:.1f} MB")
                print(f"  Individual frames kept in: {frames_dir}")
            else:
                print("⚠ No individual frames found to consolidate")
        
        print(f"\n{'='*60}")
        print(f"All data saved to: {save_dir}")
        print(f"{'='*60}\n")
        
        # Print performance profile (if enabled)
        from src.utils.profiler import get_profiler
        profiler = get_profiler()
        if profiler.enabled:
            profiler.report(min_time=0.01, top_n=15)
        
        return
    
    # INTERACTIVE GUI MODE (original behavior)
    # Setup visualization
    from src.config import VISUALIZATION_CONFIG
    dt = VISUALIZATION_CONFIG['dt']
    
    fig, ax_sonar, ax_map, ax_gt = setup_figure(scene_type)
    
    # Create keyboard handler
    on_key = create_keyboard_handler(sonar, robot, grid, dynamic_objects, scene_module, 
                                     ax_sonar, ax_map, ax_gt, world_size, save_dir, frame_counter, dt)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display(sonar, robot, grid, dynamic_objects, scene_module, ax_sonar, ax_map, ax_gt, world_size, save_dir, frame_counter, dt)
    
    # Setup continuous animation
    anim = setup_animation(fig, sonar, robot, grid, dynamic_objects, scene_module, 
                          ax_sonar, ax_map, ax_gt, world_size, save_dir, frame_counter, dt)
    
    # Print controls
    print_controls()
    
    plt.tight_layout()
    plt.show()
    
    # Print performance profile after window closes (if enabled)
    if args.profile:
        profiler.report(min_time=0.01, top_n=15)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sonar simulation with different scenes')
    parser.add_argument('--scene', type=str, default='src.scenes.fish_cage_scene',
                       help='Scene module path (e.g., src.scenes.fish_cage_scene or src.scenes.street_scene)')
    parser.add_argument('--save', type=str, nargs='?', const=True, default=None,
                       help='Save run data. Optionally provide run name, otherwise uses timestamp.')
    parser.add_argument('--collect', type=str, choices=['circular', 'grid', 'random', 'spiral'], default=None,
                       help='Run headless data collection with specified path type (requires --save)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to collect in headless mode (default: 100)')
    parser.add_argument('--radius-variation', type=float, default=3.5,
                       help='Radius variation for circular path (default: 3.5)')
    parser.add_argument('--orientation-mode', type=str, choices=['inward', 'tangent', 'outward', 'mixed'], 
                       default='outward', help='Orientation mode for circular path (default: outward)')
    parser.add_argument('--orientation-noise', type=float, default=35.0,
                       help='Orientation noise in degrees for circular path (default: 35.0)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling and show report on exit')
    
    args = parser.parse_args()
    
    # Configure profiler
    from src.utils.profiler import get_profiler
    profiler = get_profiler()
    if not args.profile:
        profiler.disable()
    
    # Build path_kwargs from arguments
    path_kwargs = {
        'radius_variation': args.radius_variation,
        'orientation_mode': args.orientation_mode,
        'orientation_noise_deg': args.orientation_noise,
    }
    
    main(scene_path=args.scene, save_run=args.save, collect_mode=args.collect, 
         num_samples=args.num_samples, path_kwargs=path_kwargs)
