"""Scene definitions for sonar simulation.

Each scene module should define:
- create_scene() -> dict: Returns scene configuration
- update_scene(grid, scene_data) -> None: Updates dynamic objects

Scene dict structure:
{
    'grid': VoxelGrid,
    'sonar_start_pos': np.ndarray,
    'sonar_start_dir': np.ndarray,
    'sonar_range': float,
    'world_size': float,
    'scene_type': str,
    'dynamic_objects': dict,  # Scene-specific dynamic objects
    'map_renderer': callable,  # Function to render map view
}
"""
