"""Street scene with houses, trees, and moving cars."""
import numpy as np
import sys
from pathlib import Path

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from voxel_grid import VoxelGrid
from materials import CONCRETE, WOOD, FOLIAGE, METAL, GLASS
from dynamics import update_cars
from config import STREET_SCENE_CONFIG


def create_scene():
    """Create a street scene with houses, trees, and moving cars.
    
    Returns:
        dict with scene configuration
    """
    world_size = STREET_SCENE_CONFIG['world_size_m']
    street_width = STREET_SCENE_CONFIG['street_width']
    sidewalk_width = STREET_SCENE_CONFIG['sidewalk_width']
    num_houses = STREET_SCENE_CONFIG['num_houses']
    num_trees = STREET_SCENE_CONFIG['num_trees']
    num_cars = STREET_SCENE_CONFIG['num_cars']
    
    # Create larger grid (500x500 voxels = 50m x 50m)
    grid = VoxelGrid(500, 500, voxel_size=0.1)
    
    # === STREET AND SIDEWALKS ===
    street_center_y = world_size / 2
    street_y1 = street_center_y - street_width / 2
    street_y2 = street_center_y + street_width / 2
    
    # Concrete road
    grid.set_box(
        np.array([0, street_y1]),
        np.array([world_size, street_y2]),
        CONCRETE
    )
    
    # Sidewalks
    sidewalk_north_y1 = street_y1 - sidewalk_width
    sidewalk_north_y2 = street_y1
    sidewalk_south_y1 = street_y2
    sidewalk_south_y2 = street_y2 + sidewalk_width
    
    grid.set_box(
        np.array([0, sidewalk_north_y1]),
        np.array([world_size, sidewalk_north_y2]),
        CONCRETE
    )
    grid.set_box(
        np.array([0, sidewalk_south_y1]),
        np.array([world_size, sidewalk_south_y2]),
        CONCRETE
    )
    
    # === HOUSES ===
    house_spacing = world_size / (num_houses / 2 + 1)
    
    for i in range(num_houses):
        if i < num_houses // 2:
            # North side
            house_x = (i + 1) * house_spacing
            house_y = sidewalk_north_y1 - 3.0
            side_offset = -5.0
        else:
            # South side
            house_x = (i - num_houses // 2 + 1) * house_spacing
            house_y = sidewalk_south_y2 + 3.0
            side_offset = 5.0
        
        # House dimensions
        house_width = 4.0 + np.random.rand() * 2.0
        house_depth = 5.0 + np.random.rand() * 3.0
        
        # Main building
        grid.set_box(
            np.array([house_x - house_width/2, house_y + side_offset - house_depth/2]),
            np.array([house_x + house_width/2, house_y + side_offset + house_depth/2]),
            CONCRETE
        )
        
        # Windows
        for window_idx in range(2):
            window_x = house_x - house_width/3 + window_idx * house_width/2
            window_size = 0.6
            grid.set_box(
                np.array([window_x - window_size/2, house_y + side_offset - house_depth/4]),
                np.array([window_x + window_size/2, house_y + side_offset + house_depth/4]),
                GLASS
            )
        
        # Chimney
        chimney_size = 0.5
        grid.set_box(
            np.array([house_x - chimney_size/2, house_y + side_offset + house_depth/3]),
            np.array([house_x + chimney_size/2, house_y + side_offset + house_depth/2]),
            WOOD
        )
    
    # === TREES ===
    for i in range(num_trees):
        tree_x = np.random.rand() * world_size
        
        if np.random.rand() < 0.5:
            tree_y = sidewalk_north_y1 - np.random.rand() * 8.0 - 1.0
        else:
            tree_y = sidewalk_south_y2 + np.random.rand() * 8.0 + 1.0
        
        # Trunk
        trunk_radius = 0.3
        grid.set_circle(np.array([tree_x, tree_y]), trunk_radius, WOOD)
        
        # Canopy
        canopy_radius = 1.2 + np.random.rand() * 0.8
        grid.set_circle(np.array([tree_x, tree_y]), canopy_radius, FOLIAGE)
    
    # === CARS ===
    car_data = []
    for i in range(num_cars):
        car_x = np.random.rand() * world_size
        
        # Two lanes
        if np.random.rand() < 0.5:
            # Eastbound (left to right)
            car_y = street_center_y - street_width / 4
            velocity = np.array([np.random.uniform(*STREET_SCENE_CONFIG['car_speed_range']), 0.0])
        else:
            # Westbound (right to left)
            car_y = street_center_y + street_width / 4
            velocity = np.array([-np.random.uniform(*STREET_SCENE_CONFIG['car_speed_range']), 0.0])
        
        car_length = 3.5
        car_width = 1.8
        
        car_data.append({
            'pos': np.array([car_x, car_y]),
            'vel': velocity,
            'length': car_length,
            'width': car_width,
            'material': METAL
        })
    
    # Render cars initially
    for car in car_data:
        grid.set_box(
            np.array([car['pos'][0] - car['length']/2, car['pos'][1] - car['width']/2]),
            np.array([car['pos'][0] + car['length']/2, car['pos'][1] + car['width']/2]),
            car['material']
        )
    
    print(f"Created street scene: {num_houses} houses, {num_trees} trees, {num_cars} cars")
    
    return {
        'grid': grid,
        'world_size': world_size,
        'scene_type': 'street',
        'sonar_start_pos': np.array([world_size/2, world_size/2 - 4.0]),
        'sonar_start_dir': np.array([0.0, 1.0]),
        'dynamic_objects': {
            'car_data': car_data,
        }
    }


def update_scene(grid, scene_data, sonar_pos):
    """Update car positions in the scene."""
    car_data = scene_data['car_data']
    world_size = STREET_SCENE_CONFIG['world_size_m']
    
    update_cars(grid, car_data, world_size)


def render_map(ax, scene_data, sonar):
    """Render the street map view."""
    import matplotlib.pyplot as plt
    
    world_size = STREET_SCENE_CONFIG['world_size_m']
    street_width = STREET_SCENE_CONFIG['street_width']
    street_center_y = world_size / 2
    car_data = scene_data['car_data']
    car_data = scene_data['car_data']
    world_size = STREET_SCENE_CONFIG['world_size_m']
    
    # Draw street
    ax.add_patch(plt.Rectangle(
        (0, street_center_y - street_width/2),
        world_size, street_width,
        facecolor='gray', alpha=0.3, label='Street'
    ))
    
    # Draw cars
    for car in car_data:
        ax.add_patch(plt.Rectangle(
            (car['pos'][0] - car['length']/2, car['pos'][1] - car['width']/2),
            car['length'], car['width'],
            facecolor='blue', alpha=0.7
        ))
