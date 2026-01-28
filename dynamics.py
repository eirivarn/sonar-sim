"""Dynamic object update functions for simulation."""
import numpy as np
from voxel_grid import VoxelGrid
from materials import FISH, EMPTY


def update_fish(grid: VoxelGrid, fish_data: list, cage_center: np.ndarray, cage_radius: float, sonar_pos: np.ndarray):
    """Update fish positions and redraw them in the grid."""
    # Clear existing fish
    grid.clear_fish()
    
    # Species behavior parameters
    # A: Schooling (strong same-species attraction)
    # B: Solitary (avoid all fish)
    # C: Mixed (moderate attraction to all)
    behavior = {
        'A': {'same_attract': 2.0, 'other_attract': 0.1, 'avoid': 0.8, 'sonar_avoid': 1.0},
        'B': {'same_attract': 0.0, 'other_attract': 0.0, 'avoid': 2.0, 'sonar_avoid': 2.5},
        'C': {'same_attract': 0.8, 'other_attract': 0.6, 'avoid': 1.0, 'sonar_avoid': 1.8}
    }
    
    # Update each fish
    for i, fish in enumerate(fish_data):
        # Update position
        fish['pos'] += fish['vel']
        
        # Flocking behavior (computed every frame for responsiveness)
        species = fish['species']
        avoid_vec = np.zeros(2)
        same_attract_vec = np.zeros(2)
        other_attract_vec = np.zeros(2)
        same_count = 0
        other_count = 0
        
        # Check nearby fish (within 3m for efficiency)
        for j, other in enumerate(fish_data):
            if i == j:
                continue
            
            diff = other['pos'] - fish['pos']
            dist = np.linalg.norm(diff)
            
            if dist < 3.0 and dist > 0.01:
                # Avoidance (short range)
                if dist < 0.8:
                    avoid_vec -= diff / (dist * dist + 0.1)
                
                # Attraction (medium range)
                if dist < 3.0:
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
        
        # SONAR AVOIDANCE: Fish flee from the robot/sonar
        diff_from_sonar = fish['pos'] - sonar_pos
        dist_from_sonar = np.linalg.norm(diff_from_sonar)
        
        if dist_from_sonar < 5.0 and dist_from_sonar > 0.01:  # Within 5m detection range
            # Flee away from sonar with inverse square law
            flee_strength = params['sonar_avoid'] / (dist_from_sonar**2 + 0.5)
            steer += flee_strength * diff_from_sonar / dist_from_sonar
        
        # PERIMETER PREFERENCE: Fish naturally want to swim toward outer net area
        # This simulates fish preferring to be near the net structure
        dx = fish['pos'][0] - cage_center[0]
        dy = fish['pos'][1] - cage_center[1]
        dist_from_center = np.sqrt(dx*dx + dy*dy)
        
        # Target the outer perimeter (0.85 of radius - close but not at the net)
        target_radius = cage_radius * 0.85
        if dist_from_center > 0.01:
            # If too far inside, gently push toward perimeter
            if dist_from_center < target_radius * 0.7:
                to_perimeter = np.array([dx, dy]) / dist_from_center
                # Gentle attraction to outer area (0.4 strength)
                steer += to_perimeter * 0.4
        
        # Add small random component
        steer += (np.random.rand(2) - 0.5) * 0.3
        
        # Apply steering with momentum
        if np.linalg.norm(steer) > 0.01:
            fish['vel'] += steer * 0.03
            
            # Limit speed
            speed = np.linalg.norm(fish['vel'])
            max_speed = 0.2
            min_speed = 0.05
            if speed > max_speed:
                fish['vel'] = fish['vel'] / speed * max_speed
            elif speed < min_speed:
                fish['vel'] = fish['vel'] / speed * min_speed
            
            # Update orientation
            fish['orientation'] = np.arctan2(fish['vel'][1], fish['vel'][0])
        
        # Keep fish inside stretched cage bounds (matching current deflection)
        # Calculate what the boundary position should be at this fish's angle
        angle = np.arctan2(dy, dx)
        
        # Calculate deflected boundary at this angle (matching net deflection)
        boundary_x_base = cage_center[0] + cage_radius * np.cos(angle)
        boundary_y_base = cage_center[1] + cage_radius * np.sin(angle)
        
        # Apply same current deflection as net
        current_direction = np.array([0.0, 1.0])  # Southward
        current_strength = 6.5
        deflection = current_strength * max(0, (boundary_y_base - cage_center[1]) / cage_radius) * current_direction
        lateral_factor = np.sin(angle) * 0.4
        deflection += np.array([lateral_factor * current_strength * 0.3, 0])
        
        boundary_x = boundary_x_base + deflection[0]
        boundary_y = boundary_y_base + deflection[1]
        
        # Check if fish is outside deflected boundary
        dx_to_boundary = fish['pos'][0] - boundary_x
        dy_to_boundary = fish['pos'][1] - boundary_y
        # Check if fish is beyond boundary in the radial direction
        if dx_to_boundary * dx > 0 or dy_to_boundary * dy > 0:
            if dist_from_center > cage_radius - 1.0:
                # Bounce back toward cage center
                angle_to_center = np.arctan2(-dy, -dx)
                swim_speed = np.linalg.norm(fish['vel'])
                fish['vel'][0] = swim_speed * np.cos(angle_to_center)
                fish['vel'][1] = swim_speed * np.sin(angle_to_center)
                fish['orientation'] = angle_to_center
        
        # Draw fish at new position
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)


def update_debris(grid: VoxelGrid, debris_data: list, cage_center: np.ndarray, cage_radius: float):
    """Update debris positions and redraw them in the grid."""
    # Clear existing debris
    grid.clear_debris()
    
    # Update each debris piece
    for debris in debris_data:
        # Add random turbulence/drift
        turbulence = (np.random.rand(2) - 0.5) * 0.008  # Small random drift
        debris['vel'] += turbulence
        
        # Limit drift speed
        speed = np.linalg.norm(debris['vel'])
        max_speed = 0.05
        if speed > max_speed:
            debris['vel'] = debris['vel'] / speed * max_speed
        
        # Update position
        debris['pos'] += debris['vel']
        
        # Keep debris inside cage bounds (bounce off walls)
        dx = debris['pos'][0] - cage_center[0]
        dy = debris['pos'][1] - cage_center[1]
        dist_from_center = np.sqrt(dx*dx + dy*dy)
        
        if dist_from_center > cage_radius - 0.5:
            # Reflect velocity off wall
            normal = np.array([dx, dy]) / (dist_from_center + 0.001)
            debris['vel'] = debris['vel'] - 2 * np.dot(debris['vel'], normal) * normal
            # Also push back inside slightly
            debris['pos'] = cage_center + normal * (cage_radius - 0.5)
        
        # Draw debris at new position
        grid.set_circle(debris['pos'], debris['size'], debris['material'])


def update_cars(grid: VoxelGrid, car_data: list, world_size: float):
    """Update car positions and redraw them in the grid."""
    # Clear existing cars
    grid.clear_cars()
    
    # Update each car
    for car in car_data:
        # Update position
        car['pos'] += car['vel']
        
        # Wrap around at world boundaries (cars loop around)
        if car['pos'][0] < 0:
            car['pos'][0] += world_size
        elif car['pos'][0] > world_size:
            car['pos'][0] -= world_size
        
        # Draw car at new position
        grid.set_box(
            np.array([car['pos'][0] - car['length']/2, car['pos'][1] - car['width']/2]),
            np.array([car['pos'][0] + car['length']/2, car['pos'][1] + car['width']/2]),
            car['material']
        )
