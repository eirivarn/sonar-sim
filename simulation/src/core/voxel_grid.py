"""Voxel grid for spatial representation of materials.

OVERVIEW:
---------
The VoxelGrid is a 2D grid that discretizes the world into small cells (voxels).
Each voxel stores material properties used for sonar ray marching simulation.

ARCHITECTURE:
-------------
The grid maintains four parallel 2D numpy arrays:
- density[x, y]: Material density (0-1)
- reflectivity[x, y]: Acoustic backscatter strength (0-1)  
- absorption[x, y]: Energy loss coefficient (0-1)
- material_id[x, y]: Ground truth label for segmentation (uint8)

RESOLUTION & COORDINATES:
------------------------
voxel_size determines spatial resolution (typically 0.1m = 10cm):
- World coordinates: meters (continuous, float)
- Voxel coordinates: grid indices (discrete, int)
- Conversion: voxel_index = int(world_position / voxel_size)

Example: 300x300 grid with voxel_size=0.1 represents a 30m x 30m world

SHAPE PRIMITIVES:
----------------
The grid provides methods to fill regions with materials:

1. set_circle(center, radius, material)
   - Fills circular region
   - Used for: fish, debris, simple objects
   
2. set_ellipse(center, radii, orientation, material)
   - Fills elliptical region with rotation
   - Used for: elongated fish, angled objects
   - Smart collision: won't overwrite static structures with dynamic objects
   
3. set_box(min_pos, max_pos, material)  
   - Fills rectangular region
   - Used for: buildings, walls, roads, vehicles
   
4. set_net_plane(x_pos, y_range, z_range, mesh_size, rope_thickness)
   - Creates fishing net mesh pattern
   - Sparse grid of rope with some net material

DYNAMIC OBJECT UPDATES:
----------------------
For animated objects, use clear_*() methods before redrawing:
- clear_fish(): Removes all fish material
- clear_debris(): Removes all debris materials
- clear_cars(): Removes all metal (vehicle) material

This allows objects to move each frame without leaving trails.

CLEARING MECHANISM:
Identifies material by reflectivity value (not perfect, but fast):
    mask = np.abs(self.reflectivity - FISH.reflectivity) < 0.01
    self.density[mask] = EMPTY.density  # Clear to empty

STATIC vs DYNAMIC:
------------------
set_ellipse() has special logic to prevent dynamic objects from overwriting
static structures:
- Dynamic: FISH, DEBRIS (should move)
- Static: NET, ROPE, WALL, BIOMASS (should not be overwritten)

This prevents fish from erasing the net cage as they swim.

USAGE IN SCENES:
----------------
Scene creation:
    grid = VoxelGrid(size_x=300, size_y=300, voxel_size=0.1)
    grid.set_box([0, 10], [50, 12], CONCRETE)  # Road
    grid.set_circle([15, 15], 0.4, FISH)       # Add fish

Scene updates (each frame):
    grid.clear_fish()  # Remove old positions
    for fish in fish_data:
        grid.set_ellipse(fish['pos'], fish['radii'], fish['angle'], FISH)

RELATIONSHIP TO OTHER MODULES:
-----------------------------
- materials.py: Defines Material objects stored in voxels
- sonar.py: Reads voxel properties during ray marching
- dynamics.py: Uses clear/set methods to update object positions
- scenes/*.py: Build and maintain grid content
"""
import numpy as np
from src.core.materials import Material, EMPTY, NET, ROPE, FISH, BIOMASS, WALL, DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY, METAL


class VoxelGrid:
    """2D voxel grid storing material density and properties."""
    
    def __init__(self, size_x: int, size_y: int, voxel_size: float = 0.1):
        """Initialize empty voxel grid.
        
        Args:
            size_x, size_y: Grid dimensions in voxels
            voxel_size: Size of each voxel in meters
        """
        self.size_x = size_x
        self.size_y = size_y
        self.voxel_size = voxel_size
        
        # Grid stores density and acoustic properties per voxel
        self.density = np.zeros((size_x, size_y), dtype=np.float32)
        self.reflectivity = np.zeros((size_x, size_y), dtype=np.float32)
        self.absorption = np.zeros((size_x, size_y), dtype=np.float32)
        self.material_id = np.zeros((size_x, size_y), dtype=np.uint8)  # Ground truth material IDs
    
    def world_to_voxel(self, pos: np.ndarray) -> tuple:
        """Convert world position to voxel indices."""
        x = int(pos[0] / self.voxel_size)
        y = int(pos[1] / self.voxel_size)
        return x, y
    
    def is_inside(self, x: int, y: int) -> bool:
        """Check if voxel indices are inside grid."""
        return 0 <= x < self.size_x and 0 <= y < self.size_y
    
    def set_circle(self, center: np.ndarray, radius: float, material: Material):
        """Fill circle with material."""
        cx, cy = self.world_to_voxel(center)
        r_voxels = int(radius / self.voxel_size)
        
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                if dx*dx + dy*dy <= r_voxels*r_voxels:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        self.density[x, y] = material.density
                        self.reflectivity[x, y] = material.reflectivity
                        self.absorption[x, y] = material.absorption
                        self.material_id[x, y] = material.material_id
    
    def set_ellipse(self, center: np.ndarray, radii: np.ndarray, orientation: float, material: Material):
        """Fill ellipse with material (elongated along orientation angle).
        
        Args:
            center: Center position [x, y]
            radii: Semi-axes lengths [length_along_orientation, width]
            orientation: Rotation angle (radians)
            material: Material to fill with
        """
        cx, cy = self.world_to_voxel(center)
        rx_vox = max(1, int(radii[0] / self.voxel_size))
        ry_vox = max(1, int(radii[1] / self.voxel_size))
        
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        
        max_r = max(rx_vox, ry_vox)
        
        # Check if this is a dynamic object that shouldn't overwrite static structures
        is_dynamic = material in [FISH, DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY]
        
        for dx in range(-max_r, max_r + 1):
            for dy in range(-max_r, max_r + 1):
                # Rotate back to ellipse space
                dx_rot = dx * cos_a + dy * sin_a
                dy_rot = -dx * sin_a + dy * cos_a
                
                # Check if inside ellipse
                dist_sq = (dx_rot/rx_vox)**2 + (dy_rot/ry_vox)**2
                
                if dist_sq <= 1.0:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        # If dynamic object, don't overwrite static structures
                        if is_dynamic:
                            existing_material_is_static = (
                                np.abs(self.reflectivity[x, y] - NET.reflectivity) < 0.01 or
                                np.abs(self.reflectivity[x, y] - ROPE.reflectivity) < 0.01 or
                                np.abs(self.reflectivity[x, y] - BIOMASS.reflectivity) < 0.01 or
                                np.abs(self.reflectivity[x, y] - WALL.reflectivity) < 0.01
                            )
                            if existing_material_is_static:
                                continue
                        
                        self.density[x, y] = material.density
                        self.reflectivity[x, y] = material.reflectivity
                        self.absorption[x, y] = material.absorption
                        self.material_id[x, y] = material.material_id
    
    def clear_fish(self):
        """Clear all fish material from grid (for dynamic updates)."""
        mask = np.abs(self.reflectivity - FISH.reflectivity) < 0.01
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
        self.material_id[mask] = EMPTY.material_id
    
    def clear_debris(self):
        """Clear all debris material from grid (for dynamic updates)."""
        mask = (np.abs(self.reflectivity - DEBRIS_LIGHT.reflectivity) < 0.01) | \
               (np.abs(self.reflectivity - DEBRIS_MEDIUM.reflectivity) < 0.01) | \
               (np.abs(self.reflectivity - DEBRIS_HEAVY.reflectivity) < 0.01)
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
        self.material_id[mask] = EMPTY.material_id
    
    def clear_cars(self):
        """Clear all car/metal material from grid (for dynamic updates)."""
        mask = (np.abs(self.reflectivity - METAL.reflectivity) < 0.01)
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
        self.material_id[mask] = EMPTY.material_id
    
    def set_box(self, min_pos: np.ndarray, max_pos: np.ndarray, material: Material):
        """Fill box region with material."""
        x1, y1 = self.world_to_voxel(min_pos)
        x2, y2 = self.world_to_voxel(max_pos)
        
        x1, x2 = max(0, min(x1, x2)), min(self.size_x, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(self.size_y, max(y1, y2))
        
        self.density[x1:x2, y1:y2] = material.density
        self.reflectivity[x1:x2, y1:y2] = material.reflectivity
        self.absorption[x1:x2, y1:y2] = material.absorption
        self.material_id[x1:x2, y1:y2] = material.material_id
    
    def set_net_plane(self, x_pos: float, y_range: tuple, z_range: tuple, 
                      mesh_size: float = 0.5, rope_thickness: float = 0.05):
        """Create a net plane (vertical mesh pattern)."""
        x_vox = int(x_pos / self.voxel_size)
        
        y_min, y_max = int(y_range[0] / self.voxel_size), int(y_range[1] / self.voxel_size)
        z_min, z_max = int(z_range[0] / self.voxel_size), int(z_range[1] / self.voxel_size)
        
        mesh_voxels = int(mesh_size / self.voxel_size)
        rope_voxels = int(rope_thickness / self.voxel_size)
        
        # Fill net pattern
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                if not self.is_inside(x_vox, y, z):
                    continue
                
                # Check if on rope line (grid pattern)
                y_mod = y % mesh_voxels
                z_mod = z % mesh_voxels
                
                on_rope = (y_mod < rope_voxels or z_mod < rope_voxels)
                
                if on_rope:
                    self.density[x_vox, y, z] = ROPE.density
                    self.reflectivity[x_vox, y, z] = ROPE.reflectivity
                    self.absorption[x_vox, y, z] = ROPE.absorption
                else:
                    # Net mesh (thinner)
                    if np.random.rand() < 0.3:
                        self.density[x_vox, y, z] = NET.density
                        self.reflectivity[x_vox, y, z] = NET.reflectivity
                        self.absorption[x_vox, y, z] = NET.absorption
