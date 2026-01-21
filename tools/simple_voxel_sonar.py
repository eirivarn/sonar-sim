"""Simple voxel-based sonar simulation with density field ray marching.

This is a clean reimplementation using volumetric ray marching:
- No surface raycasting
- Voxel grid with density/material properties
- Ray marching integrates scattering continuously
- Returns build up across volume
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass


@dataclass
class Material:
    """Material properties for voxel grid."""
    name: str
    density: float          # 0-1: how much matter is here
    reflectivity: float     # 0-1: acoustic backscatter strength
    absorption: float       # 0-1: how much energy is absorbed per meter
    

# Material library
EMPTY = Material("empty", 0.0, 0.0, 0.0)
NET = Material("net", 0.3, 0.2, 0.1)
ROPE = Material("rope", 0.8, 0.4, 0.2)
FISH = Material("fish", 0.7, 0.5, 0.3)  # Increased reflectivity from 0.3 to 0.5
WALL = Material("wall", 1.0, 0.5, 0.4)


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
    
    def set_ellipse(self, center: np.ndarray, radii: np.ndarray, orientation: float, material: Material):
        """Fill ellipse with material (elongated along orientation angle).
        
        Args:
            center: Center position [x, y]
            radii: Semi-axes lengths [length_along_orientation, width]
            orientation: Rotation angle (radians)
            material: Material to fill with
        """
        cx, cy = self.world_to_voxel(center)
        # radii[0] is length (along orientation), radii[1] is width
        rx_vox = max(1, int(radii[0] / self.voxel_size))  # Length along orientation
        ry_vox = max(1, int(radii[1] / self.voxel_size))  # Width perpendicular
        
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        
        max_r = max(rx_vox, ry_vox)
        
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
                        self.density[x, y] = material.density
                        self.reflectivity[x, y] = material.reflectivity
                        self.absorption[x, y] = material.absorption
    
    def clear_fish(self):
        """Clear all fish material from grid (for dynamic updates)."""
        # Use reflectivity to identify fish (FISH has 0.5, NET has 0.2, ROPE has 0.4)
        mask = np.abs(self.reflectivity - FISH.reflectivity) < 0.01
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
    
    def set_box(self, min_pos: np.ndarray, max_pos: np.ndarray, material: Material):
        """Fill box region with material."""
        x1, y1 = self.world_to_voxel(min_pos)
        x2, y2 = self.world_to_voxel(max_pos)
        
        x1, x2 = max(0, min(x1, x2)), min(self.size_x, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(self.size_y, max(y1, y2))
        
        self.density[x1:x2, y1:y2] = material.density
        self.reflectivity[x1:x2, y1:y2] = material.reflectivity
        self.absorption[x1:x2, y1:y2] = material.absorption
    
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
                    # Rope
                    self.density[x_vox, y, z] = ROPE.density
                    self.reflectivity[x_vox, y, z] = ROPE.reflectivity
                    self.absorption[x_vox, y, z] = ROPE.absorption
                else:
                    # Net mesh (thinner)
                    if np.random.rand() < 0.3:  # Sparse mesh
                        self.density[x_vox, y, z] = NET.density
                        self.reflectivity[x_vox, y, z] = NET.reflectivity
                        self.absorption[x_vox, y, z] = NET.absorption


class VoxelSonar:
    """Sonar using voxel ray marching (no surface raycasting)."""
    
    def __init__(self, position: np.ndarray, direction: np.ndarray,
                 range_m: float = 10.0, fov_deg: float = 90.0, num_beams: int = 128):
        """Initialize sonar.
        
        Args:
            position: Sonar position in world (meters)
            direction: Forward direction (normalized)
            range_m: Maximum range
            fov_deg: Field of view in degrees
            num_beams: Number of beams
        """
        self.position = position.copy()
        self.direction = direction / (np.linalg.norm(direction) + 1e-9)
        self.range_m = range_m
        self.fov_deg = fov_deg
        self.num_beams = num_beams
        self.range_bins = 256
    
    def scan(self, grid: VoxelGrid) -> np.ndarray:
        """Scan scene using volumetric ray marching.
        
        Returns:
            (range_bins, num_beams) array of accumulated returns
        """
        image = np.zeros((self.range_bins, self.num_beams), dtype=np.float32)
        
        fov_rad = np.deg2rad(self.fov_deg)
        
        for beam_idx in range(self.num_beams):
            # Beam direction
            t = beam_idx / (self.num_beams - 1) if self.num_beams > 1 else 0.5
            angle = (-fov_rad / 2) + t * fov_rad
            
            # Rotate direction by angle
            dir_angle = np.arctan2(self.direction[1], self.direction[0])
            beam_angle = dir_angle + angle
            beam_dir = np.array([np.cos(beam_angle), np.sin(beam_angle)])
            
            # Ray march through volume
            self._march_ray(grid, self.position, beam_dir, image[:, beam_idx])
        
        # TEMPORAL DECORRELATION: Additional frame-to-frame variability on objects only
        # Small random fluctuations representing net sway, water movement, etc.
        # Only apply where there's actual signal (not background)
        signal_mask = image > 1e-8
        decorrelation_noise = np.random.gamma(shape=5.0, scale=1.0/5.0, size=image.shape)
        image[signal_mask] *= decorrelation_noise[signal_mask]
        
        return image
    
    def _march_ray(self, grid: VoxelGrid, origin: np.ndarray, direction: np.ndarray,
                   output_bins: np.ndarray):
        """March ray through voxel grid, accumulating returns.
        
        This is the core volumetric integration - no surface hits!
        """
        # DDA-style voxel traversal
        step_size = grid.voxel_size * 0.5  # Sub-voxel steps for smooth integration
        num_steps = int(self.range_m / step_size)
        
        current_pos = origin.copy()
        energy = 1.0  # Energy budget
        
        for step in range(num_steps):
            distance = step * step_size
            if distance >= self.range_m or energy < 0.01:
                break
            
            # Get voxel properties at current position
            x, y = grid.world_to_voxel(current_pos)
            
            if not grid.is_inside(x, y):
                current_pos += direction * step_size
                continue
            
            density = grid.density[x, y]
            reflectivity = grid.reflectivity[x, y]
            absorption = grid.absorption[x, y]
            
            # VOLUME SCATTERING: Deposit return proportional to density and reflectivity
            if density > 0.01:
                # ACOUSTIC SPECKLE: Multiplicative noise from coherent interference
                # Real sonar sees interference between many sub-resolution scatterers
                # This creates Rayleigh/Rician distributed amplitude variations
                speckle = np.random.gamma(shape=1.2, scale=1.0/1.2)  # More speckle variation
                
                # ASPECT ANGLE VARIATION: Return strength varies with small angle changes
                # Model as random modulation representing micro-scale roughness/orientation
                aspect_variation = 0.5 + 0.8 * np.random.randn()  # Increased variation (~55% std dev)
                aspect_variation = np.clip(aspect_variation, 0.2, 2.0)  # Wider range
                
                # Backscatter strength with temporal variations
                scatter = energy * density * reflectivity * step_size * speckle * aspect_variation
                
                # Two-way propagation loss
                spreading_loss = 1.0 / (distance**2 + 1.0)
                water_absorption = np.exp(-0.05 * distance * 2 * 0.115)  # Two-way
                
                return_energy = scatter * spreading_loss * water_absorption
                
                # Deposit into range bin
                bin_idx = int((distance / self.range_m) * (len(output_bins) - 1))
                if 0 <= bin_idx < len(output_bins):
                    output_bins[bin_idx] += return_energy
            
            # ABSORPTION: Reduce forward energy
            if density > 0.01:
                energy *= np.exp(-absorption * step_size)
            
            # Move forward
            current_pos += direction * step_size
    
    def move(self, delta: np.ndarray):
        """Move sonar position."""
        self.position += delta
    
    def rotate(self, angle_deg: float):
        """Rotate sonar."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate in 2D
        new_x = self.direction[0] * cos_a - self.direction[1] * sin_a
        new_y = self.direction[0] * sin_a + self.direction[1] * cos_a
        self.direction = np.array([new_x, new_y])
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-9)


def create_demo_scene() -> VoxelGrid:
    """Create fish farm net cage with fish."""
    # Create 50m x 50m world at 10cm resolution
    grid = VoxelGrid(500, 500, voxel_size=0.1)
    
    # Net cage parameters (centered at 25, 25)
    cage_center = np.array([25.0, 25.0])
    cage_radius = 20.0  # 20m radius
    num_sides = 12      # Dodecagon cage
    
    # Create circular net panels
    for i in range(num_sides):
        angle1 = (i / num_sides) * 2 * np.pi
        angle2 = ((i + 1) / num_sides) * 2 * np.pi
        
        # Panel corners
        x1 = cage_center[0] + cage_radius * np.cos(angle1)
        y1 = cage_center[1] + cage_radius * np.sin(angle1)
        x2 = cage_center[0] + cage_radius * np.cos(angle2)
        y2 = cage_center[1] + cage_radius * np.sin(angle2)
        
        # Create net line for this panel
        for t in np.linspace(0, 1, 80):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Net mesh
            grid.set_box(
                np.array([x - 0.08, y - 0.08]),
                np.array([x + 0.08, y + 0.08]),
                NET
            )
            
            # Add occasional rope structure
            if t % (1.0 / 7) < 0.02:
                grid.set_box(
                    np.array([x - 0.12, y - 0.12]),
                    np.array([x + 0.12, y + 0.12]),
                    ROPE
                )
    
    # Add fish scattered throughout cage - store as dynamic objects
    np.random.seed(42)  # Reproducible fish positions
    num_fish = 400
    
    # Store fish data for animation
    fish_data = []
    for _ in range(num_fish):
        # Random position within cage (prefer near perimeter)
        angle = np.random.rand() * 2 * np.pi
        # Bias toward perimeter (0.6-0.95 of radius)
        r_fraction = 0.6 + 0.35 * np.random.rand()
        r = cage_radius * r_fraction
        
        x = cage_center[0] + r * np.cos(angle)
        y = cage_center[1] + r * np.sin(angle)
        
        # Random swimming direction and speed
        swim_angle = np.random.rand() * 2 * np.pi
        swim_speed = 0.08 + np.random.rand() * 0.12  # 8-20 cm/frame
        vx = swim_speed * np.cos(swim_angle)
        vy = swim_speed * np.sin(swim_angle)
        
        # Fish size
        fish_length = 0.4 + np.random.rand() * 0.4  # 40-80cm long
        fish_width = fish_length * 0.25  # Width is 25% of length
        
        fish_data.append({
            'pos': np.array([x, y]),
            'vel': np.array([vx, vy]),
            'orientation': swim_angle,
            'radii': np.array([fish_length, fish_width]),
            'turn_timer': np.random.rand() * 100  # Random phase for turning
        })
    
    # Initial fish rendering
    for fish in fish_data:
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)
    
    # Debug: Count how many fish voxels were created
    fish_voxel_count = np.sum(np.abs(grid.reflectivity - FISH.reflectivity) < 0.01)
    print(f"Fish voxels created: {fish_voxel_count}")
    
    # Return both grid and fish data
    return grid, fish_data


def update_fish(grid: VoxelGrid, fish_data: list, cage_center: np.ndarray, cage_radius: float):
    """Update fish positions and redraw them in the grid."""
    # Clear existing fish
    grid.clear_fish()
    
    # Update each fish
    for fish in fish_data:
        # Update position
        fish['pos'] += fish['vel']
        
        # Occasional direction changes
        fish['turn_timer'] -= 1
        if fish['turn_timer'] <= 0:
            # Turn toward cage center with some randomness
            to_center = cage_center - fish['pos']
            to_center_angle = np.arctan2(to_center[1], to_center[0])
            
            # Add randomness
            new_angle = to_center_angle + (np.random.rand() - 0.5) * np.pi
            
            swim_speed = np.linalg.norm(fish['vel'])
            fish['vel'][0] = swim_speed * np.cos(new_angle)
            fish['vel'][1] = swim_speed * np.sin(new_angle)
            fish['orientation'] = new_angle
            
            fish['turn_timer'] = 50 + np.random.rand() * 100
        
        # Keep fish inside cage bounds
        dx = fish['pos'][0] - cage_center[0]
        dy = fish['pos'][1] - cage_center[1]
        dist_from_center = np.sqrt(dx*dx + dy*dy)
        
        if dist_from_center > cage_radius - 1.0:
            # Bounce back toward center
            angle_to_center = np.arctan2(-dy, -dx)
            swim_speed = np.linalg.norm(fish['vel'])
            fish['vel'][0] = swim_speed * np.cos(angle_to_center)
            fish['vel'][1] = swim_speed * np.sin(angle_to_center)
            fish['orientation'] = angle_to_center
        
        # Draw fish at new position
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)
    
    print(f"Created fish cage:")
    print(f"  Center: {cage_center}")
    print(f"  Radius: {cage_radius}m")
    print(f"  Fish: {len(fish_data)}")
    
    return grid, fish_data


def main():
    """Run interactive voxel sonar viewer."""
    print("Building fish farm cage scene (this may take a moment)...")
    grid, fish_data = create_demo_scene()
    
    # Cage parameters for fish updates
    cage_center = np.array([25.0, 25.0])
    cage_radius = 20.0
    
    print("Initializing sonar...")
    # Position sonar inside cage, near center where fish are
    sonar = VoxelSonar(
        position=np.array([25.0, 25.0]),  # Inside cage center
        direction=np.array([1.0, 0.0]),     # Looking outward
        range_m=10.0,
        fov_deg=120.0,
        num_beams=180
    )
    
    # Create sonar figure
    fig_sonar, ax_sonar = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    plt.figure(fig_sonar.number)
    plt.subplots_adjust(bottom=0.15)
    
    # Create map figure
    fig_map, ax_map = plt.subplots(figsize=(8, 8))
    fig_map.canvas.manager.set_window_title('World Map')
    
    # Store cage parameters for map display
    num_sides = 12
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
    
    
    def update_display():
        """Update sonar and map displays."""
        # Update fish positions
        update_fish(grid, fish_data, cage_center, cage_radius)
        
        # Extract current fish positions for map
        fish_positions = np.array([[f['pos'][0], f['pos'][1]] for f in fish_data])
        
        print(f"Scanning from position {sonar.position}...")
        image = sonar.scan(grid)
        
        # Display polar plot
        ax_sonar.clear()
        
        fov_rad = np.deg2rad(sonar.fov_deg)
        angles = np.linspace(-fov_rad/2, fov_rad/2, sonar.num_beams)
        ranges = np.linspace(0, sonar.range_m, image.shape[0])
        
        theta, r = np.meshgrid(angles, ranges)
        
        # Log compression for display (no background noise)
        image_db = 10 * np.log10(np.maximum(image, 1e-10))  # Avoid log(0)
        image_db = np.clip((image_db + 60) / 60, 0, 1)  # Normalize
        
        ax_sonar.contourf(theta, r, image_db, levels=20, cmap='gray')
        ax_sonar.set_theta_zero_location('N')
        ax_sonar.set_theta_direction(1)
        ax_sonar.set_thetamin(-sonar.fov_deg/2)
        ax_sonar.set_thetamax(sonar.fov_deg/2)
        ax_sonar.set_ylim(0, sonar.range_m)
        ax_sonar.set_title(f'Voxel Sonar - Pos: [{sonar.position[0]:.1f}, {sonar.position[1]:.1f}]')
        
        # Update map display
        ax_map.clear()
        
        # Draw cage outline (polygon)
        cage_x = []
        cage_y = []
        for i in range(num_sides + 1):
            angle = (i / num_sides) * 2 * np.pi
            cage_x.append(cage_center[0] + cage_radius * np.cos(angle))
            cage_y.append(cage_center[1] + cage_radius * np.sin(angle))
        ax_map.plot(cage_x, cage_y, 'b-', linewidth=2, label='Cage')
        
        # Draw fish
        ax_map.scatter(fish_positions[:, 0], fish_positions[:, 1], c='orange', s=3, alpha=0.6, label='Fish')
        
        # Draw sonar position and direction
        ax_map.scatter(sonar.position[0], sonar.position[1], c='red', s=100, marker='^', label='Sonar', zorder=5)
        
        # Draw sonar direction vector
        arrow_length = 3.0
        ax_map.arrow(sonar.position[0], sonar.position[1], 
                     sonar.direction[0] * arrow_length, sonar.direction[1] * arrow_length,
                     head_width=0.8, head_length=0.5, fc='red', ec='red', zorder=5)
        
        # Draw FOV cone
        fov_rad = np.deg2rad(sonar.fov_deg)
        # Rotate direction by +/- fov/2
        dir_angle = np.arctan2(sonar.direction[1], sonar.direction[0])
        left_angle = dir_angle + fov_rad / 2
        right_angle = dir_angle - fov_rad / 2
        
        cone_length = sonar.range_m * 0.5
        left_x = sonar.position[0] + cone_length * np.cos(left_angle)
        left_y = sonar.position[1] + cone_length * np.sin(left_angle)
        right_x = sonar.position[0] + cone_length * np.cos(right_angle)
        right_y = sonar.position[1] + cone_length * np.sin(right_angle)
        
        ax_map.plot([sonar.position[0], left_x], [sonar.position[1], left_y], 'r--', alpha=0.3, linewidth=1)
        ax_map.plot([sonar.position[0], right_x], [sonar.position[1], right_y], 'r--', alpha=0.3, linewidth=1)
        
        ax_map.set_xlim(0, 50)
        ax_map.set_ylim(0, 50)
        ax_map.set_aspect('equal')
        ax_map.set_xlabel('X (m)')
        ax_map.set_ylabel('Y (m)')
        ax_map.set_title('World Map (Top View)')
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(loc='upper right')
        
        fig_sonar.canvas.draw()
        fig_map.canvas.draw()
    
    def on_key(event):
        """Handle keyboard input."""
        move_speed = 0.5
        rotate_speed = 10
        
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
            sonar.rotate(rotate_speed)
        elif event.key == 'right':
            sonar.rotate(-rotate_speed)
        else:
            return
        
        update_display()
    
    fig_sonar.canvas.mpl_connect('key_press_event', on_key)
    fig_map.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    # Start continuous animation to show flickering
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig_sonar, lambda frame: update_display() or [], interval=100, cache_frame_data=False)
    
    print("\nControls:")
    print("  W/S: Move forward/back")
    print("  A/D: Move left/right")
    print("  Left/Right arrows: Rotate")
    print("\nNote: Image flickers continuously due to:")
    print("  - Acoustic speckle (coherent interference)")
    print("  - Aspect angle variations (micro-scale roughness)")
    print("  - Temporal decorrelation (net sway, water movement)")
    print("  - Receiver noise (electronic/thermal)")
    
    plt.show()


if __name__ == '__main__':
    main()
