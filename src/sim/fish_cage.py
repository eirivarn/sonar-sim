"""Fish farm cage primitives with nets and fish."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .primitives import Primitive, Hit
from .config import FishConfig

@dataclass
class NetCage(Primitive):
    """Polygonal fish farm cage with flat net panels.
    
    The cage is made of straight vertical panels (polygon) rather than a smooth cylinder.
    Each panel is defined by vertical ropes at the corners, creating a more realistic
    cage structure.
    """
    obj_id: str
    center: np.ndarray      # Center point (x, y) at water surface (z=0)
    radius_top: float       # Radius at the top (surface)
    radius_bottom: float    # Radius at the bottom (can be smaller for tapering)
    depth: float            # Depth of the cage
    num_sides: int = 20     # Number of straight panels (polygon sides)
    mesh_size: float = 0.5  # Distance between rope sections
    rope_thickness: float = 0.02  # Thickness of structural ropes
    net_reflectivity: float = 0.25  # Reflectivity of net mesh
    rope_reflectivity: float = 0.45  # Reflectivity of structural ropes
    has_bottom: bool = True  # Whether the cage has a bottom net
    sag_factor: float = 0.1  # How much the net sags between ropes (0-1)
    
    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """Ray intersection with net cage structure."""
        # Check intersection with polygonal net walls (flat panels)
        hit = self._intersect_polygon_net(ro, rd)
        
        # Check bottom net if present
        if self.has_bottom and hit is None:
            bottom_hit = self._intersect_bottom_net(ro, rd)
            if bottom_hit is not None:
                return bottom_hit
        
        return hit
    
    def _intersect_polygon_net(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """Intersect with vertical net panels (polygon cage)."""
        # Generate polygon vertices at top and bottom
        angles = np.linspace(0, 2 * np.pi, self.num_sides + 1)[:-1]  # Don't duplicate last point
        
        closest_hit = None
        closest_t = float('inf')
        
        # Check each vertical panel
        for i in range(self.num_sides):
            angle1 = angles[i]
            angle2 = angles[(i + 1) % self.num_sides]
            
            # Panel corners at top
            top1 = self.center + np.array([
                self.radius_top * np.cos(angle1),
                self.radius_top * np.sin(angle1),
                0.0
            ])
            top2 = self.center + np.array([
                self.radius_top * np.cos(angle2),
                self.radius_top * np.sin(angle2),
                0.0
            ])
            
            # Panel corners at bottom
            bottom1 = self.center + np.array([
                self.radius_bottom * np.cos(angle1),
                self.radius_bottom * np.sin(angle1),
                -self.depth
            ])
            bottom2 = self.center + np.array([
                self.radius_bottom * np.cos(angle2),
                self.radius_bottom * np.sin(angle2),
                -self.depth
            ])
            
            # Intersect with this panel (vertical quadrilateral)
            hit = self._intersect_panel(ro, rd, top1, top2, bottom1, bottom2, angle1, angle2)
            
            if hit is not None and hit.t < closest_t:
                closest_t = hit.t
                closest_hit = hit
        
        return closest_hit
    
    def _intersect_panel(self, ro: np.ndarray, rd: np.ndarray,
                        top1: np.ndarray, top2: np.ndarray,
                        bottom1: np.ndarray, bottom2: np.ndarray,
                        angle1: float, angle2: float) -> Hit | None:
        """Intersect with a single vertical panel (quadrilateral net segment)."""
        # Panel center for reference
        panel_center = (top1 + top2 + bottom1 + bottom2) / 4.0
        
        # Calculate normal from cage center (more robust)
        to_panel_center = panel_center[:2] - self.center[:2]
        if np.linalg.norm(to_panel_center) < 1e-9:
            return None
        
        # Normal points outward from cage center in XY plane
        normal_xy = to_panel_center / np.linalg.norm(to_panel_center)
        normal = np.array([normal_xy[0], normal_xy[1], 0.0])
        
        # Plane intersection (use top1 as reference point)
        denom = np.dot(normal, rd)
        if abs(denom) < 1e-6:
            return None
        
        t = np.dot(normal, (top1 - ro)) / denom
        if t <= 0.001 or t > 150.0:
            return None
        
        # Hit point
        p = ro + t * rd
        
        # Check if point is within panel bounds
        # Project onto panel plane using 2D coordinates
        panel_x = top2 - top1
        panel_width = np.linalg.norm(panel_x)
        if panel_width < 1e-9:
            return None
        panel_x = panel_x / panel_width
        
        # Point relative to top1
        rel = p - top1
        u = np.dot(rel, panel_x)  # Horizontal position along panel (0 to panel_width)
        v = p[2] - top1[2]  # Vertical position: negative means below top1 (0 to -depth)
        
        # Check bounds
        if u < -0.1 or u > panel_width + 0.1:
            return None
        if v > 0.1 or v < -self.depth - 0.1:
            return None
        
        # Calculate position on panel for mesh pattern
        angle_mid = (angle1 + angle2) / 2.0
        angle_deg = np.degrees(angle_mid) % 360
        z_rel = -v  # Depth below surface (positive value)
        
        # Check if on vertical structural ropes (at panel edges)
        is_on_vertical_rope = (u < self.rope_thickness or 
                               u > panel_width - self.rope_thickness)
        
        # Check if on horizontal rope
        num_horizontal_ropes = int(self.depth / self.mesh_size)
        nearest_h_rope_depth = round(z_rel / self.mesh_size) * self.mesh_size
        dist_to_h_rope = abs(z_rel - nearest_h_rope_depth)
        is_on_horizontal_rope = dist_to_h_rope < self.rope_thickness
        
        # STRUCTURAL ROPES: top band, bottom band
        top_band_thickness = 0.3
        bottom_band_thickness = 0.4
        
        is_on_top_band = z_rel < top_band_thickness
        is_on_bottom_band = abs(z_rel - self.depth) < bottom_band_thickness
        
        is_on_rope = (is_on_vertical_rope or is_on_horizontal_rope or
                     is_on_top_band or is_on_bottom_band)
        
        # Mesh pattern - not all rays hit the net
        if not is_on_rope:
            mesh_hit_prob = 0.4
            hash_val = ((u * 100) % 1.0 + (z_rel * 100) % 1.0) / 2.0
            if hash_val > mesh_hit_prob:
                return None  # Ray passed through mesh hole
        
        # Add sagging effect (minimal for vertical panels)
        # Sag is maximum at panel center horizontally
        if not is_on_rope:
            sag_offset = self.sag_factor * self.mesh_size * 0.5
            horizontal_center_dist = abs(u - panel_width / 2.0)
            sag_factor_local = 1.0 - (horizontal_center_dist / (panel_width / 2.0 + 0.001))
            sag_amount = sag_offset * sag_factor_local
            # Sag inward (toward center)
            p -= normal * sag_amount
            t = np.linalg.norm(p - ro)
        
        reflectivity = self.rope_reflectivity if is_on_rope else self.net_reflectivity
        
        # BIOFOULING: modulate reflectivity with spatial noise
        biofouling_factor = self._biofouling_modulation(angle_deg, z_rel)
        reflectivity *= biofouling_factor
        
        return Hit(t=float(t), point=p, normal=normal,
                  obj_id=self.obj_id, reflectivity=reflectivity)
    
    def _intersect_cylinder_net(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """DEPRECATED: Old cylinder intersection method."""
        # This method is no longer used - kept for reference
        return None
    
    def _biofouling_modulation(self, angle_deg: float, z_rel: float, time: float = 0.0) -> float:
        """Calculate biofouling reflectivity modulation using deterministic noise.
        
        Args:
            angle_deg: Angular position around cage (0-360)
            z_rel: Depth relative to top of cage (0 to depth)
            time: Optional time parameter for evolution (default 0)
            
        Returns:
            Multiplicative factor for reflectivity (0.7 to 1.6)
        """
        # Deterministic noise based on position
        def noise01(x, y, seed=42):
            """Simple hash-based noise returning [0, 1]."""
            h = int((x * 127.1 + y * 311.7 + seed * 758.5) * 43758.5453) % 100000
            return (h % 10000) / 10000.0
        
        # Multi-scale noise for patchy appearance
        n1 = noise01(angle_deg / 30.0, z_rel / 2.0, 42)
        n2 = noise01(angle_deg / 10.0, z_rel / 0.5, 123)
        combined_noise = (n1 * 0.7 + n2 * 0.3)
        
        # Depth weighting - more fouling near surface (upper 3m)
        depth_weight = np.exp(-z_rel / 3.0)
        
        # Also add deeper band (mid-depth fouling)
        mid_depth_weight = np.exp(-abs(z_rel - self.depth * 0.5) / 2.0) * 0.3
        
        total_depth_weight = depth_weight + mid_depth_weight
        
        # Modulation strength
        k = 0.6 * total_depth_weight
        
        # Calculate factor: baseline 1.0, add variation
        factor = 1.0 + k * (combined_noise - 0.5)
        
        # Clamp to reasonable range
        return np.clip(factor, 0.7, 1.6)
    
    def _intersect_bottom_net(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """Intersect with the bottom net (horizontal plane with mesh)."""
        bottom_z = self.center[2] - self.depth
        
        # Plane intersection
        if abs(rd[2]) < 1e-9:
            return None
        
        t = (bottom_z - ro[2]) / rd[2]
        if t <= 0.001:
            return None
        
        hit_point = ro + t * rd
        
        # Check if within bottom radius (tapered)
        dist_from_center = np.linalg.norm(hit_point[:2] - self.center[:2])
        if dist_from_center > self.radius_bottom:
            return None
        
        # Grid pattern on bottom net
        x_rel = hit_point[0] - self.center[0]
        y_rel = hit_point[1] - self.center[1]
        
        # Grid pattern
        x_to_rope = abs(x_rel % self.mesh_size - self.mesh_size/2)
        y_to_rope = abs(y_rel % self.mesh_size - self.mesh_size/2)
        
        is_on_rope = (x_to_rope < self.rope_thickness or 
                      y_to_rope < self.rope_thickness)
        
        if not is_on_rope:
            # Check if ray goes through mesh hole
            hash_val = ((x_rel * 100) % 1.0 + (y_rel * 100) % 1.0) / 2.0
            if hash_val > 0.4:
                return None
        
        normal = np.array([0.0, 0.0, 1.0])  # Pointing up
        reflectivity = self.rope_reflectivity if is_on_rope else self.net_reflectivity
        
        return Hit(t=float(t), point=hit_point, normal=normal, 
                  obj_id=self.obj_id, reflectivity=reflectivity)


@dataclass
class Fish:
    """A single fish in the simulation."""
    position: np.ndarray
    velocity: np.ndarray
    size: float = 0.2  # Fish length
    reflectivity: float = 0.35  # Base reflectivity
    preferred_depth: float = 0.5  # Preferred relative depth (0-1)
    circling_direction: float = 1.0  # 1.0 for counter-clockwise, -1.0 for clockwise
    target_strength: float = 0.0  # Acoustic target strength modulation (AR(1) process)
    burst_timer: float = 0.0  # Timer for burst swimming events
    
    def update(self, dt: float, cage_center: np.ndarray, cage_radius_top: float, 
               cage_radius_bottom: float, cage_depth: float, neighbors: list):
        """Update fish position with realistic schooling behavior."""
        
        # Calculate current depth and radius at this depth
        z_rel = max(0, min(cage_depth, cage_center[2] - self.position[2]))
        depth_ratio = z_rel / cage_depth
        current_cage_radius = cage_radius_top + (cage_radius_bottom - cage_radius_top) * depth_ratio
        
        # Distance from cage center in XY
        dist_from_center_2d = np.linalg.norm(self.position[:2] - cage_center[:2])
        
        # Schooling behavior forces
        accel = np.zeros(3)
        
        # 1. CIRCULAR SWIMMING - main behavior in fish cages
        # Fish naturally swim in circles following the cage wall
        to_center_2d = cage_center[:2] - self.position[:2]
        if dist_from_center_2d > 0.01:
            to_center_2d = to_center_2d / dist_from_center_2d
            
            # Tangent direction for circular motion
            tangent = np.array([-to_center_2d[1], to_center_2d[0]]) * self.circling_direction
            
            # Prefer swimming in circles at PERIMETER
            preferred_radius = current_cage_radius * FishConfig.PERIMETER_RADIUS_MEAN
            radius_error = dist_from_center_2d - preferred_radius
            
            # Circular motion and centering forces
            circular_force = tangent * FishConfig.CIRCULAR_MOTION_STRENGTH
            centering_force = to_center_2d * radius_error * FishConfig.CENTERING_FORCE
            
            accel[:2] += circular_force + centering_force
        
        # 2. COHESION - attraction to stay close to nearby fish (tight schooling)
        if len(neighbors) > 0:
            center_of_mass = np.mean([n.position for n in neighbors], axis=0)
            cohesion = (center_of_mass - self.position) * FishConfig.COHESION_STRENGTH
            accel += cohesion
        
        # 3. ALIGNMENT - tendency to match velocity with neighbors
        if len(neighbors) > 0:
            avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
            alignment = (avg_velocity - self.velocity) * FishConfig.ALIGNMENT_STRENGTH
            accel += alignment
        
        # 4. SEPARATION - avoid crowding
        separation = np.zeros(3)
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist < FishConfig.SEPARATION_DISTANCE and dist > 0.01:
                separation += diff / (dist * dist) * FishConfig.SEPARATION_STRENGTH
        accel += separation
        
        # 5. DEPTH PREFERENCE - maintain preferred depth
        target_depth = cage_center[2] - cage_depth * self.preferred_depth
        depth_error = target_depth - self.position[2]
        accel[2] += depth_error * FishConfig.DEPTH_PREFERENCE_STRENGTH
        
        # 6. AVOID WALLS - strong force near boundaries
        wall_threshold = FishConfig.WALL_AVOIDANCE_THRESHOLD
        if dist_from_center_2d > current_cage_radius - wall_threshold:
            # Push away from wall
            to_center = (cage_center[:2] - self.position[:2])
            if np.linalg.norm(to_center) > 0.01:
                to_center = to_center / np.linalg.norm(to_center)
                wall_avoidance = to_center * FishConfig.WALL_AVOIDANCE_STRENGTH
                accel[:2] += wall_avoidance
        
        # Avoid bottom
        if self.position[2] < cage_center[2] - cage_depth + wall_threshold:
            accel[2] += 2.0
        
        # Avoid surface
        if self.position[2] > cage_center[2] - wall_threshold:
            accel[2] -= 2.0
        
        # 7. BURST EVENTS - occasional rapid acceleration (realistic flicker in sonar)
        burst_force = 0.0
        if self.burst_timer > 0:
            burst_force = FishConfig.BURST_FORCE
            self.burst_timer -= dt
        elif np.random.rand() < FishConfig.BURST_PROBABILITY:
            self.burst_timer = (FishConfig.BURST_DURATION_MIN + 
                              np.random.rand() * (FishConfig.BURST_DURATION_MAX - FishConfig.BURST_DURATION_MIN))
            burst_force = FishConfig.BURST_FORCE
        
        # Random exploration
        accel += np.random.randn(3) * FishConfig.RANDOM_EXPLORATION
        if burst_force > 0:
            # Burst in current swimming direction
            if np.linalg.norm(self.velocity) > 0.1:
                accel += (self.velocity / np.linalg.norm(self.velocity)) * burst_force
        
        # Update velocity with damping
        self.velocity += accel * dt
        self.velocity *= FishConfig.VELOCITY_DAMPING
        
        # UPDATE TARGET STRENGTH (acoustic aspect angle variation)
        # AR(1) process: ts = 0.98*ts + 0.02*randn()
        self.target_strength = 0.98 * self.target_strength + 0.02 * np.random.randn()
        self.target_strength = np.clip(self.target_strength, -2.0, 2.0)
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > FishConfig.MAX_SPEED:
            self.velocity = self.velocity * (FishConfig.MAX_SPEED / speed)
        
        # Update position
        self.position += self.velocity * dt
        
        # Hard constraints (safety bounds)
        dist_2d = np.linalg.norm(self.position[:2] - cage_center[:2])
        if dist_2d > current_cage_radius - 0.1:
            direction = (self.position[:2] - cage_center[:2]) / (dist_2d + 0.001)
            self.position[:2] = cage_center[:2] + direction * (current_cage_radius - 0.1)
            # Bounce velocity
            self.velocity[:2] -= 2 * np.dot(self.velocity[:2], direction) * direction
            self.velocity[:2] *= 0.5
        
        if self.position[2] > cage_center[2] - 0.1:
            self.position[2] = cage_center[2] - 0.1
            self.velocity[2] *= -0.5
        elif self.position[2] < cage_center[2] - cage_depth + 0.1:
            self.position[2] = cage_center[2] - cage_depth + 0.1
            self.velocity[2] *= -0.5

@dataclass  
class FishSchool(Primitive):
    """A school of fish represented as dynamic point targets."""
    obj_id: str
    cage_center: np.ndarray
    cage_radius_top: float
    cage_radius_bottom: float
    cage_depth: float
    num_fish: int = 200
    reflectivity: float = 0.35
    neighbor_distance: float = 2.0  # Distance for schooling behavior
    use_spatial_grid: bool = True  # Use spatial grid for optimization
    grid_cell_size: float = 3.0  # Cell size for spatial hashing
    
    # Vectorized arrays for fast intersection
    positions: np.ndarray = None  # (N, 3)
    radii: np.ndarray = None  # (N,)
    reflectivities: np.ndarray = None  # (N,)
    
    # Spatial grid for neighbor queries and intersection
    spatial_grid: dict = None  # Maps (ix, iy, iz) -> list of fish indices
    school_direction: float = 1.0  # Common circling direction (70-90% alignment)
    
    def __post_init__(self):
        """Initialize swimming fish with dense schools near the perimeter."""
        self.fish = []
        
        # Dense schools near the perimeter (realistic fish farm behavior)
        num_clusters = FishConfig.NUM_CLUSTERS
        
        cluster_angles = np.random.rand(num_clusters) * 2 * np.pi
        cluster_depths = np.random.rand(num_clusters) * 0.8 + 0.1
        
        for i in range(self.num_fish):
            rand = np.random.rand()
            
            if rand < FishConfig.PERIMETER_CLUSTER_RATIO:
                cluster_idx = i % num_clusters
                angle = cluster_angles[cluster_idx] + np.random.randn() * FishConfig.CLUSTER_ANGLE_STD
                depth_ratio = np.clip(cluster_depths[cluster_idx] + np.random.randn() * FishConfig.CLUSTER_DEPTH_STD, 0.1, 0.9)
                radius_factor = np.clip(FishConfig.PERIMETER_RADIUS_MEAN + np.random.randn() * FishConfig.CLUSTER_RADIUS_STD, 
                                       FishConfig.PERIMETER_RADIUS_MIN, FishConfig.PERIMETER_RADIUS_MAX)
                
            else:  # Scattered throughout cage
                angle = np.random.rand() * 2 * np.pi
                depth_ratio = np.random.rand() * 0.8 + 0.1
                radius_factor = (np.random.rand() * (FishConfig.SCATTERED_RADIUS_MAX - FishConfig.SCATTERED_RADIUS_MIN) 
                               + FishConfig.SCATTERED_RADIUS_MIN)
            
            depth = depth_ratio * self.cage_depth
            radius_at_depth = self.cage_radius_top + (self.cage_radius_bottom - self.cage_radius_top) * depth_ratio
            r = radius_at_depth * radius_factor
            
            pos = np.array([
                self.cage_center[0] + r * np.cos(angle),
                self.cage_center[1] + r * np.sin(angle),
                self.cage_center[2] - depth
            ])
            
            # Initial velocity for circular swimming
            speed = FishConfig.SWIM_SPEED_MIN + np.random.rand() * (FishConfig.SWIM_SPEED_MAX - FishConfig.SWIM_SPEED_MIN)
            tangent_angle = angle + np.pi/2
            vel = np.array([
                speed * np.cos(tangent_angle),
                speed * np.sin(tangent_angle),
                0.0
            ])
            
            # SIZE DISTRIBUTION: lognormal for realistic size variation
            fish_size = np.random.lognormal(np.log(FishConfig.FISH_SIZE_MEAN) - 0.5 * (FishConfig.FISH_SIZE_STD/FishConfig.FISH_SIZE_MEAN)**2, 
                                           FishConfig.FISH_SIZE_STD/FishConfig.FISH_SIZE_MEAN)
            fish_size = np.clip(fish_size, FishConfig.FISH_SIZE_MIN, FishConfig.FISH_SIZE_MAX)
            
            # Most fish circle in same direction
            circling_dir = self.school_direction if np.random.rand() < FishConfig.SCHOOL_DIRECTION_ALIGNMENT else -self.school_direction
            preferred_depth = depth_ratio
            
            self.fish.append(Fish(pos, vel, 
                                 size=fish_size, 
                                 reflectivity=self.reflectivity,
                                 preferred_depth=preferred_depth,
                                 circling_direction=circling_dir))
        
        # Initialize vectorized arrays
        self._update_vectorized_arrays()
        
        # Initialize spatial grid for neighbor queries
        if self.use_spatial_grid:
            self.spatial_grid = {}
            self._rebuild_spatial_grid()
    
    def update(self, dt: float):
        """Update fish positions with schooling behavior."""
        # Update spatial grid for neighbor queries
        if self.use_spatial_grid:
            self._rebuild_spatial_grid()
        
        # Update each fish
        for i, fish in enumerate(self.fish):
            # Find neighbors efficiently
            if self.use_spatial_grid:
                neighbors = self._get_neighbors_grid(i, fish.position)
            else:
                neighbors = self._get_neighbors_bruteforce(fish)
            
            # Update fish with schooling behavior
            fish.update(dt, self.cage_center, self.cage_radius_top, 
                       self.cage_radius_bottom, self.cage_depth, neighbors)
        
        # Update vectorized arrays for fast intersection
        self._update_vectorized_arrays()
    
    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """VECTORIZED ray intersection with all fish (fast NumPy version)."""
        if len(self.fish) == 0:
            return None
        
        # Vectorized sphere intersection for ALL fish at once
        # oc = ro - positions (broadcast to (N, 3))
        oc = ro - self.positions  # (N, 3)
        
        # b = dot(oc, rd) for each fish
        b = np.sum(oc * rd, axis=1)  # (N,)
        
        # c = dot(oc, oc) - radius^2 for each fish
        c = np.sum(oc * oc, axis=1) - self.radii * self.radii  # (N,)
        
        # Discriminant
        disc = b * b - c  # (N,)
        
        # Mask for valid intersections
        valid = disc >= 0
        
        if not np.any(valid):
            return None
        
        # Calculate t values only for valid intersections
        s = np.sqrt(disc[valid])
        b_valid = b[valid]
        t0 = -b_valid - s
        t1 = -b_valid + s
        
        # Choose closest positive t
        t = np.where(t0 > 0.001, t0, np.where(t1 > 0.001, t1, np.inf))
        
        # Find minimum t
        if np.all(t == np.inf):
            return None
        
        min_idx_local = np.argmin(t)
        closest_t = t[min_idx_local]
        
        if closest_t == np.inf:
            return None
        
        # Map back to original fish index
        valid_indices = np.where(valid)[0]
        fish_idx = valid_indices[min_idx_local]
        
        fish = self.fish[fish_idx]
        p = ro + closest_t * rd
        n = (p - fish.position) / fish.size
        
        # Apply acoustic target strength modulation
        ts_factor = 1.0 + 0.5 * fish.target_strength
        modulated_refl = fish.reflectivity * np.clip(ts_factor, 0.3, 1.7)
        
        return Hit(t=float(closest_t), point=p, normal=n, 
                  obj_id=f"{self.obj_id}_fish", 
                  reflectivity=modulated_refl)
    
    def _update_vectorized_arrays(self):
        """Update vectorized arrays from fish list for fast intersection."""
        n = len(self.fish)
        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.radii = np.zeros(n, dtype=np.float32)
        self.reflectivities = np.zeros(n, dtype=np.float32)
        
        for i, fish in enumerate(self.fish):
            self.positions[i] = fish.position
            self.radii[i] = fish.size
            self.reflectivities[i] = fish.reflectivity
    
    def _rebuild_spatial_grid(self):
        """Rebuild spatial hash grid for fast neighbor queries."""
        self.spatial_grid = {}
        
        for i, fish in enumerate(self.fish):
            cell = self._position_to_cell(fish.position)
            if cell not in self.spatial_grid:
                self.spatial_grid[cell] = []
            self.spatial_grid[cell].append(i)
    
    def _position_to_cell(self, pos: np.ndarray) -> tuple:
        """Convert 3D position to grid cell coordinates."""
        return (
            int(np.floor(pos[0] / self.grid_cell_size)),
            int(np.floor(pos[1] / self.grid_cell_size)),
            int(np.floor(pos[2] / self.grid_cell_size))
        )
    
    def _get_neighbors_grid(self, fish_idx: int, pos: np.ndarray) -> list:
        """Get neighbors using spatial grid (O(k) instead of O(N))."""
        neighbors = []
        cell = self._position_to_cell(pos)
        
        # Check own cell and 26 adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if check_cell in self.spatial_grid:
                        for other_idx in self.spatial_grid[check_cell]:
                            if other_idx != fish_idx:
                                other = self.fish[other_idx]
                                dist = np.linalg.norm(pos - other.position)
                                if dist < self.neighbor_distance:
                                    neighbors.append(other)
        
        return neighbors
