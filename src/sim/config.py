"""Configuration file for fish farm sonar simulation.

All tunable parameters are centralized here for easy experimentation.
Adjust these values to change the simulation behavior.
"""
import numpy as np


# ==============================================================================
# WORLD / ENVIRONMENT
# ==============================================================================

class WorldConfig:
    """Global world environment parameters."""
    
    # Seafloor depth (meters below surface, z=0)
    # Higher values = deeper water, more distance for sound to travel
    SEAFLOOR_DEPTH = 35.0
    
    # Seafloor reflectivity (0-1)
    # Higher = stronger bottom echoes, lower = softer mud/sand
    SEAFLOOR_REFLECTIVITY = 0.15


# ==============================================================================
# FISH CAGE GEOMETRY
# ==============================================================================

class CageConfig:
    """Fish cage physical structure parameters."""
    
    # Cage position (x, y, z in meters, z=0 is water surface)
    CAGE_CENTER = np.array([25.0, 0.0, 0.0])
    
    # Cage dimensions
    CAGE_RADIUS_TOP = 25.0      # Radius at surface (meters) - larger = bigger cage
    CAGE_RADIUS_BOTTOM = 23.0   # Radius at bottom (meters) - smaller = tapered shape
    CAGE_DEPTH = 25.0           # Cage depth (meters) - deeper = more fish volume
    
    # Polygon sides for cage structure
    # Higher = more circular appearance (12-20 typical)
    # Lower = more obvious flat panels (4=square, 6=hexagon)
    NUM_SIDES = 12
    
    # Net mesh properties
    MESH_SIZE = 0.8             # Distance between rope intersections (meters)
                                # Smaller = finer mesh, more reflections
    
    ROPE_THICKNESS = 0.04       # Thickness of structural ropes (meters)
                                # Thicker = stronger reflections at rope positions
    
    NET_REFLECTIVITY = 0.25     # Net mesh reflectivity (0-1)
                                # Higher = brighter net in sonar image
    
    ROPE_REFLECTIVITY = 0.30    # Structural rope reflectivity (0-1)
                                # Higher = structural elements more visible
    
    HAS_BOTTOM = True           # Whether cage has bottom net (vs open bottom)
    
    SAG_FACTOR = 0.12           # How much net sags between ropes (0-1)
                                # Higher = more realistic droop, affects geometry


# ==============================================================================
# FISH BEHAVIOR & DISTRIBUTION
# ==============================================================================

class FishConfig:
    """Fish school behavior and distribution parameters."""
    
    # Population
    NUM_FISH = 1000             # Total number of fish in simulation
                                # More = denser schools but slower simulation
    
    # Visibility
    ENABLE_FISH_IN_WORLD = True # Toggle fish in sonar simulation
                                # False = fish not added to world (no sonar detection)
    
    # Movement
    ENABLE_FISH_MOVEMENT = False # Toggle fish swimming on/off
                                # False = static fish positions (for testing)
    
    # Distribution (percentages should sum to ~1.0)
    PERIMETER_CLUSTER_RATIO = 0.75  # Fraction clustered near perimeter (0-1)
                                     # Higher = more fish at cage walls
    
    NUM_CLUSTERS = 8            # Number of dense schools around perimeter
                                # More = smaller, more distributed schools
    
    CLUSTER_ANGLE_STD = 0.2     # Angular spread of clusters (radians)
                                # Smaller = tighter schools, larger = looser
    
    CLUSTER_DEPTH_STD = 0.15    # Vertical spread of clusters (fraction of depth)
                                # Smaller = fish at similar depths
    
    CLUSTER_RADIUS_STD = 0.1   # Radial spread of clusters (fraction)
                                # Smaller = fish closer to same radius
    
    # Perimeter positioning
    PERIMETER_RADIUS_MEAN = 0.98    # Mean radius for perimeter fish (0-1)
                                     # Higher = closer to cage wall
    
    PERIMETER_RADIUS_MIN = 0.85     # Minimum radius for perimeter fish (0-1) - increased so fish reach the net
    PERIMETER_RADIUS_MAX = 1.0     # Maximum radius for perimeter fish (0-1)
    
    # Scattered fish (non-clustered)
    SCATTERED_RADIUS_MIN = 0.0      # Min radius for scattered fish (0-1)
    SCATTERED_RADIUS_MAX = 1.1      # Max radius for scattered fish (0-1)
    
    # Swimming behavior
    SWIM_SPEED_MIN = 0.5        # Minimum swim speed (m/s)
    SWIM_SPEED_MAX = 0.8        # Maximum swim speed (m/s)
                                # Faster = more dynamic, more motion blur
    
    SCHOOL_DIRECTION_ALIGNMENT = 0.8    # Fraction swimming same direction (0-1)
                                         # Higher = more coordinated circular motion
    
    # Schooling forces (tune these to adjust fish interactions)
    CIRCULAR_MOTION_STRENGTH = 2.5      # Tangential swimming force
                                         # Higher = stronger circular pattern
    
    CENTERING_FORCE = 0.8       # Force to maintain preferred radius
                                # Higher = fish stay closer to target radius
    
    COHESION_STRENGTH = 0.6     # Attraction to nearby fish (schooling)
                                # Higher = tighter schools, lower = looser
    
    ALIGNMENT_STRENGTH = 0.5    # Tendency to match neighbor velocities
                                # Higher = more synchronized swimming
    
    SEPARATION_STRENGTH = 0.3   # Avoidance of crowding
                                # Higher = more personal space
    
    SEPARATION_DISTANCE = 0.3   # Distance at which separation activates (m)
                                # Larger = fish keep more distance
    
    DEPTH_PREFERENCE_STRENGTH = 0.3     # Force to maintain preferred depth
                                         # Higher = less vertical wandering
    
    WALL_AVOIDANCE_THRESHOLD = 0.15      # Distance from wall to start avoiding (m) - reduced so fish get closer
    WALL_AVOIDANCE_STRENGTH = 3.0       # Force pushing away from walls
    
    BURST_PROBABILITY = 0.002   # Per-frame probability of burst swimming (0-1)
                                # Higher = more frequent rapid movements
    
    BURST_DURATION_MIN = 0.3    # Minimum burst duration (seconds)
    BURST_DURATION_MAX = 0.8    # Maximum burst duration (seconds)
    BURST_FORCE = 5.0           # Acceleration during burst
    
    RANDOM_EXPLORATION = 0.1    # Random movement strength
                                # Higher = more erratic swimming
    
    VELOCITY_DAMPING = 0.92     # Velocity decay per frame (0-1)
                                # Lower = faster deceleration, smoother motion
    
    MAX_SPEED = 1.2             # Maximum speed cap (m/s)
    
    # Fish physical properties
    FISH_SIZE_MEAN = 0.20       # Mean fish length (meters)
    FISH_SIZE_STD = 0.08        # Standard deviation of fish size
    FISH_SIZE_MIN = 0.10        # Minimum fish size
    FISH_SIZE_MAX = 0.40        # Maximum fish size
    
    FISH_REFLECTIVITY = 0.25    # Base acoustic reflectivity (0-1)
                                # Higher = brighter fish in sonar
    
    # Neighbor search
    NEIGHBOR_DISTANCE = 2.0     # Range for schooling interactions (meters)
                                # Larger = fish interact over greater distances
    
    SPATIAL_GRID_CELL_SIZE = 3.0    # Cell size for spatial grid optimization (m)
                                     # Should be > NEIGHBOR_DISTANCE for efficiency


# ==============================================================================
# SONAR SENSOR
# ==============================================================================

class SonarConfig:
    """Imaging sonar sensor parameters."""
    
    # Position and orientation
    SONAR_POSITION = np.array([25.0, 0.0, -5.0])  # Position (x, y, z) in meters
    SONAR_ORIENTATION = np.array([0.0, 0.0, 90.0])  # Roll, pitch, yaw (degrees)
    
    # Field of view
    HFOV_DEG = 90.0             # Horizontal field of view (degrees)
                                # Wider = see more area, but lower angular resolution
    
    H_BEAMS = 181               # Number of horizontal beams (angular samples)
                                # More = better angular resolution, slower
    
    # Beam cone spreading (multibeam sonar characteristic)
    BEAMWIDTH_DEG = 10          # Beam cone half-angle (degrees)
                                # At range R, beam footprint diameter = 2 * R * tan(beamwidth)
                                # Larger = more spreading, lower resolution at far ranges
    
    RAYS_PER_BEAM = 5           # Number of rays to cast per beam for cone sampling
                                # More = better cone coverage, slower (1 = pencil beam)
    
    # Multi-hit tracing (for returns behind porous objects like nets)
    MAX_HITS_PER_RAY = 3        # Maximum hits to trace along each ray (2-3 recommended)
                                # Higher = see fish behind nets, slower
    
    MIN_HIT_STRENGTH = 0.025     # Minimum signal strength to continue tracing
                                # Lower = trace weaker returns, slower
    
    TRANSMISSION_BOOST = 0.8    # Multiplier for transmitted signal through objects (0.5-2.0)
                                # 1.0 = realistic (transmitted = 1 - reflectivity)
                                # >1.0 = less signal loss, see through objects better
                                # <1.0 = more signal loss, objects more opaque
    
    # Beam pattern weighting
    BEAM_PATTERN_SIGMA_DEG = 0.8  # Gaussian sigma for beam pattern weighting (degrees)
                                  # Controls how rays in cone are weighted by angle
    
    USE_DETERMINISTIC_CONE = True  # Use precomputed cone pattern (vs random per frame)
                                   # True = less flicker, more realistic
    
    # Range settings
    RANGE_M = 10.0              # Maximum range (meters)
                                # Limited by physics: absorption, spreading loss
    
    RANGE_BINS = 1024           # Number of range bins (samples per beam)
                                # More = better range resolution (Δr = range/bins)
                                # 1024 bins at 35m ≈ 3.4cm resolution (realistic)
    
    # Physics parameters
    ALPHA_DB_PER_M = 0.05       # Water absorption coefficient (dB/m)
                                # Higher = faster signal decay with distance
                                # 0.05 typical for ~700 kHz in seawater
    
    EDGE_STRENGTH_DB = 6.0      # Edge rolloff for beam pattern (dB)
                                # Higher = sharper beam edges, more sidelobe
    
    # Feature flags
    ENABLE_MULTIPATH = True     # Enable multipath reflections (seafloor bounce)
    ENABLE_STRUCTURED_MULTIPATH = True  # Use mirror method for surface/seafloor (vs generic bounces)
    SURFACE_REFLECTIVITY = 0.8  # Surface (air-water) reflection coefficient
    
    ENABLE_NOISE = True         # Enable measurement noise
    ENABLE_REALISTIC_EFFECTS = True  # Enable full realistic processing pipeline


# ==============================================================================
# SONAR SIGNAL PROCESSING
# ==============================================================================

class SignalProcessingConfig:
    """Realistic sonar effects and signal processing parameters."""
    
    # Speckle noise (coherent interference)
    SPECKLE_LOOKS = 1.5         # Number of looks for speckle averaging (1-8)
                                # 1.0 = very grainy/scattery (raw sonar)
                                # 2-3 = moderate texture
                                # 4-8 = smooth (multi-looked)
    
    # Noise floor
    NOISE_FLOOR = 3e-6          # Receiver noise level (linear scale)
                                # Higher = more background noise
    
    # Shadowing (acoustic occlusion behind strong targets)
    SHADOW_THRESHOLD_PERCENTILE = 0.995  # Percentile for shadow threshold
                                         # Only strongest returns cast shadows
    
    SHADOW_STRENGTH = 0.50      # Shadow attenuation factor (0-1)
                                # Higher = darker shadows behind objects
    
    SHADOW_RECOVERY_RATE = 0.01 # Rate of shadow recovery along beam
                                # Higher = shadows dissipate faster
    
    # Time-varied gain (TVG) compensation
    TVG_EXPONENT = 2.0          # Exponent for range compensation (typically 2)
                                # Compensates for r^-2 spreading loss
    
    # Enhancement parameters
    PERCENTILE_LOW = 0.01       # Lower percentile for contrast stretching
    PERCENTILE_HIGH = 0.995     # Upper percentile for contrast stretching
    GAMMA_CORRECTION = 0.75     # Gamma for final display enhancement
                                # Lower = brighter midtones, higher = more contrast
    
    # Water column clutter (injected directly into mu)
    CLUTTER_DENSITY = 0.0008    # Probability of clutter per range-beam cell
                                # Higher = more sparse returns in water column
    
    CLUTTER_INTENSITY_MIN = 0.02  # Minimum clutter reflectivity
    CLUTTER_INTENSITY_MAX = 0.08  # Maximum clutter reflectivity
    
    CLUTTER_RANGE_DECAY = 0.02  # Clutter density decay with range (1/m)
                                # Higher = less clutter at far range


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class VisualizationConfig:
    """3D world and sonar display parameters."""
    
    # Wireframe cage visualization
    LINES_PER_PANEL = 5         # Vertical lines drawn per polygon panel
                                # More = denser wireframe appearance
    
    CORNER_LINE_WIDTH = 1.5     # Width of corner/edge lines
    INTERIOR_LINE_WIDTH = 0.5   # Width of interior mesh lines
    
    CORNER_LINE_ALPHA = 0.5     # Transparency of corner lines (0-1)
    INTERIOR_LINE_ALPHA = 0.3   # Transparency of interior lines (0-1)
    
    NUM_HORIZONTAL_RINGS = 10   # Number of horizontal rings in wireframe
                                # More = shows depth structure better
    
    # Display size defaults
    SONAR_DISPLAY_SIZE = 8      # Sonar polar display size (inches)
                                # Use +/- keys to adjust during runtime
    
    WORLD_VIEW_SIZE = (10, 8)   # 3D world view size (width, height in inches)


# ==============================================================================
# SIMULATION
# ==============================================================================

class SimulationConfig:
    """Simulation timing and update parameters."""
    
    DT = 0.05                   # Time step for fish updates (seconds)
                                # Smaller = smoother motion, more compute
    
    FPS = 20                    # Target frames per second for visualization
                                # Higher = smoother animation, more CPU intensive
