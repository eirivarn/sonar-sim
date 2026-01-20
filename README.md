# Sonar-Sim: Realistic Fish Farm Sonar Simulation

A high-fidelity 3D imaging sonar simulator for fish farm monitoring with realistic physics, animated fish schools, and polygon cage geometry. Built for research, training, and algorithm development.

## Key Features

### 🎯 Realistic Sonar Physics
- **Accurate signal propagation**: r⁻² spreading loss (imaging sonar model)
- **Water absorption**: Frequency-dependent attenuation (0.05 dB/m @ 700 kHz)
- **Speckle noise**: Gamma-distributed multiplicative noise (configurable looks)
- **Shadowing**: Acoustic occlusion behind strong targets
- **TVG compensation**: Time-varied gain with logarithmic enhancement
- **High resolution**: 1024 range bins (~3.4cm @ 35m range, matching real hardware)

### 🐟 Advanced Fish Behavior
- **1200 animated fish** with realistic schooling dynamics
- **Dense perimeter clusters** (75% of fish near cage walls)
- **Circular swimming** patterns following cage contours
- **Neighbor-aware behavior**: Cohesion, alignment, separation forces
- **Burst swimming**: Occasional rapid movements for realistic flicker
- **Spatial grid optimization**: O(k) neighbor queries for performance

### 🏗️ Polygon Cage Geometry
- **Commercial-scale cages**: 50m diameter, 25m depth
- **Flat panel construction**: 12-sided polygon (appears circular but made of straight segments)
- **Realistic net mesh**: 0.8m spacing with structural ropes
- **Analytical intersection**: Proper plane-based ray intersection for flat panels
- **Material differentiation**: Mesh (0.25) vs. rope (0.45) reflectivity

### 📊 Interactive Visualization
- **3D world view**: Wireframe cage with swimming fish
- **Polar sonar display**: Imaging sonar format with realistic artifacts
- **Real-time control**: WASD movement, arrow key rotation
- **Pause/unpause**: Freeze fish animation for inspection
- **Colormap cycling**: 8 built-in color schemes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sonar-sim.git
cd sonar-sim

# Install dependencies
pip install -r requirements.txt

# Run the main viewer
python tools/fish_farm_viewer.py
```

**Requirements:** Python 3.10+, NumPy, Matplotlib, SciPy

## Quick Start

### Fish Farm Viewer (Recommended)
View a complete fish farm with animated schools in a large polygon cage:

```bash
python tools/fish_farm_viewer.py
```

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| `W/S` | Move forward/backward |
| `A/D` | Move left/right |
| `Q/E` | Move down/up |
| `Arrow Keys` | Rotate (yaw) |
| `I/K` | Pitch up/down |
| `J/L` | Roll left/right |
| `R` | Reset to default position |
| `SPACE` | Pause/unpause fish animation |
| `C` | Cycle colormap themes |

**Display Windows:**
1. **3D View** - Polygon cage wireframe with 1200 swimming fish
2. **Polar Sonar** - Imaging sonar display (90° FOV, 35m range)
3. **Range Plot** - Traditional beam intensity vs. distance

## Configuration

All parameters are centralized in **`src/sim/config.py`** with detailed documentation. Simply edit this file to customize the simulation.

### Configuration Groups

#### 1. World Environment (`WorldConfig`)
```python
SEAFLOOR_DEPTH = 35.0          # Depth below surface (meters)
SEAFLOOR_REFLECTIVITY = 0.15   # Ground echo strength (0-1)
```

#### 2. Fish Cage (`CageConfig`)
```python
CAGE_RADIUS_TOP = 25.0         # Surface radius (meters)
CAGE_RADIUS_BOTTOM = 23.0      # Bottom radius (tapered)
CAGE_DEPTH = 25.0              # Cage depth (meters)
NUM_SIDES = 12                 # Polygon sides (12=near-circular)
MESH_SIZE = 0.8                # Net mesh spacing (meters)
NET_REFLECTIVITY = 0.25        # Net echo strength
ROPE_REFLECTIVITY = 0.45       # Structural rope strength
```

#### 3. Fish Behavior (`FishConfig`)
```python
NUM_FISH = 1200                # Total population
PERIMETER_CLUSTER_RATIO = 0.75 # Fraction near walls
NUM_CLUSTERS = 8               # Number of schools
SWIM_SPEED_MIN = 0.5           # Min speed (m/s)
SWIM_SPEED_MAX = 0.8           # Max speed (m/s)
COHESION_STRENGTH = 0.6        # Schooling attraction
CIRCULAR_MOTION_STRENGTH = 2.5 # Cage-following force
```

#### 4. Sonar Sensor (`SonarConfig`)
```python
RANGE_M = 35.0                 # Maximum range (meters)
HFOV_DEG = 90.0                # Horizontal field of view
H_BEAMS = 181                  # Number of beams
RANGE_BINS = 1024              # Range resolution (3.4cm)
ALPHA_DB_PER_M = 0.05          # Absorption (dB/m)
```

#### 5. Signal Processing (`SignalProcessingConfig`)
```python
SPECKLE_LOOKS = 1.2            # Speckle grain (1=grainy, 8=smooth)
SHADOW_STRENGTH = 0.85         # Occlusion darkness (0-1)
TVG_EXPONENT = 2.0             # Range compensation
GAMMA_CORRECTION = 0.75        # Display gamma
```

#### 6. Visualization (`VisualizationConfig`)
```python
LINES_PER_PANEL = 5            # Wireframe density
POLAR_GAUSSIAN_SIGMA = 0.5     # Display smoothing
CONTOUR_LEVELS = 20            # Polar plot granularity
```

### Quick Tuning Guide

**Make fish more scattered:**
```python
FishConfig.PERIMETER_CLUSTER_RATIO = 0.5  # Less clustering
FishConfig.CLUSTER_ANGLE_STD = 0.4        # Wider spread
```

**Make cage more angular (square-like):**
```python
CageConfig.NUM_SIDES = 4  # 4=square, 6=hexagon, 12=circular
```

**Make sonar grainier (more texture):**
```python
SignalProcessingConfig.SPECKLE_LOOKS = 1.0  # Very grainy
VisualizationConfig.POLAR_GAUSSIAN_SIGMA = 0.0  # No smoothing
```

**Increase sonar range:**
```python
SonarConfig.RANGE_M = 50.0           # Extended range
SonarConfig.ALPHA_DB_PER_M = 0.03    # Less absorption
```

**Make fish swim faster:**
```python
FishConfig.SWIM_SPEED_MAX = 1.2           # Faster swimming
FishConfig.CIRCULAR_MOTION_STRENGTH = 3.5 # Stronger force
```

See **`src/sim/config.py`** for all 100+ tunable parameters with detailed comments.

## Architecture

```
sonar-sim/
├── src/sim/
│   ├── config.py              # 🔧 Centralized configuration
│   ├── primitives.py          # Geometric primitives (planes, spheres, AABB, clutter)
│   ├── fish_cage.py           # NetCage polygon geometry + FishSchool behavior
│   ├── world.py               # World container + ray-casting engine
│   ├── sonar.py               # Sonar sensor with scan_2d()
│   ├── sonar_effects.py       # Realistic signal processing pipeline
│   ├── visualization.py       # 3D rendering utilities
│   ├── math3d.py              # 3D math (rotation matrices, vectors)
│   └── fish_farm_world.py     # World builder
│
├── tools/
│   ├── fish_farm_viewer.py    # ⭐ Main application (recommended)
│   ├── interactive_viewer.py  # Simple demo world
│   └── world_viewer_3d.py     # Static world viewer
│
└── requirements.txt
```

### Core Classes

**`Sonar`** - Imaging sonar sensor
- Casts 181 beams across 90° horizontal FOV
- 1024 range bins for ~3.4cm resolution
- Returns distances and intensities per beam
- Applies realistic effects pipeline

**`NetCage`** - Polygon fish cage primitive
- 12-sided polygon with flat panels (not cylindrical)
- Analytical plane intersection per panel
- Net mesh pattern with structural ropes
- Biofouling modulation (spatial variation)

**`FishSchool`** - Animated fish population
- 1200 individual fish agents
- Vectorized sphere intersection (O(N) rays × O(M) fish)
- Spatial grid for O(k) neighbor queries
- Realistic schooling forces (cohesion, alignment, separation)

**`World`** - Scene container
- Holds all primitives (cage, fish, floor, clutter)
- Multi-primitive ray-casting with closest-hit logic
- Material-based reflectivity system

## Technical Details

### Sonar Physics Pipeline
1. **Ray generation**: 181 beams across FOV
2. **Intersection**: Test all primitives, find closest hit
3. **Two-way loss**: r⁻² spreading (imaging sonar, not point target r⁻⁴)
4. **Absorption**: exp(-α × r) frequency-dependent decay
5. **Angle rolloff**: Beam pattern edge effects
6. **Range/beam PSF**: Blurring from limited resolution
7. **Speckle**: Gamma-distributed multiplicative noise
8. **Noise floor**: Additive exponential noise
9. **Shadowing**: Acoustic occlusion along beams
10. **TVG + log + gamma**: Display enhancement

### Fish Schooling Algorithm
Forces on each fish:
- **Circular motion**: Tangential force for cage-following (2.5× strength)
- **Centering**: Maintain preferred radius (0.92 × cage radius)
- **Cohesion**: Attraction to nearby fish (0.6× strength)
- **Alignment**: Match neighbor velocities (0.5× strength)
- **Separation**: Avoid crowding < 0.3m (0.3× strength)
- **Depth preference**: Maintain target depth (0.3× strength)
- **Wall avoidance**: Strong repulsion near boundaries
- **Burst events**: Random rapid acceleration (0.2% probability)
- **Damping**: 0.92 velocity decay for smooth motion

### Polygon Cage Intersection
- Generate N polygon vertices at top and bottom circles
- For each of N panels:
  - Compute panel normal perpendicular to edge (not radial!)
  - Analytical plane intersection: t = dot(normal, top - ray_origin) / dot(normal, ray_dir)
  - Bounds check: u ∈ [0, panel_width], v ∈ [0, -depth]
- Return closest hit across all panels

**Key insight:** Panel normals must be perpendicular to panel edges, not pointing to panel center, otherwise flat panels are invisible!

## Advanced Usage

### Creating Custom Worlds
Edit `src/sim/fish_farm_world.py`:

```python
def build_fish_farm_world():
    w = World()
    
    # Add seafloor
    w.objects.append(Plane("seafloor", 
                          point=np.array([0, 0, -35]),
                          normal=np.array([0, 0, 1]),
                          reflectivity=0.15))
    
    # Add custom cage
    cage = NetCage(
        obj_id="my_cage",
        center=np.array([20, 0, 0]),
        radius_top=30.0,
        radius_bottom=28.0,
        depth=30.0,
        num_sides=8,  # Octagon
        mesh_size=1.0,
        rope_thickness=0.05,
        net_reflectivity=0.3,
        rope_reflectivity=0.5
    )
    w.objects.append(cage)
    
    # Add fish school
    fish = FishSchool(
        obj_id="salmon",
        cage_center=cage.center,
        cage_radius_top=cage.radius_top,
        cage_radius_bottom=cage.radius_bottom,
        cage_depth=cage.depth,
        num_fish=2000,
        reflectivity=0.4
    )
    w.objects.append(fish)
    
    return w, cage, fish
```

### Programmatic Control
```python
from src.sim.sonar import Sonar
from src.sim.fish_farm_world import build_fish_farm_world
import numpy as np

# Build world
world, cage, fish_school = build_fish_farm_world()

# Create sonar
sonar = Sonar(
    pos=np.array([0, 0, -5]),
    rpy=np.array([0, 0, 90]),  # Degrees
    range_m=35.0,
    hfov_deg=90.0,
    h_beams=181
)

# Scan
result = sonar.scan_2d(world)
distances = result['distances']      # (181,) array
intensities = result['intensities']  # (181,) array

# Update fish (time step)
fish_school.update(dt=0.05)

# Move sonar
sonar.pos += np.array([0.1, 0, 0])  # Move 10cm forward
```

## Material Reflectivity Reference

| Material | Reflectivity | Usage |
|----------|-------------|-------|
| Structural ropes | 0.45 | Cage framework |
| Fish (salmon) | 0.35 | Biological targets |
| Net mesh | 0.25 | Cage netting |
| Seafloor | 0.15 | Bottom surface |
| Water clutter | 0.02-0.12 | Suspended particles |

## Performance

- **Frame rate**: 20 FPS (configurable)
- **Fish update**: ~10ms for 1200 fish with spatial grid
- **Sonar scan**: ~5ms for 181 beams
- **Total**: ~50ms/frame (real-time capable)

Optimizations:
- Vectorized fish intersection (NumPy broadcasting)
- Spatial grid for O(k) neighbor search (k ≈ 10-20)
- Ray marching with early termination
- Efficient polygon panel iteration

## Troubleshooting

**Fish not moving:**
- Check `SPACE` key (pause/unpause toggle)
- Verify `SimulationConfig.DT > 0`

**Cage looks circular instead of polygonal:**
- In 3D view: This is expected with 12 sides
- Reduce `CageConfig.NUM_SIDES` to 4-6 for obvious panels
- In sonar: Flat panels should be visible if normals are correct

**Sonar image too smooth:**
- Decrease `SignalProcessingConfig.SPECKLE_LOOKS` (try 1.0)
- Decrease `VisualizationConfig.POLAR_GAUSSIAN_SIGMA` (try 0.2)

**No signal at long range:**
- Increase `SonarConfig.RANGE_M`
- Decrease `SonarConfig.ALPHA_DB_PER_M` (less absorption)
- Check spreading loss (r⁻² limits effective range)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Realistic sonar effects based on SonoptixECHO specifications
- Fish schooling inspired by Reynolds' Boids algorithm
- Polygon cage geometry for commercial aquaculture applications
