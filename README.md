# Sonar Simulation

Voxel-based sonar simulator with realistic acoustic effects for underwater and urban environments.

## Project Structure

```
sonar-sim/
├── simulation.py           # Main entry point - run interactive or headless simulation
├── requirements.txt        # Python dependencies
├── data/                   # Saved run data
│   └── runs/              # Individual run directories
└── src/                   # Source code modules
    ├── config.py          # Centralized configuration parameters
    ├── core/              # Core simulation components
    │   ├── materials.py   # Material definitions and acoustic properties
    │   ├── voxel_grid.py  # 2D spatial grid with primitives
    │   ├── sonar.py       # Volumetric ray marching sonar
    │   └── dynamics.py    # Dynamic object behavior (fish, cars, debris)
    ├── scenes/            # Scene definitions
    │   ├── fish_cage_scene.py   # Underwater fish cage
    │   └── street_scene.py      # Urban street
    ├── scripts/           # Visualization and data collection scripts
    │   ├── visualization.py     # Interactive display
    │   ├── visualize_run.py     # Saved run visualization
    │   └── data_collection.py   # Headless data collection paths
    └── utils/             # Utility functions
        └── utils.py
```

## Quick Start

### Interactive Simulation
```bash
# Fish cage scene (default)
python simulation.py

# Street scene
python simulation.py --scene src.scenes.street_scene
```

### Data Collection
```bash
# Collect 200 samples with circular path
python simulation.py --collect circular --save my_run --num-samples 200

# View as video
python -m src.scripts.visualize_run data/runs/my_run --output video.mp4

# Interactive playback
python -m src.scripts.visualize_run data/runs/my_run --interactive
```

### Controls (Interactive Mode)
- **WASD**: Move sonar position
- **Arrow keys**: Rotate sonar direction
- **Q**: Quit

## Creating New Scenes

See `src/scenes/fish_cage_scene.py` for a complete example. Each scene requires:

1. `create_scene()` - Initialize world and return config
2. `update_scene()` - Update dynamic objects each frame
3. `render_map()` - Draw world map view

## Configuration

All simulation parameters are in `src/config.py`:
- Sonar parameters (range, FOV, acoustic effects)
- Scene parameters (world size, object counts)
- Visualization settings (colors, update rate)

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- See `requirements.txt` for complete list
