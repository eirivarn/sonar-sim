"""
Centralized configuration for sonar simulation.
"""

# ============================================================================
# SONAR PARAMETERS
# ============================================================================
SONAR_CONFIG = {
    # Basic sonar properties
    'range_m': 20.0,           # Maximum range in meters
    'fov_deg': 120.0,          # Field of view in degrees
    'num_beams': 256,          # Number of acoustic beams
    'range_bins': 1024,        # Number of range bins (output resolution)
    
    # Ray marching parameters
    'step_size_factor': 0.5,   # Step size as fraction of voxel size
    'energy_threshold': 0.01,  # Minimum energy to continue ray marching
    
    # Acoustic effects
    'water_absorption': 0.115, # Water absorption coefficient (dB/m at ~100kHz)
    'spreading_loss_min': 1.0, # Minimum range for spreading loss calculation
    
    # Beam pattern
    'beam_pattern_falloff': 2.5,  # Gaussian falloff toward beam edges (higher = stronger falloff)
    
    # Noise and artifacts (speckle, jitter, decorrelation)
    'speckle_shape': 1.2,      # Gamma distribution shape for acoustic speckle
    'aspect_variation_std': 0.8,  # Aspect angle variation std dev
    'aspect_variation_range': [0.2, 2.0],  # Min/max aspect variation
    
    'jitter_probability': 0.5,  # Probability of range jitter per return
    'jitter_std_base': 1.5,     # Base standard deviation for range jitter (bins)
    'jitter_range_factor': 3.0, # How much jitter increases with range
    'jitter_max_offset': 8,     # Maximum jitter offset in bins
    
    'spread_probability': 0.3,  # Probability of multi-bin spreading
    'spread_bin_options': [2, 3, 4],  # Possible spread widths
    'spread_bin_probs': [0.5, 0.35, 0.15],  # Probabilities for each width
    
    'temporal_decorrelation_shape': 5.0,  # Gamma shape for frame-to-frame variation
    
    # Absorption and shadow parameters
    'absorption_factor': 2.0,   # Absorption strength multiplier
    'scattering_loss_factor': 3.0,  # Energy loss from scattering
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
VISUALIZATION_CONFIG = {
    # Figure layout
    'figure_size': (22, 6),    # Figure size in inches (width, height)
    'num_subplots': 3,         # Number of subplot panels
    'subplot_layout': (1, 3),  # Subplot grid layout (rows, cols)
    
    # Animation
    'animation_interval': 100, # Update interval in milliseconds
    'animation_cache': False,  # Cache animation frames
    
    # Display normalization
    'db_normalization': 60,    # dB range for display normalization
    'sonar_colormap': 'hot',   # Colormap for sonar image
    
    # Ground truth visualization colors (RGB)
    'material_colors': {
        0: [0, 0, 0],           # Empty - Black
        1: [0, 0, 255],         # Net - Blue
        2: [128, 128, 255],     # Rope - Light blue
        3: [255, 165, 0],       # Fish - Orange
        4: [128, 128, 128],     # Wall - Gray
        5: [0, 255, 0],         # Biomass - Green
        6: [255, 255, 0],       # Debris_Light - Yellow
        7: [255, 192, 0],       # Debris_Medium - Dark yellow
        8: [255, 128, 0],       # Debris_Heavy - Orange-red
        9: [160, 160, 160],     # Concrete - Light gray
        10: [139, 90, 43],      # Wood - Brown
        11: [34, 139, 34],      # Foliage - Forest green
        12: [192, 192, 192],    # Metal - Silver
        13: [173, 216, 230],    # Glass - Light blue
    },
    
    # Controls
    'move_speed': 0.5,         # Movement speed in meters per key press
    'rotate_speed': 5.0,       # Rotation speed in degrees per key press
}

# ============================================================================
# SCENE PARAMETERS
# ============================================================================
SCENE_CONFIG = {
    'world_size_m': 30.0,      # World is 30m x 30m
    'cage_center': [15.0, 15.0],  # Center of cage
    'cage_radius': 12.0,       # Cage radius in meters
    'num_sides': 24,           # Number of cage panel segments
    'current_strength': 2.0,   # Current deflection strength
    'net_sag': 0.25,          # Maximum net sag in meters
    
    # Fish parameters
    'num_fish': 200,           # Number of fish in cage
    'fish_length_range': [0.4, 0.6],  # Fish length range [min, max]
    'fish_width_ratio': 0.20,  # Width as fraction of length
}

# Street scene configuration
STREET_SCENE_CONFIG = {
    'world_size_m': 50.0,      # Larger world for street scene
    'street_width': 8.0,       # Width of main street
    'sidewalk_width': 2.0,     # Width of sidewalks
    'num_houses': 8,           # Number of houses along street
    'num_trees': 12,           # Number of trees
    'num_cars': 4,             # Number of moving cars
    'car_speed_range': [0.1, 0.3],  # Car speed range in m/s
}

# ============================================================================
# MATERIAL PROPERTIES (Sonar Physics)
# ============================================================================
MATERIAL_CONFIG = {
    # Each material: [reflectivity, scattering, absorption]
    # reflectivity: 0-1, how bright the object appears
    # scattering: 0-1, how much energy scatters
    # absorption: 0-1, how much energy is absorbed (controls shadow strength)
    
    'net': {
        'reflectivity': 0.4,
        'scattering': 0.2,
        'absorption': 0.1,    # Light shadow - nets are thin/sparse
    },
    'rope': {
        'reflectivity': 0.8,
        'scattering': 0.4,
        'absorption': 0.4,     # Strong shadow
    },
    'fish': {
        'reflectivity': 0.7,
        'scattering': 0.5,
        'absorption': 0.2,     # Moderate shadow
    },
    'biomass': {
        'reflectivity': 0.9,
        'scattering': 0.7,
        'absorption': 0.1,     # Light shadow - algae/fouling
    },
    'wall': {
        'reflectivity': 1.0,
        'scattering': 0.5,
        'absorption': 1.0,     # Strong shadow
    },
    'debris_light': {
        'reflectivity': 0.8,
        'scattering': 0.6,
        'absorption': 0.1,
    },
    'debris_medium': {
        'reflectivity': 0.9,
        'scattering': 0.7,
        'absorption': 0.2,
    },
    'debris_heavy': {
        'reflectivity': 1.0,
        'scattering': 0.8,
        'absorption': 0.3,
    },
    
    # Urban materials (for street scene)
    'concrete': {
        'reflectivity': 0.9,
        'scattering': 0.3,
        'absorption': 0.6,     # Moderate shadow - buildings/roads
    },
    'wood': {
        'reflectivity': 0.6,
        'scattering': 0.4,
        'absorption': 0.3,     # Light shadow - wooden structures
    },
    'foliage': {
        'reflectivity': 0.5,
        'scattering': 0.8,
        'absorption': 0.2,     # Scattered reflection - trees/plants
    },
    'metal': {
        'reflectivity': 1.0,
        'scattering': 0.2,
        'absorption': 0.9,     # Strong reflection and shadow - cars/metal
    },
    'glass': {
        'reflectivity': 0.3,
        'scattering': 0.1,
        'absorption': 0.5,     # Low reflection - windows
    },
}

