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
    
    # Overall intensity calibration (post-processing gain)
    'intensity_gain': 203.92,  # Multiply all returns by this factor to match real sonar intensity levels
    
    # Ray marching parameters
    'step_size_factor': 0.5,   # Step size as fraction of voxel size
    'energy_threshold': 0.01,  # Minimum energy to continue ray marching
    
    # Acoustic effects
    'water_absorption': 0.115, # Water absorption coefficient (dB/m at ~100kHz)
    'spreading_loss_min': 1.0, # Minimum range for spreading loss calculation
    
    # Beam pattern
    'beam_pattern_falloff': 0.50,  # Gaussian falloff toward beam edges (higher = stronger falloff)
                                   # Reduced from 2.5 to 0.45 (0.18x) to match real data edge intensity
    
    # Noise and artifacts (speckle, jitter, decorrelation)
    'speckle_shape': 6.0,      # Gamma distribution shape for acoustic speckle (increased from 3.5 - less noise)
    'aspect_variation_std': 1.5,  # Aspect angle variation std dev (increased for more scatter)
    'aspect_variation_range': [0.1, 3.5],  # Min/max aspect variation (wider range)
    
    'jitter_probability': 0.35,  # Probability of range jitter per return (reduced from 0.65)
    'jitter_std_base': 2.0,     # Base standard deviation for range jitter (bins) (increased)
    'jitter_range_factor': 4.0, # How much jitter increases with range (increased)
    'jitter_max_offset': 12,     # Maximum jitter offset in bins (increased)
    
    'spread_probability': 0.25,  # Probability of multi-bin spreading (reduced from 0.50)
    'spread_bin_options': [2, 3, 4, 5, 6],  # Possible spread widths (added more)
    'spread_bin_probs': [0.3, 0.2, 0.1, 0.2, 0.1],  # Probabilities for each width
    
    'temporal_decorrelation_shape': 25.0,  # Gamma shape for frame-to-frame variation (increased from 12.0 - less flicker)
    
    # Angle-dependent scattering (new parameters)
    'angle_scatter_strength': 3.0,  # Multiplier for off-axis scatter intensity
    'angle_scatter_power': 5.5,     # How quickly scatter increases toward edges (higher = more extreme)
    
    # Density-dependent scattering (new parameters)
    'density_scatter_threshold': 0.3,  # Density above which extra scatter kicks in
    'density_scatter_strength': 1.8,   # Multiplier for high-density scatter
    'density_noise_boost': 0.3,        # Additional jitter probability in dense areas
    
    # Absorption and shadow parameters
    'absorption_factor': 5.0,   # Absorption strength multiplier
    'scattering_loss_factor': 5.0,  # Energy loss from scattering
    'proximity_shadow_enabled': True,   # Enable distance-dependent shadowing
    'proximity_shadow_strength': 3.0,   # Multiplier for shadow strength (closer = stronger)
    'proximity_shadow_max_distance': 10.0,  # Distance at which proximity effect stops (meters)
    
    # Azimuth streaking (range-dependent gain saturation)
    'azimuth_streak_enabled': True,     # Enable range-slice saturation effects
    'azimuth_streak_threshold': 0.3,    # Energy threshold to trigger (increased from 0.4)
    'azimuth_streak_probability': 0.3,  # Probability when threshold exceeded (reduced from 0.6)
    'azimuth_streak_strength': 5.0,     # Gain adjustment strength (±120%, way up from 40%)
    'azimuth_streak_width': 50,         # Width of streak in beams (increased from 30)
    
    # Grouped scattering (coherent noise patches)
    'grouped_scatter_enabled': True,    # Enable coherent noise patches
    'grouped_scatter_probability': 0.15, # Probability per beam (reduced from 0.4 for lower frequency)
    'grouped_scatter_width': 10,         # How many adjacent beams affected (increased from 5)
    'grouped_scatter_coherence': 2.0,   # How similar the scatter is (increased for visibility)
    'grouped_scatter_strength': 2.0,    # Scatter magnitude (±150%, way up from 60%)
    'grouped_scatter_additive_prob': 0.5,  # Probability that scatter adds energy (vs multiply)
    'grouped_scatter_disappear_prob': 0.35, # Probability that scatter removes energy
    'grouped_scatter_additive_boost': 2.0, # Energy multiplier when adding (creates bright spots)
    
    # Single-pixel jitter scatter behavior
    'jitter_additive_prob': 0.3,       # Probability that jittered pixel adds energy
    'jitter_disappear_prob': 0.25,      # Probability that jittered pixel disappears
    'jitter_additive_boost': 1.5,       # Energy multiplier for additive jitter
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
    'animation_interval': 33,  # Update interval in milliseconds (33ms = ~30 FPS)
    'dt': 0.033,               # Time step in seconds (MUST match interval: dt = interval/1000)
                               # For smooth 30 FPS video playback
                               # For 60 FPS (16ms): set interval=16, dt=0.016
    'animation_cache': False,  # Cache animation frames
    
    # Display normalization
    'db_normalization': 60,    # dB range for display normalization
    'sonar_colormap': 'gray',   # Colormap for sonar image (gray, hot, viridis, etc.)
    
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
    'move_speed': 0.01,         # Movement speed in meters per key press (0.1m = 10cm)
                               # For smooth 30 FPS: 0.1m feels like ~3 m/s if held
                               # Reduce to 0.05 for even slower, more deliberate movement
    'rotate_speed': 0.5,       # Rotation speed in degrees per key press
}

# ============================================================================
# FLOATING PARTICLE PARAMETERS
# ============================================================================
PARTICLE_CONFIG = {
    'enabled': True,            # Enable/disable floating particle system
    'max_particles': 500,       # Maximum number of active particles (increased from 200)
    'spawn_rate': 20.0,         # Particles spawned per second (increased from 5.0)
    'size_mean': 0.08,          # Mean particle size (exponential distribution)
    'size_min': 0.03,           # Minimum particle size
    'size_max': 0.25,           # Maximum particle size
    'lifetime_min': 2.0,        # Minimum particle lifetime (seconds)
    'lifetime_max': 10.0,       # Maximum particle lifetime (seconds)
    'drift_speed': 0.03,        # Maximum drift velocity
    'vertical_bob_speed': 2.0,  # Vertical bobbing frequency
    'vertical_bob_amplitude': 0.02,  # Vertical bobbing amplitude
    'density_bias': 0.7,        # How much to bias spawning toward dense areas (0-1)
    'density_threshold': 0.2,   # Density above which to boost particle spawning
}

# ============================================================================
# SCENE PARAMETERS
# ============================================================================
SCENE_CONFIG = {
    'world_size_m': 30.0,      # World is 30m x 30m
    'cage_center': [15.0, 15.0],  # Center of cage
    'cage_radius': 12.0,       # Cage radius in meters
    'num_sides': 12,           # Number of cage panel segments
    'current_direction': [0.0, 1.0],  # Southward
    'current_strength': 1.0,   # Current deflection strength
    'net_sag': 0.1,          # Maximum net sag in meters
    
    # Fish parameters
    'num_fish': 80,             # Number of fish in cage (0 = no fish)
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
# DATA COLLECTION PATH PARAMETERS
# ============================================================================
DATA_COLLECTION_CONFIG = {
    # Circular path movement speed
    'circular_path_duration_seconds': 200.0,  # Time to complete one full circle (at 30 FPS)
    
    # Circular path smooth variation (uses multiple sine waves)
    # Distance variation: r = base_radius + sum(sin(angle * freq) * amp) * radius_variation
    'radius_sine_waves': [
        {'frequency': 3.0, 'amplitude': 0.7},    # Main in/out pattern
        {'frequency': 7.0, 'amplitude': 0.5},    # High frequency detail
        {'frequency': 1.5, 'amplitude': 0.6},    # Slow drift
        {'frequency': 11.0, 'amplitude': 0.4},   # Even higher frequency variation
    ],
    
    # Orientation variation: angle_offset = sum(sin(angle * freq + phase) * amp) * orientation_noise
    # SET TO 0 to always face net directly with no orientation changes
    'orientation_sine_waves': [
        {'frequency': 5.0, 'amplitude': 0.0, 'phase_offset': 0.0},      # Disabled - always face net
        {'frequency': 11.0, 'amplitude': 0.0, 'phase_offset': 0.0},     # Disabled - always face net
        {'frequency': 2.0, 'amplitude': 0.0, 'phase_offset': 0.0},      # Disabled - always face net
    ],
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
        'reflectivity': 0.95,  # Increased from 0.8 for better visibility
        'scattering': 0.7,     # Increased from 0.6
        'absorption': 0.05,    # Decreased from 0.1 for less attenuation
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

