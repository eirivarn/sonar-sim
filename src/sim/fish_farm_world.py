"""Fish farm world configuration and setup."""
import numpy as np

from .world import World
from .primitives import Plane, Sphere, ClutterVolume
from .fish_cage import NetCage, FishSchool
from .config import WorldConfig, CageConfig, FishConfig


def build_fish_farm_world() -> tuple[World, NetCage, FishSchool]:
    """Create a realistic fish farm environment.
    
    Returns:
        tuple: (world, net_cage, fish_school)
    """
    w = World()
    
    # Fish cage - large tapered polygon net cage
    net_cage = NetCage(
        obj_id="fish_cage",
        center=CageConfig.CAGE_CENTER,
        radius_top=CageConfig.CAGE_RADIUS_TOP,
        radius_bottom=CageConfig.CAGE_RADIUS_BOTTOM,
        depth=CageConfig.CAGE_DEPTH,
        num_sides=CageConfig.NUM_SIDES,
        mesh_size=CageConfig.MESH_SIZE,
        rope_thickness=CageConfig.ROPE_THICKNESS,
        net_reflectivity=CageConfig.NET_REFLECTIVITY,
        rope_reflectivity=CageConfig.ROPE_REFLECTIVITY,
        has_bottom=CageConfig.HAS_BOTTOM,
        sag_factor=CageConfig.SAG_FACTOR
    )
    w.objects.append(net_cage)
    
    # Fish school - many fish scattered near perimeter with clustering
    fish_school = FishSchool(
        obj_id="fish_school",
        cage_center=CageConfig.CAGE_CENTER,
        cage_radius_top=CageConfig.CAGE_RADIUS_TOP,
        cage_radius_bottom=CageConfig.CAGE_RADIUS_BOTTOM,
        cage_depth=CageConfig.CAGE_DEPTH,
        num_fish=FishConfig.NUM_FISH,
        reflectivity=FishConfig.FISH_REFLECTIVITY,
        neighbor_distance=FishConfig.NEIGHBOR_DISTANCE
    )
    
    # Add fish to world only if enabled in config
    if FishConfig.ENABLE_FISH_IN_WORLD:
        w.objects.append(fish_school)
    
    # WATER COLUMN CLUTTER - simulates plankton, particles, suspended matter
    # Covers the entire operational area with probabilistic scatterers
    clutter_volume = ClutterVolume(
        obj_id="water_clutter",
        bmin=np.array([-30.0, -40.0, -40.0]),  # Larger volume for bigger cage
        bmax=np.array([80.0, 40.0, 0.0]),      # Up to surface
        base_prob=0.20,  # 20% base chance of hit per ray - increased for more debris
        reflectivity_min=0.03,  # Weak scatterers with some reflection
        reflectivity_max=0.18,  # Higher reflectivity for visible debris
        depth_influence=0.3,  # More clutter near surface
        surface_depth=0.0,
        feeding_mode=False  # Can be toggled for high-clutter events
    )
    w.objects.append(clutter_volume)
    
    return w, net_cage, fish_school


def get_default_sonar_config():
    """Get default sonar configuration for fish farm viewing.
    
    Returns:
        dict: SonarV2 configuration parameters
    """
    return {
        'pos': np.array([0.0, -10.0, -12.0], dtype=float),  # Further back and deeper for larger cage
        'rpy': np.array([0.0, 0.0, np.deg2rad(90)], dtype=float),  # Looking toward cage
        'range_m': 20.0,  # Shorter range for focused view
        'hfov_deg': 120.0,
        'h_beams': 256,
        'enable_multipath': True,
        'enable_realistic_effects': True,
        'range_bins': 1024,  
        'alpha_db_per_m': 0.05,
        'speckle_looks': 2.0,
        'edge_strength_db': 6.0,
    }

