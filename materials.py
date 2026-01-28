"""Material definitions and properties for sonar simulation."""
from dataclasses import dataclass
from config import MATERIAL_CONFIG


@dataclass
class Material:
    """Material properties for voxel grid."""
    name: str
    density: float          # 0-1: how much matter is here
    reflectivity: float     # 0-1: acoustic backscatter strength
    absorption: float       # 0-1: how much energy is absorbed per meter
    material_id: int = 0    # Unique ID for ground truth segmentation


# Material IDs for ground truth segmentation
MATERIAL_ID_EMPTY = 0
MATERIAL_ID_NET = 1
MATERIAL_ID_ROPE = 2
MATERIAL_ID_FISH = 3
MATERIAL_ID_WALL = 4
MATERIAL_ID_BIOMASS = 5
MATERIAL_ID_DEBRIS_LIGHT = 6
MATERIAL_ID_DEBRIS_MEDIUM = 7
MATERIAL_ID_DEBRIS_HEAVY = 8
# Urban material IDs
MATERIAL_ID_CONCRETE = 9
MATERIAL_ID_WOOD = 10
MATERIAL_ID_FOLIAGE = 11
MATERIAL_ID_METAL = 12
MATERIAL_ID_GLASS = 13


# Material library - load from config
EMPTY = Material("empty", 0.0, 0.0, 0.0, MATERIAL_ID_EMPTY)

NET = Material("net", MATERIAL_CONFIG['net']['reflectivity'], 
               MATERIAL_CONFIG['net']['scattering'], 
               MATERIAL_CONFIG['net']['absorption'], MATERIAL_ID_NET)

ROPE = Material("rope", MATERIAL_CONFIG['rope']['reflectivity'], 
                MATERIAL_CONFIG['rope']['scattering'], 
                MATERIAL_CONFIG['rope']['absorption'], MATERIAL_ID_ROPE)

FISH = Material("fish", MATERIAL_CONFIG['fish']['reflectivity'], 
                MATERIAL_CONFIG['fish']['scattering'], 
                MATERIAL_CONFIG['fish']['absorption'], MATERIAL_ID_FISH)

WALL = Material("wall", MATERIAL_CONFIG['wall']['reflectivity'], 
                MATERIAL_CONFIG['wall']['scattering'], 
                MATERIAL_CONFIG['wall']['absorption'], MATERIAL_ID_WALL)

BIOMASS = Material("biomass", MATERIAL_CONFIG['biomass']['reflectivity'], 
                   MATERIAL_CONFIG['biomass']['scattering'], 
                   MATERIAL_CONFIG['biomass']['absorption'], MATERIAL_ID_BIOMASS)

DEBRIS_LIGHT = Material("debris_light", MATERIAL_CONFIG['debris_light']['reflectivity'], 
                        MATERIAL_CONFIG['debris_light']['scattering'], 
                        MATERIAL_CONFIG['debris_light']['absorption'], MATERIAL_ID_DEBRIS_LIGHT)

DEBRIS_MEDIUM = Material("debris_medium", MATERIAL_CONFIG['debris_medium']['reflectivity'], 
                         MATERIAL_CONFIG['debris_medium']['scattering'], 
                         MATERIAL_CONFIG['debris_medium']['absorption'], MATERIAL_ID_DEBRIS_MEDIUM)

DEBRIS_HEAVY = Material("debris_heavy", MATERIAL_CONFIG['debris_heavy']['reflectivity'], 
                        MATERIAL_CONFIG['debris_heavy']['scattering'], 
                        MATERIAL_CONFIG['debris_heavy']['absorption'], MATERIAL_ID_DEBRIS_HEAVY)

# Urban materials
CONCRETE = Material("concrete", MATERIAL_CONFIG['concrete']['reflectivity'], 
                    MATERIAL_CONFIG['concrete']['scattering'], 
                    MATERIAL_CONFIG['concrete']['absorption'], MATERIAL_ID_CONCRETE)

WOOD = Material("wood", MATERIAL_CONFIG['wood']['reflectivity'], 
                MATERIAL_CONFIG['wood']['scattering'], 
                MATERIAL_CONFIG['wood']['absorption'], MATERIAL_ID_WOOD)

FOLIAGE = Material("foliage", MATERIAL_CONFIG['foliage']['reflectivity'], 
                   MATERIAL_CONFIG['foliage']['scattering'], 
                   MATERIAL_CONFIG['foliage']['absorption'], MATERIAL_ID_FOLIAGE)

METAL = Material("metal", MATERIAL_CONFIG['metal']['reflectivity'], 
                 MATERIAL_CONFIG['metal']['scattering'], 
                 MATERIAL_CONFIG['metal']['absorption'], MATERIAL_ID_METAL)

GLASS = Material("glass", MATERIAL_CONFIG['glass']['reflectivity'], 
                 MATERIAL_CONFIG['glass']['scattering'], 
                 MATERIAL_CONFIG['glass']['absorption'], MATERIAL_ID_GLASS)
