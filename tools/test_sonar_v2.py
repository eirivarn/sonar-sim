"""Example script showing how to use the new high-realism SonarV2.

The new sonar implements all the realism improvements:
- Direct 2D polar accumulation (no early collapse to 1D)
- Deterministic cone sampling with beam pattern weighting
- Multi-hit tracing for porous objects (nets)
- Structured multipath (surface/seafloor mirror method)
- Water column clutter injection
- Broadphase bounds optimization
"""
import sys
sys.path.insert(0, '/Users/eirikvarnes/code/sonar-sim')

import numpy as np
import matplotlib.pyplot as plt

from src.sim.sonar import Sonar
from src.sim.fish_farm_world import build_fish_farm_world

# Build the world
world, net_cage, fish_school = build_fish_farm_world()

# Create high-realism sonar
sonar = Sonar(
    pos=np.array([0.0, -10.0, -12.0]),
    rpy=np.array([0.0, 0.0, np.deg2rad(90)])
)

print("Sonar Configuration:")
print(f"  Beams: {sonar.h_beams}")
print(f"  Range bins: {sonar.range_bins}")
print(f"  Rays per beam: {sonar.rays_per_beam}")
print(f"  Max hits per ray: {sonar.max_hits_per_ray}")
print(f"  Deterministic cone: {sonar.use_deterministic_cone}")
print(f"  Structured multipath: {sonar.enable_structured_multipath}")

# Scan the scene
print("\nScanning...")
scan_data = sonar.scan_2d(world)

# Display results
polar_image = scan_data['polar_image']
print(f"\nPolar image shape: {polar_image.shape}")
print(f"Peak intensity: {np.max(polar_image):.4f}")
print(f"Mean intensity: {np.mean(polar_image[polar_image > 0]):.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 2D polar image (new!)
ax1.imshow(polar_image, aspect='auto', origin='lower', cmap='hot')
ax1.set_xlabel('Beam Index')
ax1.set_ylabel('Range Bin')
ax1.set_title('2D Polar Sonar Image (High Realism)')

# 1D range plot (legacy compatibility)
distances = np.array(scan_data['distances'])
intensities = np.array(scan_data['intensities'])

beam_angles = np.linspace(-45, 45, len(distances))
ax2.plot(beam_angles, distances, 'b-', alpha=0.7)
ax2.fill_between(beam_angles, 0, distances, alpha=0.3)
ax2.set_xlabel('Beam Angle (deg)')
ax2.set_ylabel('Range (m)')
ax2.set_title('Peak Range per Beam')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('sonar_v2_example.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to sonar_v2_example.png")
plt.show()
