"""Compare old Sonar vs new SonarV2 side-by-side.

NOTE: This script is for documentation purposes only.
The old Sonar class has been removed. This script demonstrates
the conceptual differences between the old approach (1D collapse)
and the new approach (2D accumulation).
"""
import sys
sys.path.insert(0, '/Users/eirikvarnes/code/sonar-sim')

import numpy as np
import matplotlib.pyplot as plt

from src.sim.sonar import Sonar
from src.sim.fish_farm_world import build_fish_farm_world

# Build the world
world, net_cage, fish_school = build_fish_farm_world()

sonar_pos = np.array([0.0, -10.0, -12.0])
sonar_rpy = np.array([0.0, 0.0, np.deg2rad(90)])

# Simulate "old" behavior by using Sonar with minimal features
print("Simulating old sonar (Sonar with legacy settings)...")
sonar_old = Sonar(
    pos=sonar_pos, 
    rpy=sonar_rpy,
    max_hits_per_ray=1,  # Single hit only
    use_deterministic_cone=False,  # Random cone
    enable_structured_multipath=False,  # No structured multipath
)
sonar_old._cone_pattern = None  # Force random sampling
scan_old = sonar_old.scan_2d(world)

# New sonar with all features
print("Scanning with new Sonar (full features)...")
sonar_new = Sonar(pos=sonar_pos, rpy=sonar_rpy)
scan_new = sonar_new.scan_2d(world)

# Create comparison plot
fig = plt.figure(figsize=(16, 10))

# Old-style sonar - extract polar image
ax1 = plt.subplot(2, 2, 1)
polar_old = scan_old['polar_image']
ax1.imshow(polar_old, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=np.percentile(polar_old, 99))
ax1.set_title('Legacy Mode (Single Hit, Random Cone)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Beam Index')
ax1.set_ylabel('Range Bin')

# New sonar - full 2D polar image
ax2 = plt.subplot(2, 2, 2)
polar_new = scan_new['polar_image']
ax2.imshow(polar_new, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=np.percentile(polar_new, 99))
ax2.set_title('High-Realism Mode (Multi-Hit, Deterministic)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Beam Index')
ax2.set_ylabel('Range Bin')

# Range comparison
ax3 = plt.subplot(2, 2, 3)
beam_angles = np.linspace(-45, 45, len(scan_old['distances']))
distances_old = np.array(scan_old['distances'])
ax3.plot(beam_angles, distances_old, 'b-', alpha=0.7, label='Legacy Mode', linewidth=2)

distances_new = np.array(scan_new['distances'])
ax3.plot(beam_angles, distances_new, 'r-', alpha=0.7, label='High-Realism Mode', linewidth=2)

ax3.fill_between(beam_angles, 0, distances_old, alpha=0.2, color='blue')
ax3.fill_between(beam_angles, 0, distances_new, alpha=0.2, color='red')
ax3.set_xlabel('Beam Angle (deg)')
ax3.set_ylabel('Range (m)')
ax3.set_title('Peak Range per Beam Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_yaxis()

# Feature comparison text
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

comparison_text = f"""
FEATURE COMPARISON

Legacy Mode (minimal features):
  • Random cone sampling (flicker)
  • Single hit per ray
  • No structured multipath
  • Still uses 2D accumulation
  
High-Realism Mode (all features):
  • Deterministic cone pattern
  • Multi-hit tracing ({sonar_new.max_hits_per_ray} hits/ray)
  • Structured multipath (mirrors)
  • Direct 2D polar accumulation
  • Water column clutter
  • Broadphase bounds optimization
  
Performance:
  Legacy: {len(distances_old)} beams × {sonar_old.rays_per_beam} rays
          = {len(distances_old) * sonar_old.rays_per_beam} total rays
  
  Realism: {len(distances_new)} beams × {sonar_new.rays_per_beam} rays
           × {sonar_new.max_hits_per_ray} hits
           = ~{len(distances_new) * sonar_new.rays_per_beam * sonar_new.max_hits_per_ray} raycasts
           (+ multipath + broadphase speedup)

Output Format:
  Both: 2D polar image ({polar_new.shape}) + 1D legacy peaks
  
Note: Old Sonar class removed - using current Sonar 
      with different feature flags for comparison.
"""

ax4.text(0.1, 0.5, comparison_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Sonar Comparison: Legacy vs High-Realism', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sonar_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved comparison to sonar_comparison.png")
plt.show()
