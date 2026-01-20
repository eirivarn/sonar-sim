from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .math3d import rpy_to_R, unit
from .world import World
from .sonar_effects import render_realistic_sonar
from .config import SonarConfig, SignalProcessingConfig

@dataclass
class Sonar:
    pos: np.ndarray              # shape (3,)
    rpy: np.ndarray              # (roll,pitch,yaw) in rad
    range_m: float = SonarConfig.RANGE_M
    hfov_deg: float = SonarConfig.HFOV_DEG
    h_beams: int = SonarConfig.H_BEAMS
    attenuation: float = SonarConfig.ATTENUATION
    noise_std: float = SonarConfig.NOISE_STD
    enable_multipath: bool = SonarConfig.ENABLE_MULTIPATH
    enable_noise: bool = SonarConfig.ENABLE_NOISE
    enable_realistic_effects: bool = SonarConfig.ENABLE_REALISTIC_EFFECTS
    range_bins: int = SonarConfig.RANGE_BINS
    # Realistic sonar parameters
    alpha_db_per_m: float = SonarConfig.ALPHA_DB_PER_M
    speckle_looks: float = SignalProcessingConfig.SPECKLE_LOOKS
    edge_strength_db: float = SonarConfig.EDGE_STRENGTH_DB

    def scan_2d(self, world: World) -> dict:
        """Returns dict with distances and intensities with realistic artifacts."""
        R = rpy_to_R(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2]))
        hfov = np.deg2rad(self.hfov_deg)

        distances = np.full(self.h_beams, self.range_m, dtype=np.float32)
        intensities = np.zeros(self.h_beams, dtype=np.float32)
        multipath = []  # Store secondary returns

        for i in range(self.h_beams):
            t = 0.5 if self.h_beams == 1 else i / (self.h_beams - 1)
            yaw = (-hfov * 0.5) + t * hfov

            # yaw around local +Z (up), so the scan is in XY plane
            cy, sy = np.cos(yaw), np.sin(yaw)
            dir_local = np.array([cy, sy, 0.0], dtype=float)

            rd = unit(R @ dir_local)
            
            # Primary return with multipath
            incident_strength = 1.0
            beam_returns = []
            
            if self.enable_multipath:
                # Cast multiple times for multipath
                current_pos = self.pos.copy()
                for bounce in range(3):  # Up to 3 bounces
                    hit = world.raycast(current_pos, rd, self.range_m)
                    if hit is None:
                        break
                    
                    # Apply water attenuation
                    total_dist = np.linalg.norm(hit.point - self.pos)
                    attenuated_strength = incident_strength * np.exp(-self.attenuation * total_dist) * 0.8
                    
                    # Material reflection/transmission
                    reflected = attenuated_strength * hit.reflectivity
                    transmitted = attenuated_strength * (1 - hit.reflectivity)
                    
                    if reflected > 0.05:  # Only record if strong enough
                        beam_returns.append((total_dist, reflected))
                    
                    # Continue with transmitted ray
                    incident_strength = transmitted
                    if incident_strength < 0.1:
                        break
                    
                    # Continue from hit point (slight offset to avoid self-intersection)
                    current_pos = hit.point + rd * 0.01
            else:
                # Simple single return
                hit = world.raycast(self.pos, rd, self.range_m)
                if hit is not None:
                    dist = float(hit.t)
                    attenuated = np.exp(-self.attenuation * dist) * 0.8
                    reflected = attenuated * hit.reflectivity
                    beam_returns.append((dist, reflected))
            
            # Take strongest return as primary
            if beam_returns:
                beam_returns.sort(key=lambda x: x[1], reverse=True)  # Sort by intensity
                primary_dist, primary_intensity = beam_returns[0]
                
                # Add noise if enabled (legacy mode only)

                if self.enable_noise and not self.enable_realistic_effects:
                    range_noise = np.random.normal(0, self.noise_std)
                    intensity_noise = np.random.uniform(0.9, 1.1)
                    primary_dist = max(0, primary_dist + range_noise)
                    primary_intensity *= intensity_noise
                
                distances[i] = primary_dist
                intensities[i] = primary_intensity
                
                # Store multipath for visualization
                if len(beam_returns) > 1:
                    multipath.extend([(i, r[0], r[1]) for r in beam_returns[1:]])

        # Apply realistic sonar effects if enabled
        if self.enable_realistic_effects:
            intensities = self._apply_realistic_effects(distances, intensities)

        return {
            "pos": self.pos.tolist(),
            "rpy": self.rpy.tolist(),
            "range_m": self.range_m,
            "hfov_deg": self.hfov_deg,
            "h_beams": self.h_beams,
            "distances": distances.tolist(),
            "intensities": intensities.tolist(),
            "multipath": multipath,  # (beam_idx, distance, intensity)
        }
    
    def _apply_realistic_effects(self, distances, intensities):
        """Apply realistic sonar processing effects to create polar image.
        
        Converts 1D beam data (distances, intensities) into 2D polar grid,
        applies realistic sonar effects, then extracts peak intensities.
        """
        H = self.range_bins
        W = self.h_beams
        
        # Build 2D polar intensity grid (range x beam)
        mu = np.zeros((H, W), dtype=np.float32)
        r_m = np.linspace(0, self.range_m, H)
        
        # Fill grid: for each beam, place intensity at appropriate range bin
        for j in range(W):
            if distances[j] < self.range_m and intensities[j] > 0:
                # Find closest range bin
                r_idx = int((distances[j] / self.range_m) * (H - 1))
                if 0 <= r_idx < H:
                    # Spread intensity over a few bins (simulate pulse length)
                    for offset in range(-2, 3):
                        idx = r_idx + offset
                        if 0 <= idx < H:
                            weight = np.exp(-0.5 * (offset / 1.5)**2)  # Gaussian spread
                            mu[idx, j] += intensities[j] * weight
        
        # Apply full realistic sonar signal chain
        enhanced = render_realistic_sonar(
            mu, r_m,
            alpha_db_per_m=self.alpha_db_per_m,
            edge_strength_db=self.edge_strength_db,
            looks=self.speckle_looks
        )
        
        # Extract peak intensities per beam (for backward compatibility)
        # In realistic mode, we return the enhanced 2D image characteristics
        output_intensities = np.zeros(W, dtype=np.float32)
        for j in range(W):
            # Find peak in this beam's range profile
            output_intensities[j] = np.max(enhanced[:, j])
        
        return output_intensities

