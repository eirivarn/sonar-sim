"""Improved sonar with direct 2D polar accumulation and realistic physics.

This module implements the high-realism sonar measurement model:
- Direct accumulation into 2D polar grid (no early 1D collapse)
- Deterministic cone sampling with beam pattern weighting
- Multi-hit tracing for porous objects (nets)
- Structured multipath (surface/seafloor mirror method)
- Water column clutter injection
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import convolve1d
from .math3d import rpy_to_R
from .world import World
from .sonar_effects import render_realistic_sonar
from .config import SonarConfig, SignalProcessingConfig, WorldConfig

@dataclass
class Sonar:
    """High-realism sonar with direct 2D polar accumulation."""
    pos: np.ndarray              # shape (3,)
    rpy: np.ndarray              # (roll,pitch,yaw) in rad
    range_m: float = SonarConfig.RANGE_M
    hfov_deg: float = SonarConfig.HFOV_DEG
    h_beams: int = SonarConfig.H_BEAMS
    beamwidth_deg: float = SonarConfig.BEAMWIDTH_DEG
    rays_per_beam: int = SonarConfig.RAYS_PER_BEAM
    max_hits_per_ray: int = SonarConfig.MAX_HITS_PER_RAY
    min_hit_strength: float = SonarConfig.MIN_HIT_STRENGTH
    transmission_boost: float = SonarConfig.TRANSMISSION_BOOST
    range_bins: int = SonarConfig.RANGE_BINS
    alpha_db_per_m: float = SonarConfig.ALPHA_DB_PER_M
    speckle_looks: float = SignalProcessingConfig.SPECKLE_LOOKS
    edge_strength_db: float = SonarConfig.EDGE_STRENGTH_DB
    enable_realistic_effects: bool = SonarConfig.ENABLE_REALISTIC_EFFECTS
    enable_multipath: bool = SonarConfig.ENABLE_MULTIPATH
    enable_structured_multipath: bool = SonarConfig.ENABLE_STRUCTURED_MULTIPATH
    use_deterministic_cone: bool = SonarConfig.USE_DETERMINISTIC_CONE
    
    # Precomputed cone pattern (deterministic)
    _cone_pattern: np.ndarray = field(default=None, init=False, repr=False)
    _beam_weights: np.ndarray = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Precompute deterministic cone sampling pattern."""
        if self.use_deterministic_cone and self.rays_per_beam > 1:
            self._precompute_cone_pattern()
    
    def _precompute_cone_pattern(self):
        """Generate deterministic ray offsets and beam pattern weights."""
        beamwidth_rad = np.deg2rad(self.beamwidth_deg)
        sigma_rad = np.deg2rad(SonarConfig.BEAM_PATTERN_SIGMA_DEG)
        
        if self.rays_per_beam == 1:
            self._cone_pattern = np.array([[0.0, 0.0]])
            self._beam_weights = np.array([1.0])
        else:
            # Generate pattern: center ray + circular arrangement
            offsets = []
            weights = []
            
            # Center ray
            offsets.append([0.0, 0.0])
            weights.append(1.0)
            
            # Arrange remaining rays in concentric circles
            n_remaining = self.rays_per_beam - 1
            
            if n_remaining >= 4:
                # Inner ring (40% of beamwidth)
                n_inner = n_remaining // 2
                for i in range(n_inner):
                    angle = 2.0 * np.pi * i / n_inner
                    r = 0.4 * beamwidth_rad
                    yaw_off = r * np.cos(angle)
                    pitch_off = r * np.sin(angle)
                    offsets.append([yaw_off, pitch_off])
                    
                    # Gaussian beam pattern weight
                    r_sq = yaw_off**2 + pitch_off**2
                    w = np.exp(-r_sq / (2 * sigma_rad**2))
                    weights.append(w)
                
                # Outer ring (80% of beamwidth)
                n_outer = n_remaining - n_inner
                for i in range(n_outer):
                    angle = 2.0 * np.pi * i / n_outer + np.pi / n_outer  # Offset for stagger
                    r = 0.8 * beamwidth_rad
                    yaw_off = r * np.cos(angle)
                    pitch_off = r * np.sin(angle)
                    offsets.append([yaw_off, pitch_off])
                    
                    r_sq = yaw_off**2 + pitch_off**2
                    w = np.exp(-r_sq / (2 * sigma_rad**2))
                    weights.append(w)
            else:
                # Simple circular pattern for few rays
                for i in range(n_remaining):
                    angle = 2.0 * np.pi * i / n_remaining
                    r = 0.6 * beamwidth_rad
                    yaw_off = r * np.cos(angle)
                    pitch_off = r * np.sin(angle)
                    offsets.append([yaw_off, pitch_off])
                    
                    r_sq = yaw_off**2 + pitch_off**2
                    w = np.exp(-r_sq / (2 * sigma_rad**2))
                    weights.append(w)
            
            self._cone_pattern = np.array(offsets)
            self._beam_weights = np.array(weights)
            # Normalize weights to sum to rays_per_beam (preserve energy)
            self._beam_weights *= self.rays_per_beam / np.sum(self._beam_weights)
    
    def scan_2d(self, world: World) -> dict:
        """Scan world and return 2D polar image with realistic artifacts.
        
        Returns:
            dict with:
                - polar_image: (range_bins, h_beams) enhanced sonar image
                - distances: (h_beams,) peak range per beam (legacy)
                - intensities: (h_beams,) peak intensity per beam (legacy)
                - pos, rpy, range_m, hfov_deg, h_beams: sonar parameters
        """
        # Initialize 2D polar accumulation grid
        mu = np.zeros((self.range_bins, self.h_beams), dtype=np.float32)
        
        # Get rotation matrix
        R = rpy_to_R(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2]))
        hfov = np.deg2rad(self.hfov_deg)
        
        # Range array for binning
        r_bins = np.linspace(0, self.range_m, self.range_bins)
        dr = self.range_m / self.range_bins
        
        # Scan each beam
        for beam_idx in range(self.h_beams):
            t = 0.5 if self.h_beams == 1 else beam_idx / (self.h_beams - 1)
            yaw_center = (-hfov * 0.5) + t * hfov
            
            # Cast rays in cone for this beam
            for ray_idx in range(self.rays_per_beam):
                # Get ray direction with cone offset
                if self.use_deterministic_cone and self._cone_pattern is not None:
                    yaw_off, pitch_off = self._cone_pattern[ray_idx]
                    weight = self._beam_weights[ray_idx]
                else:
                    # Random cone sampling (legacy)
                    beamwidth_rad = np.deg2rad(self.beamwidth_deg)
                    if ray_idx == 0:
                        yaw_off, pitch_off = 0.0, 0.0
                        weight = 1.0
                    else:
                        angle = 2.0 * np.pi * (ray_idx - 1) / max(1, self.rays_per_beam - 1)
                        r = beamwidth_rad * np.sqrt(np.random.uniform(0.3, 1.0))
                        yaw_off = r * np.cos(angle)
                        pitch_off = r * np.sin(angle)
                        sigma_rad = np.deg2rad(SonarConfig.BEAM_PATTERN_SIGMA_DEG)
                        weight = np.exp(-(yaw_off**2 + pitch_off**2) / (2 * sigma_rad**2))
                
                yaw_total = yaw_center + yaw_off
                pitch_total = pitch_off
                
                cy, sy = np.cos(yaw_total), np.sin(yaw_total)
                cp, sp = np.cos(pitch_total), np.sin(pitch_total)
                rd = R @ np.array([cy * cp, sy * cp, sp], dtype=float)
                rd = rd / np.linalg.norm(rd)
                
                # Multi-hit tracing along this ray
                hits = self._trace_ray_multihit(world, self.pos, rd)
                
                # Deposit hits into mu
                for hit_dist, hit_intensity in hits:
                    if hit_dist < self.range_m:
                        # Weighted energy deposition
                        energy = hit_intensity * weight
                        
                        # Find range bin and deposit directly
                        r_idx = int(hit_dist / dr)
                        if 0 <= r_idx < self.range_bins:
                            mu[r_idx, beam_idx] += energy
                
                # Structured multipath (surface/seafloor mirrors)
                if self.enable_multipath and self.enable_structured_multipath:
                    multipath_hits = self._trace_structured_multipath(world, rd)
                    for hit_dist, hit_intensity in multipath_hits:
                        if hit_dist < self.range_m:
                            energy = hit_intensity * weight * 0.5  # Attenuate multipath
                            r_idx = int(hit_dist / dr)
                            if 0 <= r_idx < self.range_bins:
                                mu[r_idx, beam_idx] += energy
        
        # Inject water column clutter
        self._inject_clutter(mu, r_bins)
        
        # Apply 2D spreading to create continuous surfaces
        # This handles both angular (beam width) and range (pulse length) spreading
        
        # 1. Range spreading - creates continuous lines in range direction
        range_spread_sigma = SonarConfig.PULSE_LENGTH_BINS
        if range_spread_sigma > 0:
            kernel_width = int(np.ceil(range_spread_sigma * 3))
            kernel = np.exp(-0.5 * (np.arange(-kernel_width, kernel_width + 1) / range_spread_sigma)**2)
            kernel = kernel / kernel.sum()
            # Apply along range axis (axis=0)
            mu = convolve1d(mu, kernel, axis=0, mode='nearest')
        
        # 2. Angular spreading - creates continuous lines in beam direction
        beam_spread_sigma = SonarConfig.ANGULAR_SPREAD_BEAMS
        if beam_spread_sigma > 0:
            kernel_width = int(np.ceil(beam_spread_sigma * 3))
            kernel = np.exp(-0.5 * (np.arange(-kernel_width, kernel_width + 1) / beam_spread_sigma)**2)
            kernel = kernel / kernel.sum()
            # Apply along beam axis (axis=1)
            mu = convolve1d(mu, kernel, axis=1, mode='wrap')
        
        # Apply realistic sonar effects
        if self.enable_realistic_effects:
            enhanced = render_realistic_sonar(
                mu, r_bins,
                alpha_db_per_m=self.alpha_db_per_m,
                edge_strength_db=self.edge_strength_db,
                looks=self.speckle_looks
            )
        else:
            enhanced = mu
        
        # Extract 1D peaks for legacy compatibility
        distances = np.zeros(self.h_beams, dtype=np.float32)
        intensities = np.zeros(self.h_beams, dtype=np.float32)
        
        for j in range(self.h_beams):
            peak_idx = np.argmax(enhanced[:, j])
            if enhanced[peak_idx, j] > 1e-6:
                distances[j] = r_bins[peak_idx]
                intensities[j] = enhanced[peak_idx, j]
            else:
                distances[j] = self.range_m
                intensities[j] = 0.0
        
        return {
            "polar_image": enhanced,  # NEW: Full 2D image
            "distances": distances.tolist(),
            "intensities": intensities.tolist(),
            "pos": self.pos.tolist(),
            "rpy": self.rpy.tolist(),
            "range_m": self.range_m,
            "hfov_deg": self.hfov_deg,
            "h_beams": self.h_beams,
        }
    
    def _trace_ray_multihit(self, world: World, ro: np.ndarray, rd: np.ndarray) -> list:
        """Trace ray and collect multiple hits (for porous objects like nets).
        
        Returns:
            List of (distance, intensity) tuples
        """
        hits = []
        current_pos = ro.copy()
        incident_strength = 1.0
        remaining_range = self.range_m
        
        for _ in range(self.max_hits_per_ray):
            if incident_strength < self.min_hit_strength or remaining_range <= 0:
                break
            
            # Cast ray from current position
            hit = world.raycast(current_pos, rd, remaining_range)
            if hit is None:
                break
            
            # Calculate distance from original source
            total_dist = np.linalg.norm(hit.point - ro)
            
            # Apply attenuation (water absorption + spreading)
            # Convert dB/m to linear attenuation: alpha_linear = ln(10) * alpha_db / 10
            alpha_linear = np.log(10) * self.alpha_db_per_m / 10.0
            attenuated_strength = incident_strength * np.exp(-alpha_linear * total_dist)
            attenuated_strength *= (1.0 / (1.0 + total_dist / 10.0))  # Range-dependent spreading
            
            # Reflection and transmission
            reflected = attenuated_strength * hit.reflectivity
            # Apply transmission boost to reduce signal loss through objects
            transmitted = attenuated_strength * (1.0 - hit.reflectivity) * self.transmission_boost
            
            # Record hit
            if reflected > self.min_hit_strength:
                hits.append((total_dist, reflected))
            
            # Continue tracing with transmitted energy
            incident_strength = transmitted
            remaining_range -= hit.t
            current_pos = hit.point + rd * 0.01  # Offset to avoid self-intersection
        
        return hits
    
    def _trace_structured_multipath(self, world: World, rd: np.ndarray) -> list:
        """Trace structured multipath (surface/seafloor mirror reflections).
        
        Returns:
            List of (distance, intensity) tuples for multipath returns
        """
        multipath_hits = []
        
        # Surface reflection (mirror across z=0)
        if self.pos[2] < 0:  # Sonar below surface
            # Mirror sonar position across surface
            ro_mirror = self.pos.copy()
            ro_mirror[2] = -ro_mirror[2]
            
            # Mirror ray direction across surface
            rd_mirror = rd.copy()
            rd_mirror[2] = -rd_mirror[2]
            
            # Trace from mirror position
            hit = world.raycast(ro_mirror, rd_mirror, self.range_m)
            if hit is not None:
                # Total path: sonar -> surface -> target -> surface -> sonar
                path_to_surface = abs(self.pos[2])
                surface_to_target = np.linalg.norm(hit.point - ro_mirror)
                target_to_sonar = np.linalg.norm(hit.point - self.pos)
                total_dist = 2 * path_to_surface + surface_to_target
                
                # Attenuate: two surface bounces + water path
                surface_refl = SonarConfig.SURFACE_REFLECTIVITY
                strength = surface_refl**2 * hit.reflectivity
                alpha_linear = np.log(10) * self.alpha_db_per_m / 10.0
                strength *= np.exp(-alpha_linear * total_dist)
                strength *= (1.0 / (1.0 + total_dist / 10.0))
                
                if strength > self.min_hit_strength:
                    multipath_hits.append((total_dist, strength))
        
        # Seafloor reflection (mirror across seafloor)
        seafloor_z = -WorldConfig.SEAFLOOR_DEPTH
        if self.pos[2] > seafloor_z:  # Sonar above seafloor
            # Mirror sonar position across seafloor
            ro_mirror = self.pos.copy()
            ro_mirror[2] = 2 * seafloor_z - ro_mirror[2]
            
            # Mirror ray direction
            rd_mirror = rd.copy()
            rd_mirror[2] = -rd_mirror[2]
            
            # Trace from mirror position
            hit = world.raycast(ro_mirror, rd_mirror, self.range_m)
            if hit is not None:
                path_to_floor = abs(self.pos[2] - seafloor_z)
                floor_to_target = np.linalg.norm(hit.point - ro_mirror)
                total_dist = 2 * path_to_floor + floor_to_target
                
                # Attenuate: two floor bounces + water path
                floor_refl = WorldConfig.SEAFLOOR_REFLECTIVITY
                strength = floor_refl**2 * hit.reflectivity
                alpha_linear = np.log(10) * self.alpha_db_per_m / 10.0
                strength *= np.exp(-alpha_linear * total_dist)
                strength *= (1.0 / (1.0 + total_dist / 10.0))
                
                if strength > self.min_hit_strength:
                    multipath_hits.append((total_dist, strength))
        
        return multipath_hits
    
    def _inject_clutter(self, mu: np.ndarray, r_bins: np.ndarray):
        """Inject water column clutter directly into mu."""
        H, W = mu.shape
        density = SignalProcessingConfig.CLUTTER_DENSITY
        intensity_min = SignalProcessingConfig.CLUTTER_INTENSITY_MIN
        intensity_max = SignalProcessingConfig.CLUTTER_INTENSITY_MAX
        decay = SignalProcessingConfig.CLUTTER_RANGE_DECAY
        
        # Sparse random clutter
        for i in range(H):
            for j in range(W):
                range_m = r_bins[i]
                # Range-dependent density (less clutter far away)
                eff_density = density * np.exp(-decay * range_m)
                
                if np.random.rand() < eff_density:
                    # Add clutter impulse
                    intensity = np.random.uniform(intensity_min, intensity_max)
                    mu[i, j] += intensity
