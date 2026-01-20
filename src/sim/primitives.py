from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPS = 1e-9

@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    obj_id: str
    reflectivity: float = 0.75  # Material reflectivity (0-1)

class Primitive:
    obj_id: str
    reflectivity: float = 0.75  # Default reflectivity
    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        raise NotImplementedError

@dataclass
class Plane(Primitive):
    obj_id: str
    point: np.ndarray     # point on plane
    normal: np.ndarray    # unit normal
    reflectivity: float = 0.15  # Weak reflector (ground/sediment)

    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        n = self.normal
        denom = float(np.dot(n, rd))
        if abs(denom) < EPS:
            return None
        t = float(np.dot(n, (self.point - ro)) / denom)
        if t <= 0:
            return None
        p = ro + t * rd
        # Ensure normal points "against" the ray for consistency
        nn = n if np.dot(n, rd) < 0 else -n
        return Hit(t=t, point=p, normal=nn, obj_id=self.obj_id, reflectivity=self.reflectivity)

@dataclass
class Sphere(Primitive):
    obj_id: str
    center: np.ndarray
    radius: float
    reflectivity: float = 0.33  # Moderate reflector (debris)

    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        oc = ro - self.center
        b = float(np.dot(oc, rd))
        c = float(np.dot(oc, oc) - self.radius * self.radius)
        disc = b*b - c
        if disc < 0:
            return None
        s = float(np.sqrt(disc))
        t0 = -b - s
        t1 = -b + s
        t = t0 if t0 > EPS else (t1 if t1 > EPS else None)
        if t is None:
            return None
        p = ro + t * rd
        n = (p - self.center) / self.radius
        return Hit(t=float(t), point=p, normal=n, obj_id=self.obj_id, reflectivity=self.reflectivity)

@dataclass
class AABB(Primitive):
    """Axis-aligned bounding box defined by min and max corners."""
    obj_id: str
    bmin: np.ndarray
    bmax: np.ndarray
    reflectivity: float = 0.75  # Strong reflector (metal/concrete)

    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        inv = 1.0 / np.where(np.abs(rd) < EPS, np.sign(rd) * EPS + EPS, rd)
        t1 = (self.bmin - ro) * inv
        t2 = (self.bmax - ro) * inv

        tmin = float(np.max(np.minimum(t1, t2)))
        tmax = float(np.min(np.maximum(t1, t2)))

        if tmax < max(tmin, EPS):
            return None

        t = tmin if tmin > EPS else tmax
        p = ro + t * rd

        # Approximate normal by which face is closest
        # Determine axis of maximum entry
        face_eps = 1e-6
        n = np.zeros(3, dtype=float)
        for axis in range(3):
            if abs(p[axis] - self.bmin[axis]) < face_eps:
                n[axis] = -1.0
                break
            if abs(p[axis] - self.bmax[axis]) < face_eps:
                n[axis] = 1.0
                break
        if np.linalg.norm(n) < EPS:
            # Fallback: choose axis with largest |rd|
            axis = int(np.argmax(np.abs(rd)))
        return Hit(t=float(t), point=p, normal=n, obj_id=self.obj_id, reflectivity=self.reflectivity)


@dataclass
class ClutterVolume(Primitive):
    """Probabilistic water column clutter (plankton, particles, suspended matter).
    
    Returns a random hit along the ray to simulate volumetric scatterers
    without adding explicit geometry. This is O(1) per ray.
    """
    obj_id: str
    bmin: np.ndarray  # Volume bounds
    bmax: np.ndarray
    base_prob: float = 0.15  # Base probability of hit per ray
    reflectivity_min: float = 0.02  # Minimum clutter reflectivity
    reflectivity_max: float = 0.12  # Maximum clutter reflectivity
    depth_influence: float = 0.3  # How much depth affects probability
    surface_depth: float = 0.0  # Z coordinate of surface
    feeding_mode: bool = False  # High clutter during feeding
    
    def intersect(self, ro: np.ndarray, rd: np.ndarray) -> Hit | None:
        """Probabilistically return a hit within the volume."""
        # First check if ray enters the volume at all (AABB test)
        inv = 1.0 / np.where(np.abs(rd) < EPS, np.sign(rd) * EPS + EPS, rd)
        t1 = (self.bmin - ro) * inv
        t2 = (self.bmax - ro) * inv
        
        tmin = float(np.max(np.minimum(t1, t2)))
        tmax = float(np.min(np.maximum(t1, t2)))
        
        if tmax < max(tmin, EPS):
            return None  # Ray doesn't intersect volume
        
        t_entry = max(tmin, EPS)
        t_exit = tmax
        
        if t_exit <= t_entry:
            return None
        
        # Calculate probability based on path length and depth
        path_length = t_exit - t_entry
        
        # Sample a point along the ray within the volume
        sample_t = t_entry + path_length * 0.5  # Middle of volume
        sample_point = ro + sample_t * rd
        
        # Depth influence (more clutter near surface in many cases)
        depth_below_surface = self.surface_depth - sample_point[2]
        depth_factor = 1.0 + self.depth_influence * np.exp(-depth_below_surface / 10.0)
        
        # Range influence (slightly more clutter at longer ranges due to larger volume)
        range_factor = 1.0 + 0.1 * np.log1p(sample_t / 10.0)
        
        # Feeding mode increases clutter significantly
        feeding_factor = 3.0 if self.feeding_mode else 1.0
        
        # Combined probability
        hit_prob = self.base_prob * depth_factor * range_factor * feeding_factor
        hit_prob = min(hit_prob, 0.95)  # Cap at 95%
        
        # Use deterministic hash for consistency (same ray = same result)
        hash_seed = int((ro[0] * 1000 + ro[1] * 1331 + ro[2] * 1777 + 
                        rd[0] * 2003 + rd[1] * 2111 + rd[2] * 2221) % 10000)
        np.random.seed(hash_seed % 2**31)
        
        if np.random.rand() > hit_prob:
            return None
        
        # Generate random hit distance within volume
        t = t_entry + np.random.rand() * (t_exit - t_entry)
        p = ro + t * rd
        
        # Random reflectivity (weak scatterers)
        refl = self.reflectivity_min + np.random.rand() * (self.reflectivity_max - self.reflectivity_min)
        
        # Normal can just be opposite of ray direction (backscatter)
        n = -rd
        
        return Hit(t=float(t), point=p, normal=n, obj_id=self.obj_id, reflectivity=refl)
