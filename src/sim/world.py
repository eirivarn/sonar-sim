from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .primitives import Primitive, Hit

@dataclass
class World:
    objects: list[Primitive] = field(default_factory=list)

    def raycast(self, ro: np.ndarray, rd: np.ndarray, t_max: float) -> Hit | None:
        """Cast ray and find closest intersection with broadphase culling."""
        best: Hit | None = None
        best_t = t_max
        
        for obj in self.objects:
            # Broadphase: check bounding sphere first
            bounds = obj.bounds_sphere()
            if bounds is not None:
                center, radius = bounds
                # Quick sphere-ray intersection test
                oc = ro - center
                b = np.dot(oc, rd)
                c = np.dot(oc, oc) - radius * radius
                disc = b * b - c
                
                if disc < 0:
                    continue  # Ray misses bounding sphere entirely
                
                # Check if sphere is in valid range
                s = np.sqrt(disc)
                t_near = -b - s
                t_far = -b + s
                
                if t_far < 0 or t_near > best_t:
                    continue  # Sphere is behind ray or beyond current best
            
            # Passed broadphase, do full intersection test
            hit = obj.intersect(ro, rd)
            if hit is None:
                continue
            if hit.t < best_t and hit.t <= t_max:
                best_t = hit.t
                best = hit
        
        return best
