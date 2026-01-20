from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .primitives import Primitive, Hit

@dataclass
class World:
    objects: list[Primitive] = field(default_factory=list)

    def raycast(self, ro: np.ndarray, rd: np.ndarray, t_max: float) -> Hit | None:
        best: Hit | None = None
        best_t = t_max
        for obj in self.objects:
            hit = obj.intersect(ro, rd)
            if hit is None:
                continue
            if hit.t < best_t and hit.t <= t_max:
                best_t = hit.t
                best = hit
        return best
