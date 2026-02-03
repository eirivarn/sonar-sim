#!/usr/bin/env python3
"""
Minimal Bag-to-NetTracker Video Pipeline

This script provides a simple, self-contained way to:
1. Load sonar data from a ROS bag file
2. Run the NetTracker on each frame
3. Save a debug video showing the tracking process

USAGE:
------
python run_tracker.py /path/to/bag/file.bag output_video.mp4

REQUIREMENTS:
-------------
pip install numpy pandas opencv-python rosbags tqdm
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import SOLAQUA utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_enhancement import preprocess_edges
from utils.sonar_tracking import NetTracker
from utils.config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG


# ============================================================================
# BAG FILE LOADING WITH NPZ CACHING
# ============================================================================

def load_or_extract_sonar_data(
    bag_path: Path, 
    cache_dir: Path | None = None,
    use_cache: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Load sonar data from cached NPZ if available, otherwise extract from bag.
    
    Args:
        bag_path: Path to ROS bag file
        cache_dir: Directory for NPZ cache files (default: bag_path.parent / "cache")
        use_cache: If True, try to load from cache and save to cache
        
    Returns:
        Tuple of (DataFrame with raw polar data, metadata dict)
    """
    if cache_dir is None:
        cache_dir = bag_path.parent / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate cache filename
    cache_file = cache_dir / f"{bag_path.stem}_raw_polar.npz"
    
    # Try loading from cache
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file.name}")
        return load_raw_polar_npz(cache_file)
    
    # Extract from bag
    df, metadata = extract_sonar_from_bag(bag_path)
    
    # Save to cache if requested
    if use_cache:
        print(f"Saving to cache: {cache_file.name}")
        save_raw_polar_to_npz(df, cache_file, metadata)
    
    return df, metadata


def save_raw_polar_to_npz(df: pd.DataFrame, npz_path: Path, metadata: dict):
    """Save raw polar sonar DataFrame to NPZ file for fast loading."""
    # Extract all frames
    frames = []
    timestamps = []
    dim_info = None
    
    for idx in range(len(df)):
        frame = get_sonar_frame_polar(df, idx)
        frames.append(frame)
        timestamps.append(df.iloc[idx]["t"])
        if dim_info is None:
            dim_info = {
                "dim_labels": df.iloc[idx].get("dim_labels", ["range_bins", "beams"]),
                "dim_sizes": list(frame.shape)
            }
    
    # Stack into 3D array
    raw_polar = np.stack(frames, axis=0).astype(np.float32)
    ts_array = np.array(timestamps, dtype=np.float64)
    
    # Save compressed
    np.savez_compressed(
        npz_path,
        raw_polar=raw_polar,
        timestamps=ts_array,
        dim_labels=dim_info["dim_labels"],
        dim_sizes=dim_info["dim_sizes"],
        metadata=json.dumps(metadata)
    )
    print(f"  ✓ Saved {len(frames)} frames to NPZ cache")


def load_raw_polar_npz(npz_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load raw polar sonar data from NPZ cache file."""
    with np.load(npz_path, allow_pickle=True) as data:
        raw_polar = data["raw_polar"]
        timestamps = data["timestamps"]
        dim_labels = data["dim_labels"].tolist() if "dim_labels" in data else ["range_bins", "beams"]
        dim_sizes = data["dim_sizes"].tolist() if "dim_sizes" in data else list(raw_polar[0].shape)
        metadata = json.loads(str(data["metadata"]))
    
    # Reconstruct DataFrame
    rows = []
    for idx in range(len(raw_polar)):
        rows.append({
            "t": timestamps[idx],
            "topic": metadata.get("topic", ""),
            "data": raw_polar[idx].ravel().tolist(),
            "dim_labels": dim_labels,
            "dim_sizes": dim_sizes,
        })
    
    df = pd.DataFrame(rows)
    print(f"  ✓ Loaded {len(df)} frames from NPZ cache")
    return df, metadata


def extract_sonar_from_bag(bag_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Extract Sonoptix sonar data from ROS bag file.
    
    Returns:
        (DataFrame with sonar data, metadata dict)
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError(
            "rosbags is required.\nInstall with: pip install rosbags"
        ) from exc

    print(f"Loading sonar data from: {bag_path.name}")
    rows = []
    
    with AnyReader([bag_path]) as reader:
        # Find Sonoptix topics
        sonar_conns = [c for c in reader.connections if "sonoptix" in c.topic.lower()]
        
        if not sonar_conns:
            raise RuntimeError(f"No Sonoptix topics found in bag")
        
        for conn in sonar_conns:
            for idx, (connection, timestamp, rawdata) in enumerate(reader.messages([conn]), start=1):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # Extract timestamp
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                ts = pd.Timestamp(sec, unit="s", tz="UTC") + pd.to_timedelta(nsec, unit="ns")
                
                # Extract dimensions
                dims = msg.array_data.layout.dim
                dim_labels = [d.label or f"dim_{i}" for i, d in enumerate(dims)] or ["range", "beam"]
                dim_sizes = [int(d.size) for d in dims]
                
                # Extract raw data
                data_vals = [float(v) for v in msg.array_data.data]
                
                rows.append({
                    "topic": conn.topic,
                    "ts_utc": ts.isoformat(),
                    "t": sec + nsec / 1e9,
                    "dim_labels": dim_labels,
                    "dim_sizes": dim_sizes,
                    "data": data_vals,
                })
                
                if idx % 100 == 0:
                    print(f"  Extracted {idx} frames...", end="\r")
    
    print(f"\n✓ Extracted {len(rows)} sonar frames")
    
    df = pd.DataFrame(rows)
    
    # Create metadata
    metadata = {
        "bag_name": bag_path.name,
        "num_frames": len(df),
        "topic": df.iloc[0]["topic"] if len(df) > 0 else None,
    }
    
    return df, metadata


def get_sonar_frame_polar(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Extract a single sonar frame as numpy array in polar format (range_bins, beams)."""
    row = df.iloc[idx]
    data = row["data"]
    dim_sizes = row["dim_sizes"]
    dim_labels = row.get("dim_labels", [])
    
    # Infer H, W from dimensions
    # Sonoptix format: typically range_bins (rows) × beams (cols)
    if len(dim_sizes) >= 2:
        # Check labels to determine correct orientation
        labels_str = str(dim_labels).lower()
        if "beam" in labels_str and "range" in labels_str:
            # Find which dimension is which
            beam_idx = next((i for i, l in enumerate(dim_labels) if "beam" in str(l).lower()), 0)
            range_idx = next((i for i, l in enumerate(dim_labels) if "range" in str(l).lower() or "bin" in str(l).lower()), 1)
            
            # Range bins should be rows (H), beams should be columns (W)
            H = dim_sizes[range_idx]
            W = dim_sizes[beam_idx]
        else:
            # Default: assume first dim is range, second is beams
            # But check if dimensions seem swapped (beams usually < range_bins)
            if dim_sizes[0] < dim_sizes[1]:
                # Likely beams × range_bins, need to transpose
                W, H = dim_sizes[0], dim_sizes[1]
                frame = np.array(data, dtype=np.float32).reshape(W, H).T
                return frame
            else:
                H, W = dim_sizes[0], dim_sizes[1]
    else:
        H, W = dim_sizes[0], dim_sizes[1] if len(dim_sizes) > 1 else 1
    
    # Reshape to (H, W) = (range_bins, beams)
    frame = np.array(data, dtype=np.float32).reshape(H, W)
    return frame


# ============================================================================
# POLAR TO CONE-VIEW CONVERSION
# ============================================================================

def rasterize_cone(
    polar_frame_normalized: np.ndarray,
    fov_deg: float = 120.0,
    rmin: float = 0.5,
    rmax: float = 20.0,
    img_h: int = 700,
    img_w: int = 900,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Convert NORMALIZED polar sonar image to Cartesian cone-view.
    
    IMPORTANT: Input should be pre-normalized to [0,1] range (after dB scaling).
    This matches the notebook's cone_raster_like_display_cell function.
    
    Args:
        polar_frame_normalized: Normalized polar image [0,1] (range_bins, beams)
        fov_deg: Field of view in degrees
        rmin: Minimum range in meters
        rmax: Maximum range in meters
        img_h: Output image height
        img_w: Output image width
        
    Returns:
        Tuple of (cone_image, extent=(xmin, xmax, ymin, ymax))
    """
    H, W = polar_frame_normalized.shape  # H=range_bins, W=beams
    
    half_fov = np.deg2rad(0.5 * fov_deg)
    y_min = max(0.0, rmin)
    y_max = rmax
    x_max = np.sin(half_fov) * y_max
    x_min = -x_max
    
    # Create Cartesian grid
    x = np.linspace(x_min, x_max, img_w)
    y = np.linspace(y_min, y_max, img_h)
    Xg, Yg = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    theta = np.arctan2(Xg, Yg)  # angle from +Y axis
    r = np.hypot(Xg, Yg)
    
    # Mask valid region
    mask = (r >= rmin) & (r <= y_max) & (theta >= -half_fov) & (theta <= half_fov)
    
    # Map (r, theta) to (row, col) in polar image
    rowf = (r - rmin) / max((rmax - rmin), 1e-12) * (H - 1)
    colf = (theta + half_fov) / max((2 * half_fov), 1e-12) * (W - 1)
    rows = np.rint(np.clip(rowf, 0, H - 1)).astype(np.int32)
    cols = np.rint(np.clip(colf, 0, W - 1)).astype(np.int32)
    
    # Sample from polar image
    cone = np.full((img_h, img_w), np.nan, dtype=np.float32)
    mflat = mask.ravel()
    cone.ravel()[mflat] = polar_frame_normalized[rows.ravel()[mflat], cols.ravel()[mflat]]
    
    extent = (x_min, x_max, y_min, y_max)
    return cone, extent


def polar_to_db_normalized(raw_polar: np.ndarray, db_norm: float = 60.0) -> np.ndarray:
    """
    Convert raw polar sonar to dB scale and normalize to [0,1].
    
    This is Step 1 of the notebook pipeline.
    
    Args:
        raw_polar: Raw polar image (any scale)
        db_norm: dB normalization constant
        
    Returns:
        Normalized image in [0,1] range
    """
    image_db = 10 * np.log10(np.maximum(raw_polar, 1e-10))
    display_frame = np.clip((image_db + db_norm) / db_norm, 0, 1)
    return display_frame


def to_uint8_gray(frame01: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] frame to uint8, handling NaN values."""
    safe = np.nan_to_num(frame01, nan=0.0, posinf=1.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    return (safe * 255.0).astype(np.uint8)


def get_sonar_frame_cone(
    df: pd.DataFrame, 
    idx: int,
    fov_deg: float = 120.0,
    rmin: float = 0.5,
    rmax: float = 20.0,
    img_h: int = 700,
    img_w: int = 900,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Extract sonar frame and convert to cone-view (Cartesian coordinates).
    This is what notebook uses - NOT raw polar data!
    """
    polar_frame = get_sonar_frame_polar(df, idx)
    cone_frame, extent = rasterize_cone(polar_frame, fov_deg, rmin, rmax, img_h, img_w)
    return cone_frame, extent


# ============================================================================
# NET TRACKER (Simplified)
# ============================================================================

class NetTracker:
    """
    Full SOLAQUA net tracker with corridor splitting, linearity scoring,
    and adaptive search regions.
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Current ellipse state
        self.center = None
        self.size = None
        self.angle = None
        
        # Tracking
        self.last_distance = None
        self.frames_lost = 0
    
    def find_and_update(self, edges: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Find net in edges, update tracking state.
        Returns best contour or None.
        """
        H, W = image_shape
        
        # Get search mask (ellipse + corridor)
        search_mask = self._get_search_mask((H, W))
        
        # Apply mask to edges
        if search_mask is not None:
            masked_edges = cv2.bitwise_and(edges, search_mask)
        else:
            masked_edges = edges
        
        # Find contours
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best with advanced scoring
        best = self._find_best_contour(contours)
        
        # Update state with smoothing
        if best is not None and len(best) >= 5:
            self._update_from_contour(best)
            self.frames_lost = 0
        else:
            self.frames_lost += 1
            if self.frames_lost > self.config.get('max_frames_without_detection', 30):
                self._reset()
        
        return best
    
    def calculate_distance(self, image_width: int, image_height: int) -> tuple:
        """Calculate distance and angle from contour."""
        if self.center is None or self.angle is None:
            return self.last_distance, None
        
        cx, cy = self.center
        
        # The red line angle is perpendicular to major axis
        major_axis_angle = self.angle
        red_line_angle = (major_axis_angle + 90.0) % 360.0
        
        # Calculate intersection with center line
        center_x = image_width / 2
        ang_r = np.radians(red_line_angle)
        cos_ang = np.cos(ang_r)
        
        if abs(cos_ang) > 1e-6:
            t = (center_x - cx) / cos_ang
            intersect_y = cy + t * np.sin(ang_r)
            distance = intersect_y
        else:
            distance = cy
        
        distance = np.clip(distance, 0, image_height - 1)
        
        # Limit maximum change per frame
        if self.last_distance is not None:
            max_change = self.config.get('max_distance_change_pixels', 20)
            change = abs(distance - self.last_distance)
            if change > max_change:
                direction = 1 if distance > self.last_distance else -1
                distance = self.last_distance + (direction * max_change)
        
        self.last_distance = distance
        
        return float(distance), float(red_line_angle)
    
    def _get_search_mask(self, shape: tuple) -> np.ndarray:
        """Create search mask with ellipse + corridor."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = shape
        
        # Expand ellipse based on tracking status
        expansion = self.config.get('ellipse_expansion_factor', 0.5)
        if self.frames_lost > 0:
            decay = self.config.get('aoi_decay_factor', 0.98)
            growth = 1.0 + (1.0 - decay) * self.frames_lost
            expansion *= growth
        
        w, h = self.size
        expanded_size = (w * (1 + expansion), h * (1 + expansion))
        
        mask = np.zeros((H, W), dtype=np.uint8)
        
        try:
            ellipse = (
                (int(self.center[0]), int(self.center[1])),
                (int(expanded_size[0]), int(expanded_size[1])),
                self.angle
            )
            cv2.ellipse(mask, ellipse, 255, -1)
        except:
            return None
        
        # Add corridor along major axis
        if self.config.get('use_corridor_splitting', True):
            try:
                corridor = self._make_corridor_mask((H, W))
                if corridor is not None:
                    mask = cv2.bitwise_or(mask, corridor)
            except:
                pass
        
        return mask
    
    def _make_corridor_mask(self, image_shape: tuple) -> np.ndarray:
        """Make corridor mask along major axis for far-field search."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = image_shape
        cx, cy = self.center
        w, h = self.size
        
        # Corridor parameters
        band_k = self.config.get('corridor_band_k', 2.0)
        length_factor = self.config.get('corridor_length_factor', 2.0)
        
        half_width = band_k * min(w, h) / 2.0
        half_length = length_factor * max(w, h) / 2.0
        
        # Rectangle in local coordinates
        local_pts = np.array([
            [-half_length, -half_width],
            [+half_length, -half_width],
            [+half_length, +half_width],
            [-half_length, +half_width],
        ], dtype=np.float32)
        
        # Determine major axis angle
        major_angle = self.angle
        if h > w:
            major_angle = (self.angle + 90.0) % 360.0
        
        ang_r = np.radians(major_angle)
        cos_a, sin_a = np.cos(ang_r), np.sin(ang_r)
        
        # Rotation matrix
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = local_pts @ R.T
        
        # Translate to center
        world_pts = rotated + np.array([cx, cy])
        
        # Draw
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [world_pts.astype(np.int32)], 255)
        
        return mask
    
    def _find_best_contour(self, contours: list) -> np.ndarray:
        """Find best contour with linearity and aspect ratio scoring."""
        min_area = self.config.get('min_contour_area', 200)
        if self.center is not None:
            min_area *= 0.3  # Lower threshold when tracking
        
        best = None
        best_score = 0.0
        
        for c in contours:
            if c is None or len(c) < 5:
                continue
            
            try:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                
                score = area
                
                # Proximity bonus
                if self.center is not None:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        dist = np.sqrt((cx - self.center[0])**2 + (cy - self.center[1])**2)
                        proximity = max(0.1, 1.0 - dist / 100.0)
                        score *= proximity
                
                # Linearity score (prefer straight lines)
                linearity_score = self._calculate_contour_linearity(c)
                score += linearity_score * self.config.get('linearity_score_weight', 1.0)
                
                # Aspect ratio score (prefer wide rectangles)
                aspect_score = self._calculate_aspect_ratio_score(c)
                score += aspect_score * self.config.get('aspect_ratio_score_weight', 1.0)
                
                if score > best_score:
                    best = c
                    best_score = score
            except:
                continue
        
        return best
    
    def _calculate_contour_linearity(self, contour: np.ndarray) -> float:
        """Calculate how straight/linear a contour is using PCA."""
        if len(contour) < 2:
            return 0.0
        
        points = contour.reshape(-1, 2).astype(np.float32)
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        cov = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eig(cov)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        
        if eigenvalues[1] < 1e-6:
            return 1.0
        
        linearity_ratio = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1])
        return float(linearity_ratio)
    
    def _calculate_aspect_ratio_score(self, contour: np.ndarray) -> float:
        """Score based on aspect ratio - prefer wide rectangles (net edge)."""
        if len(contour) < 5:
            return 0.5
        
        try:
            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), _ = rect
            
            if height > width:
                width, height = height, width
            
            if height < 1e-6:
                return 0.0
            
            aspect_ratio = width / height
            
            # Ideal: 2-5 (wide but not extreme), peaks at 3
            ideal_ratio = 3.0
            
            if aspect_ratio < 1.0:
                return 0.3
            elif aspect_ratio <= ideal_ratio:
                return 0.5 + 0.5 * (aspect_ratio - 1.0) / (ideal_ratio - 1.0)
            elif aspect_ratio <= ideal_ratio * 2:
                return 1.0 - 0.3 * (aspect_ratio - ideal_ratio) / ideal_ratio
            else:
                return 0.4
        except:
            return 0.5
    
    def _update_from_contour(self, contour: np.ndarray):
        """Update ellipse state with adaptive smoothing."""
        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        except:
            return
        
        # Get smoothing alphas
        alpha_center = self.config.get('center_smoothing_alpha', 0.4)
        alpha_size = self.config.get('ellipse_size_smoothing_alpha', 0.01)
        alpha_angle = self.config.get('ellipse_orientation_smoothing_alpha', 0.2)
        
        # Smooth center
        if self.center is None:
            self.center = (cx, cy)
        else:
            old_cx, old_cy = self.center
            new_cx = old_cx * (1 - alpha_center) + cx * alpha_center
            new_cy = old_cy * (1 - alpha_center) + cy * alpha_center
            
            # Limit movement
            max_move = self.config.get('ellipse_max_movement_pixels', 30.0)
            dx = new_cx - old_cx
            dy = new_cy - old_cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > max_move:
                scale = max_move / dist
                new_cx = old_cx + dx * scale
                new_cy = old_cy + dy * scale
            
            self.center = (new_cx, new_cy)
        
        # Smooth size
        if self.size is None:
            self.size = (w, h)
        else:
            old_w, old_h = self.size
            new_w = old_w * (1 - alpha_size) + w * alpha_size
            new_h = old_h * (1 - alpha_size) + h * alpha_size
            self.size = (new_w, new_h)
        
        # Smooth angle (handle wrap-around)
        if self.angle is None:
            self.angle = angle
        else:
            angle_diff = angle - self.angle
            if angle_diff > 90:
                angle_diff -= 180
            elif angle_diff < -90:
                angle_diff += 180
            self.angle = self.angle + angle_diff * alpha_angle
    
    def _reset(self):
        """Reset tracking state."""
        self.center = None
        self.size = None
        self.angle = None
        self.last_distance = None
        self.frames_lost = 0
    
    def get_status(self) -> str:
        """Get tracking status."""
        if self.center is None:
            return "LOST"
        elif self.frames_lost == 0:
            return "TRACKED"
        else:
            max_frames = self.config.get('max_frames_without_detection', 30)
            return f"SEARCHING ({self.frames_lost}/{max_frames})"


# ============================================================================
# IMAGE PROCESSING (SOLAQUA-style with adaptive momentum merging)
# ============================================================================

def compute_structure_tensor_field(grad_x: np.ndarray, grad_y: np.ndarray, 
                                   sigma: float = 1.0) -> tuple:
    """
    Fast vectorized structure tensor computation for orientation detection.
    """
    # Structure tensor components
    Jxx = grad_x * grad_x
    Jyy = grad_y * grad_y  
    Jxy = grad_x * grad_y
    
    # Apply Gaussian smoothing
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_gauss = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel_gauss @ kernel_gauss.T
    
    Jxx_smooth = cv2.filter2D(Jxx, -1, kernel_2d)
    Jyy_smooth = cv2.filter2D(Jyy, -1, kernel_2d)
    Jxy_smooth = cv2.filter2D(Jxy, -1, kernel_2d)
    
    # Vectorized eigenvalue analysis
    trace = Jxx_smooth + Jyy_smooth
    det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
    
    # Compute orientation and coherency
    orientation_map = np.zeros_like(Jxx_smooth)
    coherency_map = np.zeros_like(Jxx_smooth)
    
    valid_mask = np.abs(Jxy_smooth) > 1e-6
    orientation_map[valid_mask] = 0.5 * np.arctan2(2 * Jxy_smooth[valid_mask], 
                                                   Jxx_smooth[valid_mask] - Jyy_smooth[valid_mask])
    orientation_map = (orientation_map * 180 / np.pi + 180) % 180
    
    valid_coherency_mask = (trace > 1e-6) & (det >= 0)
    coherency_map[valid_coherency_mask] = ((trace[valid_coherency_mask] - 
                                          2 * np.sqrt(det[valid_coherency_mask])) / 
                                         trace[valid_coherency_mask])
    coherency_map = np.clip(coherency_map, 0, 1)
    
    return orientation_map, coherency_map


def adaptive_linear_momentum_merge(frame: np.ndarray, config: dict) -> np.ndarray:
    """
    Advanced adaptive linear momentum merge using structure tensor analysis.
    This is the key SOLAQUA enhancement that improves net edge detection.
    """
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = frame.astype(np.float32)
    h, w = result.shape
    
    # Early exit for low contrast
    frame_std = np.std(result)
    if frame_std < 5.0:
        enhanced = cv2.GaussianBlur(result, (7, 7), 1.0)
        final_result = result + 0.24 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Downscale for efficiency
    downscale = config.get('momentum_downscale', 2)
    h_small = max(h // downscale, 32)
    w_small = max(w // downscale, 32)
    frame_small = cv2.resize(result, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # Compute structure tensor
    grad_x = cv2.Sobel(frame_small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_small, cv2.CV_64F, 0, 1, ksize=3)
    
    orientations, linearity_map_small = compute_structure_tensor_field(grad_x, grad_y, sigma=1.5)
    
    # Quantize orientations
    angle_steps = config.get('angle_steps', 36)
    orientations_normalized = orientations / 180.0
    direction_bin_map_small = np.round(orientations_normalized * (angle_steps - 1)).astype(np.int32)
    direction_bin_map_small = np.clip(direction_bin_map_small, 0, angle_steps - 1)
    
    # Normalize linearity
    max_linearity = np.max(linearity_map_small)
    if max_linearity > 0:
        linearity_map_small = linearity_map_small / max_linearity
    else:
        enhanced = cv2.GaussianBlur(result, (7, 7), 1.0)
        return np.clip(result + 0.4 * enhanced, 0, 255).astype(np.uint8)
    
    # Upsample maps
    linearity_map = cv2.resize(linearity_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
    direction_bin_map = cv2.resize(direction_bin_map_small.astype(np.float32), (w, h), 
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    # Create linear mask
    linearity_threshold = config.get('linearity_threshold', 0.15)
    linear_mask = linearity_map > linearity_threshold
    
    if np.sum(linear_mask) == 0:
        enhanced = cv2.GaussianBlur(result, (7, 7), 1.0)
        return np.clip(result + 0.4 * enhanced, 0, 255).astype(np.uint8)
    
    # Process top angle bins
    unique_bins, bin_counts = np.unique(direction_bin_map[linear_mask], return_counts=True)
    total_linear_pixels = np.sum(linear_mask)
    
    bin_info = []
    for bin_id, count in zip(unique_bins, bin_counts):
        coverage_pct = 100.0 * count / total_linear_pixels
        if coverage_pct >= config.get('min_coverage_percent', 0.5):
            bin_mask = linear_mask & (direction_bin_map == bin_id)
            avg_linearity = np.mean(linearity_map[bin_mask])
            bin_info.append((bin_id, count, avg_linearity))
    
    # Sort by linearity and take top K
    bin_info.sort(key=lambda x: x[2], reverse=True)
    top_k = config.get('top_k_bins', 8)
    bins_to_process = [b[0] for b in bin_info[:top_k]]
    
    # Process each bin with separable Gaussian
    momentum_boost = config.get('momentum_boost', 0.8)
    base_radius = config.get('base_radius', 3)
    
    for bin_id in bins_to_process:
        bin_mask = linear_mask & (direction_bin_map == bin_id)
        angle_deg = (bin_id * 180.0) / angle_steps
        
        # Create anisotropic Gaussian kernel
        kernel_size = 2 * base_radius + 1
        gaussian_1d = cv2.getGaussianKernel(kernel_size, base_radius / 2.0)
        
        # Apply oriented filtering
        angle_rad = np.radians(angle_deg)
        enhanced_slice = cv2.sepFilter2D(result, -1, gaussian_1d, gaussian_1d.T)
        
        # Weighted blend based on linearity
        weights = linearity_map[bin_mask]
        boost = momentum_boost * weights
        
        result_flat = result.ravel()
        enhanced_flat = enhanced_slice.ravel()
        mask_flat = bin_mask.ravel()
        boost_flat = np.zeros(result.size)
        boost_flat[mask_flat] = boost
        
        result_flat += boost_flat * enhanced_flat
    
    return np.clip(result, 0, 255).astype(np.uint8)


def process_sonar_image_notebook(frame_u8: np.ndarray, config: dict) -> tuple:
    """
    Process sonar frame using notebook's exact pipeline.
    
    Returns:
        (binary_frame, momentum_merged, edges)
    """
    # Step 1: Binary thresholding
    binary_threshold = config.get('binary_threshold', 30)
    binary = (frame_u8 > binary_threshold).astype(np.uint8) * 255
    
    # Step 2: Edge enhancement with momentum merging (from utils)
    momentum_merged, edges = preprocess_edges(binary, config)
    
    return binary, momentum_merged, edges


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def create_debug_frame(sonar_img: np.ndarray, binary: np.ndarray, 
                      momentum: np.ndarray, edges: np.ndarray,
                      contour: np.ndarray, tracker: NetTracker,
                      distance: float, angle: float,
                      frame_idx: int) -> np.ndarray:
    """Create debug visualization with 4-panel view."""
    H, W = sonar_img.shape
    
    # Convert all to color
    sonar_color = cv2.cvtColor(
        cv2.normalize(sonar_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        cv2.COLOR_GRAY2BGR
    )
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    momentum_color = cv2.cvtColor(momentum, cv2.COLOR_GRAY2BGR)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Draw tracking on edges
    if contour is not None:
        cv2.drawContours(edges_color, [contour], -1, (0, 255, 0), 2)
        
        # Draw ellipse
        if tracker.center is not None:
            ellipse = (tracker.center, tracker.size, tracker.angle)
            cv2.ellipse(edges_color, ellipse, (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(edges_color, (int(tracker.center[0]), int(tracker.center[1])), 
                      3, (0, 0, 255), -1)
            
            # Draw search region as ellipse outline (expanded)
            expansion = tracker.config.get('ellipse_expansion_factor', 0.5)
            w, h = tracker.size
            expanded_size = (w * (1 + expansion), h * (1 + expansion))
            search_ellipse = (tracker.center, expanded_size, tracker.angle)
            cv2.ellipse(edges_color, search_ellipse, (255, 128, 0), 1)  # Orange outline
    
    # Draw net detection line (red, oriented correctly)
    # angle is already perpendicular to the detection ray (calculated as major_axis + 90 in tracker)
    if distance is not None and angle is not None and tracker.center is not None and tracker.size is not None:
        # Use angle directly - it's already the net orientation (perpendicular to detection ray)
        ang_r = np.radians(angle)
        half_len = max(tracker.size) / 2  # Half the major axis length
        
        # Calculate endpoints from center
        p1x = int(tracker.center[0] + half_len * np.cos(ang_r))
        p1y = int(tracker.center[1] + half_len * np.sin(ang_r))
        p2x = int(tracker.center[0] - half_len * np.cos(ang_r))
        p2y = int(tracker.center[1] - half_len * np.sin(ang_r))
        
        # Draw red net line
        cv2.line(edges_color, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)
    
    # Add labels
    cv2.putText(sonar_color, "1. Cone-View", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(binary_color, "2. Binary", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(momentum_color, "3. Momentum Merged", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(edges_color, "4. Tracking", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add status
    status = f"Frame: {frame_idx} | {tracker.get_status()}"
    if distance is not None:
        status += f" | Dist: {distance:.1f}px | Angle: {angle:.1f}deg"
    
    cv2.putText(edges_color, status, (10, H - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Stack: 2x2 grid
    top_row = np.hstack([sonar_color, binary_color])
    bottom_row = np.hstack([momentum_color, edges_color])
    debug_frame = np.vstack([top_row, bottom_row])
    
    return debug_frame


def generate_tracking_video(df: pd.DataFrame, output_path: Path, 
                           start_frame: int = 0, 
                           end_frame: int = None,
                           stride: int = 1):
    """
    Generate tracking debug video with SOLAQUA processing pipeline.
    """
    # Use SOLAQUA configuration from utils
    tracker_config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    
    # Override binary threshold for cone-view data
    tracker_config['binary_threshold'] = 30
    
    tracker = NetTracker(tracker_config)
    
    # Frame range
    N = len(df)
    start_frame = max(0, start_frame)
    end_frame = min(N, end_frame if end_frame is not None else N)
    frame_indices = list(range(start_frame, end_frame, stride))
    
    print(f"\nProcessing {len(frame_indices)} frames with SOLAQUA pipeline...")
    print(f"  - Using notebook's preprocess_edges for momentum merging")
    print(f"  - Corridor splitting: {tracker_config.get('use_corridor_splitting', False)}")
    
    # Initialize video writer
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15.0
    
    # Process frames
    for idx in tqdm(frame_indices):
        try:
            # Follow EXACT notebook pipeline:
            # 1. Get raw polar data
            polar_img = get_sonar_frame_polar(df, idx)
            
            # 2. Convert to dB and normalize to [0,1]
            polar_db_normalized = polar_to_db_normalized(polar_img, db_norm=60.0)
            
            # 3. Convert to Cartesian cone-view
            sonar_img, extent = rasterize_cone(polar_db_normalized, 120.0, 0.5, 20.0)
            
            # 4. Convert to uint8
            sonar_img_u8 = to_uint8_gray(sonar_img)
            H, W = sonar_img_u8.shape
            
            # Debug: print first frame info
            if idx == frame_indices[0]:
                print(f"\n\u2713 First frame info:")
                print(f"  Polar shape: {polar_img.shape} (range_bins \u00d7 beams)")
                print(f"  Cone-view shape: {H} \u00d7 {W}")
                print(f"  Extent (x_min, x_max, y_min, y_max): {extent}")
                print(f"  Data range: [{np.nanmin(sonar_img):.2f}, {np.nanmax(sonar_img):.2f}]")
                print(f"  Valid pixels: {np.sum(~np.isnan(sonar_img))} / {sonar_img.size} ({100*np.sum(~np.isnan(sonar_img))/sonar_img.size:.1f}%)")
            
            # 5. Binary threshold and edge extraction (using notebook's preprocess_edges)
            tracker_config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
            binary, momentum, edges = process_sonar_image_notebook(sonar_img_u8, tracker_config)
            
            # Debug: print processing results on first frame
            if idx == frame_indices[0]:
                print(f"  Binary pixels: {np.sum(binary > 0)} ({100*np.sum(binary > 0)/(H*W):.1f}%)")
                print(f"  Momentum pixels: {np.sum(momentum > 0)} ({100*np.sum(momentum > 0)/(H*W):.1f}%)")
                print(f"  Edge pixels: {np.sum(edges > 0)} ({100*np.sum(edges > 0)/(H*W):.1f}%)")
            
            # Track
            contour = tracker.find_and_update(edges, (H, W))
            distance, angle = tracker.calculate_distance(W, H)
            
            # Create debug frame (4-panel)
            debug_frame = create_debug_frame(sonar_img_u8, binary, momentum, edges,
                                            contour, tracker, distance, angle, idx)
            
            # Initialize writer on first frame
            if writer is None:
                frame_h, frame_w = debug_frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                        (frame_w, frame_h), True)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer: {output_path}")
                print(f"✓ Video writer initialized: {frame_w}x{frame_h} @ {fps} FPS")
                print(f"  Output: {output_path}")
            
            # Verify frame is uint8 BGR before writing
            if debug_frame.dtype != np.uint8:
                debug_frame = debug_frame.astype(np.uint8)
            if len(debug_frame.shape) != 3 or debug_frame.shape[2] != 3:
                raise ValueError(f"Invalid frame format: {debug_frame.shape}, expected (H, W, 3)")
            
            # Write frame
            writer.write(debug_frame)
            
        except Exception as e:
            print(f"\nError processing frame {idx}: {e}")
            continue
    
    # Cleanup
    if writer is not None:
        writer.release()
    
    print(f"\n✓ Video saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run NetTracker on ROS bag file and save debug video"
    )
    parser.add_argument("bag_file", type=Path, help="Path to ROS bag file")
    parser.add_argument("output_video", type=Path, help="Output video path (e.g., output.mp4)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (None for all)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (1 for every frame)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.bag_file.exists():
        print(f"Error: Bag file not found: {args.bag_file}")
        return 1
    
    # Load data
    print("\n" + "=" * 60)
    print("MINIMAL NET TRACKER PIPELINE")
    print("=" * 60)
    
    df, metadata = load_or_extract_sonar_data(args.bag_file, use_cache=True)
    
    # Generate video
    generate_tracking_video(
        df, 
        args.output_video,
        start_frame=args.start,
        end_frame=args.end,
        stride=args.stride
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
