#!/usr/bin/env python3
"""Data loading utilities: ROS bags, NPZ caching, and frame extraction."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


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
    Full pipeline: polar -> dB normalized -> cone-view.
    """
    from .coordinate_transform import rasterize_cone, polar_to_db_normalized
    
    polar_frame = get_sonar_frame_polar(df, idx)
    polar_normalized = polar_to_db_normalized(polar_frame)
    cone_frame, extent = rasterize_cone(polar_normalized, fov_deg, rmin, rmax, img_h, img_w)
    return cone_frame, extent
