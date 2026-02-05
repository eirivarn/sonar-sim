"""
Temporal smoothing for segmentation predictions.

Reduces prediction flickering by leveraging temporal consistency
across consecutive frames without retraining the model.
"""

import numpy as np
from collections import deque
from typing import Optional, Literal


class TemporalSmoother:
    """
    Apply temporal smoothing to prediction masks over time.
    
    Methods:
    - 'median': Robust to outliers (recommended for noisy sonar)
    - 'mean': Simple average
    - 'exponential': Weighted by recency (alpha smoothing)
    """
    
    def __init__(
        self, 
        window_size: int = 5,
        method: Literal['median', 'mean', 'exponential'] = 'median',
        alpha: float = 0.3
    ):
        """
        Args:
            window_size: Number of frames to consider (3-7 recommended)
            method: Smoothing method
            alpha: For exponential smoothing (0.1-0.5, lower = more smoothing)
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        
        # History buffer
        self.history = deque(maxlen=window_size)
        
        # For exponential smoothing
        self.smoothed_mask: Optional[np.ndarray] = None
        
    def update(self, mask: np.ndarray) -> np.ndarray:
        """
        Add new mask and return temporally smoothed result.
        
        Args:
            mask: Binary mask (0-255) or probability map (0.0-1.0)
        
        Returns:
            Smoothed mask in same format as input
        """
        # Normalize to 0-1 for processing
        is_uint8 = mask.dtype == np.uint8
        if is_uint8:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)
        
        # Apply smoothing
        if self.method == 'exponential':
            smoothed = self._exponential_smooth(mask_norm)
        else:
            self.history.append(mask_norm)
            if len(self.history) < 2:
                smoothed = mask_norm
            else:
                smoothed = self._window_smooth()
        
        # Convert back to original format
        if is_uint8:
            return (smoothed * 255).astype(np.uint8)
        else:
            return smoothed
    
    def _window_smooth(self) -> np.ndarray:
        """Apply window-based smoothing (median or mean)."""
        history_array = np.array(self.history)  # Shape: (T, H, W)
        
        if self.method == 'median':
            # Median filter - best for sonar noise
            return np.median(history_array, axis=0)
        elif self.method == 'mean':
            # Simple average
            return np.mean(history_array, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _exponential_smooth(self, mask: np.ndarray) -> np.ndarray:
        """Exponential moving average: smooth = smooth * (1-α) + new * α"""
        if self.smoothed_mask is None:
            self.smoothed_mask = mask.copy()
        else:
            self.smoothed_mask = (
                self.smoothed_mask * (1 - self.alpha) + 
                mask * self.alpha
            )
        return self.smoothed_mask
    
    def reset(self):
        """Clear history (call when video sequence breaks)."""
        self.history.clear()
        self.smoothed_mask = None
    
    def is_warmed_up(self) -> bool:
        """Check if history buffer is filled."""
        if self.method == 'exponential':
            return self.smoothed_mask is not None
        else:
            return len(self.history) >= self.window_size


class ProbabilityTemporalSmoother(TemporalSmoother):
    """
    Temporal smoother that operates on probability maps before thresholding.
    
    More effective than smoothing binary masks since it preserves
    uncertainty information during smoothing.
    """
    
    def __init__(self, window_size: int = 5, method: str = 'median', 
                 alpha: float = 0.3, threshold: float = 0.5):
        super().__init__(window_size, method, alpha)
        self.threshold = threshold
    
    def update_and_threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Smooth probability map, then apply threshold.
        
        Args:
            prob_map: Probability map (0.0-1.0)
        
        Returns:
            Binary mask (0-255)
        """
        # Smooth probabilities
        smoothed_prob = self.update(prob_map)
        
        # Threshold
        binary = (smoothed_prob > self.threshold).astype(np.uint8) * 255
        
        return binary


def smooth_batch(masks: np.ndarray, method: str = 'median', window_size: int = 5) -> np.ndarray:
    """
    Smooth a batch of masks offline (for post-processing saved predictions).
    
    Args:
        masks: Array of shape (T, H, W) where T is number of frames
        method: 'median', 'mean', or 'gaussian'
        window_size: Temporal window size (must be odd for median/gaussian)
    
    Returns:
        Smoothed masks of same shape
    """
    from scipy.ndimage import median_filter, uniform_filter, gaussian_filter1d
    
    if method == 'median':
        # Median filter along time axis
        return median_filter(masks, size=(window_size, 1, 1))
    elif method == 'mean':
        # Mean filter along time axis
        return uniform_filter(masks, size=(window_size, 1, 1))
    elif method == 'gaussian':
        # Gaussian filter along time axis
        sigma = window_size / 6.0  # ~99% within window
        return gaussian_filter1d(masks, sigma=sigma, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
