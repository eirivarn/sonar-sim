"""Realistic sonar signal processing effects.

This module implements physics-based sonar artifacts to make simulated frames
look and behave like real imaging multibeam sonar (SonoptixECHO-style):
- Two-way transmission loss (spreading + absorption)
- Angle-dependent rolloff (grazing angle effects)
- Beam pattern PSF (azimuth point spread function with sidelobes)
- Range PSF (pulse length / matched filter effects)
- Multiplicative speckle (Gamma model for coherent interference)
- Noise floor and volume reverberation
- Shadowing (occlusion effects)
- TVG, log compression, and dynamic range enhancement
"""
import numpy as np


def apply_two_way_loss(mu, r_m, alpha_db_per_m=0.0, eps=1e-9):
    """Apply two-way transmission loss (spreading + absorption).
    
    For imaging sonar, use r^-2 spreading (effective one-way for extended targets)
    rather than r^-4 which is too aggressive and kills long-range signals.
    
    Args:
        mu: (H,W) expected intensity (linear units)
        r_m: (H,1) or (H,) range in meters for each range bin center
        alpha_db_per_m: absorption coefficient in dB/m (default 0.0)
        eps: small value to avoid division by zero
        
    Returns:
        (H,W) intensity with transmission loss applied
    """
    # Ensure r_m is column vector for broadcasting with (H,W) mu
    r = np.maximum(np.atleast_2d(r_m).T if r_m.ndim == 1 else r_m, eps)
    # Imaging sonar effective spreading: intensity ~ 1/r^2 (not r^4)
    # r^4 is for point targets; r^2 is appropriate for extended surface scattering
    spread = 1.0 / (r**2)
    # Absorption: intensity scale = 10^(-2*alpha*r / 10) = 10^(-alpha*r/5)
    absorb = 10.0 ** (-(alpha_db_per_m * r) / 5.0)
    return mu * spread * absorb


def angle_rolloff(W, strength_db=6.0, shape=2.0):
    """Create symmetric rolloff (1 at center, down towards edges).
    
    Models grazing angle effects and larger footprint at sector edges.
    
    Args:
        W: number of beams (width)
        strength_db: how much weaker at edges in dB
        shape: exponent controlling rolloff steepness
        
    Returns:
        (W,) array with rolloff weights
    """
    x = np.linspace(-1.0, 1.0, W)
    w = 10.0 ** (-(strength_db * (np.abs(x) ** shape)) / 10.0)
    return w


def conv1d_axis(x, kernel, axis):
    """Apply 1D convolution along specified axis.
    
    Args:
        x: input array
        kernel: 1D convolution kernel
        axis: axis along which to convolve
        
    Returns:
        convolved array same shape as x
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel = kernel / (np.sum(kernel) + 1e-12)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, x)


def add_speckle_gamma(mu_lin, looks=2.0, rng=None):
    """Add multiplicative speckle using Gamma distribution.
    
    Models coherent interference patterns characteristic of sonar imagery.
    
    Args:
        mu_lin: (H,W) mean intensity (linear units)
        looks: number of looks (1=very speckly, 3-8=smoother)
        rng: numpy random generator (optional)
        
    Returns:
        (H,W) intensity with speckle applied
    """
    if rng is None:
        rng = np.random.default_rng()
    L = float(looks)
    # Gamma(L, 1/L) has mean 1, variance 1/L
    speckle = rng.gamma(shape=L, scale=1.0/L, size=mu_lin.shape)
    return mu_lin * speckle


def add_noise_floor(I_lin, noise_floor=1e-6, rng=None):
    """Add receiver noise floor.
    
    Args:
        I_lin: (H,W) intensity
        noise_floor: noise level (exponential scale parameter)
        rng: numpy random generator (optional)
        
    Returns:
        (H,W) intensity with noise added
    """
    if rng is None:
        rng = np.random.default_rng()
    # Exponential is reasonable intensity-noise model after envelope detection
    n = rng.exponential(scale=noise_floor, size=I_lin.shape)
    return I_lin + n


def apply_shadowing(I_lin, shadow_thresh=None, shadow_strength=0.85):
    """Apply acoustic shadowing (occlusion) effects.
    
    Strong targets reduce energy behind them along each beam.
    
    Args:
        I_lin: (H,W) intensity
        shadow_thresh: threshold for strong returns (auto if None)
        shadow_strength: shadowing factor (0.85 = 85% attenuation)
        
    Returns:
        (H,W) intensity with shadows applied
    """
    H, W = I_lin.shape
    out = I_lin.copy()
    if shadow_thresh is None:
        # Robust threshold based on global percentile
        shadow_thresh = np.quantile(out, 0.995)

    for j in range(W):
        atten = 1.0
        for i in range(H):
            out[i, j] *= atten
            if out[i, j] > shadow_thresh:
                atten *= (1.0 - shadow_strength)  # Strong hit -> strong shadow
            else:
                # Slowly recover
                atten = min(1.0, atten + 0.01)
    return out


def sonar_postprocess_tvg_log_gamma(I_lin, r_m, tvg_k=2.0, eps=1e-12,
                                    p_lo=0.01, p_hi=0.995, gamma=0.75):
    """Apply sonar-style enhancement: TVG -> log -> stretch -> gamma.
    
    Implements the enhancement pipeline described in SonoptixECHO processing:
    1. Time-Varied Gain (TVG) to compensate range
    2. Log compression
    3. Percentile contrast stretch
    4. Gamma correction to brighten faint echoes
    
    Args:
        I_lin: (H,W) intensity in linear units
        r_m: (H,) range bin centers in meters
        tvg_k: TVG exponent (2.0 typical)
        eps: small value for log
        p_lo: lower percentile for stretch (0.01 = 1%)
        p_hi: upper percentile for stretch (0.995 = 99.5%)
        gamma: gamma correction factor (<1 brightens faint echoes)
        
    Returns:
        (H,W) enhanced intensity in [0,1]
    """
    # TVG: amplify with range^k
    tvg = (np.maximum(r_m, 1e-6) ** tvg_k).reshape(-1, 1)
    x = I_lin * tvg

    # Log compression
    x_db = 10.0 * np.log10(x + eps)

    # Percentile stretch to [0,1]
    lo = np.quantile(x_db, p_lo)
    hi = np.quantile(x_db, p_hi)
    y = (x_db - lo) / (hi - lo + 1e-12)
    y = np.clip(y, 0.0, 1.0)

    # Gamma (<1 brightens faint echoes)
    y = y ** gamma
    return y


def render_realistic_sonar(mu, r_m, *,
                           alpha_db_per_m=0.05,
                           edge_strength_db=6.0,
                           beam_psf=(0.05, 0.15, 0.60, 0.15, 0.05),
                           range_psf=(0.10, 0.20, 0.40, 0.20, 0.10),
                           looks=2.0,
                           noise_floor=1e-6,
                           shadow_strength=0.85,
                           tvg_k=2.0,
                           gamma=0.75,
                           rng=None):
    """Apply complete realistic sonar signal chain.
    
    This is the main function that transforms a clean raytraced intensity
    map into a realistic sonar image with all physical effects.
    
    Processing chain:
    1. Two-way transmission loss (spreading + absorption)
    2. Angle-dependent rolloff (grazing effects)
    3. Beam pattern PSF (cross-beam leakage)
    4. Range PSF (pulse length smearing)
    5. Multiplicative speckle (coherent interference)
    6. Noise floor
    7. Shadowing (occlusion)
    8. Enhancement (TVG, log, stretch, gamma)
    
    Args:
        mu: (H,W) mean backscatter intensity before sensor effects
        r_m: (H,) or (H,1) range bin centers in meters
        alpha_db_per_m: absorption coefficient in dB/m (0.05 typical for 700kHz)
        edge_strength_db: edge rolloff strength in dB (6.0 typical)
        beam_psf: beam pattern kernel (cross-beam)
        range_psf: range kernel (along-beam)
        looks: number of looks for speckle (2.0 typical)
        noise_floor: noise level (1e-6 typical)
        shadow_strength: shadowing factor (0.85 typical)
        tvg_k: TVG exponent (2.0 typical)
        gamma: gamma correction (0.75 typical)
        rng: numpy random generator (optional)
        
    Returns:
        (H,W) enhanced intensity in [0,1], ready for display
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = mu.shape
    r_m = np.asarray(r_m).reshape(H,)

    # 1) Physics-inspired loss
    I = apply_two_way_loss(mu, r_m, alpha_db_per_m=alpha_db_per_m)

    # 2) Fan edge rolloff
    I *= angle_rolloff(W, strength_db=edge_strength_db)[None, :]

    # 3) PSFs: beam + range
    I = conv1d_axis(I, beam_psf, axis=1)
    I = conv1d_axis(I, range_psf, axis=0)

    # 4) Speckle + noise + shadow
    I = add_speckle_gamma(I, looks=looks, rng=rng)
    I = add_noise_floor(I, noise_floor=noise_floor, rng=rng)
    I = apply_shadowing(I, shadow_strength=shadow_strength)

    # 5) Enhancement: TVG/log/stretch/gamma (like SonoptixECHO pipeline)
    out = sonar_postprocess_tvg_log_gamma(I, r_m, tvg_k=tvg_k, gamma=gamma)
    return out
