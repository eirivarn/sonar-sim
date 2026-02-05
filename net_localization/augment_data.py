"""
Generate augmented training data with simulated physics variations.

AUGMENTATION STRATEGY (BATCH PROCESSING):
- Keep FOV and range CONSTANT (120°, 20m)
- Process in batches: complete ALL images for one config before next
- For each configuration:
  * Apply image-level approximations of different physics
  * Use frames from simulation
  * Generate 1 image per frame
  * Total: images_per_config images per configuration
- num_configs configurations × images_per_config images = total

Simulated physics variations:
- Attenuation (water absorption)
- Speckle noise (acoustic speckle)
- Blur (beam pattern/resolution)
- Gain variations (intensity calibration)
- Gaussian noise (sensor noise)
- Azimuth streaks (gain saturation)
- And many more...

Usage:
    # Generate images
    python augment_data.py --num_configs 20 --images_per_config 5000 --frame_step 2
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


def polar_to_cartesian(polar_image, max_range_m=20.0, fov_deg=120.0, output_size=400):
    """Convert polar sonar image to Cartesian coordinates."""
    r_bins, n_beams = polar_image.shape
    half_width = max_range_m * np.sin(np.radians(fov_deg / 2))
    x = np.linspace(-half_width, half_width, output_size)
    y = np.linspace(0, max_range_m, output_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(X, Y)
    fov_rad = np.radians(fov_deg)
    beam_idx = (Theta + fov_rad / 2) / fov_rad * (n_beams - 1)
    range_idx = (R / max_range_m) * (r_bins - 1)
    beam_idx = np.clip(beam_idx, 0, n_beams - 1).astype(np.float32)
    range_idx = np.clip(range_idx, 0, r_bins - 1).astype(np.float32)
    cart = cv2.remap(polar_image.astype(np.float32), beam_idx, range_idx, 
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cart[(R > max_range_m) | (Theta < -fov_rad/2) | (Theta > fov_rad/2)] = 0
    return cart


def generate_physics_configurations(num_configs=20):
    """Generate physics simulation configurations."""
    configs = []
    
    for i in range(num_configs):
        t = i / max(1, num_configs - 1)  # 0 to 1
        
        config = {
            'name': f'Config_{i+1:02d}',
            
            # Attenuation (simulates water_absorption, absorption_factor)
            'attenuation_strength': np.interp(t, [0, 1], [0.88, 1.12]),
            'attenuation_power': np.interp(t, [0, 1], [0.7, 1.3]),
            
            # Speckle (simulates speckle_shape, aspect_variation)
            'speckle_intensity': np.interp(t, [0, 1], [0.02, 0.10]),
            'speckle_scale': np.interp(t, [0, 1], [3, 8]),
            'speckle_aspect_variation': np.interp(t, [0, 1], [0.5, 2.5]),
            
            # Blur (simulates beam_pattern_falloff, resolution)
            'blur_kernel': int(np.interp(t, [0, 1], [1, 7])),
            'blur_sigma': np.interp(t, [0, 1], [0.3, 1.5]),
            
            # Gain (simulates intensity variations)
            'gain_mean': np.interp(t, [0, 1], [0.80, 1.20]),
            'gain_std': np.interp(t, [0, 1], [0.02, 0.12]),
            
            # Gaussian noise (simulates sensor noise, jitter_probability)
            'gaussian_noise': np.interp(t, [0, 1], [0.003, 0.030]),
            
            # Jitter effects
            'jitter_probability': np.interp(t, [0, 1], [0.25, 0.45]),
            'jitter_range_factor': np.interp(t, [0, 1], [2.5, 5.5]),
            'jitter_max_shift': int(np.interp(t, [0, 1], [1, 4])),
            
            # Spread effects
            'spread_probability': np.interp(t, [0, 1], [0.15, 0.35]),
            'spread_kernel_size': int(np.interp(t, [0, 1], [3, 7])),
            
            # Angle-dependent scattering
            'edge_scatter_strength': np.interp(t, [0, 1], [1.5, 4.5]),
            'edge_scatter_power': np.interp(t, [0, 1], [3.0, 8.0]),
            
            # Density-dependent effects
            'density_boost': np.interp(t, [0, 1], [1.0, 2.2]),
            'density_threshold': np.interp(t, [0, 1], [0.2, 0.4]),
            
            # Shadow/absorption
            'shadow_strength': np.interp(t, [0, 1], [0.85, 1.0]),
            'shadow_falloff': np.interp(t, [0, 1], [0.015, 0.035]),
            
            # Azimuth streaking
            'streak_probability': np.interp(t, [0, 1], [0.1, 0.4]),
            'streak_intensity': np.interp(t, [0, 1], [0.85, 1.15]),
            'streak_width': int(np.interp(t, [0, 1], [20, 80])),
            
            # Grouped scatter patches
            'grouped_scatter_prob': np.interp(t, [0, 1], [0.05, 0.25]),
            'grouped_scatter_size': int(np.interp(t, [0, 1], [15, 40])),
            'grouped_scatter_strength': np.interp(t, [0, 1], [0.7, 1.3]),
            
            # Temporal decorrelation
            'temporal_noise': np.interp(t, [0, 1], [0.01, 0.04]),
        }
        
        configs.append(config)
    
    return configs


def apply_physics_augmentation(image, config):
    """Apply image-level physics approximations based on config."""
    img = image.copy()
    rows, cols = img.shape
    
    # 1. Range-dependent attenuation
    attenuation = np.linspace(1.0, config['attenuation_strength'], rows)
    attenuation = np.power(attenuation, config['attenuation_power'])
    img = img * attenuation[:, np.newaxis]
    
    # 2. Anisotropic speckle noise
    speckle = np.random.gamma(config['speckle_scale'], 
                             config['speckle_intensity'] / config['speckle_scale'], 
                             img.shape)
    if config['speckle_aspect_variation'] > 1.0:
        aspect_kernel = int(config['speckle_aspect_variation'] * 2) | 1
        speckle = cv2.GaussianBlur(speckle, (1, aspect_kernel), 0)
    img = img * (1 + speckle - config['speckle_intensity'])
    
    # 3. Blur
    if config['blur_kernel'] > 1:
        kernel_size = config['blur_kernel'] if config['blur_kernel'] % 2 == 1 else config['blur_kernel'] + 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), config['blur_sigma'])
    
    # 4. Range-dependent jitter
    if np.random.random() < config['jitter_probability']:
        for col in range(0, cols, max(1, cols // 32)):
            if np.random.random() < 0.5:
                shift = int(np.random.uniform(-config['jitter_max_shift'], config['jitter_max_shift']))
                if shift != 0:
                    col_end = min(col + cols // 32, cols)
                    img[:, col:col_end] = np.roll(img[:, col:col_end], shift, axis=0)
    
    # 5. Spread/dilation effects
    if np.random.random() < config['spread_probability']:
        kernel = np.ones((config['spread_kernel_size'], config['spread_kernel_size']), np.uint8)
        img_uint8 = (img * 255).astype(np.uint8)
        img_dilated = cv2.dilate(img_uint8, kernel, iterations=1)
        img = (img_dilated.astype(np.float32) / 255.0) * 0.7 + img * 0.3
    
    # 6. Edge/angle-dependent scattering
    beam_angles = np.linspace(-1, 1, cols)
    angle_factor = 1 + (np.abs(beam_angles) ** config['edge_scatter_power']) * (config['edge_scatter_strength'] - 1)
    img = img * angle_factor[np.newaxis, :]
    
    # 7. Density-dependent effects
    high_density_mask = img > config['density_threshold']
    img[high_density_mask] *= config['density_boost']
    if high_density_mask.any():
        density_noise = np.random.normal(0, config['gaussian_noise'] * 2, img.shape)
        img = np.where(high_density_mask, img + density_noise, img)
    
    # 8. Proximity shadow
    shadow_map = np.linspace(config['shadow_strength'], 1.0, rows)
    shadow_map = np.exp(-config['shadow_falloff'] * np.arange(rows))
    shadow_map = np.clip(shadow_map, config['shadow_strength'], 1.0)
    img = img * shadow_map[:, np.newaxis]
    
    # 9. Gain variations
    gain = np.random.normal(config['gain_mean'], config['gain_std'])
    img = img * gain
    
    # 10. Gaussian noise
    img = img + np.random.normal(0, config['gaussian_noise'], img.shape)
    
    # 11. Azimuth streaks
    if np.random.random() < config['streak_probability']:
        streak_range = np.random.randint(rows // 4, 3 * rows // 4)
        streak_height = np.random.randint(5, 15)
        start_beam = np.random.randint(0, max(1, cols - config['streak_width']))
        img[streak_range:streak_range+streak_height, 
            start_beam:start_beam+config['streak_width']] *= config['streak_intensity']
    
    # 12. Grouped scatter patches
    if np.random.random() < config['grouped_scatter_prob']:
        num_patches = np.random.randint(2, 6)
        for _ in range(num_patches):
            patch_r = np.random.randint(0, rows - config['grouped_scatter_size'])
            patch_c = np.random.randint(0, cols - config['grouped_scatter_size'])
            patch_region = img[patch_r:patch_r+config['grouped_scatter_size'], 
                             patch_c:patch_c+config['grouped_scatter_size']]
            patch_region *= config['grouped_scatter_strength']
    
    # 13. Temporal decorrelation
    temporal_variation = np.random.gamma(25.0, config['temporal_noise'] / 25.0, img.shape)
    img = img * (1 + temporal_variation - config['temporal_noise'])
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def augment_frame(sonar_img, ground_truth, physics_config, fov=120.0, max_range=20.0,
                  net_material_id=1):
    """Generate augmented version with physics variations."""
    # Create binary mask
    mask_polar = (ground_truth == net_material_id).astype(np.uint8) * 255
    mask_cart = polar_to_cartesian(mask_polar.astype(np.float32) / 255.0, max_range, fov, 400)
    mask_cart = (mask_cart > 0.5).astype(np.uint8) * 255
    
    # Apply physics-based augmentation (random noise each time)
    img_aug = apply_physics_augmentation(sonar_img, physics_config)
    
    # Convert to Cartesian
    img_cart = polar_to_cartesian(img_aug, max_range, fov, 400)
    
    return img_cart, mask_cart


def export_augmented_dataset(input_dir, output_dir, net_material_id=1,
                            num_configs=20, images_per_config=5000,
                            base_frames=1000, frame_step=3, fov=120.0, max_range=20.0,
                            train_split=0.8):
    """Generate augmented training dataset with physics variations."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    frames_dir = input_path / 'frames'
    
    # Create output directories
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Get frame files (with frame step)
    all_frames = sorted(frames_dir.glob('frame_*.npz'))
    
    # Limit by base_frames if specified
    if base_frames > 0:
        all_frames = all_frames[:base_frames]
    
    frame_files = all_frames[::frame_step]  # Every Nth frame
    
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
    
    # Check if we have enough frames
    if len(frame_files) < images_per_config:
        print(f"\n⚠️  WARNING: Not enough frames!")
        print(f"   Requested: {images_per_config:,} images per config")
        print(f"   Available: {len(frame_files):,} frames (after frame_step={frame_step})")
        print(f"   Need: {images_per_config * frame_step:,} total simulation frames")
        print(f"   Current: {len(all_frames):,} simulation frames")
        print(f"\n   → Will generate only {len(frame_files):,} images per config")
        print(f"   → To get {images_per_config:,} images, run simulation for {images_per_config * frame_step:,} frames\n")
        images_per_config = len(frame_files)  # Adjust to available
    
    # Generate physics configurations
    configs = generate_physics_configurations(num_configs)
    
    total_images = num_configs * images_per_config
    train_count_per_config = int(images_per_config * train_split)
    
    print(f"=" * 70)
    print(f"BATCH PHYSICS AUGMENTATION")
    print(f"=" * 70)
    print(f"FOV: {fov}° (CONSTANT)")
    print(f"Range: {max_range}m (CONSTANT)")
    print(f"Physics configs: {num_configs}")
    print(f"Images per config: {images_per_config:,}")
    print(f"Available frames: {len(all_frames):,}")
    print(f"Frames to process: {len(frame_files):,} (every {frame_step}{'st' if frame_step==1 else 'nd' if frame_step==2 else 'rd' if frame_step==3 else 'th'})")
    print(f"Variants per frame: 1 (1:1 mapping)")
    print(f"Total images: {total_images:,}")
    print(f"Train/Val: {train_split:.0%} / {1-train_split:.0%}")
    print(f"\nSimulated physics variations:")
    print(f"  • Attenuation (water absorption, absorption_factor)")
    print(f"  • Anisotropic speckle (speckle_shape, aspect_variation)")
    print(f"  • Blur (beam_pattern_falloff)")
    print(f"  • Range jitter (jitter_probability, jitter_range_factor, jitter_max_offset)")
    print(f"  • Spread effects (spread_probability, multi-bin spreading)")
    print(f"  • Edge scattering (angle_scatter_strength, angle_scatter_power)")
    print(f"  • Density effects (density_scatter_strength, density_noise_boost)")
    print(f"  • Proximity shadows (proximity_shadow_strength, scattering_loss)")
    print(f"  • Gain variations (intensity calibration)")
    print(f"  • Gaussian noise (sensor noise)")
    print(f"  • Azimuth streaks (azimuth_streak parameters)")
    print(f"  • Grouped scatter (grouped_scatter parameters)")
    print(f"  • Temporal decorrelation (temporal_decorrelation_shape)")
    print(f"\nBatch: Complete {images_per_config:,} images per config before next")
    print(f"=" * 70)
    
    sample_idx = 0
    
    # Process each configuration (batch)
    for config_idx, physics_config in enumerate(configs, 1):
        print(f"\n[{config_idx}/{num_configs}] {physics_config['name']}")
        print(f"  atten={physics_config['attenuation_strength']:.2f}, "
              f"speckle={physics_config['speckle_intensity']:.3f}, "
              f"jitter_p={physics_config['jitter_probability']:.2f}, "
              f"edge={physics_config['edge_scatter_strength']:.2f}, "
              f"streak_p={physics_config['streak_probability']:.2f}")
        
        config_sample_count = 0
        
        for frame_path in tqdm(frame_files, desc=f"  Frames", leave=False):
            if config_sample_count >= images_per_config:
                break
            
            # Load data
            data = np.load(frame_path)
            
            if 'sonar_image' not in data or 'ground_truth' not in data:
                data.close()
                continue
            
            sonar_img = data['sonar_image']
            ground_truth = data['ground_truth']
            data.close()
            
            # Generate single augmented image (1 variant per frame)
            img_cart, mask_cart = augment_frame(sonar_img, ground_truth, physics_config, 
                                               fov, max_range, net_material_id)
            
            split = 'train' if config_sample_count < train_count_per_config else 'val'
            
            # Normalize to 0-255
            img_norm = ((img_cart - img_cart.min()) / 
                       (img_cart.max() - img_cart.min() + 1e-8) * 255).astype(np.uint8)
            
            # Save as PNG
            Image.fromarray(img_norm).save(output_path / split / 'images' / f'{sample_idx:06d}.png')
            Image.fromarray(mask_cart).save(output_path / split / 'masks' / f'{sample_idx:06d}.png')
            
            sample_idx += 1
            config_sample_count += 1
        
        print(f"  ✓ Generated {config_sample_count:,} images")
    
    # Count final samples
    train_imgs = len(list((output_path / 'train' / 'images').glob('*.png')))
    val_imgs = len(list((output_path / 'val' / 'images').glob('*.png')))
    
    print(f"\n" + "=" * 70)
    print(f"✅ DATASET COMPLETE")
    print(f"=" * 70)
    print(f"Output: {output_path.absolute()}")
    print(f"Train: {train_imgs:,} images")
    print(f"Val: {val_imgs:,} images")
    print(f"Total: {train_imgs + val_imgs:,} images")
    print(f"=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate augmented sonar data with simulated physics variations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images (20 physics configs × 5000 each)
  python augment_data.py --num_configs 20 --images_per_config 5000
  
  # Use every 2nd frame
  python augment_data.py --num_configs 20 --images_per_config 5000 --frame_step 2

NOTE: Applies image-level approximations of physics variations.
      FOV and range remain constant (120°, 20m by default).
        """
    )
    parser.add_argument('--input_data', type=str, 
                       default='../simulation/data/runs/net_following_fish',
                       help='Path to simulation data')
    parser.add_argument('--output_dir', type=str, 
                       default='training_data_augmented',
                       help='Output directory')
    parser.add_argument('--net_material_id', type=int, default=1,
                       help='Material ID for net')
    parser.add_argument('--num_configs', type=int, default=20,
                       help='Number of physics configs')
    parser.add_argument('--images_per_config', type=int, default=5000,
                       help='Images per config')
    parser.add_argument('--base_frames', type=int, default=1000,
                       help='Base frames to use')
    parser.add_argument('--frame_step', type=int, default=3,
                       help='Process every Nth frame (1=all, 3=every 3rd)')
    parser.add_argument('--fov', type=float, default=120.0,
                       help='FOV in degrees (CONSTANT)')
    parser.add_argument('--max_range', type=float, default=20.0,
                       help='Max range in meters (CONSTANT)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/val split')
    
    args = parser.parse_args()
    
    export_augmented_dataset(
        input_dir=args.input_data,
        output_dir=args.output_dir,
        net_material_id=args.net_material_id,
        num_configs=args.num_configs,
        images_per_config=args.images_per_config,
        base_frames=args.base_frames,
        frame_step=args.frame_step,
        fov=args.fov,
        max_range=args.max_range,
        train_split=args.train_split
    )
