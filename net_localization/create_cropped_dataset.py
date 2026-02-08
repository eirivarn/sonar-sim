"""
Create 10x10m dataset from existing 20x20m data.

Two methods:
1. Re-convert from polar sonar data (best quality)
2. Crop existing Cartesian images (faster)

Usage:
    # Method 1: Re-convert from polar (RECOMMENDED - better quality)
    python create_cropped_dataset.py --method polar --input_runs ../data/runs/temp_augmentation_* --output_dir train_val_10m --new_range 10.0
    
    # Method 2: Crop existing images (faster, but lower quality)
    python create_cropped_dataset.py --method crop --input_dir train_val --output_dir train_val_10m --crop_factor 0.5
"""

import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import glob
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def polar_to_cartesian_custom_range(polar_image, range_m=10.0, fov_deg=120.0, output_size=256, original_range_m=20.0):
    """
    Convert polar sonar image to Cartesian with custom range.
    
    This allows us to "zoom in" by using a smaller range_m value than the
    original simulation used.
    
    Args:
        polar_image: (range_bins, num_beams) polar sonar image
        range_m: Maximum range to visualize (meters) - the "zoom" level
        fov_deg: Field of view in degrees
        output_size: Output image size (square)
        original_range_m: The range the polar data was originally collected at (default 20.0m)
        
    Returns:
        Cartesian image (output_size, output_size) in [0, 1] range
    """
    num_range_bins, num_beams = polar_image.shape
    
    # Calculate extent for the desired range
    x_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    y_extent = range_m
    
    # Create coordinate grids
    height, width = output_size, output_size
    j_grid, i_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixels to meters in sonar frame
    z_m = (j_grid - width/2) * (2 * x_extent / width)
    y_m = i_grid * (y_extent / height)
    
    # Convert to polar coordinates
    r_m = np.sqrt(z_m**2 + y_m**2)
    theta_rad = np.arctan2(z_m, y_m)
    
    # Create mask for points within sonar cone
    valid_mask = (r_m <= range_m) & (np.abs(theta_rad) <= np.deg2rad(fov_deg/2))
    
    # Find corresponding indices in polar image
    # CRITICAL: Map to the original range, not the display range!
    # This way, if original_range_m=20 and range_m=10, we only sample the first half of the polar image
    r_idx = ((r_m / original_range_m) * (num_range_bins - 1)).astype(np.int32)
    theta_idx = ((theta_rad + np.deg2rad(fov_deg/2)) / np.deg2rad(fov_deg) * (num_beams - 1)).astype(np.int32)
    
    # Clamp indices
    r_idx = np.clip(r_idx, 0, num_range_bins - 1)
    theta_idx = np.clip(theta_idx, 0, num_beams - 1)
    
    # Create output
    output = np.zeros((height, width), dtype=np.float32)
    output[valid_mask] = polar_image[r_idx[valid_mask], theta_idx[valid_mask]]
    
    # Set invalid regions to black
    output[~valid_mask] = 0
    
    return output


def show_preview_polar(npz_files, new_range=10.0, image_size=256):
    """Show preview of first frame before/after conversion"""
    # Load first npz file
    data = np.load(npz_files[0], allow_pickle=True)
    
    # Handle both formats: 'cones' (consolidated) or 'sonar_images' (old)
    if 'cones' in data:
        polar = data['cones'][0]
    elif 'sonar_images' in data:
        polar = data['sonar_images'][0]
    else:
        raise KeyError(f"NPZ file must contain either 'cones' or 'sonar_images'. Found: {list(data.keys())}")
    
    # Convert with original 20m range
    cartesian_20m = polar_to_cartesian_custom_range(polar, 20.0, 120.0, image_size, original_range_m=20.0)
    
    # Convert with new range (zoomed in)
    cartesian_new = polar_to_cartesian_custom_range(polar, new_range, 120.0, image_size, original_range_m=20.0)
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(cartesian_20m, cmap='gray')
    ax1.set_title(f'Original (20.0m range)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(cartesian_new, cmap='gray')
    ax2.set_title(f'New ({new_range}m range - ZOOMED IN)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Ask for confirmation
    response = input(f"\n👀 Preview shown above. Proceed with conversion to {new_range}m? (y/n): ")
    return response.lower() == 'y'


def show_preview_crop(input_dir, crop_factor=0.5):
    """Show preview of first frame before/after cropping"""
    input_dir = Path(input_dir)
    
    # Find first image
    img_dir = input_dir / 'train' / 'images'
    if not img_dir.exists():
        img_dir = input_dir / 'val' / 'images'
    
    if not img_dir.exists():
        print("❌ Could not find images for preview")
        return False
    
    img_files = list(img_dir.glob('*.png'))
    if not img_files:
        print("❌ No images found for preview")
        return False
    
    # Load first image
    img = np.array(Image.open(img_files[0]))
    
    # Calculate crop
    h, w = img.shape[:2]
    crop_h = int(h * crop_factor)
    crop_w = int(w * crop_factor)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    img_cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Original ({h}x{w} pixels)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    # Draw crop rectangle
    rect = plt.Rectangle((start_w, start_h), crop_w, crop_h, 
                         fill=False, edgecolor='lime', linewidth=2)
    ax1.add_patch(rect)
    
    ax2.imshow(img_cropped, cmap='gray')
    ax2.set_title(f'Cropped ({crop_h}x{crop_w} pixels - {crop_factor*100:.0f}%)', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Ask for confirmation
    response = input(f"\n👀 Preview shown above. Proceed with cropping (factor={crop_factor})? (y/n): ")
    return response.lower() == 'y'


def process_polar_method(input_runs, output_dir, new_range=10.0, image_size=256, skip_preview=False):
    """
    Re-convert polar sonar data to Cartesian with smaller range.
    
    Args:
        input_runs: List of run directories containing .npz files
        output_dir: Output directory for images
        new_range: New range in meters (e.g., 10.0)
        image_size: Output image size
        skip_preview: Skip preview confirmation
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .npz files
    npz_files = []
    for pattern in input_runs:
        npz_files.extend(glob.glob(pattern + '/*.npz'))
    
    if not npz_files:
        print(f"❌ No .npz files found in: {input_runs}")
        return
    
    print(f"Found {len(npz_files)} .npz files")
    print(f"Converting to {new_range}m range at {image_size}x{image_size} pixels")
    
    # Show preview and ask for confirmation
    if not skip_preview:
        if not show_preview_polar(npz_files, new_range, image_size):
            print("❌ Cancelled by user")
            return
    
    total_images = 0
    train_count = 0
    val_count = 0
    
    for npz_path in tqdm(npz_files, desc='Processing runs'):
        npz_path = Path(npz_path)
        
        # Load data
        data = np.load(npz_path, allow_pickle=True)
        
        # Handle both formats: 'cones' (consolidated) or 'sonar_images' (old)
        if 'cones' in data:
            sonar_images = data['cones']  # (N, range_bins, beams)
        elif 'sonar_images' in data:
            sonar_images = data['sonar_images']
        else:
            print(f"⚠️  Unknown format in {npz_path.name}, skipping")
            continue
        
        # Get run directory - check for frames/ subdirectory with images
        run_dir = npz_path.parent
        frames_dir = run_dir / 'frames'
        
        # For simulation data, we need to generate masks if they don't exist
        # For now, we'll create simple masks based on the sonar data itself
        # This is a temporary solution - ideally masks should be pre-generated
        
        # Process each frame
        for idx in range(len(sonar_images)):
            # Convert sonar to Cartesian with new range
            polar = sonar_images[idx]
            cartesian = polar_to_cartesian_custom_range(polar, new_range, 120.0, image_size, original_range_m=20.0)
            
            # Normalize: convert to dB scale then to [0, 255]
            # Handle raw intensity properly
            polar_db = 10 * np.log10(np.maximum(polar, 1e-10))
            cartesian_db = 10 * np.log10(np.maximum(cartesian, 1e-10))
            # Normalize to [0, 255] using a reasonable dB range (-60 to 0 dB)
            cartesian_normalized = np.clip((cartesian_db + 60) / 60, 0, 1)
            cartesian_u8 = (cartesian_normalized * 255).astype(np.uint8)
            
            # For now, create a simple mask based on intensity threshold
            # TODO: Use proper ground truth masks when available
            # This creates a binary mask where high-intensity regions are marked
            threshold = 0.3  # Adjust this based on your data
            mask = (cartesian_normalized > threshold).astype(np.uint8) * 255
            
            # Alternatively, check if masks exist in frames directory
            if frames_dir.exists():
                mask_path = frames_dir / f'mask_{idx:06d}.png'
                if mask_path.exists():
                    mask = np.array(Image.open(mask_path))
                    if mask.shape[0] != image_size or mask.shape[1] != image_size:
                        mask = np.array(Image.fromarray(mask).resize((image_size, image_size), Image.NEAREST))
            
            # Determine train/val split (80/20)
            if np.random.rand() < 0.8:
                img_out = train_dir / 'images' / f'{run_dir.name}_{idx:06d}.png'
                mask_out = train_dir / 'masks' / f'{run_dir.name}_{idx:06d}.png'
                train_count += 1
            else:
                img_out = val_dir / 'images' / f'{run_dir.name}_{idx:06d}.png'
                mask_out = val_dir / 'masks' / f'{run_dir.name}_{idx:06d}.png'
                val_count += 1
            
            # Save
            img_out.parent.mkdir(parents=True, exist_ok=True)
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cartesian_u8).save(img_out)
            Image.fromarray(mask).save(mask_out)
            
            total_images += 1
    
    print(f"\n✅ Dataset created:")
    print(f"   Total images: {total_images}")
    print(f"   Train: {train_count}")
    print(f"   Val: {val_count}")
    print(f"   Output: {output_dir}")


def process_crop_method(input_dir, output_dir, crop_factor=0.5, skip_preview=False):
    """
    Crop existing Cartesian images to smaller size.
    
    Args:
        input_dir: Input directory with train/val structure
        output_dir: Output directory
        crop_factor: Fraction to keep (0.5 = keep center 50% = 10m from 20m)
        skip_preview: Skip preview confirmation
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Show preview and ask for confirmation
    if not skip_preview:
        if not show_preview_crop(input_dir, crop_factor):
            print("❌ Cancelled by user")
            return
    
    # Process train and val sets
    for split in ['train', 'val']:
        split_input = input_dir / split
        split_output = output_dir / split
        
        if not split_input.exists():
            print(f"⚠️  {split} directory not found, skipping")
            continue
        
        print(f"\n📁 Processing {split} set...")
        
        # Process images
        img_dir = split_input / 'images'
        mask_dir = split_input / 'masks'
        
        if not img_dir.exists() or not mask_dir.exists():
            print(f"⚠️  Missing images or masks directory")
            continue
        
        img_output = split_output / 'images'
        mask_output = split_output / 'masks'
        img_output.mkdir(parents=True, exist_ok=True)
        mask_output.mkdir(parents=True, exist_ok=True)
        
        image_files = list(img_dir.glob('*.png'))
        
        for img_path in tqdm(image_files, desc=f'Cropping {split}'):
            # Load image and mask
            img = np.array(Image.open(img_path))
            mask_path = mask_dir / img_path.name
            
            if not mask_path.exists():
                continue
            
            mask = np.array(Image.open(mask_path))
            
            # Calculate crop dimensions
            h, w = img.shape[:2]
            crop_h = int(h * crop_factor)
            crop_w = int(w * crop_factor)
            
            # Center crop
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            
            img_cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
            mask_cropped = mask[start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Save
            Image.fromarray(img_cropped).save(img_output / img_path.name)
            Image.fromarray(mask_cropped).save(mask_output / img_path.name)
        
        print(f"   ✅ {len(image_files)} images processed")
    
    print(f"\n✅ Cropped dataset created in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create 10x10m dataset from existing 20x20m data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Re-convert from polar (BEST QUALITY)
    python create_cropped_dataset.py --method polar --input_runs ../data/runs/temp_augmentation_01 ../data/runs/temp_augmentation_02 --output_dir train_val_10m
    
    # Using glob pattern
    python create_cropped_dataset.py --method polar --input_runs "../data/runs/temp_augmentation_*" --output_dir train_val_10m
    
    # Crop existing images (FASTER)
    python create_cropped_dataset.py --method crop --input_dir train_val --output_dir train_val_10m --crop_factor 0.5
    
Notes:
    - Method 'polar' gives better quality (re-renders from raw sonar data)
    - Method 'crop' is faster (just crops existing images)
    - crop_factor=0.5 means 10m from 20m (50% of each dimension)
    - For 10m from 20m, use either method='polar' with new_range=10.0 or method='crop' with crop_factor=0.5
        """
    )
    
    parser.add_argument('--method', choices=['polar', 'crop'], required=True,
                       help='Conversion method: polar (re-convert) or crop (crop existing)')
    
    # Polar method args
    parser.add_argument('--input_runs', nargs='+',
                       help='Input run directories (for polar method). Can use glob patterns.')
    parser.add_argument('--new_range', type=float, default=10.0,
                       help='New range in meters (for polar method). Default: 10.0')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Output image size (for polar method). Default: 256')
    
    # Crop method args
    parser.add_argument('--input_dir', type=str,
                       help='Input directory with train/val structure (for crop method)')
    parser.add_argument('--crop_factor', type=float, default=0.5,
                       help='Crop factor (for crop method). 0.5 = keep center 50%. Default: 0.5')
    
    # Common args
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/val split. Default: 42')
    parser.add_argument('--skip_preview', action='store_true',
                       help='Skip preview confirmation and process immediately')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    if args.method == 'polar':
        if not args.input_runs:
            print("❌ --input_runs required for polar method")
            return
        
        # Expand glob patterns
        expanded_runs = []
        for pattern in args.input_runs:
            matches = glob.glob(pattern)
            if matches:
                expanded_runs.extend(matches)
            else:
                expanded_runs.append(pattern)  # Keep original if no matches
        
        print(f"Processing {len(expanded_runs)} run directories")
        process_polar_method(expanded_runs, args.output_dir, args.new_range, args.image_size, args.skip_preview)
        
    elif args.method == 'crop':
        if not args.input_dir:
            print("❌ --input_dir required for crop method")
            return
        
        process_crop_method(args.input_dir, args.output_dir, args.crop_factor, args.skip_preview)


if __name__ == '__main__':
    main()
