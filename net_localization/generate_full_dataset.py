"""
Generate complete training dataset by running simulation + augmentation multiple times.

This script:
1. Runs simulation for N frames
2. Applies one physics config to generate augmented images
3. Cleans up frames
4. Repeats for next physics config

This ensures each config uses completely different simulation data for maximum diversity.
"""

import subprocess
import shutil
from pathlib import Path
import argparse
import sys


def run_simulation(run_name, num_samples):
    """
    Run simulation data collection.
    
    Args:
        run_name: Name for the simulation run (saves to data/runs/<run_name>)
        num_samples: Number of samples to collect
    """
    print(f"\n{'='*70}")
    print(f"RUNNING SIMULATION")
    print(f"{'='*70}")
    print(f"Samples: {num_samples:,}")
    print(f"Run name: {run_name}")
    print(f"Output: data/runs/{run_name}")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable, 'simulation.py',
        '--save', run_name,
        '--collect', 'random',  # Random robot exploration
        '--num-samples', str(num_samples)
    ]
    
    # Run from simulation directory
    result = subprocess.run(
        cmd,
        cwd='../simulation',
        check=True
    )
    
    return result.returncode == 0


def run_augmentation(input_dir, output_dir, config_start, config_end, 
                    images_per_config, frame_step, train_split):
    """
    Run augmentation on simulation data.
    
    Args:
        input_dir: Simulation data directory
        output_dir: Output directory for augmented images
        config_start: Starting config number (1-based)
        config_end: Ending config number (1-based, inclusive)
        images_per_config: Images per config
        frame_step: Process every Nth frame
        train_split: Train/val split ratio
    """
    num_configs = config_end - config_start + 1
    
    print(f"\n{'='*70}")
    print(f"RUNNING AUGMENTATION")
    print(f"{'='*70}")
    print(f"Configs: {config_start} to {config_end} ({num_configs} total)")
    print(f"Images per config: {images_per_config:,}")
    print(f"Frame step: {frame_step}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable, 'augment_data.py',
        '--input_data', str(input_dir),
        '--output_dir', str(output_dir),
        '--num_configs', str(num_configs),
        '--images_per_config', str(images_per_config),
        '--frame_step', str(frame_step),
        '--base_frames', '0',  # Use all available frames
        '--train_split', str(train_split)
    ]
    
    # Run from net_localization directory
    script_dir = Path(__file__).parent
    result = subprocess.run(cmd, cwd=script_dir, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate complete training dataset with multiple simulation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Generate 100k images (20 configs × 5k each)
  # Runs simulation 20 times, 10k steps each
  python generate_full_dataset.py --total_configs 20 --images_per_config 5000 \\
      --sim_steps 10000 --frame_step 2

  # Quick test (2 configs × 100 images)
  python generate_full_dataset.py --total_configs 2 --images_per_config 100 \\
      --sim_steps 200 --frame_step 2

Note: Each simulation run uses different random paths for maximum diversity.
        """
    )
    
    parser.add_argument('--total_configs', type=int, default=20,
                       help='Total number of physics configs (default: 20)')
    parser.add_argument('--images_per_config', type=int, default=5000,
                       help='Images per config (default: 5000)')
    parser.add_argument('--sim_samples', type=int, default=10000,
                       help='Simulation samples per run (default: 10000)')
    parser.add_argument('--frame_step', type=int, default=2,
                       help='Process every Nth frame in augmentation (default: 2)')
    parser.add_argument('--output_dir', type=str, default='training_data_augmented',
                       help='Final output directory')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/val split (default: 0.8)')
    parser.add_argument('--keep_sim_data', action='store_true',
                       help='Keep simulation frames after augmentation (uses more disk space)')
    parser.add_argument('--configs_per_batch', type=int, default=1,
                       help='Number of configs to apply per simulation run (default: 1)')
    
    args = parser.parse_args()
    
    # Validate
    frames_per_run = args.sim_samples
    frames_after_step = frames_per_run // args.frame_step
    images_needed = args.images_per_config * args.configs_per_batch
    
    if frames_after_step < images_needed:
        print(f"❌ ERROR: Not enough frames after frame_step!")
        print(f"   Simulation will generate: {frames_per_run:,} frames")
        print(f"   After frame_step={args.frame_step}: {frames_after_step:,} frames")
        print(f"   Augmentation needs: {images_needed:,} frames")
        print(f"   Solutions:")
        print(f"     1. Increase --sim_samples to at least {images_needed * args.frame_step:,}")
        print(f"     2. Decrease --images_per_config (current: {args.images_per_config:,})")
        print(f"     3. Decrease --frame_step (current: {args.frame_step})")
        sys.exit(1)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate runs needed
    num_runs = (args.total_configs + args.configs_per_batch - 1) // args.configs_per_batch
    
    print(f"\n{'='*70}")
    print(f"DATASET GENERATION PLAN")
    print(f"{'='*70}")
    print(f"Total configs: {args.total_configs}")
    print(f"Images per config: {args.images_per_config:,}")
    print(f"Total images: {args.total_configs * args.images_per_config:,}")
    print(f"")
    print(f"Simulation runs: {num_runs}")
    print(f"Samples per run: {args.sim_samples:,}")
    print(f"Frame step (augmentation): {args.frame_step}")
    print(f"Frames used per run: {frames_after_step:,}")
    print(f"Configs per run: {args.configs_per_batch}")
    print(f"")
    print(f"Output: {output_dir.absolute()}")
    print(f"Train/Val: {args.train_split:.0%} / {1-args.train_split:.0%}")
    print(f"{'='*70}\n")
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run simulation + augmentation in batches
    config_idx = 1
    
    for run_num in range(1, num_runs + 1):
        config_start = config_idx
        config_end = min(config_idx + args.configs_per_batch - 1, args.total_configs)
        run_name = f'temp_augmentation_{config_start:02d}'
        sim_dir = Path(f'../simulation/data/runs/{run_name}')
        
        print(f"\n\n{'#'*70}")
        print(f"# RUN {run_num}/{num_runs}: Configs {config_start}-{config_end}")
        print(f"{'#'*70}\n")
        
        # 1. Run simulation
        try:
            run_simulation(run_name, args.sim_samples)
        except subprocess.CalledProcessError as e:
            print(f"❌ Simulation failed: {e}")
            sys.exit(1)
        
        # 2. Run augmentation
        try:
            run_augmentation(
                sim_dir,
                output_dir,
                config_start,
                config_end,
                args.images_per_config,
                args.frame_step,
                args.train_split
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Augmentation failed: {e}")
            sys.exit(1)
        
        # 3. Clean up simulation data (unless keeping)
        if not args.keep_sim_data:
            print(f"\n🗑️  Cleaning up simulation data...")
            if sim_dir.exists():
                shutil.rmtree(sim_dir)
            print(f"   ✓ Removed {sim_dir}")
        
        config_idx = config_end + 1
    
    # Final summary
    train_imgs = len(list((output_dir / 'train' / 'images').glob('*.png')))
    val_imgs = len(list((output_dir / 'val' / 'images').glob('*.png')))
    
    print(f"\n\n{'='*70}")
    print(f"✅ DATASET GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Train images: {train_imgs:,}")
    print(f"Val images: {val_imgs:,}")
    print(f"Total: {train_imgs + val_imgs:,}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
