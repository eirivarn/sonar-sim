"""Reconstruct and visualize saved runs as videos or interactive playback.

This script loads saved sonar data and recreates the visualization, either as:
- MP4/AVI video file
- Interactive matplotlib playback with controls
- Individual frame images

USAGE:
------
Create video from saved run:
    python visualize_run.py data/runs/my_experiment_01 --output video.mp4

Interactive playback:
    python visualize_run.py data/runs/my_experiment_01 --interactive

Save as image sequence:
    python visualize_run.py data/runs/my_experiment_01 --output-dir frames/

REQUIREMENTS:
-------------
For video output:
    pip install opencv-python
    or
    pip install ffmpeg-python
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import json
from pathlib import Path
import argparse


def load_run_data(run_dir):
    """Load all data from a saved run.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        dict with:
            - run_config: Run configuration
            - frames: List of frame data dicts
            - num_frames: Total number of frames
    """
    run_dir = Path(run_dir)
    
    # Load run configuration
    with open(run_dir / 'run_config.json') as f:
        run_config = json.load(f)
    
    # Find all frames
    sonar_files = sorted((run_dir / 'sonar').glob('frame_*.npy'))
    num_frames = len(sonar_files)
    
    print(f"Found {num_frames} frames in {run_dir}")
    
    # Load all frame data
    frames = []
    for i in range(num_frames):
        frame_data = {
            'sonar': np.load(run_dir / 'sonar' / f'frame_{i:06d}.npy'),
            'ground_truth': np.load(run_dir / 'ground_truth' / f'frame_{i:06d}.npy'),
        }
        
        # Load metadata if available
        meta_path = run_dir / 'metadata' / f'frame_{i:06d}.json'
        if meta_path.exists():
            with open(meta_path) as f:
                frame_data['metadata'] = json.load(f)
        
        frames.append(frame_data)
    
    return {
        'run_config': run_config,
        'frames': frames,
        'num_frames': num_frames
    }


def create_visualization(frame_data, run_config, material_colors=None):
    """Create visualization for a single frame.
    
    Args:
        frame_data: Dict with sonar, ground_truth, metadata
        run_config: Run configuration dict
        material_colors: Optional material color mapping
        
    Returns:
        fig: Matplotlib figure
    """
    if material_colors is None:
        # Default colors (from VISUALIZATION_CONFIG)
        material_colors = {
            0: [0, 0, 0],           # Empty - black
            1: [0, 100, 255],       # Net - blue
            2: [0, 150, 200],       # Rope - cyan
            3: [255, 140, 0],       # Fish - orange
            4: [128, 128, 128],     # Wall - gray
            5: [0, 200, 0],         # Biomass - green
            6: [200, 200, 100],     # Debris light - tan
            7: [150, 150, 70],      # Debris medium - brown
            8: [100, 100, 40],      # Debris heavy - dark brown
            9: [180, 180, 180],     # Concrete - light gray
            10: [139, 90, 43],      # Wood - brown
            11: [34, 139, 34],      # Foliage - forest green
            12: [192, 192, 192],    # Metal - silver
            13: [173, 216, 230],    # Glass - light blue
        }
    
    # Create figure with 3 panels
    fig, (ax_sonar, ax_map, ax_gt) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Run: {run_config['run_name']} - Frame {frame_data.get('metadata', {}).get('frame', 0)}", 
                 fontsize=14, weight='bold')
    
    # === SONAR PANEL ===
    sonar_image = frame_data['sonar']
    
    # Convert to dB and normalize
    image_db = 10 * np.log10(np.maximum(sonar_image, 1e-10))
    db_norm = 60.0  # Default normalization
    image_normalized = np.clip((image_db + db_norm) / db_norm, 0, 1)
    
    ax_sonar.imshow(image_normalized, cmap='viridis', aspect='auto', origin='lower')
    
    if 'metadata' in frame_data:
        pos = frame_data['metadata']['sonar_position']
        ax_sonar.set_title(f'Sonar View - Polar\nPos: [{pos[0]:.1f}, {pos[1]:.1f}]')
    else:
        ax_sonar.set_title('Sonar View - Polar')
    
    ax_sonar.set_xlabel('Beams')
    ax_sonar.set_ylabel('Range Bins')
    ax_sonar.grid(False)
    
    # === MAP PANEL ===
    # For now, show placeholder (would need scene info to reconstruct properly)
    ax_map.text(0.5, 0.5, 'Map view requires scene reconstruction\n(Use interactive mode with --scene)', 
                ha='center', va='center', transform=ax_map.transAxes)
    ax_map.set_title('World Map')
    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    
    if 'metadata' in frame_data:
        # Draw sonar position if available
        world_size = run_config.get('world_size', 30.0)
        ax_map.set_xlim(0, world_size)
        ax_map.set_ylim(world_size, 0)
        ax_map.set_aspect('equal')
        
        pos = frame_data['metadata']['sonar_position']
        direction = frame_data['metadata']['sonar_direction']
        
        ax_map.scatter(pos[0], pos[1], c='red', s=100, marker='^', zorder=5)
        ax_map.arrow(pos[0], pos[1], direction[0]*2, direction[1]*2,
                    head_width=0.5, head_length=0.3, fc='red', ec='red')
    
    # === GROUND TRUTH PANEL ===
    ground_truth = frame_data['ground_truth']
    
    # Convert to RGB
    gt_rgb = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
    for mat_id, color in material_colors.items():
        mask = ground_truth == mat_id
        gt_rgb[mask] = color
    
    ax_gt.imshow(gt_rgb, aspect='auto', origin='lower')
    ax_gt.set_title('Ground Truth - Material Segmentation')
    ax_gt.set_xlabel('Beams')
    ax_gt.set_ylabel('Range Bins')
    ax_gt.grid(False)
    
    # Add legend
    unique_materials = np.unique(ground_truth)
    material_names = {
        0: 'Empty', 1: 'Net', 2: 'Rope', 3: 'Fish',
        4: 'Wall', 5: 'Biomass', 6: 'Debris Light',
        7: 'Debris Medium', 8: 'Debris Heavy', 9: 'Concrete',
        10: 'Wood', 11: 'Foliage', 12: 'Metal', 13: 'Glass'
    }
    
    legend_elements = []
    for mat_id in unique_materials[:5]:  # Limit legend size
        if mat_id in material_colors:
            legend_elements.append(
                Patch(facecolor=np.array(material_colors[mat_id])/255, 
                     label=material_names.get(mat_id, f'Mat{mat_id}'))
            )
    
    ax_gt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


def create_video(run_data, output_path, fps=10, dpi=100):
    """Create video from run data.
    
    Args:
        run_data: Dict from load_run_data()
        output_path: Output video file path (.mp4, .avi, .gif)
        fps: Frames per second
        dpi: Resolution (dots per inch)
    """
    output_path = Path(output_path)
    frames = run_data['frames']
    run_config = run_data['run_config']
    
    print(f"Creating video: {output_path}")
    print(f"Frames: {len(frames)}, FPS: {fps}, DPI: {dpi}")
    
    # Create figure for first frame
    fig = create_visualization(frames[0], run_config)
    
    def update(frame_idx):
        """Update function for animation."""
        print(f"\rRendering frame {frame_idx+1}/{len(frames)}", end='', flush=True)
        
        # Clear all axes
        for ax in fig.axes:
            ax.clear()
        
        # Recreate visualization (simpler than updating)
        frame_data = frames[frame_idx]
        
        ax_sonar, ax_map, ax_gt = fig.axes
        
        # Sonar panel
        sonar_image = frame_data['sonar']
        image_db = 10 * np.log10(np.maximum(sonar_image, 1e-10))
        image_normalized = np.clip((image_db + 60.0) / 60.0, 0, 1)
        ax_sonar.imshow(image_normalized, cmap='viridis', aspect='auto', origin='lower')
        
        if 'metadata' in frame_data:
            pos = frame_data['metadata']['sonar_position']
            ax_sonar.set_title(f'Sonar View - Polar\nPos: [{pos[0]:.1f}, {pos[1]:.1f}]')
        ax_sonar.set_xlabel('Beams')
        ax_sonar.set_ylabel('Range Bins')
        
        # Map panel (placeholder)
        ax_map.text(0.5, 0.5, 'Map view', ha='center', va='center', transform=ax_map.transAxes)
        ax_map.set_title('World Map')
        
        if 'metadata' in frame_data:
            world_size = run_config.get('world_size', 30.0)
            ax_map.set_xlim(0, world_size)
            ax_map.set_ylim(world_size, 0)
            pos = frame_data['metadata']['sonar_position']
            direction = frame_data['metadata']['sonar_direction']
            ax_map.scatter(pos[0], pos[1], c='red', s=100, marker='^')
            ax_map.arrow(pos[0], pos[1], direction[0]*2, direction[1]*2,
                        head_width=0.5, fc='red', ec='red')
        
        # Ground truth panel
        ground_truth = frame_data['ground_truth']
        material_colors = {
            0: [0, 0, 0], 1: [0, 100, 255], 2: [0, 150, 200], 3: [255, 140, 0],
            4: [128, 128, 128], 5: [0, 200, 0], 6: [200, 200, 100]
        }
        gt_rgb = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
        for mat_id, color in material_colors.items():
            gt_rgb[ground_truth == mat_id] = color
        
        ax_gt.imshow(gt_rgb, aspect='auto', origin='lower')
        ax_gt.set_title('Ground Truth')
        ax_gt.set_xlabel('Beams')
        ax_gt.set_ylabel('Range Bins')
        
        fig.suptitle(f"Run: {run_config['run_name']} - Frame {frame_idx}", fontsize=14, weight='bold')
        
        return fig.axes
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    
    # Save with appropriate writer
    if output_path.suffix == '.gif':
        writer = PillowWriter(fps=fps)
    else:
        # Try FFMpeg, fall back to Pillow
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
        except:
            print("\nFFMpeg not available, using Pillow (slower)")
            writer = PillowWriter(fps=fps)
    
    anim.save(output_path, writer=writer, dpi=dpi)
    print(f"\n\nVideo saved: {output_path}")


def interactive_playback(run_data):
    """Interactive playback with keyboard controls.
    
    Controls:
        Space: Play/Pause
        Left/Right: Previous/Next frame
        Home/End: First/Last frame
    """
    frames = run_data['frames']
    run_config = run_data['run_config']
    
    state = {'current_frame': 0, 'playing': False}
    
    fig = create_visualization(frames[0], run_config)
    
    def update_frame(frame_idx):
        """Update display to show frame_idx."""
        state['current_frame'] = frame_idx
        
        for ax in fig.axes:
            ax.clear()
        
        # Redraw (similar to video update)
        frame_data = frames[frame_idx]
        ax_sonar, ax_map, ax_gt = fig.axes
        
        # [Same rendering code as in create_video]
        sonar_image = frame_data['sonar']
        image_db = 10 * np.log10(np.maximum(sonar_image, 1e-10))
        image_normalized = np.clip((image_db + 60.0) / 60.0, 0, 1)
        ax_sonar.imshow(image_normalized, cmap='viridis', aspect='auto', origin='lower')
        
        if 'metadata' in frame_data:
            pos = frame_data['metadata']['sonar_position']
            ax_sonar.set_title(f'Sonar - Frame {frame_idx}\nPos: [{pos[0]:.1f}, {pos[1]:.1f}]')
        
        # Ground truth
        ground_truth = frame_data['ground_truth']
        material_colors = {0: [0, 0, 0], 1: [0, 100, 255], 3: [255, 140, 0]}
        gt_rgb = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        for mat_id, color in material_colors.items():
            gt_rgb[ground_truth == mat_id] = color
        ax_gt.imshow(gt_rgb, aspect='auto', origin='lower')
        ax_gt.set_title(f'Ground Truth - Frame {frame_idx}')
        
        fig.canvas.draw()
    
    def on_key(event):
        """Handle keyboard input."""
        if event.key == ' ':  # Space - play/pause
            state['playing'] = not state['playing']
        elif event.key == 'right':
            state['current_frame'] = min(state['current_frame'] + 1, len(frames) - 1)
            update_frame(state['current_frame'])
        elif event.key == 'left':
            state['current_frame'] = max(state['current_frame'] - 1, 0)
            update_frame(state['current_frame'])
        elif event.key == 'home':
            state['current_frame'] = 0
            update_frame(state['current_frame'])
        elif event.key == 'end':
            state['current_frame'] = len(frames) - 1
            update_frame(state['current_frame'])
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nInteractive Playback Controls:")
    print("  Space: Play/Pause")
    print("  Left/Right: Previous/Next frame")
    print("  Home/End: First/Last frame")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize saved sonar run data')
    parser.add_argument('run_dir', type=str, help='Path to run directory')
    parser.add_argument('--output', type=str, help='Output video file (.mp4, .avi, .gif)')
    parser.add_argument('--interactive', action='store_true', help='Interactive playback')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for video (default: 10)')
    parser.add_argument('--dpi', type=int, default=100, help='Resolution for video (default: 100)')
    
    args = parser.parse_args()
    
    # Load run data
    run_data = load_run_data(args.run_dir)
    
    if args.interactive:
        interactive_playback(run_data)
    elif args.output:
        create_video(run_data, args.output, fps=args.fps, dpi=args.dpi)
    else:
        print("Error: Specify --output for video or --interactive for playback")
        print("\nExamples:")
        print("  python visualize_run.py data/runs/my_experiment_01 --output video.mp4")
        print("  python visualize_run.py data/runs/my_experiment_01 --interactive")


if __name__ == '__main__':
    main()
