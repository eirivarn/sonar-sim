"""Record sonar scan as animated GIF for debugging.

Moves sonar slowly in one direction and captures frames.
"""
import sys
sys.path.insert(0, '/Users/eirikvarnes/code/sonar-sim')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import io

from src.sim.fish_farm_world import build_fish_farm_world, get_default_sonar_config
from src.sim.sonar import Sonar
from src.sim.config import SonarConfig


def capture_sonar_frame(sonar, world, colormap='viridis'):
    """Capture a single sonar frame as numpy array.
    
    Args:
        sonar: Sonar instance
        world: World instance
        colormap: matplotlib colormap name
        
    Returns:
        RGB image as numpy array
    """
    # Get scan data
    scan_data = sonar.scan_2d(world)
    polar_image = np.array(scan_data['polar_image'])
    
    # Create figure
    fig = Figure(figsize=(8, 8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='polar')
    
    # Configure polar plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)  # Counter-clockwise (standard math convention)
    
    hfov_rad = np.deg2rad(sonar.hfov_deg)
    angles = np.linspace(-hfov_rad/2, hfov_rad/2, sonar.h_beams)
    
    # Set limits
    ax.set_thetamin(-sonar.hfov_deg/2)
    ax.set_thetamax(sonar.hfov_deg/2)
    ax.set_ylim(0, sonar.range_m)
    
    # Plot
    theta_mesh, r_mesh = np.meshgrid(angles, np.linspace(0, sonar.range_m, polar_image.shape[0]))
    ax.contourf(theta_mesh, r_mesh, polar_image, levels=20, cmap=colormap)
    ax.set_title(f'Sonar @ ({sonar.pos[0]:.1f}, {sonar.pos[1]:.1f}, {sonar.pos[2]:.1f})')
    
    # Render to buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    
    plt.close(fig)
    return img


def main():
    """Record sonar movement as GIF."""
    print("Building fish farm world...")
    world, net_cage, fish_school = build_fish_farm_world()
    
    print("Initializing sonar with reduced resolution for faster recording...")
    sonar_config = get_default_sonar_config()
    # Reduce resolution for faster recording
    sonar_config['h_beams'] = 90  # Half the beams (was ~181)
    sonar_config['range_bins'] = 512  # Half the range bins (was 1024)
    sonar_config['rays_per_beam'] = 3  # Fewer rays per beam (was 5)
    sonar = Sonar(**sonar_config)
    
    # Animation parameters
    duration = 3.0  # seconds
    fps = 20
    num_frames = int(duration * fps)
    
    # Movement: move forward slowly
    start_pos = sonar.pos.copy()
    movement_distance = 10.0  # meters total
    movement_per_frame = movement_distance / num_frames
    
    # Direction: move forward in sonar's local X axis
    from src.sim.math3d import rpy_to_R
    R = rpy_to_R(sonar.rpy[0], sonar.rpy[1], sonar.rpy[2])
    move_direction = R[:, 0]  # Forward direction
    
    print(f"Recording {num_frames} frames at {fps} fps...")
    frames = []
    
    import time
    start_time = time.time()
    
    for i in range(num_frames):
        # Update position
        sonar.pos = start_pos + move_direction * (movement_per_frame * i)
        
        # Capture frame
        frame_start = time.time()
        img = capture_sonar_frame(sonar, world, colormap='viridis')
        frames.append(Image.fromarray(img))
        frame_time = time.time() - frame_start
        
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (num_frames - i - 1)
            print(f"  Frame {i+1}/{num_frames} ({frame_time:.2f}s/frame, ETA: {eta:.1f}s)")
    
    # Save as GIF
    output_path = '/Users/eirikvarnes/code/sonar-sim/sonar_recording.gif'
    print(f"Saving GIF to {output_path}...")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,  # milliseconds per frame
        loop=0
    )
    
    print(f"Done! Saved {len(frames)} frames to {output_path}")
    print(f"GIF duration: {duration} seconds @ {fps} fps")


if __name__ == "__main__":
    main()
