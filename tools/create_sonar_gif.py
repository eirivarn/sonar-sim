"""Create animated GIF of sonar scanning the fish cage from different positions and orientations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
from simple_voxel_sonar import create_demo_scene, update_fish, update_debris, VoxelSonar

def create_sonar_animation_gif(output_file='sonar_scan.gif', duration_seconds=5, fps=40):
    """Create GIF showing sonar scanning fish cage from different angles.
    
    Args:
        output_file: Output GIF filename
        duration_seconds: Total duration of animation in seconds
        fps: Frames per second
    """
    print("Creating fish farm scene...")
    grid, fish_data, debris_data = create_demo_scene()
    
    # Cage parameters
    cage_center = np.array([25.0, 25.0])
    cage_radius = 20.0
    
    # Initialize sonar
    sonar = VoxelSonar(
        position=np.array([25.0, 25.0]),
        direction=np.array([1.0, 0.0]),
        range_m=10.0,
        fov_deg=120.0,
        num_beams=180
    )
    
    # Create figure for sonar view only
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Calculate total frames
    total_frames = int(duration_seconds * fps)
    
    # Define camera path - slow realistic movement to inspect the net
    def get_sonar_pose(frame_idx):
        """Calculate sonar position and direction for given frame.
        
        Simulates realistic ROV movement: slow translation (~0.5 m/s) and rotation.
        Keeps sonar aimed at the cage net for inspection.
        """
        t = frame_idx / total_frames  # 0 to 1
        
        # Realistic ROV speed: ~0.3-0.5 m/s
        # Over 10 seconds, travel ~4-5 meters total
        rov_speed = 0.4  # m/s
        total_distance = rov_speed * duration_seconds  # ~4 meters
        
        # Start position: 5m from cage wall, looking outward at the net
        # Move along an arc near the cage perimeter to inspect different net sections
        start_radius = cage_radius - 5.0  # 5m inside the cage, looking out at net
        
        # Arc path: move along a quarter circle (90 degrees) near the cage wall
        # This gives ~31m arc length at r=20m, but we only travel ~4m (about 11 degrees)
        arc_angle_range = total_distance / cage_radius  # radians traveled
        
        # Start at angle 0, move to arc_angle_range
        current_arc_angle = t * arc_angle_range
        
        # Position stays at fixed radius from center, but moves along arc
        angle_from_center = current_arc_angle
        pos_x = cage_center[0] + start_radius * np.cos(angle_from_center)
        pos_y = cage_center[1] + start_radius * np.sin(angle_from_center)
        pos = np.array([pos_x, pos_y])
        
        # Look direction: point toward the cage wall (outward from center)
        # Add slow scanning rotation ±15 degrees
        radial_outward = pos - cage_center
        radial_outward = radial_outward / (np.linalg.norm(radial_outward) + 1e-9)
        
        # Slow scanning rotation: oscillate ±15 degrees over the duration
        scan_angle = np.sin(t * 2 * np.pi * 2) * 0.26  # ±15 degrees, 2 full scans
        cos_a = np.cos(scan_angle)
        sin_a = np.sin(scan_angle)
        direction = np.array([
            radial_outward[0] * cos_a - radial_outward[1] * sin_a,
            radial_outward[0] * sin_a + radial_outward[1] * cos_a
        ])
        
        return pos, direction
    
    print(f"Generating {total_frames} frames at {fps} fps...")
    
    def update_frame(frame_idx):
        """Update function for animation."""
        # Update sonar pose
        sonar.position, sonar.direction = get_sonar_pose(frame_idx)
        
        # Update dynamic objects
        update_fish(grid, fish_data, cage_center, cage_radius, sonar.position)
        update_debris(grid, debris_data, cage_center, cage_radius)
        
        # Scan
        image = sonar.scan(grid)
        
        # Clear and redraw
        ax.clear()
        
        fov_rad = np.deg2rad(sonar.fov_deg)
        angles = np.linspace(-fov_rad/2, fov_rad/2, sonar.num_beams)
        ranges = np.linspace(0, sonar.range_m, image.shape[0])
        
        theta, r = np.meshgrid(angles, ranges)
        
        # Log compression for display
        image_db = 10 * np.log10(np.maximum(image, 1e-10))
        image_db = np.clip((image_db + 60) / 60, 0, 1)
        
        ax.contourf(theta, r, image_db, levels=20, cmap='gray')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_thetamin(-sonar.fov_deg/2)
        ax.set_thetamax(sonar.fov_deg/2)
        ax.set_ylim(0, sonar.range_m)
        ax.grid(False)
        ax.set_title(f'Sonar Scan - Frame {frame_idx+1}/{total_frames}', fontsize=14, pad=20)
        
        # Progress indicator
        if (frame_idx + 1) % 10 == 0:
            progress = (frame_idx + 1) / total_frames * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx+1}/{total_frames} frames)")
        
        return ax,
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update_frame, frames=total_frames, 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    print(f"Saving to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=100)
    
    plt.close()
    print(f"✓ GIF saved successfully to {output_file}")
    print(f"  Duration: {duration_seconds}s at {fps} fps ({total_frames} frames)")
    print(f"  Resolution: ~{1000}x{800} pixels")


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Create sonar scanning animation GIF')
    parser.add_argument('--output', '-o', default='sonar_scan.gif', 
                       help='Output GIF filename (default: sonar_scan.gif)')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                       help='Animation duration in seconds (default: 10)')
    parser.add_argument('--fps', '-f', type=int, default=40,
                       help='Frames per second (default: 40)')
    
    args = parser.parse_args()
    
    create_sonar_animation_gif(
        output_file=args.output,
        duration_seconds=args.duration,
        fps=args.fps
    )
