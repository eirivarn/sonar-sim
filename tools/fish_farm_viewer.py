"""Interactive fish farm cage viewer with animated fish.

This is the main viewer application for the sonar-sim fish farm simulation.
Features real-time 3D visualization, polar sonar view, and range plots.
"""
import sys
sys.path.insert(0, '/Users/eirikvarnes/code/sonar-sim')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import time

from src.sim.sonar import Sonar
from src.sim.fish_cage import NetCage, FishSchool
from src.sim.primitives import Plane, Sphere
from src.sim.fish_farm_world import build_fish_farm_world, get_default_sonar_config
from src.sim.visualization import draw_cage_wireframe, draw_plane, draw_sonar_position, draw_sonar_fov
from src.sim.config import FishConfig, VisualizationConfig, SonarConfig, SimulationConfig
from src.sim.math3d import rpy_to_R


class FishFarmViewer:
    """Interactive viewer for fish farm sonar simulation.
    
    Provides three synchronized views:
    - 3D world view with cage and fish
    - Polar sonar view (imaging sonar style)
    - Cartesian range plot
    
    Keyboard controls:
        W/S/A/D/Q/E: Move sonar position
        Arrow keys: Rotate yaw
        I/K: Pitch up/down
        J/L: Roll left/right
        R: Reset position
        SPACE: Pause/unpause animation
        F: Toggle fish visibility
        N: Toggle noise
        M: Toggle multipath
        C: Cycle colormap
    """
    
    def __init__(self, world, net_cage, fish_school):
        """Initialize the viewer.
        
        Args:
            world: World object containing all primitives
            net_cage: NetCage object for reference
            fish_school: FishSchool object to animate
        """
        self.world = world
        self.net_cage = net_cage
        self.fish_school = fish_school
        
        # Initialize sonar with default config
        sonar_config = get_default_sonar_config()
        self.sonar = Sonar(**sonar_config)
        
        self.last_update = time.time()
        self.paused = False
        self.show_fish = True  # Toggle fish visibility
        self.colormap = 'viridis'  # Sonar colormap theme
        self.available_cmaps = ['hot', 'viridis', 'plasma', 'inferno', 'turbo', 'gray', 'bone', 'ocean']
        
        # Display sizing
        self.sonar_display_size = VisualizationConfig.SONAR_DISPLAY_SIZE
        
        # Create separate figures
        self.fig_3d = plt.figure(figsize=VisualizationConfig.WORLD_VIEW_SIZE, num='3D View - Fish Farm')
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        
        self.fig_polar = plt.figure(figsize=(self.sonar_display_size, self.sonar_display_size), num='Polar Sonar View')
        self.ax_polar = self.fig_polar.add_subplot(111, projection='polar')
        
        # Setup polar plot
        self.ax_polar.set_theta_zero_location('N')
        self.ax_polar.set_theta_direction(-1)
        
        # Connect keyboard to all figures
        for fig in [self.fig_3d, self.fig_polar]:
            fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Add controls to 3D figure
        self.fig_3d.text(0.01, 0.98, 
                     'FISH FARM SONAR CONTROLS\n'
                     '════════════════════════\n'
                     'W/S: Forward/Back\n'
                     'A/D: Left/Right\n'
                     'Q/E: Down/Up\n'
                     'Arrows: Yaw\n'
                     'I/K: Pitch\n'
                     'R: Reset\n'
                     'SPACE: Pause\n'
                     'F: Toggle Fish\n'
                     'N: Toggle Noise\n'
                     'M: Toggle Multipath\n'
                     'C: Cycle Colormap\n'
                     '+/-: Display Size\n'
                     '[/]: Range',
                     fontsize=9, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                     family='monospace')
        
        # Start animation on all figures
        self.anim_3d = FuncAnimation(self.fig_3d, self.animate, interval=50, blit=False)
        self.anim_polar = FuncAnimation(self.fig_polar, self.animate_dummy, interval=50, blit=False)

        
    def on_key(self, event):
        """Handle keyboard input."""
        step = 0.5
        angle_step = np.deg2rad(5)
        
        if event.key == 'w':
            R = rpy_to_R(self.sonar.rpy[0], self.sonar.rpy[1], self.sonar.rpy[2])
            self.sonar.pos += step * R[:, 0]
        elif event.key == 's':
            R = rpy_to_R(self.sonar.rpy[0], self.sonar.rpy[1], self.sonar.rpy[2])
            self.sonar.pos -= step * R[:, 0]
        elif event.key == 'a':
            R = rpy_to_R(self.sonar.rpy[0], self.sonar.rpy[1], self.sonar.rpy[2])
            self.sonar.pos -= step * R[:, 1]
        elif event.key == 'd':
            R = rpy_to_R(self.sonar.rpy[0], self.sonar.rpy[1], self.sonar.rpy[2])
            self.sonar.pos += step * R[:, 1]
        elif event.key == 'q':
            self.sonar.pos[2] -= step
        elif event.key == 'e':
            self.sonar.pos[2] += step
        elif event.key == 'left':
            self.sonar.rpy[2] += angle_step
        elif event.key == 'right':
            self.sonar.rpy[2] -= angle_step
        elif event.key == 'i':
            self.sonar.rpy[1] += angle_step
        elif event.key == 'k':
            self.sonar.rpy[1] -= angle_step
        elif event.key == 'r':
            self.sonar.pos = np.array([0.0, 0.0, -7.0])
            self.sonar.rpy = np.array([0.0, 0.0, 0.0])
        elif event.key == ' ':
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Running'}")
        elif event.key == 'n':
            self.sonar.enable_noise = not self.sonar.enable_noise
            print(f"Noise: {'ON' if self.sonar.enable_noise else 'OFF'}")
        elif event.key == 'c':
            # Cycle through colormaps
            current_idx = self.available_cmaps.index(self.colormap)
            self.colormap = self.available_cmaps[(current_idx + 1) % len(self.available_cmaps)]
            print(f"Colormap: {self.colormap}")
        elif event.key == 'm':
            self.sonar.enable_multipath = not self.sonar.enable_multipath
            print(f"Multipath: {'ON' if self.sonar.enable_multipath else 'OFF'}")
        elif event.key == 'f':
            self.show_fish = not self.show_fish
            # Toggle fish in world for sonar detection
            FishConfig.ENABLE_FISH_IN_WORLD = self.show_fish
            # Rebuild world to add/remove fish
            from src.sim.fish_farm_world import build_fish_farm_world
            self.world, self.net_cage, self.fish_school = build_fish_farm_world()
            print(f"Fish: {'ON' if self.show_fish else 'OFF'} (sonar {'detects' if self.show_fish else 'ignores'} fish)")
        elif event.key == '+':
            self.sonar_display_size = min(self.sonar_display_size + 1, 20)
            self.fig_polar.set_size_inches(self.sonar_display_size, self.sonar_display_size)
            print(f"Sonar display size: {self.sonar_display_size}")
        elif event.key == '-':
            self.sonar_display_size = max(self.sonar_display_size - 1, 4)
            self.fig_polar.set_size_inches(self.sonar_display_size, self.sonar_display_size)
            print(f"Sonar display size: {self.sonar_display_size}")
        elif event.key == '[':
            self.sonar.range_m = max(self.sonar.range_m - 5, 5)
            print(f"Sonar range: {self.sonar.range_m}m")
        elif event.key == ']':
            self.sonar.range_m = min(self.sonar.range_m + 5, 100)
            print(f"Sonar range: {self.sonar.range_m}m")
    
    def animate(self, frame):
        """Animation update function."""
        # Update fish positions
        current_time = time.time()
        dt = min(current_time - self.last_update, 0.1)  # Cap dt
        self.last_update = current_time
        
        if not self.paused:
            self.fish_school.update(dt)
        
        self.update_plot()
    
    def animate_dummy(self, frame):
        """Dummy animation for other windows (actual updates happen in animate())."""
        pass
    
    def update_plot(self):
        """Redraw all views."""
        # Get scan data
        scan_data = self.sonar.scan_2d(self.world)
        polar_image = np.array(scan_data['polar_image'])  # (range_bins, h_beams)
        
        # Clear and redraw 3D view
        self.ax3d.clear()
        
        # Draw sonar FOV cone
        draw_sonar_fov(self.ax3d, self.sonar.pos, self.sonar.rpy, 
                      self.sonar.hfov_deg, self.sonar.range_m, 
                      alpha=0.15, color='cyan')
        
        # Draw cage wireframe
        draw_cage_wireframe(self.ax3d, self.net_cage.center, 
                          self.net_cage.radius_top, 
                          self.net_cage.radius_bottom, 
                          self.net_cage.depth,
                          self.net_cage.num_sides)
        
        # Draw other objects
        for obj in self.world.objects:
            if isinstance(obj, Plane):
                draw_plane(self.ax3d, obj.point, obj.normal, size=30, color='tan', alpha=0.3)
            elif isinstance(obj, Sphere) and "buoy" in obj.obj_id:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obj.center[0] + obj.radius * np.outer(np.cos(u), np.sin(v))
                y = obj.center[1] + obj.radius * np.outer(np.sin(u), np.sin(v))
                z = obj.center[2] + obj.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                self.ax3d.plot_surface(x, y, z, color='orange', alpha=0.6)
        
        # Draw fish (if enabled)
        if self.show_fish:
            fish_positions = np.array([f.position for f in self.fish_school.fish])
            self.ax3d.scatter(fish_positions[:, 0], fish_positions[:, 1], fish_positions[:, 2],
                             c='red', s=10, alpha=0.6, marker='o')
        
        # Draw sonar
        pos = self.sonar.pos
        self.ax3d.scatter([pos[0]], [pos[1]], [pos[2]], color='black', s=100, marker='o')
        R = rpy_to_R(self.sonar.rpy[0], self.sonar.rpy[1], self.sonar.rpy[2])
        scale = 1.0
        self.ax3d.quiver(pos[0], pos[1], pos[2], R[0,0]*scale, R[1,0]*scale, R[2,0]*scale, 
                        color='red', arrow_length_ratio=0.3, linewidth=2)
        
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')
        fish_status = f'{len(self.fish_school.fish)} fish' if self.show_fish else 'fish hidden'
        self.ax3d.set_title(f'Fish Farm - {fish_status}\n'
                           f'Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')
        
        # Set view limits based on cage
        self.ax3d.set_xlim([self.net_cage.center[0] - 15, self.net_cage.center[0] + 15])
        self.ax3d.set_ylim([self.net_cage.center[1] - 15, self.net_cage.center[1] + 15])
        self.ax3d.set_zlim([self.net_cage.center[2] - self.net_cage.depth - 2, 
                           self.net_cage.center[2] + 2])
        
        # Update polar sonar view
        self.ax_polar.clear()
        self.ax_polar.set_theta_zero_location('N')
        self.ax_polar.set_theta_direction(-1)
        
        hfov_rad = np.deg2rad(self.sonar.hfov_deg)
        angles = np.linspace(-hfov_rad/2, hfov_rad/2, self.sonar.h_beams)
        
        # Set limits to only show the sonar cone
        self.ax_polar.set_thetamin(-self.sonar.hfov_deg/2)
        self.ax_polar.set_thetamax(self.sonar.hfov_deg/2)
        self.ax_polar.set_ylim(0, self.sonar.range_m)
        
        # Use the polar image directly (already has realistic effects applied)
        theta_mesh, r_mesh = np.meshgrid(angles, np.linspace(0, self.sonar.range_m, polar_image.shape[0]))
        self.ax_polar.contourf(theta_mesh, r_mesh, polar_image, 
                              levels=20, 
                              cmap=self.colormap)
        self.ax_polar.set_title(f'Imaging Sonar (Polar)\nColormap: {self.colormap}')
        
        # Force redraw of all figures
        self.fig_3d.canvas.draw_idle()
        self.fig_polar.canvas.draw_idle()

def main():
    # Disable matplotlib's global keybindings
    plt.rcParams['keymap.quit'] = []
    plt.rcParams['keymap.save'] = []
    plt.rcParams['keymap.fullscreen'] = []
    plt.rcParams['keymap.home'] = []
    plt.rcParams['keymap.back'] = []
    plt.rcParams['keymap.forward'] = []
    plt.rcParams['keymap.pan'] = []
    plt.rcParams['keymap.zoom'] = []
    plt.rcParams['keymap.grid'] = []
    plt.rcParams['keymap.yscale'] = []
    plt.rcParams['keymap.xscale'] = []
    
    # Build the fish farm world
    world, net_cage, fish_school = build_fish_farm_world()
    
    viewer = FishFarmViewer(world, net_cage, fish_school)
    plt.show()

if __name__ == "__main__":
    main()
