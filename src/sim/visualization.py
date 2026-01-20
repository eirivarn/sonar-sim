"""Visualization utilities for 3D rendering."""
import numpy as np
from .config import VisualizationConfig
from .math3d import rpy_to_R


def draw_sonar_fov(ax, sonar_pos, sonar_rpy, hfov_deg, range_m, alpha=0.15, color='blue'):
    """Draw a transparent cone showing the sonar field of view.
    
    Args:
        ax: Matplotlib 3D axes
        sonar_pos: Sonar position [x, y, z]
        sonar_rpy: Sonar orientation [roll, pitch, yaw] in radians
        hfov_deg: Horizontal field of view in degrees
        range_m: Sonar range in meters
        alpha: Transparency (0-1)
        color: Color of the FOV cone
    """
    # Get rotation matrix
    R = rpy_to_R(float(sonar_rpy[0]), float(sonar_rpy[1]), float(sonar_rpy[2]))
    
    hfov_rad = np.deg2rad(hfov_deg)
    
    # Create fan shape in local coordinates
    num_angles = 30
    angles = np.linspace(-hfov_rad/2, hfov_rad/2, num_angles)
    
    # Create vertices for the FOV cone
    vertices = [sonar_pos]  # Apex at sonar
    
    for angle in angles:
        # Direction in local frame (horizontal fan)
        cy, sy = np.cos(angle), np.sin(angle)
        dir_local = np.array([cy, sy, 0.0]) * range_m
        
        # Transform to world frame
        dir_world = R @ dir_local
        vertex = sonar_pos + dir_world
        vertices.append(vertex)
    
    vertices = np.array(vertices)
    
    # Draw the cone as a surface
    # Create triangular faces from apex to each edge segment
    for i in range(1, len(vertices) - 1):
        tri = np.array([vertices[0], vertices[i], vertices[i+1]])
        ax.plot_trisurf(tri[:, 0], tri[:, 1], tri[:, 2], 
                       color=color, alpha=alpha, edgecolor='none')
    
    # Draw outline edges
    for i in [1, -1]:  # First and last edge
        ax.plot([sonar_pos[0], vertices[i][0]], 
               [sonar_pos[1], vertices[i][1]], 
               [sonar_pos[2], vertices[i][2]], 
               color=color, alpha=alpha*3, linewidth=1.5)
    
    # Draw arc at max range
    ax.plot(vertices[1:, 0], vertices[1:, 1], vertices[1:, 2], 
           color=color, alpha=alpha*3, linewidth=1.5)


def draw_cage_wireframe(ax, center, radius_top, radius_bottom, depth, num_sides=4):
    """Draw a wireframe representation of the tapered polygon cage.
    
    Args:
        ax: Matplotlib 3D axes
        center: Cage center position [x, y, z]
        radius_top: Radius at the top (surface)
        radius_bottom: Radius at the bottom
        depth: Cage depth
        num_sides: Number of polygon sides (4 for square, 6 for hexagon, etc.)
    """
    # Generate polygon vertices
    angles = np.linspace(0, 2*np.pi, num_sides + 1)  # Include duplicate to close the polygon
    
    # Draw vertical lines on each panel (not just corners)
    lines_per_panel = VisualizationConfig.LINES_PER_PANEL
    for i in range(num_sides):
        angle1 = angles[i]
        angle2 = angles[i + 1]
        
        # Draw multiple vertical lines across this panel
        for j in range(lines_per_panel):
            t = j / (lines_per_panel - 1)  # Interpolation parameter 0 to 1
            angle = angle1 + t * (angle2 - angle1)
            
            x_top = center[0] + radius_top * np.cos(angle)
            y_top = center[1] + radius_top * np.sin(angle)
            x_bottom = center[0] + radius_bottom * np.cos(angle)
            y_bottom = center[1] + radius_bottom * np.sin(angle)
            
            # Make corner lines thicker
            if j == 0 or j == lines_per_panel - 1:
                ax.plot([x_top, x_bottom], [y_top, y_bottom], [center[2], center[2] - depth], 
                       'b-', alpha=VisualizationConfig.CORNER_LINE_ALPHA, 
                       linewidth=VisualizationConfig.CORNER_LINE_WIDTH)
            else:
                ax.plot([x_top, x_bottom], [y_top, y_bottom], [center[2], center[2] - depth], 
                       'b-', alpha=VisualizationConfig.INTERIOR_LINE_ALPHA, 
                       linewidth=VisualizationConfig.INTERIOR_LINE_WIDTH)
    
    # Horizontal polygon rings at different depths
    num_horizontal = VisualizationConfig.NUM_HORIZONTAL_RINGS
    depths = np.linspace(0, depth, num_horizontal)
    for d in depths:
        depth_ratio = d / depth
        radius_at_depth = radius_top + (radius_bottom - radius_top) * depth_ratio
        
        # Draw polygon at this depth
        x = center[0] + radius_at_depth * np.cos(angles)
        y = center[1] + radius_at_depth * np.sin(angles)
        z = np.full_like(angles, center[2] - d)
        ax.plot(x, y, z, 'b-', alpha=0.3, linewidth=0.5)


def draw_plane(ax, point, normal, size=10, color='green', alpha=0.2):
    """Draw a finite representation of a plane.
    
    Args:
        ax: Matplotlib 3D axes
        point: Point on the plane
        normal: Plane normal vector
        size: Size of the plane representation
        color: Plane color
        alpha: Transparency
    """
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, np.array([0, 0, 1]))
    else:
        v1 = np.cross(normal, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    u = np.linspace(-size, size, 10)
    v = np.linspace(-size, size, 10)
    U, V = np.meshgrid(u, v)
    X = point[0] + U * v1[0] + V * v2[0]
    Y = point[1] + U * v1[1] + V * v2[1]
    Z = point[2] + U * v1[2] + V * v2[2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)


def draw_sonar_position(ax, pos, rpy, scale=1.0):
    """Draw sonar position and orientation indicator.
    
    Args:
        ax: Matplotlib 3D axes
        pos: Sonar position [x, y, z]
        rpy: Sonar orientation [roll, pitch, yaw]
        scale: Scale factor for orientation arrows
    """
    from .math3d import rpy_to_R
    
    # Draw sonar as a small sphere
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color='red', s=100, marker='o')
    
    # Draw orientation axes
    R = rpy_to_R(rpy[0], rpy[1], rpy[2])
    forward = R @ np.array([scale, 0, 0])
    right = R @ np.array([0, scale, 0])
    up = R @ np.array([0, 0, scale])
    
    # Forward (red), Right (green), Up (blue)
    ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2], 
             color='red', arrow_length_ratio=0.3)
    ax.quiver(pos[0], pos[1], pos[2], right[0], right[1], right[2], 
             color='green', arrow_length_ratio=0.3)
    ax.quiver(pos[0], pos[1], pos[2], up[0], up[1], up[2], 
             color='blue', arrow_length_ratio=0.3)
