"""
Robot class - simulates underwater robot with thruster control.
"""

import numpy as np


class Robot:
    """2D underwater robot with thruster-based control."""
    
    def __init__(self, initial_x=15.0, initial_y=15.0, initial_yaw=0.0,
                 max_linear_speed=1.2, max_angular_speed=1.0,
                 thruster_force=4.0, drag_coefficient=3.0, mass=10.0,
                 angular_thruster_torque=3.0, angular_drag_coefficient=2.0,
                 moment_of_inertia=5.0):
        """
        Initialize robot.
        
        Args:
            initial_x, initial_y: Starting position (m)
            initial_yaw: Starting orientation (radians)
            max_linear_speed: Maximum velocity (m/s)
            max_angular_speed: Maximum yaw rate (rad/s)
            thruster_force: Thruster force (N)
            drag_coefficient: Linear drag coefficient
            mass: Robot mass (kg)
            angular_thruster_torque: Torque from rotational thrusters (N·m)
            angular_drag_coefficient: Rotational drag coefficient
            moment_of_inertia: Rotational inertia (kg·m²)
        """
        # State
        self.position = np.array([initial_x, initial_y], dtype=float)
        self.yaw = initial_yaw
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.angular_velocity = 0.0
        
        # Linear motion parameters
        self.max_linear_speed = max_linear_speed
        self.thruster_force = thruster_force
        self.drag_coef = drag_coefficient
        self.mass = mass
        
        # Angular motion parameters
        self.max_angular_speed = max_angular_speed
        self.angular_thruster_torque = angular_thruster_torque
        self.angular_drag_coef = angular_drag_coefficient
        self.moment_of_inertia = moment_of_inertia
        
        # Control inputs (set by controller)
        self.thrust_cmd = np.array([0.0, 0.0], dtype=float)  # [forward/back, left/right]
        self.yaw_rate_cmd = 0.0
    
    def set_thrust(self, forward=0.0, lateral=0.0, yaw_rate=0.0):
        """
        Set thruster commands.
        
        Args:
            forward: Forward/backward thrust (-1.0 to 1.0)
            lateral: Left/right thrust (-1.0 to 1.0)
            yaw_rate: Yaw rate command (-1.0 to 1.0)
        """
        self.thrust_cmd[0] = np.clip(forward, -1.0, 1.0)
        self.thrust_cmd[1] = np.clip(lateral, -1.0, 1.0)
        self.yaw_rate_cmd = np.clip(yaw_rate, -1.0, 1.0)
    
    def update(self, dt):
        """
        Update robot physics.
        
        Args:
            dt: Time step in seconds
        """
        # Convert thrust commands to world frame forces
        thrust_force_body = self.thrust_cmd * self.thruster_force
        
        # Rotate to world frame
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        thrust_force_world = rotation_matrix @ thrust_force_body
        
        # Drag force (opposing velocity)
        drag_force = -self.drag_coef * self.velocity
        
        # Total force and acceleration
        total_force = thrust_force_world + drag_force
        acceleration = total_force / self.mass
        
        # Update velocity and position
        self.velocity += acceleration * dt
        
        # Limit max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_linear_speed:
            self.velocity = self.velocity / speed * self.max_linear_speed
        
        self.position += self.velocity * dt
        
        # Angular motion physics (torque-based, similar to linear motion)
        # Torque from rotational thrusters
        torque = self.yaw_rate_cmd * self.angular_thruster_torque
        
        # Angular drag (opposing angular velocity)
        angular_drag = -self.angular_drag_coef * self.angular_velocity
        
        # Total torque and angular acceleration
        total_torque = torque + angular_drag
        angular_acceleration = total_torque / self.moment_of_inertia
        
        # Update angular velocity
        self.angular_velocity += angular_acceleration * dt
        
        # Limit max angular speed
        if abs(self.angular_velocity) > self.max_angular_speed:
            self.angular_velocity = np.sign(self.angular_velocity) * self.max_angular_speed
        
        self.yaw += self.angular_velocity * dt
        
        # Normalize yaw to [-pi, pi]
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))
    
    def get_direction_vector(self):
        """Get unit direction vector from current yaw."""
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])
    
    def get_state(self):
        """
        Get current robot state.
        
        Returns:
            dict with position, yaw, velocity, angular_velocity
        """
        return {
            'position': self.position.copy(),
            'yaw': self.yaw,
            'velocity': self.velocity.copy(),
            'angular_velocity': self.angular_velocity,
            'direction': self.get_direction_vector()
        }
