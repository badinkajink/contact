from math import sin, cos, pi
import numpy as np

class Cartpole():
    def __init__(self, timestep=0.05, mp=0.1, mc=1.0, length=1.0, 
                 gravity=9.81, damping=0.01, control_min=-10, control_max=10):
        self.timestep = timestep
        self.mp = mp  # pole mass
        self.mc = mc  # cart mass
        self.l = length  # pole length
        self.gravity = gravity
        self.damping = damping
        self.control_min = control_min
        self.control_max = control_max

    def get_state_size(self):
        return 4  # [x, x_dot, theta, theta_dot]
    
    def get_control_size(self):
        return 1  # [force]

    def clamp(self, n, smallest, largest): 
        return max(smallest, min(n, largest))

    def wrap_angles(self, angle):
        upper_bound = pi
        lower_bound = -pi
        while angle < lower_bound:
            angle += 2*pi
        while angle > upper_bound:
            angle -= 2*pi
        return angle

    def state_delta(self, state1, state2):
        """Compute state difference with angle wrapping"""
        return np.array([
            state1[0] - state2[0], 
            state1[1] - state2[1], 
            self.wrap_angles(state1[2] - state2[2]), 
            state1[3] - state2[3]
        ])

    def next_state(self, x, u):
        """Compute next state using Euler integration"""
        x_pos = x[0]
        x_dot = x[1]
        theta = x[2]
        theta_dot = x[3]
        
        # Extract and clamp control
        u_val = self.clamp(u if np.isscalar(u) else u[0], 
                          self.control_min, self.control_max)

        # Cartpole dynamics
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        total_mass = self.mc + self.mp
        
        temp = (u_val + self.mp * self.l * theta_dot**2 * sin_theta) / total_mass
        numerator = self.gravity * sin_theta - cos_theta * temp
        denominator = self.l * (4.0/3.0 - self.mp * cos_theta**2 / total_mass)
        theta_dot_dot = numerator / denominator
        
        x_dot_dot = temp - self.mp * self.l * theta_dot_dot * cos_theta / total_mass
        
        # Apply damping
        theta_dot_dot -= self.damping * theta_dot
        x_dot_dot -= self.damping * x_dot

        # Euler integration
        new_x_pos = x_pos + self.timestep * x_dot
        new_x_dot = x_dot + self.timestep * x_dot_dot
        new_theta = self.wrap_angles(theta + self.timestep * theta_dot)
        new_theta_dot = theta_dot + self.timestep * theta_dot_dot

        return np.array([new_x_pos, new_x_dot, new_theta, new_theta_dot])

    def next_state_gradient(self, x, u):
        """Compute linearized dynamics: A = df/dx, B = df/du"""
        theta = x[2]
        theta_dot = x[3]
        
        u_val = u if np.isscalar(u) else u[0]
        
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        total_mass = self.mc + self.mp
        pole_mass_length = self.mp * self.l
        
        # Simplified linearization (analytic Jacobians for cartpole are complex)
        # This is an approximation - for better results, use finite differences
        
        # A matrix (approximate)
        A = np.array([
            [1, self.timestep, 0, 0],
            [0, 1 - self.timestep*self.damping, 
             self.timestep * self.mp * self.gravity * cos_theta / total_mass,
             self.timestep * 2 * pole_mass_length * theta_dot * sin_theta / total_mass],
            [0, 0, 1, self.timestep],
            [0, 0,
             self.timestep * self.gravity * total_mass * cos_theta / (self.l * total_mass),
             1 - self.timestep*self.damping]
        ])
        
        # B matrix
        B = np.array([
            [0],
            [self.timestep / total_mass],
            [0],
            [-self.timestep * cos_theta / (self.l * total_mass)]
        ])
        
        return A, B

    def set_Q(self, Q):
        self.Q = np.array(Q)
    
    def set_R(self, R):
        self.R = np.array(R)
    
    def set_goal(self, goal):
        self.goal = np.array(goal)

    def cost_value(self, x, u=None):
        """Compute quadratic cost"""
        delta = self.state_delta(x, self.goal)
        cost = 0.5 * delta @ self.Q @ delta
        
        if u is not None:
            u_val = u if np.isscalar(u) else u[0]
            cost += 0.5 * u_val * self.R[0] * u_val
        
        return cost

    def gradient(self, x, u=None):
        """Compute cost gradient"""
        delta = self.state_delta(x, self.goal)
        grad_x = self.Q @ delta
        
        if u is not None:
            u_val = u if np.isscalar(u) else u[0]
            grad_u = self.R[0] * u_val
            return np.concatenate([grad_x, [grad_u]])
        else:
            return grad_x

    def hessian(self, x, u=None):
        """Compute cost Hessian"""
        if u is not None:
            # Joint Hessian [Q, 0; 0, R]
            H = np.zeros((5, 5))
            H[0:4, 0:4] = self.Q
            H[4, 4] = self.R[0]
            return H
        else:
            return self.Q

    def gradient_hessian(self, x, u=None):
        """Return both gradient and Hessian"""
        return self.hessian(x, u), self.gradient(x, u)