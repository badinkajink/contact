"""
Plotting utilities for trajectory optimization visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

class TrajectoryPlotter:
    """Plot trajectory optimization results"""
    
    def __init__(self, dt=0.1):
        self.dt = dt
        
    def plot_trajectory(self, x, u, x_goal, save_path='trajectory.png'):
        """
        Plot state and control trajectories.
        
        Args:
            x: State trajectory (nx, N)
            u: Control trajectory (nu, N-1)
            x_goal: Goal state
            save_path: Where to save the plot
        """
        N = x.shape[1]
        t = np.arange(N) * self.dt
        t_u = t[:-1]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        
        # Position (angle)
        axes[0].plot(t, x[0, :], 'b-', linewidth=2.5, label='Trajectory')
        axes[0].axhline(x_goal[0], color='r', linestyle='--', linewidth=2, label='Goal')
        axes[0].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        axes[0].set_ylabel('Angle θ (rad)', fontsize=13, fontweight='bold')
        axes[0].set_title('Pendulum Swing-Up Trajectory', fontsize=15, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, t[-1]])
        
        # Velocity
        axes[1].plot(t, x[1, :], 'g-', linewidth=2.5, label='Trajectory')
        axes[1].axhline(x_goal[1], color='r', linestyle='--', linewidth=2, label='Goal')
        axes[1].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        axes[1].set_ylabel('Angular velocity ω (rad/s)', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, t[-1]])
        
        # Control
        axes[2].plot(t_u, u[0, :], 'r-', linewidth=2.5, label='Control')
        axes[2].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        axes[2].set_ylabel('Torque u (Nm)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([0, t[-1]])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTrajectory plot saved to: {save_path}")
        plt.close()
    
    def plot_phase_portrait(self, x, x_goal, save_path='phase_portrait.png'):
        """
        Plot phase portrait (angle vs velocity).
        
        Args:
            x: State trajectory (nx, N)
            x_goal: Goal state
            save_path: Where to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectory
        ax.plot(x[0, :], x[1, :], 'b-', linewidth=2.5, alpha=0.8, label='Trajectory')
        
        # Mark start and end
        ax.plot(x[0, 0], x[1, 0], 'go', markersize=15, label='Start', zorder=5)
        ax.plot(x[0, -1], x[1, -1], 'bs', markersize=15, label='End', zorder=5)
        ax.plot(x_goal[0], x_goal[1], 'r*', markersize=20, label='Goal', zorder=5)
        
        # Add direction arrows
        N = x.shape[1]
        arrow_spacing = max(1, N // 10)
        for i in range(0, N-1, arrow_spacing):
            dx = x[0, i+1] - x[0, i]
            dy = x[1, i+1] - x[1, i]
            ax.arrow(x[0, i], x[1, i], dx*0.5, dy*0.5, 
                    head_width=0.2, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        
        ax.set_xlabel('Angle θ (rad)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Angular velocity ω (rad/s)', fontsize=13, fontweight='bold')
        ax.set_title('Phase Portrait', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Phase portrait saved to: {save_path}")
        plt.close()


class iLQRDiagnosticsPlotter:
    """Plot iLQR algorithm diagnostics"""
    
    def __init__(self):
        self.costs = []
        self.regularizations = []
        self.alphas = []
        self.iterations = []
        
    def record_iteration(self, iteration, cost, regularization, alpha):
        """Record data from one iteration"""
        self.iterations.append(iteration)
        self.costs.append(cost)
        self.regularizations.append(regularization)
        self.alphas.append(alpha)
    
    def plot_convergence(self, save_path='convergence.png'):
        """
        Plot convergence diagnostics.
        
        Shows:
        - Cost vs iteration
        - Regularization vs iteration  
        - Line search step size vs iteration
        """
        if not self.costs:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Cost
        axes[0].semilogy(self.iterations, self.costs, 'b-o', linewidth=2, markersize=6)
        axes[0].set_ylabel('Cost', fontsize=13, fontweight='bold')
        axes[0].set_title('iLQR Convergence Diagnostics', fontsize=15, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([min(self.iterations)-0.5, max(self.iterations)+0.5])
        
        # Regularization
        axes[1].semilogy(self.iterations, self.regularizations, 'r-s', linewidth=2, markersize=6)
        axes[1].set_ylabel('Regularization λ', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([min(self.iterations)-0.5, max(self.iterations)+0.5])
        
        # Line search alpha
        axes[2].plot(self.iterations, self.alphas, 'g-^', linewidth=2, markersize=6)
        axes[2].set_ylabel('Step size α', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Iteration', fontsize=13, fontweight='bold')
        axes[2].set_ylim([0, 1.1])
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([min(self.iterations)-0.5, max(self.iterations)+0.5])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence diagnostics saved to: {save_path}")
        plt.close()
    
    def plot_gains_heatmap(self, K_seq, save_path='gains_heatmap.png'):
        """
        Plot feedback gain matrices over time.
        
        Args:
            K_seq: Feedback gains (nu, nx, N-1)
            save_path: Where to save the plot
        """
        nu, nx, N = K_seq.shape
        
        fig, axes = plt.subplots(nu, nx, figsize=(12, 4*nu))
        if nu == 1 and nx == 1:
            axes = np.array([[axes]])
        elif nu == 1:
            axes = axes.reshape(1, -1)
        elif nx == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(nu):
            for j in range(nx):
                ax = axes[i, j]
                gains = K_seq[i, j, :]
                
                im = ax.plot(gains, linewidth=2)
                ax.set_title(f'K[{i},{j}]: u{i} ← x{j}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Time step', fontsize=10)
                ax.set_ylabel('Gain value', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feedback Gain Evolution', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feedback gains plot saved to: {save_path}")
        plt.close()
    
    def plot_control_corrections(self, du_seq, save_path='control_corrections.png'):
        """
        Plot feedforward control corrections over time.
        
        Args:
            du_seq: Control corrections (nu, N-1)
            save_path: Where to save the plot
        """
        nu, N = du_seq.shape
        
        fig, axes = plt.subplots(nu, 1, figsize=(12, 4*nu))
        if nu == 1:
            axes = [axes]
        
        for i in range(nu):
            axes[i].plot(du_seq[i, :], 'b-', linewidth=2)
            axes[i].set_ylabel(f'du[{i}]', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Time step', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='gray', linestyle=':', linewidth=1)
        
        plt.suptitle('Feedforward Control Corrections', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Control corrections plot saved to: {save_path}")
        plt.close()


def create_summary_figure(x, u, x_goal, diagnostics, save_path='summary.png'):
    """
    Create a comprehensive summary figure.
    
    Args:
        x: State trajectory (nx, N)
        u: Control trajectory (nu, N-1)
        x_goal: Goal state
        diagnostics: iLQRDiagnosticsPlotter object
        save_path: Where to save
    """
    N = x.shape[1]
    dt = 0.1
    t = np.arange(N) * dt
    t_u = t[:-1]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Trajectory plots (left side)
    ax_theta = fig.add_subplot(gs[0, 0:2])
    ax_theta.plot(t, x[0, :], 'b-', linewidth=2.5)
    ax_theta.axhline(x_goal[0], color='r', linestyle='--', linewidth=2)
    ax_theta.set_ylabel('Angle θ (rad)', fontsize=11, fontweight='bold')
    ax_theta.set_title('State and Control Trajectories', fontsize=13, fontweight='bold')
    ax_theta.grid(True, alpha=0.3)
    
    ax_omega = fig.add_subplot(gs[1, 0:2])
    ax_omega.plot(t, x[1, :], 'g-', linewidth=2.5)
    ax_omega.axhline(x_goal[1], color='r', linestyle='--', linewidth=2)
    ax_omega.set_ylabel('Velocity ω (rad/s)', fontsize=11, fontweight='bold')
    ax_omega.grid(True, alpha=0.3)
    
    ax_u = fig.add_subplot(gs[2, 0:2])
    ax_u.plot(t_u, u[0, :], 'r-', linewidth=2.5)
    ax_u.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax_u.set_ylabel('Torque u (Nm)', fontsize=11, fontweight='bold')
    ax_u.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax_u.grid(True, alpha=0.3)
    
    # Phase portrait (top right)
    ax_phase = fig.add_subplot(gs[0, 2])
    ax_phase.plot(x[0, :], x[1, :], 'b-', linewidth=2, alpha=0.8)
    ax_phase.plot(x[0, 0], x[1, 0], 'go', markersize=10, label='Start')
    ax_phase.plot(x[0, -1], x[1, -1], 'bs', markersize=10, label='End')
    ax_phase.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
    ax_phase.set_xlabel('θ (rad)', fontsize=10)
    ax_phase.set_ylabel('ω (rad/s)', fontsize=10)
    ax_phase.set_title('Phase Portrait', fontsize=11, fontweight='bold')
    ax_phase.legend(fontsize=8)
    ax_phase.grid(True, alpha=0.3)
    
    # Cost convergence (middle right)
    if diagnostics.costs:
        ax_cost = fig.add_subplot(gs[1, 2])
        ax_cost.semilogy(diagnostics.iterations, diagnostics.costs, 'b-o', linewidth=2, markersize=4)
        ax_cost.set_xlabel('Iteration', fontsize=10)
        ax_cost.set_ylabel('Cost', fontsize=10)
        ax_cost.set_title('Cost Convergence', fontsize=11, fontweight='bold')
        ax_cost.grid(True, alpha=0.3)
    
    # Regularization (bottom right)
    if diagnostics.regularizations:
        ax_reg = fig.add_subplot(gs[2, 2])
        ax_reg.semilogy(diagnostics.iterations, diagnostics.regularizations, 'r-s', linewidth=2, markersize=4)
        ax_reg.set_xlabel('Iteration', fontsize=10)
        ax_reg.set_ylabel('Regularization λ', fontsize=10)
        ax_reg.set_title('Adaptive Regularization', fontsize=11, fontweight='bold')
        ax_reg.grid(True, alpha=0.3)
    
    plt.suptitle('iLQR Pendulum Swing-Up - Complete Summary', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary figure saved to: {save_path}")
    plt.close()