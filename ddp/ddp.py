import sys, math, pygame, copy
from pygame.locals import *
from time import sleep
import numpy as np
from util import Util
import matplotlib.pyplot as plt
PLOTTING_AVAILABLE = True

np.set_printoptions(precision=3, suppress=True)

class DDP:
    def __init__(self, robot_object, start_node, goal_node, N, XMAX_MIN, YMAX_MIN, 
                 MAX_ITER=100, EXIT_TOL=1e-4):
        self.robot_object = robot_object
        self.MAX_ITER = MAX_ITER
        self.EXIT_TOL = EXIT_TOL
        self.N = N
        self.start_node = np.array(start_node)
        self.goal_node = np.array(goal_node)
        self.util = Util(self.robot_object)
        self.XMAX_MIN = XMAX_MIN
        self.YMAX_MIN = YMAX_MIN
        self.canvas_max = 640

    def draw_circle(self, node, size, color):
        percent_x = (node[0] + self.XMAX_MIN) / (2 * self.XMAX_MIN)
        percent_y = (node[1] + self.YMAX_MIN) / (2 * self.YMAX_MIN)
        scaled_x = int(self.canvas_max * percent_x)
        scaled_y = int(self.canvas_max * percent_y)
        pygame.draw.circle(self.screen, color, (scaled_x, scaled_y), size)

    def draw_line(self, node1, node2, color):
        percent_x1 = (node1[0] + self.XMAX_MIN) / (2 * self.XMAX_MIN)
        percent_y1 = (node1[1] + self.YMAX_MIN) / (2 * self.YMAX_MIN)
        percent_x2 = (node2[0] + self.XMAX_MIN) / (2 * self.XMAX_MIN)
        percent_y2 = (node2[1] + self.YMAX_MIN) / (2 * self.YMAX_MIN)
        
        scaled_x1 = int(self.canvas_max * percent_x1)
        scaled_y1 = int(self.canvas_max * percent_y1)
        scaled_x2 = int(self.canvas_max * percent_x2)
        scaled_y2 = int(self.canvas_max * percent_y2)

        # Don't draw lines across screen due to angle wrapping
        if abs(node1[0] - node2[0]) < math.pi:
            pygame.draw.line(self.screen, color, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2))

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.canvas_max, self.canvas_max))
        pygame.display.set_caption('PS3 - DDP/iLQR')
        black = 20, 20, 40
        blue = 0, 0, 255
        green = 0, 255, 0
        self.screen.fill(black)
        self.draw_circle(self.start_node, 5, blue)
        self.draw_circle(self.goal_node, 5, green)
        pygame.display.update()

    def draw_trajectory(self, x):
        white = 255, 240, 200
        red = 255, 0, 0
        for k in range(self.N):
            self.draw_circle(x[:, k], 2, white)
            if k > 0:
                self.draw_line(x[:, k-1], x[:, k], red)
        pygame.display.update()
        sleep(0.1)

    def wait_to_exit(self, x, u, K, DISPLAY_MODE=True):
        if not DISPLAY_MODE:
            return x, u, K
        else:
            while True:
                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")

    def iLQR(self, x, u, N, DISPLAY_MODE=False, USE_FULL_DDP=False):
        """
        Iterative Linear Quadratic Regulator with adaptive regularization
        
        Args:
            x: Initial state trajectory
            u: Initial control trajectory
            N: Number of timesteps
            DISPLAY_MODE: Whether to display pygame visualization
            USE_FULL_DDP: If True, uses full DDP with second-order Hessian terms
                         If False, uses iLQR with Gauss-Newton approximation
        """
        # Initialize diagnostics tracking for this run
        self.iteration_data = {
            'iterations': [],
            'costs': [],
            'regularizations': [],
            'alphas': []
        }
        
        if DISPLAY_MODE:
            self.init_screen()
            if USE_FULL_DDP:
                print("\n*** RUNNING FULL DDP (with Hessian terms) ***\n")
            else:
                print("\n*** RUNNING iLQR (Gauss-Newton approximation) ***\n")
        
        nx = self.robot_object.get_state_size()
        nu = self.robot_object.get_control_size()
        
        # Check if we need to initialize the trajectory
        needs_init = np.allclose(x, 0)
        
        # Always ensure trajectory starts from correct initial state
        x[:, 0] = self.start_node
        
        # If the trajectory wasn't provided, forward simulate
        if needs_init:
            for k in range(N-1):
                x[:, k+1] = self.robot_object.next_state(x[:, k], u[:, k])
        
        du = np.zeros((nu, N-1))
        K = np.zeros((nu, nx, N-1))
        
        # Line search parameters
        alpha_factor = 0.5
        alpha_min = 1e-4
        
        # Adaptive regularization parameters
        reg = 1e-6
        reg_min = 1e-6
        reg_max = 1e10
        reg_factor = 10.0
        
        # Compute initial cost
        J = self.util.compute_total_cost(x, u, nx, nu, N)
        
        if DISPLAY_MODE:
            print(f"Initial Cost: {J:.4f}")
            print(f"Initial final state: {x[:, N-1]}")
            print(f"Goal state: {self.goal_node}")
        
        iteration = 0
        last_improvement = 0
        
        while iteration < self.MAX_ITER:
            # ========== BACKWARD PASS ==========
            Vxx, Vx = self.util.initialize_CTG(x[:, N-1], nx, N)
            
            backward_success = True
            for k in range(N-2, -1, -1):
                A, B, H, g = self.util.compute_approximation(x[:, k], u[:, k], nx, nu, k)
                
                # Backpropagate cost-to-go
                # The use_hessians flag determines DDP vs iLQR:
                #   - use_hessians=True  -> Full DDP (includes cost Hessians)
                #   - use_hessians=False -> iLQR Gauss-Newton (ignores cost Hessians)
                Qxx, Qux, Quu, Qx, Qu = self.util.backpropogate_CTG(
                    A, B, H, g, Vxx, Vx, nx, nu, k, use_hessians=USE_FULL_DDP
                )
                
                # Add regularization to Quu
                Quu_reg = Quu + reg * np.eye(nu)
                
                # Try to invert
                try:
                    Quu_inv = np.linalg.inv(Quu_reg)
                except np.linalg.LinAlgError:
                    backward_success = False
                    break
                
                # Compute control law
                du[:, k], K[:, :, k] = self.util.compute_du_K(
                    Qxx, Qux, Quu_reg, Qx, Qu, Quu_inv, nx, nu, k
                )
                
                # Update cost-to-go
                Vxx, Vx = self.util.compute_new_CTG(
                    Qxx, Qux, Quu_reg, Qx, Qu, Quu_inv, du[:, k], K[:, :, k], nx, nu, k
                )
            
            if not backward_success:
                # Increase regularization and retry
                reg = min(reg * reg_factor, reg_max)
                if DISPLAY_MODE:
                    print(f"Backward pass failed, increasing regularization to {reg:.2e}")
                if reg >= reg_max:
                    if DISPLAY_MODE:
                        print("Max regularization reached, exiting")
                    break
                continue
            
            # ========== FORWARD PASS (LINE SEARCH) ==========
            alpha = 1.0
            line_search_success = False
            
            while alpha > alpha_min:
                x_new = np.copy(x)
                u_new = np.copy(u)
                
                # Rollout trajectory with new control
                for k in range(N-1):
                    delta_u = self.util.compute_control_update(
                        x[:, k], x_new[:, k], K[:, :, k], du[:, k], alpha, nx, nu, k
                    )
                    u_new[:, k] = u[:, k] + delta_u
                    x_new[:, k+1] = self.robot_object.next_state(x_new[:, k], u_new[:, k])
                
                # Compute new cost
                J_new = self.util.compute_total_cost(x_new, u_new, nx, nu, N)
                delta_J = J - J_new
                
                # Accept if cost decreased
                if delta_J > 0:
                    x = x_new
                    u = u_new
                    J = J_new
                    line_search_success = True
                    last_improvement = iteration
                    
                    # Decrease regularization on success
                    reg = max(reg / reg_factor, reg_min)
                    
                    # Record iteration data
                    self.iteration_data['iterations'].append(iteration)
                    self.iteration_data['costs'].append(J)
                    self.iteration_data['regularizations'].append(reg)
                    self.iteration_data['alphas'].append(alpha)
                    
                    if DISPLAY_MODE:
                        print(f"Iter {iteration}: Cost={J:.4f}, ΔJ={delta_J:.4f}, α={alpha:.3f}, reg={reg:.2e}, final={x[:, N-1]}")
                        self.draw_trajectory(x)
                    break
                else:
                    alpha *= alpha_factor
            
            if not line_search_success:
                # Line search failed - increase regularization
                reg = min(reg * reg_factor, reg_max)
                if DISPLAY_MODE:
                    print(f"Line search failed, increasing regularization to {reg:.2e}")
                
                # If we haven't improved in a while and regularization is maxed, give up
                if reg >= reg_max or (iteration - last_improvement > 10):
                    if DISPLAY_MODE:
                        print("Stuck in local minimum or max regularization reached")
                    break
                continue
            
            # Check convergence
            if delta_J < self.EXIT_TOL:
                if DISPLAY_MODE:
                    print(f"Converged! ΔJ={delta_J:.6f} < tol={self.EXIT_TOL}")
                break
            
            iteration += 1
        
        if DISPLAY_MODE:
            print(f"\nFinal trajectory after {iteration} iterations:")
            print(f"Final state: {x[:, N-1]}")
            print(f"Goal state:  {self.goal_node}")
            print(f"Final cost: {J:.4f}")
            dist = np.linalg.norm(self.robot_object.state_delta(x[:, N-1], self.goal_node))
            print(f"Distance to goal: {dist:.6f}")
            
            # Generate plots
        if PLOTTING_AVAILABLE:
            self.generate_plots(x, u, K, du, use_full_ddp=USE_FULL_DDP)
        
        self.wait_to_exit(x, u, K, DISPLAY_MODE)
        return x, u, K
    
    def generate_plots(self, x, u, K, du, use_full_ddp=False):
        """Generate all visualization plots"""
        # Create output directory based on solver type
        import os
        output_dir = "ddp" if use_full_ddp else "ilqr"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"GENERATING VISUALIZATIONS in {output_dir}/")
        print("="*60)
        
        dt = self.robot_object.timestep
        N = x.shape[1]
        t = np.arange(N) * dt
        t_u = t[:-1]
        
        # 1. Main trajectory plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        
        axes[0].plot(t, x[0, :], 'b-', linewidth=2.5, label='Trajectory')
        axes[0].axhline(self.goal_node[0], color='r', linestyle='--', linewidth=2, label='Goal')
        axes[0].set_ylabel('Angle θ (rad)', fontsize=13, fontweight='bold')
        title = 'Full DDP' if use_full_ddp else 'iLQR (Gauss-Newton)'
        axes[0].set_title(f'Pendulum Swing-Up Trajectory - {title}', fontsize=15, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, x[1, :], 'g-', linewidth=2.5)
        axes[1].axhline(self.goal_node[1], color='r', linestyle='--', linewidth=2)
        axes[1].set_ylabel('Angular velocity ω (rad/s)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(t_u, u[0, :], 'r-', linewidth=2.5)
        axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
        axes[2].set_ylabel('Torque u (Nm)', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trajectory.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/trajectory.png")
        plt.close()
        
        # 2. Phase portrait
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x[0, :], x[1, :], 'b-', linewidth=2.5, alpha=0.8)
        ax.plot(x[0, 0], x[1, 0], 'go', markersize=15, label='Start', zorder=5)
        ax.plot(x[0, -1], x[1, -1], 'bs', markersize=15, label='End', zorder=5)
        ax.plot(self.goal_node[0], self.goal_node[1], 'r*', markersize=20, label='Goal', zorder=5)
        ax.set_xlabel('Angle θ (rad)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Angular velocity ω (rad/s)', fontsize=13, fontweight='bold')
        ax.set_title(f'Phase Portrait - {title}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/phase_portrait.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/phase_portrait.png")
        plt.close()
        
        # 3. Convergence diagnostics
        if len(self.iteration_data['iterations']) > 0:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            axes[0].semilogy(self.iteration_data['iterations'], self.iteration_data['costs'], 
                           'b-o', linewidth=2, markersize=6)
            axes[0].set_ylabel('Cost', fontsize=13, fontweight='bold')
            axes[0].set_title(f'Convergence Diagnostics - {title}', fontsize=15, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].semilogy(self.iteration_data['iterations'], self.iteration_data['regularizations'], 
                           'r-s', linewidth=2, markersize=6)
            axes[1].set_ylabel('Regularization λ', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(self.iteration_data['iterations'], self.iteration_data['alphas'], 
                        'g-^', linewidth=2, markersize=6)
            axes[2].set_ylabel('Step size α', fontsize=13, fontweight='bold')
            axes[2].set_xlabel('Iteration', fontsize=13, fontweight='bold')
            axes[2].set_ylim([0, 1.1])
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/convergence.png', dpi=150, bbox_inches='tight')
            print(f"✓ {output_dir}/convergence.png")
            plt.close()
        
        # 4. Feedback gains
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(K[0, 0, :], linewidth=2, label='K[0,0]: u ← θ')
        axes[0].set_xlabel('Time step', fontsize=12)
        axes[0].set_ylabel('Gain value', fontsize=12)
        axes[0].set_title(f'Feedback Gain: Position - {title}', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(K[0, 1, :], linewidth=2, label='K[0,1]: u ← ω', color='orange')
        axes[1].set_xlabel('Time step', fontsize=12)
        axes[1].set_ylabel('Gain value', fontsize=12)
        axes[1].set_title(f'Feedback Gain: Velocity - {title}', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feedback_gains.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/feedback_gains.png")
        plt.close()
        
        # 5. Control corrections
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(du[0, :], 'b-', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_ylabel('du[0]', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time step', fontsize=13, fontweight='bold')
        ax.set_title(f'Feedforward Control Corrections - {title}', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/control_corrections.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/control_corrections.png")
        plt.close()
        
        print("\n" + "="*60)
        print(f"All plots saved to {output_dir}/")
        print("="*60)
    
    def compare_iLQR_vs_DDP(self, x0, u0, N, DISPLAY_MODE=False):
        """
        Run both iLQR and DDP on the same problem and compare results.
        
        Returns:
            results: Dictionary with comparison data
        """
        print("\n" + "="*70)
        print("COMPARING iLQR (GAUSS-NEWTON) vs FULL DDP")
        print("="*70)
        
        # Run iLQR
        print("\n[1/2] Running iLQR (Gauss-Newton approximation)...")
        x_ilqr = np.copy(x0)
        u_ilqr = np.copy(u0)
        x_ilqr, u_ilqr, K_ilqr = self.iLQR(x_ilqr, u_ilqr, N, DISPLAY_MODE=DISPLAY_MODE, USE_FULL_DDP=False)
        cost_ilqr = self.util.compute_total_cost(x_ilqr, u_ilqr, 
                                                 self.robot_object.get_state_size(),
                                                 self.robot_object.get_control_size(), N)
        iters_ilqr = len(self.iteration_data['iterations'])
        dist_ilqr = np.linalg.norm(self.robot_object.state_delta(x_ilqr[:, -1], self.goal_node))
        
        ilqr_data = {
            'x': x_ilqr,
            'u': u_ilqr,
            'K': K_ilqr,
            'cost': cost_ilqr,
            'iterations': iters_ilqr,
            'distance_to_goal': dist_ilqr,
            'iteration_history': dict(self.iteration_data)
        }
        
        # Run DDP
        print("\n[2/2] Running Full DDP (with Hessian terms)...")
        x_ddp = np.copy(x0)
        u_ddp = np.copy(u0)
        x_ddp, u_ddp, K_ddp = self.iLQR(x_ddp, u_ddp, N, DISPLAY_MODE=DISPLAY_MODE, USE_FULL_DDP=True)
        cost_ddp = self.util.compute_total_cost(x_ddp, u_ddp,
                                                self.robot_object.get_state_size(),
                                                self.robot_object.get_control_size(), N)
        iters_ddp = len(self.iteration_data['iterations'])
        dist_ddp = np.linalg.norm(self.robot_object.state_delta(x_ddp[:, -1], self.goal_node))
        
        ddp_data = {
            'x': x_ddp,
            'u': u_ddp,
            'K': K_ddp,
            'cost': cost_ddp,
            'iterations': iters_ddp,
            'distance_to_goal': dist_ddp,
            'iteration_history': dict(self.iteration_data)
        }
        
        # Print comparison
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        print(f"{'Metric':<30} {'iLQR (G-N)':<20} {'Full DDP':<20}")
        print("-"*70)
        print(f"{'Final Cost':<30} {cost_ilqr:<20.4f} {cost_ddp:<20.4f}")
        print(f"{'Iterations':<30} {iters_ilqr:<20d} {iters_ddp:<20d}")
        print(f"{'Distance to Goal':<30} {dist_ilqr:<20.6f} {dist_ddp:<20.6f}")
        print(f"{'Final State':<30} {str(x_ilqr[:,-1]):<20} {str(x_ddp[:,-1]):<20}")
        print("="*70)
        
        if cost_ilqr < cost_ddp:
            print("✓ iLQR found better solution (lower cost)")
        elif cost_ddp < cost_ilqr:
            print("✓ Full DDP found better solution (lower cost)")
        else:
            print("✓ Both methods achieved same cost")
        
        # Create comparison plots
        self._plot_comparison(ilqr_data, ddp_data)
        
        return {'iLQR': ilqr_data, 'DDP': ddp_data}
    
    def _plot_comparison(self, ilqr_data, ddp_data):
        """Create comparison plots between iLQR and DDP"""
        import os
        output_dir = "ddp_vs_ilqr"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating comparison plots in {output_dir}/...")
        
        dt = self.robot_object.timestep
        N = ilqr_data['x'].shape[1]
        t = np.arange(N) * dt
        
        # 1. Trajectory comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        axes[0].plot(t, ilqr_data['x'][0, :], 'b-', linewidth=2, label='iLQR (Gauss-Newton)')
        axes[0].plot(t, ddp_data['x'][0, :], 'r--', linewidth=2, label='Full DDP')
        axes[0].axhline(self.goal_node[0], color='k', linestyle=':', linewidth=1, label='Goal')
        axes[0].set_ylabel('Angle θ (rad)', fontsize=12, fontweight='bold')
        axes[0].set_title('Trajectory Comparison: iLQR vs DDP', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, ilqr_data['x'][1, :], 'b-', linewidth=2, label='iLQR')
        axes[1].plot(t, ddp_data['x'][1, :], 'r--', linewidth=2, label='DDP')
        axes[1].axhline(self.goal_node[1], color='k', linestyle=':', linewidth=1)
        axes[1].set_ylabel('Angular velocity ω (rad/s)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trajectories.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/trajectories.png")
        plt.close()
        
        # 2. Convergence comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].semilogy(ilqr_data['iteration_history']['iterations'], 
                        ilqr_data['iteration_history']['costs'],
                        'b-o', linewidth=2, markersize=5, label='iLQR')
        axes[0].semilogy(ddp_data['iteration_history']['iterations'],
                        ddp_data['iteration_history']['costs'],
                        'r-s', linewidth=2, markersize=5, label='Full DDP')
        axes[0].set_ylabel('Cost', fontsize=12, fontweight='bold')
        axes[0].set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(ilqr_data['iteration_history']['iterations'],
                    ilqr_data['iteration_history']['alphas'],
                    'b-o', linewidth=2, markersize=5, label='iLQR')
        axes[1].plot(ddp_data['iteration_history']['iterations'],
                    ddp_data['iteration_history']['alphas'],
                    'r-s', linewidth=2, markersize=5, label='Full DDP')
        axes[1].set_ylabel('Step size α', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.1])
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence.png', dpi=150, bbox_inches='tight')
        print(f"✓ {output_dir}/convergence.png")
        plt.close()
        
        print(f"Comparison plots saved to {output_dir}/")