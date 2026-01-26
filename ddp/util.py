import numpy as np

class Util:
    def __init__(self, robot_object):
        self.robot_object = robot_object

    def compute_total_cost(self, x, u, nx, nu, N):
        """Compute the total cost along the trajectory"""
        J = 0
        for i in range(N):
            state_i = tuple(x[j, i] for j in range(nx))
            
            if i == N-1:
                # Terminal cost (no control)
                J += self.robot_object.cost_value(state_i)
            else:
                # Running cost (with control)
                J += self.robot_object.cost_value(state_i, [u[0, i]])
        
        return J

    def initialize_CTG(self, x, nx, k):
        """Initialize cost-to-go at terminal state"""
        Vxx, Vx = self.robot_object.gradient_hessian(x)
        return Vxx, Vx

    def compute_approximation(self, x, u, nx, nu, k):
        """Compute local approximation: dynamics gradients and cost derivatives"""
        H, g = self.robot_object.gradient_hessian(x, u)
        A, B = self.robot_object.next_state_gradient(x, u)
        return A, B, H, g

    def backpropogate_CTG(self, A, B, H, g, Vxx, Vx, nx, nu, k, use_hessians=True):
        """
        Backpropagate cost-to-go through dynamics.
        
        Args:
            A, B: Dynamics Jacobians
            H: Cost Hessian (joint state-control)
            g: Cost gradient (joint state-control)
            Vxx, Vx: Value function derivatives
            nx, nu: State and control dimensions
            k: Time index
            use_hessians: If True, use full DDP with cost Hessians (Lxx, Luu, Lux)
                         If False, use iLQR Gauss-Newton approximation (ignores cost Hessians)
        
        Returns:
            Qxx, Qux, Quu, Qx, Qu: Q-function approximation
        """
        # Convert to numpy arrays if needed
        A = np.array(A)
        B = np.array(B)
        H = np.array(H)
        g = np.array(g)
        Vxx = np.array(Vxx)
        Vx = np.array(Vx)
        
        # ============================================================
        # DDP vs iLQR DIFFERENCE: Cost Hessian terms
        # ============================================================
        if use_hessians:
            # *** FULL DDP: Include cost Hessians (second-order terms) ***
            # Extract cost Hessian blocks from joint Hessian H
            # H = [[Lxx, Lxu], [Lux, Luu]]
            Lxx = H[0:nx, 0:nx]
            Luu = H[nx:nx+nu, nx:nx+nu]
            Lux = H[nx:nx+nu, 0:nx]
            
            # Q-function Hessians include BOTH cost curvature AND dynamics curvature
            Qxx = Lxx + A.T @ Vxx @ A  # Cost Hessian + dynamics curvature
            Quu = Luu + B.T @ Vxx @ B  # Cost Hessian + dynamics curvature
            Qux = Lux + B.T @ Vxx @ A  # Cost Hessian + dynamics curvature
        else:
            # *** iLQR (GAUSS-NEWTON): Ignore cost Hessians ***
            # Q-function uses ONLY dynamics curvature (Gauss-Newton approximation)
            # This is more stable and works well when cost is nearly quadratic
            Qxx = A.T @ Vxx @ A  # Only dynamics curvature (NO Lxx)
            Quu = B.T @ Vxx @ B  # Only dynamics curvature (NO Luu)
            Qux = B.T @ Vxx @ A  # Only dynamics curvature (NO Lux)
        # ============================================================
        
        # Gradient terms are ALWAYS the same for both DDP and iLQR
        Lx = g[0:nx]
        Lu = g[nx:nx+nu]
        Qx = Lx + A.T @ Vx
        Qu = Lu + B.T @ Vx
        
        return Qxx, Qux, Quu, Qx, Qu

    def compute_du_K(self, Qxx, Qux, Quu, Qx, Qu, Quu_inv, nx, nu, N):
        """Compute optimal control update and feedback gain"""
        K = -Quu_inv @ Qux
        du = -Quu_inv @ Qu
        return du, K

    def compute_new_CTG(self, Qxx, Qux, Quu, Qx, Qu, Quu_inv, du, K, nx, nu, N):
        """
        Update cost-to-go for next iteration.
        Using the ORIGINAL formula from your working code.
        """
        Vx = Qx - Qux.T @ du
        Vxx = Qxx - Qux.T @ K
        return Vxx, Vx

    def compute_control_update(self, x, x_new, K, du, alpha, nx, nu, N):
        """Compute control update with line search"""
        change_in_u = alpha * du + K @ np.array(self.robot_object.state_delta(x_new, x))
        return change_in_u