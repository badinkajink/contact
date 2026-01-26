from ddp import DDP
from cartpole import Cartpole
from math import pi, sin
import numpy as np

# Drawing bounds
XMAX_MIN = 5.0  # max x position for cart
YMAX_MIN = pi   # max theta for pole

# Initial state: cart at origin, pole hanging down
X_START = [0, 0, pi, 0]  # [x, x_dot, theta, theta_dot]

# Goal state: cart at origin, pole balanced upright
X_GOAL = [0, 0, 0, 0]  # [x, x_dot, theta, theta_dot]

# Cost function weights - CONSTRAIN cart position heavily!
Q = np.diag([100.0, 1.0, 50.0, 1.0])  # Heavy penalty on BOTH cart position and pole angle
R = [0.1]  # Moderate control cost

# Shorter, more realistic horizon
N = 120  # 120 × 0.05s = 6 seconds

print("="*70)
print("CARTPOLE SWING-UP WITH iLQR")
print("="*70)
print(f"Task: Swing pole from hanging (π) to upright (0)")
print(f"Horizon: {N} steps × 0.05s = {N*0.05:.1f}s")
print(f"Cost weights: Q_x={Q[0,0]}, Q_θ={Q[2,2]}, R={R[0]}")
print()
print("⚠️  IMPORTANT NOTES:")
print("   - Cartpole swing-up is MUCH harder than pendulum")
print("   - The system is underactuated (can't directly control pole)")
print("   - Local optimizers like iLQR often fail or get stuck")
print("   - Better approaches: energy shaping, LQR-trees, or MPC")
print("   - This example is mainly pedagogical to show limitations")
print("="*70)

# Create cartpole system
cartpole = Cartpole(timestep=0.05, damping=0.05)  # More damping for stability
cartpole.set_Q(Q.tolist())
cartpole.set_R(R)
cartpole.set_goal(X_GOAL)

# Initialize with energy-building control sequence
x0 = np.zeros([cartpole.get_state_size(), N])
u0 = np.zeros([cartpole.get_control_size(), N-1])

# Strategy: Smaller, more controlled oscillations
# Don't let the cart run away!
pump_phase = int(0.5 * N)

for k in range(N-1):
    if k < pump_phase:
        # Smaller oscillating force - keep cart closer to origin
        phase = 2.0 * pi * k / pump_phase
        u0[0, k] = 5.0 * sin(phase)  # Reduced from 8.0
    else:
        # Small negative force to bring cart back
        u0[0, k] = -1.0

print(f"Initial control: Gentler energy pumping for {pump_phase} steps")
print("Note: Cartpole swing-up often gets stuck in local minima\n")

# Run DDP optimization with finite differences (more accurate for cartpole)
trajopt_obj = DDP(cartpole, X_START, X_GOAL, N, XMAX_MIN, YMAX_MIN, 
                  MAX_ITER=200, EXIT_TOL=1e-3)

# Use finite differences - the analytical Jacobian is approximate
# USE_FINITE_DIFF = True
# trajopt_obj.iLQR(x0, u0, N, DISPLAY_MODE=True, USE_FULL_DDP=False, 
#                  USE_FINITE_DIFF=USE_FINITE_DIFF)
RUN_MODE = "compare"  # Options: "ilqr", "ddp", "compare"

if RUN_MODE == "ilqr":
    print("\n*** Running iLQR (Gauss-Newton) ***")
    trajopt_obj.iLQR(x0, u0, N, DISPLAY_MODE=True, USE_FULL_DDP=False)
    
elif RUN_MODE == "ddp":
    print("\n*** Running Full DDP (with Hessians) ***")
    trajopt_obj.iLQR(x0, u0, N, DISPLAY_MODE=True, USE_FULL_DDP=True)
    
elif RUN_MODE == "compare":
    print("\n*** Running Comparison: iLQR vs DDP ***")
    results = trajopt_obj.compare_iLQR_vs_DDP(x0, u0, N, DISPLAY_MODE=False)