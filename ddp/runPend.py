from ddp import DDP
from pendulum import Pendulum
from math import pi, sin, cos
import numpy as np

# Drawing bounds
XMAX_MIN = pi
YMAX_MIN = 10

# Initial and goal states
X_START = [0, 0]
X_GOAL = [pi, 0]

# # Cost matrices - VERY high position weight to prioritize reaching goal
# Q = [[50, 0], [0, 1]]  # Increased from 10 to 50 - reaching goal is MUCH more important
# R = [0.01]             # Very low control cost - allow aggressive control

# # Number of timesteps - more steps = smoother optimization
# N = 60  # Increased from 32

# How did I get these numbers? A little intuition and a lot of guess and check.
Q = [[3,0],[0,1]]
R = [0.08]
N = 50

# Create pendulum with more damping for easier stabilization
pend = Pendulum(timestep=0.1, damping=0.15)
pend.set_Q(Q)
pend.set_R(R)
pend.set_goal(X_GOAL)

# Initialize trajectory with SMART energy-based warmstart
# x0 = np.zeros([pend.get_state_size(), N])
# u0 = np.zeros([pend.get_control_size(), N-1])

# Strategy: Create a control that OVERSHOOTS the goal
# This gives iLQR something to refine (brake at the right time)

# Aggressive pumping that will overshoot
pump_duration = int(0.6 * N)  # Pump for 60% of trajectory

# use Q = [[10,0],[0,1]]
# for k in range(N-1):
#     if k < pump_duration:
#         # Strong sinusoidal pumping - will build too much energy
#         phase = 3.5 * pi * k / pump_duration  # ~1.75 full swings
#         u0[0, k] = 4.5 * sin(phase)  # Increased from 3.5 to 4.5
#     else:
#         # Let it coast (iLQR will learn to brake)
#         u0[0, k] = 0.0

# for k in range(N-1):
#     if k < pump_duration:
#         # Sinusoidal pumping - frequency tuned to natural frequency
#         phase = 4.0 * pi * k / pump_duration  # ~2 full swings
#         u0[0, k] = 3.5 * sin(phase)
#     else:
#         # Small stabilizing control at end
#         u0[0, k] = 0.0

print("="*60)
print("PENDULUM SWING-UP WITH iLQR")
print("="*60)
print(f"Horizon: {N} steps × 0.1s = {N*0.1:.1f}s")
print(f"Initial control: Aggressive energy pumping (will overshoot)")
print(f"Cost weights: Q_position={Q[0][0]}, Q_velocity={Q[1][1]}, R={R[0]}")
print("="*60)

# Create DDP optimizer with adaptive regularization
trajopt_obj = DDP(pend, X_START, X_GOAL, N, XMAX_MIN, YMAX_MIN, 
                  MAX_ITER=150, EXIT_TOL=1e-4)

# Choose what to run:
RUN_MODE = "ilqr"  # Options: "ilqr", "ddp", "compare"

if RUN_MODE == "ilqr":
    print("\n*** Running iLQR (Gauss-Newton) ***")
    trajopt_obj.iLQR(x0, u0, N, DISPLAY_MODE=True, USE_FULL_DDP=False)
    
elif RUN_MODE == "ddp":
    print("\n*** Running Full DDP (with Hessians) ***")
    trajopt_obj.iLQR(x0, u0, N, DISPLAY_MODE=True, USE_FULL_DDP=True)
    
elif RUN_MODE == "compare":
    print("\n*** Running Comparison: iLQR vs DDP ***")
    results = trajopt_obj.compare_iLQR_vs_DDP(x0, u0, N, DISPLAY_MODE=False)