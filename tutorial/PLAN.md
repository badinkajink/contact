# Plan for the unwritten decks, the lab, and the integration of decks 11-16

This file holds the detailed scope of every planned piece of the curriculum. `AGENTS.md` at the
repository root is the guide (conventions, tools, status); read it first. The Hou & Mason papers
are [arXiv 1903.02715](https://arxiv.org/abs/1903.02715) and [arXiv 2011.04872](https://arxiv.org/abs/2011.04872).

## How each piece gets built

1. Outline 30-45 slides in 2-4 sections (id, title, content, build steps, figure, notes, check-yourself questions, `data-code` function, coverage-map items). Compute every number in Python first.
2. Add notebook cells after the deck's header cell: one function per non-trivial number, markdown naming the slide id it backs.
3. Write `lib/algo/NN_short.js` for the live figures and `tools/twins/NN.json` against the notebook functions.
4. Write the deck in stages: text and math, then figures, then notes. Every slide has an id and notes; the last content slide is "Results used in later decks".
5. Verify with the tools until the deck meets the definition of done in `AGENTS.md` section 8.
6. Review twice, fixing as you go: once for mathematics and code coupling (recompute every number independently, check every derivation step, check claims against sources), once as a first-year student, a careful viewer of every PNG and every live control, and the owner applying the writing rules.

## Ownership of shared concepts

One deck teaches each shared concept in depth; every other deck that uses it gets a one-slide
recap (or a small `.example` box) with a link to the owner's slide id. Do not re-teach.

| concept | owner | users (recap + link only) |
|---|---|---|
| dot product: length, angle, perpendicular | 01 | 11 (keeps its gentle intro as an entry point, adds a link to 01) |
| projection onto lines and subspaces, orthonormal bases | 02 | 05, 06, 21 |
| rank, null space, solution sets | 01 | 02, 13, 21 |
| condition number, SVD, ill-conditioning | 02 | 05 (GD rate), 21 (crashing index) |
| least squares, min-norm solution, pseudoinverse | 02 | 06 (KKT view), 13 (four-legged table), 21 |
| Schur complement | 02 | 06, 08 |
| derivatives, gradient, Hessian, Jacobian, chain rule, Taylor | 03 | 05, 06, 08, 09, 17, 18 |
| finite differences | 03 | 10 (MJPC derivatives) |
| Gaussians, covariance, sampling, softmax weights, ESS | 04 | 10, 21, 22 |
| projected gradient on the unit sphere with restarts | 05 | 21 |
| general projection / penalty / barrier / AL methods | 06 | 19, 20 |
| complementary slackness | 06 | 07, 18 |
| LCP / NCP as problems, cones (SOC, dual cone, projection onto a cone), LP geometry | 07 | 12, 16, 18, 19, 20, 21 |
| MPC receding horizon | 08 | 10, 22 |
| friction cone physics and pyramid approximations | 12 | 07, 19, 21 |
| 6D wrenches, cross products | 15 | 17 (extends with twists and power) |
| rotations, quaternions, Omega(q), twists, adjoint, virtual work | 17 | 18, 21 |
| contact Jacobian, Signorini, impulses, quasi-static limit | 18 | 19, 20, 21 |
| MuJoCo soft contact (solref, solimp, impratio, cone types) | 19 | 20, 22 |

MuJoCo demos are assigned so they do not repeat: 10 predictive sampling swing-up; 11 tilting
board (slip at arctan(mu)); 13 four-legged table force distribution vs the min-norm one;
14 two-finger pinch slipping when the Nguyen test fails; 18 box drop with contact forces;
19 solref/solimp penetration, creep below the friction angle, pyramidal vs elliptic cones;
20 ComFree vs MuJoCo on the same drop; 21 block tilting by HFVC vs pure velocity control;
22 predictive-sampling MPC pushing a block to a goal.

## Decks to write

### 01. Vectors, matrices, and linear maps

File `tutorial/slides/01_vectors_and_matrices.html`; notebook `tutorial/00_math_toolkit.py`.

**Sources.** tex 1100-1130 (vectors, matrices, norms) plus new material; the existing vector slides in 11_coulomb_friction.html (Tool #1 arrows, Tool #2 dot product) show the voice. Optional background: linear_algebra/3251/*.ipynb (COMS 3251 labs).

**Scope.** vectors as arrows and as lists; R^n; addition, scaling; linear combinations; span; linear independence; basis and coordinates; dot product basics (length, angle, perpendicular); norms (1, 2, inf) and their unit balls; matrices as tables and as linear maps; matrix-vector product as a combination of columns; composition = matrix multiplication; transpose; identity; inverse; 2x2 determinant as area scale; systems Ax = b: row picture vs column picture; Gaussian elimination and RREF with pivots; rank; column space; row space; null space; rank-nullity; non-square matrices (2x3 flattens R^3 to R^2, 3x2 embeds a plane) as maps between spaces of different sizes; solution set = particular solution + null space; homogeneous vs non-homogeneous; consistent vs inconsistent (infeasible); "the identity is the strictest goal".

**What later decks need from it.** 02 (orthogonality, rank tests, projections), 03 (Jacobian as a matrix of partials), 05-07 (decision vectors, linear constraints Az <= b), 08 (x_{k+1} = A x_k + B u_k), 13 (four-legged table null space = statical indeterminacy), 17-18 (Jacobians, J v = 0), 21 (N v = 0, G v = b_G, rank([N; G]) - rank(N) velocity commands, Sol(N&C) contained in Sol(N&G)).

### 02. Eigenvalues, singular values, and least squares

File `tutorial/slides/02_eigenvalues_and_least_squares.html`; notebook `tutorial/00_math_toolkit.py`.

**Sources.** tex 1130-1303 (quadratic forms, eigenvalues, conditioning, positive definiteness, Cholesky, least squares, normal equations, block matrices, Schur complement); Hou & Mason 2021 Sec. IV-A (condition number, crashing index, row normalization, artificial ill-conditioning) and Sec. V (Null(.), Row(.) orthonormal bases, null-space/row-space inclusion, eq. 11-13); study plan week 2 (drawer example, fragility vs infeasibility, what T does).

**Scope.** projection onto a line and onto a subspace; orthogonality; orthonormal bases; Gram-Schmidt and QR; orthogonal complement: row space is perpendicular to null space; null-space inclusion equals reversed row-space inclusion and the rank test (Hou 2021 eq. 11-13); eigenvalues and eigenvectors (2x2 by hand, geometric meaning); symmetric matrices and the spectral theorem; quadratic forms as bowls, saddles, and valleys (level-set ellipses); positive (semi)definite; Cholesky (2x2 by hand; why it exists iff PD); SVD as circle -> ellipse, U Sigma V^T, rank from singular values; condition number sigma_max/sigma_min; nearly parallel rows; fragility (ill-conditioned) vs infeasibility (inconsistent); the drawer example (pushing a drawer along its rail with a velocity command nearly parallel to the rail constraint normal); artificial ill-conditioning and row normalization; least squares: line fitting, normal equations, geometric projection of b onto the column space; minimum-norm solution of an underdetermined system and the pseudoinverse ("pick one force distribution out of many", Hou 2019 Algorithm 2 rule); block matrices and the Schur complement (block elimination), which LQR and KKT systems use later.

**What later decks need from it.** 05 (Hessian eigenvalues, condition number kappa sets gradient descent speed), 06 (KKT matrices, Schur complement, min-norm solutions), 07 (PSD cone), 08 (Riccati: PD matrices, completing the square via Schur complement), 09 (regularizing Q_uu to be PD), 13 (least-squares force distribution for the four-legged table), 19 (Delassus matrix J M^-1 J^T is PSD; Cholesky), 21 (crashing index = cond([J_hat; C_hat]); Null(), Row() orthonormal bases; row normalization).

### 03. Derivatives, gradients, and Jacobians

File `tutorial/slides/03_derivatives_and_jacobians.html`; notebook `tutorial/00_math_toolkit.py`.

**Sources.** tex 157-175 (Taylor expansion insight box), 224-236 (steepest ascent), 1303-1322 (matrix calculus) plus new material; Hou 2019 eq. 3 (differentiating Phi(q) = 0).

**Scope.** derivative as slope of the tangent line and as a limit; rules (power, sum, product, chain) with the quartic J(z) = z^4 - 4z^2 + z + 4 as running example; second derivative as curvature; Taylor polynomials (linear and quadratic models) with a live figure; functions of several variables, contour plots; partial derivatives; the gradient; directional derivative = gradient dot direction; the gradient is perpendicular to level sets and points uphill fastest; the Hessian; multivariable Taylor expansion; vector-valued functions and the Jacobian matrix; the chain rule as a product of Jacobians; linearizing dynamics x_{k+1} = f(x_k, u_k) into A = df/dx, B = df/du; matrix-calculus identities (grad of b^T x, x^T A x, (1/2)||Ax - b||^2) derived component by component; time derivative of a constraint Phi(q(t)) = 0 gives J_Phi qdot = 0; finite differences (forward vs central, the error-vs-step-size V curve from truncation and roundoff), which MuJoCo MPC uses.

**What later decks need from it.** 05 (gradient, Hessian, Taylor models behind GD and Newton), 06 (gradient of the Lagrangian), 08-09 (linearization A, B; second-order expansion of the Q-function), 10 (finite-difference derivatives in MJPC), 17 (derivative of a rotation, angular velocity, quaternion derivative), 18 (contact Jacobian), 21 (J_Phi and N = J_Phi Omega).

### 04. Probability, Gaussians, and sampling

File `tutorial/slides/04_probability_and_sampling.html`; notebook `tutorial/00_math_toolkit.py`.

**Sources.** the needs of tex 2144-2437 (predictive sampling, MPPI weights and temperature, CEM elite refitting, DIAL-MPC annealing) and tutorial/03_sampling_mpc.py (MPPI weight tables for lambda = 1 and 5, effective sample size); new material.

**Scope.** outcomes, events, probability; random variables; mean, variance, standard deviation; histograms and the law of large numbers; uniform and Gaussian distributions (pdf, 68-95-99.7); drawing samples (pseudo-random generators, seeds, Box-Muller); multivariate Gaussians, covariance matrices, covariance ellipses (eigenvectors from deck 02); sampling N(m, Sigma) with a Cholesky factor x = m + L eps; Monte Carlo estimates of expectations and the 1/sqrt(N) error; weighted averages; softmax / Boltzmann weights with a temperature, the two limits (argmin and uniform); effective sample size; importance sampling in one slide; fitting a Gaussian to elite samples (sample mean and covariance) as used by CEM; random restarts as sampling of initial guesses; random test-problem generation (Hou 2021 tests 78,000 random problems).

**What later decks need from it.** 10 (predictive sampling, MPPI, CEM, DIAL-MPC), 21 (random restarts in Hou 2019 Algorithm 1; random test problems in Hou 2021), 22 (sampling-based contact-rich MPC).

### 05. Unconstrained optimization: gradient descent and Newton's method

File `tutorial/slides/05_unconstrained_optimization.html`; notebook `tutorial/01_optimization_fundamentals.py`.

**Sources.** tex 97-468 (general problem, standard form conventions, first- and second-order conditions and why, steepest descent, scalar quadratic, worked examples 1.1, 1.2, backtracking/Armijo, convergence rates, conditioning and zig-zag, Newton derivation and worked example 1.2b, failure mode at z0 = 0, Levenberg-Marquardt regularization, Gauss-Newton and its link to iLQR); tutorial/01_optimization_fundamentals.py (run_gradient_descent, run_newton, armijo_accepts); Hou 2019 eq. 15 and its 4-step projected gradient descent with Ns random restarts. VERIFY every number in the tex tables by computing it (the gradient-descent table at z0 = 2, alpha = 0.05 looks wrong: J(1.15) is about 1.61, not 0.948).

**Scope.** the optimization problem and its vocabulary; local vs global minima on the quartic; basins of attraction and random restarts; first-order condition and why; second-order condition and why (Hessian eigenvalues classify stationary points); gradient descent; step size too small / too large; backtracking line search and the Armijo condition; convergence rates (linear, superlinear, quadratic) with digit-counting; conditioning and the zig-zag on an ill-conditioned bowl (kappa from deck 02); Newton as minimizing the quadratic model; Newton as gradient descent in the Hessian metric; Newton failure modes and Levenberg-Marquardt; least-squares costs and Gauss-Newton; optimization on the unit sphere: projected gradient descent (step, then renormalize) with random restarts, exactly Hou 2019 eq. 15.

**What later decks need from it.** 06 (constrained methods reduce to unconstrained ones), 08 (a quadratic problem is solved by one Newton step: LQR), 09 (LM regularization of Q_uu; iLQR = Gauss-Newton along a trajectory; line search in the forward pass), 10 (MJPC gradient planner and parallel line search), 21 (Hou 2019 Algorithm 1 is projected gradient descent with restarts).

### 06. Constrained optimization: Lagrange multipliers and the KKT conditions

File `tutorial/slides/06_constrained_optimization.html`; notebook `tutorial/01_optimization_fundamentals.py`.

**Sources.** tex 469-788 (equality constraints, geometric picture, multipliers as prices, inequality constraints and complementary slackness, sign of multipliers, complete KKT, active sets, worked example 1.3, penalty, augmented Lagrangian, barrier / interior point, projection methods); tutorial/01_optimization_fundamentals.py (penalty_z, barrier_z); Hou 2019 eq. 21-23 (min ||f_free||^2 subject to M f = b solved by one KKT linear system).

**Scope.** feasible sets; equality constraints and the tangency picture (grad J parallel to grad h); Lagrangian and multipliers; multipliers as sensitivities (shadow prices) with a numeric check; inequality constraints; active vs inactive; complementary slackness; why inequality multipliers are nonnegative (a floor pushes, never pulls); the four KKT conditions; active-set enumeration and its 2^m combinatorics; worked example with numbers; equality-constrained QP solved by one KKT linear system and the min-norm connection to deck 02 (Hou 2019 eq. 22); an inequality-constrained QP solved by active set; penalty method and its ill-conditioning as rho grows; augmented Lagrangian (method of multipliers); log-barrier and the central path; projection methods and projected gradient onto a box / halfspace / cone; contact forces are the multipliers of non-penetration constraints (bridge to 18-19).

**What later decks need from it.** 07 (duality builds on the Lagrangian), 08 (the trajectory QP u* = [4.8, 1.6, -1.6, -4.8] and LQR as an equality-constrained QP), 18-19 (Signorini = complementary slackness; contact forces as multipliers), 20 (penalty-like soft contact), 21 (Algorithm 2 is an equality-constrained QP plus linear inequality guard conditions).

### 07. Convexity, duality, and complementarity

File `tutorial/slides/07_convexity_and_duality.html`; notebook `tutorial/01_optimization_fundamentals.py`.

**Sources.** tex 789-1099 (Lagrangian duality, LP, QP, SOCP, NLP, convexity, complementarity problems, LCP, mode enumeration), tex 1322-1428 (cones, second-order cone, dual cones, projection onto cones); tutorial/01_optimization_fundamentals.py (solve_lcp_enumeration and the duality cell); Hou 2019 eq. 6 and Sec. V-D (guard conditions as affine inequalities; 8-sided polyhedral friction cone, d_i = [sin(pi i/4), cos(pi i/4), 0]).

**Scope.** convex sets and convex functions (chords above the graph), why convex means every local minimum is global, and what that buys solvers; linear programs: feasible polygons, level lines of the objective, optimum at a vertex, LP feasibility (phase 1) and infeasible / unbounded cases; guard conditions as linear inequalities on forces; QPs; cones, the second-order (ice-cream) cone, polyhedral cones and the 8-sided friction pyramid; SOCPs; dual cones; projecting onto a cone; Lagrangian duality with a 2D worked example, weak and strong duality, the dual as a lower bound; complementarity problems: LCP geometry and mode enumeration (2D), the NCP; the ladder LP < QP < SOCP < NLP and which contact solvers sit where.

**What later decks need from it.** 12 (pyramids as LPs), 16 (grasp quality LP), 18-19 (LCP, CCP as a convex cone program, dual cone), 20 (dual-cone spring), 21 (LP / QP for force-controlled actions; guard conditions).

### 08. Optimal control, dynamic programming, and LQR

File `tutorial/slides/08_optimal_control_and_lqr.html`; notebook `tutorial/02_lqr_ilqr_sliding_block.py`.

**Sources.** tex 1429-1758 (optimal control problem, direct vs indirect, shooting vs collocation, Bellman equation and dynamic programming, LQR, sliding block running example); tutorial/02_lqr_ilqr_sliding_block.py (min-effort block QP u* = [4.8, 1.6, -1.6, -4.8], scalar_lqr and the golden-ratio gain). Verify every number.

**Scope.** state, control, dynamics as a difference equation (discretizing a sliding block with h); cost functions over a horizon; the trajectory optimization problem as one big optimization (links to 05-07); shooting vs collocation (direct methods) and a one-slide view of indirect methods; the minimum-effort sliding-block QP solved exactly; Bellman principle of optimality with a small grid-world or shortest-path DP example computed by hand; value function and cost-to-go; LQR derivation with every step shown (completing the square, the Riccati recursion) first scalar then matrix; the golden-ratio gain; LQR on the double integrator with Q and R sliders; infinite-horizon LQR; model predictive control as re-solving on a receding horizon.

**What later decks need from it.** 09 (Q-function, backward pass generalizes Riccati), 10 (MPC framing, sampling alternatives), 22 (trajectory optimization through contact).

### 09. Differential dynamic programming and iLQR

File `tutorial/slides/09_ddp_and_ilqr.html`; notebook `tutorial/02_lqr_ilqr_sliding_block.py`.

**Sources.** tex 1759-2143 (setup, Q-function, DDP backward pass, iLQR by dropping the Hessian, forward pass with line search, complete algorithm, regularization and control limits, worked iLQR on the sliding block); tutorial/02_lqr_ilqr_sliding_block.py (ilqr_block, backward pass debug at t = 2: Q_uu = 24.22, K_2 = [0, -2.752]). Verify every number.

**Scope.** nonlinear dynamics and a nominal trajectory; local linear-quadratic models along it (deck 03 linearization); the Q-function and its second-order expansion term by term; backward pass: Q_x, Q_u, Q_xx, Q_uu, Q_ux, gains k and K, value update; DDP vs iLQR (dropping second derivatives of the dynamics = Gauss-Newton, deck 05); forward pass rollout with line search on the feedforward term; convergence; regularization of Q_uu (Levenberg-Marquardt) and control limits (box-DDP, clamping); worked example with every number; a live iLQR on a pendulum swing-up (JS) showing iterations converge; comparison with LQR (one iteration on a linear-quadratic problem).

**What later decks need from it.** 10 (gradient-based vs sampling-based MPC), 22 (iLQR through contact, smoothing).

### 10. Sampling-based model predictive control and MuJoCo MPC

File `tutorial/slides/10_sampling_mpc.html`; notebook `tutorial/03_sampling_mpc.py`.

**Sources.** tex 2144-2584 (common framework, predictive sampling, MPPI, CEM, annealed sampling and DIAL-MPC, spline parameterization, MuJoCo MPC architecture, residuals and norms, why sensor derivatives are free, gradient descent planner); tutorial/03_sampling_mpc.py (mppi_weights, effective_sample_size, knots_to_controls, rollout_cost, run_predictive_sampling, run_mppi, run_cem). Verify every number.

**Scope.** receding-horizon MPC recap; the sampling framework (perturb, roll out, score, update); predictive sampling (keep the best); MPPI (softmax weights, temperature, effective sample size) with the lambda = 1 and 5 tables; CEM (elite fraction, refit mean and covariance); annealing and DIAL-MPC; spline parameterization of controls (zero-order, linear, cubic) and why it reduces dimension; MuJoCo MPC: residuals, norms, cost terms, planners, the parallel line search of the gradient planner, why finite-difference sensor derivatives are cheap; live demos: JS sampling on the sliding block, and REAL MuJoCo (lib/mj.js) predictive sampling swinging up a pendulum or cart-pole in the browser with the sampled rollouts drawn.

**What later decks need from it.** 22 (sampling-based MPC through contact, e.g. pushing a block).

### 17. Rigid-body motion: rotations, quaternions, twists, and wrenches

File `tutorial/slides/17_rigid_body_motion.html`; notebook `tutorial/05_contact_solvers.py`.

**Sources.** new material; Hou 2019 Sec. III-A (generalized velocity v vs qdot, qdot = Omega(q) v, generalized force f, f_u = 0), eq. 4 (virtual work tau = J^T lambda, f = Omega^T tau), Sec. V-A,B (q in R^10 with a quaternion, v in R^9 with the body twist, Omega in eq. 26, E(q) in eq. 27, spatial vs body twist and the adjoint in eq. 29); tex 2613-2632 (rigid body dynamics, free motion); study plan weeks 4 and 6 (why q-dot fails, what Omega does, rotations, SO(3), quaternions and the unit-length constraint, gimbal lock); Lynch & Park, Modern Robotics ch. 2-3 and 5 as the standard reference. The existing 15_grasps_in_3d.html introduces cross products and 6D wrenches: read it and build on it, do not contradict it.

**Scope.** configuration space and degrees of freedom (a puck: 3, a free body in 3D: 6, a robot arm: its joints), generalized coordinates; planar rotation by an angle and its matrix; 3D rotation matrices: orthonormal columns, det = 1, SO(3), composition does not commute (live 3D figure); Euler angles and gimbal lock (live); axis-angle and Rodrigues formula; quaternions: definition, unit length, rotating a vector, the double cover, composing rotations; angular velocity and the skew-symmetric matrix [w]x, Rdot = R [w_b]x; the quaternion derivative qdot = (1/2) E(q) w (Hou eq. 27) derived; why qdot is not a velocity (4 numbers for 3 DOF, the unit-norm constraint) and the generalized velocity v with qdot = Omega(q) v (Hou eq. 26); integrating orientation and renormalizing; homogeneous transforms; twists (linear and angular velocity together), body vs spatial twists, the adjoint map; wrenches (force and torque together, building on deck 15), power = wrench dot twist; virtual work: tau = J^T lambda and f = Omega^T tau (Hou eq. 4) with a planar worked example.

**What later decks need from it.** 18 (Newton-Euler equations, contact Jacobians in generalized coordinates), 21 (the block-tilting model: q = [p, quaternion, hand position], Omega, E(q), adjoint, generalized forces).

### 18. Rigid-body dynamics with contact

File `tutorial/slides/18_contact_dynamics.html`; notebook `tutorial/05_contact_solvers.py`.

**Sources.** tex 2585-2781 (why contact is hard, rigid body dynamics, Signorini condition, Coulomb friction, the full contact problem as an NCP); tutorial/05_contact_solvers.py (ball_floor_lcp); Hou 2019 Sec. III-C (holonomic constraints Phi(q) = 0, J_Phi Omega v = 0, sticking vs sliding contacts, reaction forces as multipliers, Newton second law under the quasi-static assumption, eq. 5). Verify every number.

**Scope.** why contact is hard (nonsmooth, stiff, combinatorial) with numbers; Newton-Euler for a rigid body and the manipulator equation M(q) vdot + c(q, v) = tau + J_c^T lambda for simple systems (a puck, a two-link arm) with M computed; holonomic (bilateral) constraints Phi(q) = 0 and their Jacobians; unilateral contact: gap function phi(q) >= 0; the contact Jacobian mapping generalized velocity to contact-point velocity (normal and tangent rows); sticking contact = all rows zero, sliding = normal row only (the Hou and Mason model); Signorini at position and velocity level (complementarity from 06); impacts, impulses and restitution; Coulomb friction and maximum dissipation in contact coordinates (recap of 11-12); time-stepping: semi-implicit Euler, velocity-level impulses, Stewart-Trinkle / Anitescu-Potra; the full problem as an NCP; the quasi-static limit (drop M vdot) which gives Hou eq. 5; live demos: bouncing ball with restitution, a block sliding and stopping, REAL MuJoCo box drop with contact forces.

**What later decks need from it.** 19 (solvers for this NCP), 20 (complementarity-free alternative), 21 (quasi-static Newton law, holonomic constraints of sticking and sliding contacts), 22 (trajopt through contact).

### 19. Contact solvers: LCP, CCP, and MuJoCo's soft contact

File `tutorial/slides/19_contact_solvers.html`; notebook `tutorial/05_contact_solvers.py`.

**Sources.** tex 2782-3178 (summary of models, LCP by linearizing the cone, CCP relaxing Signorini, MuJoCo approach, solref and solimp deep dive, how compliance enters the CCP, MuJoCo vs Drake vs ADMM, Drake approach, ADMM for the CCP, solving the full NCP), tex 3560-3604 (comparison and practical implications); tutorial/05_contact_solvers.py (pgs, project_soc, simulate_sliding_cube). The MuJoCo documentation (mujoco.readthedocs.io, Computation chapter: soft constraints, solref, solimp, impedance, reference acceleration) is the ground truth for MuJoCo claims: check the tex against it with WebFetch and flag every discrepancy. Verify every number.

**Scope.** the Delassus matrix A = J M^-1 J^T; LCP formulation with a pyramid cone (Stewart-Trinkle), solved by Lemke or PGS (live PGS iterations); the CCP (Anitescu) relaxation and its artifact (gliding / boundary layer), with numbers; MuJoCo: convex optimization in acceleration space, soft constraints, reference acceleration a_ref = -b v - k r, solref (timeconst, dampratio) and solimp (dmin, dmax, width, midpoint, power) with every formula from the docs; impratio; pyramidal vs elliptic cones; REAL MuJoCo live slides: a resting box with solref/solimp sliders and a strip chart of penetration and normal force compared to the formula; creep on a tilted plane below the friction angle; pyramidal vs elliptic slip direction; Drake SAP in one or two slides; ADMM for a single contact (projection onto the cone, live); a comparison table of models.

**What later decks need from it.** 20 (complementarity-free is compared against these), 22 (which simulator/solver a trajopt method assumes).

### 20. Complementarity-free contact

File `tutorial/slides/20_complementarity_free_contact.html`; notebook `tutorial/05_contact_solvers.py`.

**Sources.** tex 3179-3559 (motivation, from the CCP dual to closed-form forces, the spring in the dual cone, automatic Coulomb satisfaction, damping, ComFree-Sim GPU parallelization, the dual-cone impedance connecting to MuJoCo and Drake, differentiability via SoftPlus, contact-implicit MPC at 50-100 Hz); tutorial/05_contact_solvers.py (the 1D ComFree example with beta = 10, lambda_N = m g). Find and read the ComFree paper (Jin et al., complementarity-free multi-contact modeling; check arXiv) to verify the tex. Verify every number.

**Scope.** why bypass complementarity (nonsmoothness, iterations, gradients); the dual of the CCP; closed-form forces as a spring in the dual cone; why Coulomb holds automatically (geometric argument with a live 2D dual-cone figure); damping for stability; parallel structure for GPUs; the dual-cone impedance as a common language for MuJoCo, Drake and ComFree; SoftPlus smoothing and differentiability with a live plot; performance numbers; live demo of the 1D ball (beta = 10) and a 2D contact force field; comparison against REAL MuJoCo on the same box drop.

**What later decks need from it.** 22 (smooth / complementarity-free models make contact-implicit MPC tractable).

### 21. Hybrid force-velocity control

File `tutorial/slides/21_hybrid_force_velocity_control.html`; notebook `tutorial/07_hybrid_servoing.py`.

**Sources.** Hou & Mason 2019 ((papers: arXiv PDFs) hou2019.txt) and 2021 ((papers: arXiv PDFs) hou2021.txt) read in full; Mason 1981 natural and artificial constraints; Raibert & Craig 1981 selection matrices; Lynch & Park, Modern Robotics 11.6 hybrid motion-force control (use WebFetch on the Modern Robotics book/wiki for 11.6 if needed); the authors' MATLAB code github.com/yifan-hou/pub-icra19-hybrid-control (inspect it with WebFetch for conventions and the block-tilting numbers); the student study plan (papers: arXiv PDFs) study_plan.txt (week 6 exam covers the whole paper).

**Scope.** the execution problem (stiff velocity control breaks under position errors; soft force control drifts) with a live 1D example; Mason 1981 natural vs artificial constraints; Raibert-Craig selection matrices in the fully actuated case (Modern Robotics 11.6); systems with free objects: v = [v_u; v_a], f = [0; f_a], qdot = Omega v; goal G v = b_G; holonomic natural constraints N v = 0 with N = J_Phi Omega; quasi-static Newton law Omega^T J_Phi^T lambda + f + F = 0 and sorting forces; guard conditions as affine inequalities (polyhedral cones, minimum normal force); the transform T = diag(I, R_a), w = T v, eta = T f; which velocity commands conflict with constraints (Fig. 1-2) and the fragility (nearly parallel) vs infeasibility picture; Sol(N & C) contained in Sol(N & G); dimension counting n_av = rank([N;G]) - rank(N); Algorithm 1: basis of the solution space, the two-term cost (orthogonality + alignment with null(N)), projected gradient descent on unit vectors with random restarts, completing R_a; Algorithm 2: the equality-constrained QP / KKT system for the free forces and the LP over eta_af with the guard conditions; the block-tilting example in full (q in R^10, v in R^9, Omega, E(q), Phi, guard conditions, the solved actions); 2021: kinematic conditioning, crashing index cond([J_hat; C_hat]) with the Fig. 1 numbers (2.41, 3.87, inf, 7.10, 10.48, inf) reproduced; the goal-inclusion conditions and the rank test; OCHS closed form (U = Null(J), U_bar = Row(U S_a), bounds on n_av, K from eq. 24, C = K U_bar, R_a completion, b_C = C v*, the force QP min lambda^T lambda + eta_a^T eta_a); results (78,000 random tests, 7-40x speedup, 100/100 block tilts); limitations; live demos: planar HFVC explorer (drag the control axis; crashing index and internal force blow up as it aligns with a constraint), the drawer example, Algorithm 1 cost on the circle with restarts, and REAL MuJoCo block tilting executed by HFVC vs pure velocity control under a block-size error slider.

**What later decks need from it.** 22 (HFVC executes plans that trajopt through contact produces), the lab.

### 22. Trajectory optimization through contact

File `tutorial/slides/22_contact_trajopt.html`; notebook `tutorial/08_contact_trajopt.py`.

**Sources.** tex 3845-3977 (the challenge, strategies, the MuJoCo MPC approach in practice, notation reference, further reading); Posa, Cantu & Tedrake 2014 (contact-implicit trajectory optimization); Mordatch et al. 2012 (contact-invariant optimization); the MJPC paper (Howell et al. 2022); Pang et al. 2023 (global planning for contact-rich manipulation via local smoothing) if you can verify it; Hou & Mason 2019/2021 as the execution layer.

**Scope.** why contact breaks gradient-based trajopt (nonsmooth dynamics, mode combinatorics) with a 1D example; strategies: contact-implicit trajopt with complementarity constraints (Posa 2014) and its MPCC difficulties; mode scheduling / hybrid trajopt; smoothing (soft contact from 19-20, randomized smoothing from 04) with a live cost-landscape figure; sampling-based MPC through contact (MJPC); REAL MuJoCo predictive sampling pushing a block to a goal with a pusher, live in the browser; planning vs execution: how HFVC (deck 21) executes a contact-rich plan; how to read a contact-rich paper with this series (a checklist mapping paper sections to decks, applied to Hou & Mason 2019 and Posa 2014); notation reference; further reading.

**What later decks need from it.** end of the series.

## Lab: `tutorial/labs/hybrid_servoing/`

A reimplementation and investigation of Hou & Mason 2019 (Algorithms 1-2, block tilting) and 2021 (OCHS, crashing index, random tests), for a student who has done decks 01-20. Deck 21 is written after it and ports what its live figures need to `lib/algo/21_hfvc.js`, with twins against `hfvc.py`.

* `hfvc.py`: model builders for the planar examples of 2021 Fig. 1 and the 3D block tilting of 2019 Sec. V (q in R^10 with a quaternion, v in R^9, Omega eq. 26, E(q) eq. 27, Phi eq. 33 with an analytic Jacobian checked by finite differences, guard conditions eq. 35-36 with 8-sided cones); 2019 Algorithm 1 (eq. 13 basis, eq. 15 cost, projected gradient descent with t = 10 and Ns restarts, R_a completion eq. 16); 2019 Algorithm 2 (eq. 20-23 KKT system and the LP over eta_af with guard conditions eq. 18); 2021 OCHS (U, U_bar, n_av bounds, K from eq. 24, C = K U_bar, R_a eq. 27, b_C eq. 29, force QP eq. 30); crashing index eq. 9; a random problem generator following 2021 Table I.
* `test_hfvc.py`: the Fig. 1 crashing indexes (2.41, 3.87, inf, 7.10, 10.48, inf) from reconstructed examples; goal inclusion and guard conditions hold for every solution; OCHS never worse than Algorithm 1 on random problems; Jacobians match finite differences.
* MuJoCo study: `tutorial/models/block_tilt.xml` (75 mm cube, point finger on three slide joints) and `sim_block_tilt.py` running the plan under HFVC, pure velocity control and pure force control while perturbing block size, table height and friction; measure peak internal force, contact-mode violations and success; serial, minutes, each CSV row fsynced.
* Cells in `07_hybrid_servoing.py`: Fig. 1 table, the block-tilting solution, a small Table II-style comparison, the MuJoCo study's plots.
* `lab.py`: marimo exercises mapped to the study-plan weeks, each a stub with an automatic check and a solution toggle; runs top to bottom with stubs unfilled.
* `README.md` handout: goals, prerequisite decks, setup, tasks, deliverables, rubric, exam-style questions per week, compute budget.
* Source to cross-check conventions: the authors' MATLAB code, github.com/yifan-hou/pub-icra19-hybrid-control.

## Integration of the existing decks 11-16

For each of 11-16: titles and footer names from `AGENTS.md` section 3; title slides say "Part IV, deck NN"; deck 11's six-part overview slide replaced by one placing it in the 22-deck series; an id and 60-200 words of notes on every slide; every writing-rule violation fixed (e.g. "Coulomb's law is a model, not a law of nature", "Surprise: contact area doesn't matter", "How good is a grasp?", summary slides replaced by "Results used in later decks"); cross-links to the owning new decks instead of the old `05_contact_solvers.py` mentions; `data-code` on every computed number.

Hou & Mason hooks: in 12, the 8-sided inscribed pyramid Hou uses (d_i = [sin(pi i/4), cos(pi i/4), 0]) and its worst-case error; in 13, quasi-static equilibrium, actuated vs unactuated rows, the four-legged table as statical indeterminacy and the minimum-norm force choice of Hou 2019 Algorithm 2; in 14-16, force closure vs a hybrid servo's guard conditions, and the grasp-quality LP vs the LP of Algorithm 2.

MuJoCo demos: 11, the tilting board sliding at arctan(mu) and the soft-contact creep below it; 13, the force distribution MuJoCo picks for the four-legged table vs the minimum-norm one; 14, a two-finger pinch that slips exactly when the Nguyen test fails.
