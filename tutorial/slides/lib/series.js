/* series.js — the deck list of the series, for footers, prev/next links and the series menu.
 *
 * window.SERIES is an array of the 22 decks in reading order:
 *   { n: 5, file: "05_unconstrained_optimization.html", title: "...", short: "...",
 *     part: "II", status: "planned" | "exists" }
 * plus two properties:
 *   SERIES.title  the series name
 *   SERIES.parts  [{ id: "I", title: "Mathematical toolkit" }, ...] in order
 *
 * deck.js loads this file itself. It links a deck from footers and the menu only when its
 * status is not "planned", so a deck writer flips "planned" to "exists" when the file lands
 * (and updates the status column of AGENTS.md section 3 at the same time).
 */
(function () {
  "use strict";
  const parts = [
    { id: "I", title: "Mathematical toolkit" },
    { id: "II", title: "Optimization" },
    { id: "III", title: "Trajectory optimization and control" },
    { id: "IV", title: "Contact statics and grasping" },
    { id: "V", title: "Contact dynamics and simulation" },
    { id: "VI", title: "Synthesis" },
  ];
  const d = (n, file, title, short, part, status) => ({ n, file, title, short, part, status });
  const decks = [
    d(1, "01_vectors_and_matrices.html", "Vectors, matrices, and linear maps", "Vectors and matrices", "I", "planned"),
    d(2, "02_eigenvalues_and_least_squares.html", "Eigenvalues, singular values, and least squares", "Eigenvalues and least squares", "I", "planned"),
    d(3, "03_derivatives_and_jacobians.html", "Derivatives, gradients, and Jacobians", "Derivatives and Jacobians", "I", "planned"),
    d(4, "04_probability_and_sampling.html", "Probability, Gaussians, and sampling", "Probability and sampling", "I", "planned"),
    d(5, "05_unconstrained_optimization.html", "Unconstrained optimization: gradient descent and Newton's method", "Unconstrained optimization", "II", "planned"),
    d(6, "06_constrained_optimization.html", "Constrained optimization: Lagrange multipliers and the KKT conditions", "Constrained optimization", "II", "planned"),
    d(7, "07_convexity_and_duality.html", "Convexity, duality, and complementarity", "Convexity and duality", "II", "planned"),
    d(8, "08_optimal_control_and_lqr.html", "Optimal control, dynamic programming, and LQR", "Optimal control and LQR", "III", "planned"),
    d(9, "09_ddp_and_ilqr.html", "Differential dynamic programming and iLQR", "DDP and iLQR", "III", "planned"),
    d(10, "10_sampling_mpc.html", "Sampling-based model predictive control and MuJoCo MPC", "Sampling-based MPC", "III", "planned"),
    d(11, "11_coulomb_friction.html", "Forces and Coulomb friction", "Coulomb friction", "IV", "exists"),
    d(12, "12_friction_cone_3d.html", "The friction cone in 3D", "Friction cone in 3D", "IV", "exists"),
    d(13, "13_contact_wrench_cone.html", "Multiple contacts and the contact wrench cone", "Contact wrench cone", "IV", "exists"),
    d(14, "14_planar_force_closure.html", "Planar force closure with two fingers", "Planar force closure", "IV", "exists"),
    d(15, "15_grasps_in_3d.html", "Grasps in 3D", "Grasps in 3D", "IV", "exists"),
    d(16, "16_grasp_quality.html", "Grasp quality metrics", "Grasp quality", "IV", "exists"),
    d(17, "17_rigid_body_motion.html", "Rigid-body motion: rotations, quaternions, twists, and wrenches", "Rigid-body motion", "V", "planned"),
    d(18, "18_contact_dynamics.html", "Rigid-body dynamics with contact", "Contact dynamics", "V", "planned"),
    d(19, "19_contact_solvers.html", "Contact solvers: LCP, CCP, and MuJoCo's soft contact", "Contact solvers", "V", "planned"),
    d(20, "20_complementarity_free_contact.html", "Complementarity-free contact", "Complementarity-free contact", "V", "planned"),
    d(21, "21_hybrid_force_velocity_control.html", "Hybrid force-velocity control", "Hybrid force-velocity control", "VI", "planned"),
    d(22, "22_contact_trajopt.html", "Trajectory optimization through contact", "Trajectory optimization through contact", "VI", "planned"),
  ];
  decks.title = "Optimization, Contact, and Control from Scratch";
  decks.parts = parts;
  window.SERIES = decks;
})();
