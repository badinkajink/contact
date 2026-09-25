/* num.js — small dense numerics for the live figures.
 *
 * Load with <script src="lib/num.js"></script>. It is a classic script with no dependencies,
 * works from file://, and defines one global, window.Num. Problems are small and dense
 * (n up to about 60). The twin tests in tools/twins/num.json check it against numpy and scipy:
 *
 *     uv run tools/twins.py num
 *
 * ======================================================================================
 * CONVENTIONS
 * ======================================================================================
 * A matrix is an array of row arrays, A[i][j] = row i, column j. A vector is a plain array.
 * No function modifies its arguments; every result is a fresh array. Shape mismatches throw
 * an Error whose message names the function.
 *
 * Bases of subspaces (nullspace, rowspace, colspace, leftNullspace, gramSchmidt().Q,
 * eigSym().vectors) are returned as an ARRAY OF BASIS VECTORS, i.e. the vectors are the ROWS
 * of the returned matrix. This is Hou & Mason's Null(.) convention: N = Num.nullspace(A) has
 * orthonormal rows and A N^T = 0. Transpose to get the basis vectors as columns. An empty
 * array [] means the subspace is {0}.
 *
 * Numerical rank. A singular value counts as nonzero when it exceeds
 * tol = max(m, n) * eps * sigma_max (eps = 2.2e-16), the default of numpy.linalg.matrix_rank
 * and scipy.linalg.null_space. Every function that takes `tol` (rank, nullspace, rowspace,
 * colspace, leftNullspace, pinv, lstsq, projector, project) reads it as an ABSOLUTE threshold
 * on the singular values. cond() uses the same default to decide that a matrix is singular.
 *
 * Sign conventions, chosen so that figures do not flip between frames:
 *   qr:     diag(R) >= 0 (unique thin QR when A has full column rank, as numpy's up to signs).
 *   eigSym, svd: in each eigenvector / right singular vector the entry of largest magnitude
 *           is positive (the first such entry when several tie to within 1e-12).
 *
 * Multiplier conventions (sections OPTIMIZATION): the Lagrangian of
 *   minimize (1/2) x^T H x + g^T x   s.t.  Aeq x = beq,  Aub x <= bub
 * is  L = (1/2) x^T H x + g^T x + nu^T (Aeq x - beq) + lam^T (Aub x - bub),
 * so stationarity reads H x + g + Aeq^T nu + Aub^T lam = 0 with lam >= 0.
 * LP duals follow scipy.optimize.linprog's "marginals": d(fun)/d(b).
 *
 * ======================================================================================
 * CONSTRUCTION AND ELEMENTWISE
 * ======================================================================================
 * zeros(m, n) -> m x n matrix of zeros;  zeros(n) -> vector of n zeros.
 * ones(m, n), ones(n)                    likewise with ones.
 * eye(n)                                 n x n identity.
 * diag(v) -> square matrix with v on the diagonal;  diag(A) -> vector of A's diagonal.
 * linspace(a, b, n)                      n evenly spaced numbers from a to b inclusive.
 * clone(A)                               copy of a number, vector, or matrix.
 * shape(A) -> [m, n] for a matrix, [n] for a vector.
 * transpose(A)                           A^T. A vector v gives the n x 1 column matrix.
 * add(A, B, ...), sub(A, B)              elementwise; vectors or matrices of equal shape.
 * scale(A, k)                            k A, for a vector or a matrix.
 * lincomb(coefs, vectors)                sum_i coefs[i] * vectors[i].
 * hstack(A, B, ...)                      side by side; a vector argument is a column.
 *                                        Num.hstack(A, b) is the augmented matrix [A | b].
 * vstack(A, B, ...)                      one above the other; a vector argument is a row.
 * row(A, i), col(A, j)                   copies of row i / column j as vectors.
 * submatrix(A, rows, cols)               rows, cols are index arrays; null means "all".
 * allclose(A, B, tol = 1e-9)             true when |a - b| <= tol (1 + |b|) everywhere.
 *
 * ======================================================================================
 * PRODUCTS AND NORMS
 * ======================================================================================
 * matmul(A, B, ...)   products with numpy's @ rules: matrix @ matrix -> matrix,
 *                     matrix @ vector -> vector, vector @ matrix -> vector,
 *                     vector @ vector -> number. Several arguments multiply left to right.
 * matvec(A, x)        A x.
 * dot(a, b)           a . b.
 * outer(a, b)         a b^T.
 * cross(a, b)         3D cross product; for 2D vectors, the scalar a0 b1 - a1 b0.
 * norm(v, p = 2)      vector norm; p = 2, 1, Infinity (or 'inf'), or any p > 0.
 * norm(A, p = 'fro')  matrix norm as numpy: 'fro', 2 (sigma_max), 1 (max column sum),
 *                     Infinity (max row sum).
 * normalize(v)        v / |v| (the zero vector is returned unchanged).
 * trace(A)            sum of the diagonal.
 *
 * ======================================================================================
 * LINEAR SYSTEMS
 * ======================================================================================
 * lu(A) -> {L, U, P, perm, sign}   partial pivoting, P A = L U, L unit lower triangular.
 *                     perm[i] is the row of A that became row i; sign = det(P) = +-1.
 *                     scipy.linalg.lu returns the transposed permutation (A = P L U).
 * solve(A, b)         x with A x = b, by LU. b may be a vector or a matrix (one right-hand
 *                     side per column). Returns null when A is singular to working
 *                     precision (a pivot |u_kk| <= n eps max|a_ij|).
 * inv(A)              A^{-1}, or null when singular.
 * det(A)              determinant from the LU factors.
 * chol(A) -> L        lower triangular with A = L L^T, or null when A is not positive
 *                     definite. Reads only the lower triangle (as numpy); A must be symmetric.
 * cholSolve(L, b)     solves L L^T x = b given L = chol(A).
 *
 *   Num.solve([[2, 1], [1, 3]], [3, 5])        // [0.8, 1.4]
 *   Num.chol([[1, 2], [2, 1]])                 // null: eigenvalues 3 and -1
 *
 * ======================================================================================
 * ORTHOGONALIZATION
 * ======================================================================================
 * qr(A, mode = 'thin') -> {Q, R}   Householder QR, A = Q R, diag(R) >= 0.
 *                     'thin' (numpy's 'reduced'): Q is m x k, R is k x n, k = min(m, n).
 *                     'full': Q is m x m orthogonal, R is m x n.
 * gramSchmidt(vectors, {tol = 1e-10}) -> {Q, R, rank, steps}
 *                     Orthonormalizes a LIST of vectors (pass transpose(A) for A's columns),
 *                     modified Gram-Schmidt with one reorthogonalization pass. Q holds the
 *                     orthonormal vectors (rows); v_j = sum_i R[i][j] Q[i] for independent
 *                     v_j. A vector whose residual is below tol |v_j| is dependent: it adds
 *                     no row to Q. steps[j] = {index: j, v, projections: [{onto: i, coef,
 *                     vector}], w (residual after the projections), norm: |w|, q (null when
 *                     dependent), dependent}. coef = q_i . v_j, vector = coef q_i.
 * projector(A, tol) -> P   orthogonal projector onto the column space of A (m x m).
 *                     For basis vectors stored as ROWS B, use projector(transpose(B)).
 * project(A, b, tol)  P b, the point of col(A) closest to b.
 *
 * ======================================================================================
 * EIGENVALUES, SVD, AND THE FOUR SUBSPACES
 * ======================================================================================
 * eigSym(A) -> {values, vectors, V}   symmetric eigenproblem by cyclic Jacobi rotations on
 *                     (A + A^T)/2. values descending; vectors[i] is the unit eigenvector of
 *                     values[i]; V has them as columns, so A V = V diag(values).
 * svd(A, {full: false}) -> {U, S, V}  one-sided (Hestenes) Jacobi, A = U diag(S) V^T,
 *                     S descending, k = min(m, n) values. Thin: U m x k, V n x k.
 *                     {full: true}: U m x m and V n x n (the extra columns complete the
 *                     orthonormal bases). Works for tall, square, and wide A.
 * svdvals(A)          the singular values only.
 * rank(A, tol)        number of singular values above tol (default above).
 * cond(A)             sigma_max / sigma_min over the k = min(m, n) singular values;
 *                     Infinity when rank(A) < k.
 * pinv(A, tol)        Moore-Penrose pseudoinverse V_r diag(1/sigma) U_r^T (n x m).
 * lstsq(A, b, tol) -> {x, residual, rank, S}   minimum-norm least-squares solution
 *                     x = pinv(A) b; residual = |A x - b|.
 * nullspace(A, tol)   ROWS: orthonormal basis of {v : A v = 0}; n - r rows of length n.
 * rowspace(A, tol)    ROWS: orthonormal basis of the row space; r rows of length n.
 * colspace(A, tol)    ROWS: orthonormal basis of the column space; r rows of length m.
 * leftNullspace(A, tol)   ROWS: orthonormal basis of {y : A^T y = 0}; m - r rows.
 *
 *   const N = Num.nullspace([[1, 1, 0], [0, 0, 1]]);   // [[0.7071, -0.7071, 0]]
 *   Num.matmul([[1, 1, 0], [0, 0, 1]], Num.transpose(N)) // [[0], [0]] up to 1e-16
 *   Num.cond([[1, 1], [1, 1.0001]])                      // 4.0002e4
 *
 * ======================================================================================
 * ROW REDUCTION
 * ======================================================================================
 * rref(A, {tol, pivot: 'first'}) -> {R, pivots, rank, steps}
 *                     Gauss-Jordan elimination. R is the reduced row echelon form, pivots
 *                     the pivot column indices (0-based), rank = pivots.length.
 *                     pivot 'first' takes the first row with a nonzero entry (the hand
 *                     method); 'max' takes the largest entry (partial pivoting).
 *                     Entries with |x| <= tol are zero; tol defaults to 1e-10 max|a_ij|.
 *                     steps lists every elementary row operation in order:
 *                       {op: 'swap',  i, j,    text: 'R1 <-> R2',       tex, matrix, pivot}
 *                       {op: 'scale', i, k,    text: 'R1 <- (1/2) R1',  tex, matrix, pivot}
 *                       {op: 'add',   i, j, k, text: 'R2 <- R2 - 3 R1', tex, matrix, pivot}
 *                     i, j are 0-based rows, k the factor (row_i <- row_i + k row_j for
 *                     'add'), text is plain Unicode with 1-based rows, tex is KaTeX source,
 *                     matrix the matrix AFTER the step, pivot = [row, col] being cleared.
 *                     Operations with factor 1 (scale) or 0 (add) are not logged.
 *
 *   const r = Num.rref([[1, 2, 3], [2, 4, 7]]);
 *   r.R        // [[1, 2, 0], [0, 0, 1]];  r.pivots // [0, 2]
 *   r.steps[0].text  // "R2 ← R2 − 2 R1"
 *
 * ======================================================================================
 * RANDOM NUMBERS
 * ======================================================================================
 * rng(seed = 1) -> generator; a deterministic stream (mulberry32; normals by Box-Muller).
 *   .uniform()            in [0, 1);   .uniform(a, b) in [a, b).
 *   .normal()             standard normal;  .normal(mu, sigma).
 *   .int(k)               integer in {0, ..., k - 1}.
 *   .normalVec(n)         n independent standard normals.
 *   .uniformVec(n, a = 0, b = 1)
 *   .sphere(n)            uniform unit vector in R^n (a normalized normalVec).
 *   .mvnormal(mean, cov)  mean + L z with L = chol(cov), z = normalVec; a PSD cov without a
 *                         Cholesky factor uses L = V diag(sqrt(max(values, 0))).
 *   .shuffle(arr)         shuffled COPY (Fisher-Yates, j = int(i + 1) for i = n-1 .. 1).
 * Normals come in Box-Muller pairs: u1 = 1 - uniform(), u2 = uniform(),
 * r = sqrt(-2 ln u1), first r cos(2 pi u2), then (next call) r sin(2 pi u2).
 * tools/twins/num.json holds a Python port that reproduces the stream bit for bit, so a
 * notebook can regenerate a slide's random data exactly.
 *
 *   const R = Num.rng(42);  const z = R.normalVec(3);
 *
 * ======================================================================================
 * OPTIMIZATION
 * ======================================================================================
 * lp({c, Aub, bub, Aeq, beq, bounds}, {tol = 1e-9}) ->
 *     {status, x, fun, duals: {ub, eq, lower, upper}, iterations}
 *                     minimize c^T x s.t. Aub x <= bub, Aeq x = beq, lo <= x <= hi.
 *                     Dense two-phase simplex with Bland's rule (no cycling).
 *                     bounds follows scipy.optimize.linprog: the DEFAULT IS x >= 0.
 *                     bounds: [lo, hi] for every variable, or [[lo0, hi0], [lo1, hi1], ...];
 *                     null (or +-Infinity) means unbounded on that side, so
 *                     bounds: [null, null] makes every variable free.
 *                     status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration_limit'.
 *                     Infeasible gives x = null, fun = Infinity; unbounded gives x = null,
 *                     fun = -Infinity. duals are scipy's marginals d(fun)/d(rhs): ub <= 0,
 *                     lower >= 0, upper <= 0 (unique only for nondegenerate problems).
 * feasible({Aub, bub, Aeq, beq, bounds, n}) -> {feasible, x, status}
 *                     LP phase 1 only. Same bounds default as lp (x >= 0). n is needed only
 *                     when there are no constraint rows to read it from.
 * qp({H, g, Aeq, beq, Aub, bub}, {tol = 1e-9}) ->
 *     {status, x, fun, lamEq, lamUb, activeUb, iterations}
 *                     minimize (1/2) x^T H x + g^T x s.t. Aeq x = beq, Aub x <= bub, with H
 *                     symmetric positive SEMIdefinite (x is free; add bounds as Aub rows).
 *                     Primal active-set method started from an LP phase-1 point; each step
 *                     solves the equality-constrained subproblem exactly on the null space of
 *                     the working constraints, so x is exact to rounding (about 1e-14).
 *                     status: 'optimal' | 'infeasible' | 'unbounded' | 'nonconvex' |
 *                     'iteration_limit'. Infeasibility and unboundedness are decided by
 *                     simplex LPs, so both are exact. lamEq, lamUb follow the Lagrangian above
 *                     (lamUb >= 0, zero on inactive rows); activeUb lists the active rows.
 * eqQP(H, g, A, b) -> {x, nu, singular}
 *                     minimize (1/2) x^T H x + g^T x s.t. A x = b by ONE linear solve of the
 *                     KKT system [H A^T; A 0] [x; nu] = [-g; b]. Hou & Mason 2019 eq. 22
 *                     (min f^T f s.t. M f = b) is eqQP(2I, 0, M, b). When the KKT matrix is
 *                     singular, the minimum-norm least-squares solution is returned and
 *                     singular = true.
 * nnls(A, b) -> {x, residual, iterations}   min |A x - b| s.t. x >= 0 (Lawson-Hanson, as
 *                     scipy.optimize.nnls); residual = |A x - b|.
 *
 *   Num.lp({c: [-1, -2], Aub: [[1, 1], [1, -1]], bub: [4, 2]})   // x = [0, 4], fun = -8
 *   Num.qp({H: [[2, 0], [0, 2]], g: [-2, -5], Aub: [[1, 2]], bub: [3]})
 *   Num.eqQP(Num.scale(Num.eye(3), 2), [0, 0, 0], [[1, 1, 1]], [3])   // x = [1, 1, 1]
 *
 * ======================================================================================
 * DERIVATIVES BY CENTRAL DIFFERENCES
 * ======================================================================================
 * derivFD(f, x, h)    f'(x) for scalar f of a scalar, (f(x+h) - f(x-h)) / 2h.
 * gradFD(f, x, h)     gradient of f: R^n -> R.
 * jacFD(F, x, h)      m x n Jacobian of F: R^n -> R^m, J[i][j] = dF_i/dx_j.
 * hessFD(f, x, h)     n x n Hessian of f (symmetric by construction): second differences
 *                     (f(x+h e_i) - 2 f(x) + f(x-h e_i)) / h^2 on the diagonal and the
 *                     four-point formula off it.
 * h may be a number (absolute step) or omitted. Omitted, h_i = c max(1, |x_i|) with
 * c = eps^(1/3) = 6.1e-6 for first derivatives (error about 1e-10) and c = eps^(1/4) = 1.2e-4
 * for the Hessian (error about 1e-7).
 *
 * ======================================================================================
 * FORMATTING
 * ======================================================================================
 * fmt(x, d = 3)       number, vector, or matrix as text with d decimals; a matrix becomes
 *                     aligned rows separated by "\n" (for a monospace .readout).
 * frac(x, maxDen = 1000)   "p/q" when x is within 1e-9 of a fraction with q <= maxDen,
 *                     "p" for integers, otherwise 4 significant digits.
 * tex(A, {digits})    KaTeX source for a number, vector (a column), or matrix (bmatrix).
 *                     Without digits, entries are fractions where possible (\tfrac{1}{3}).
 */
(function (root) {
  "use strict";

  const EPS = 2.220446049250313e-16;

  // =====================================================================================
  // Shapes and construction
  // =====================================================================================

  const isArr = (x) => Array.isArray(x) || ArrayBuffer.isView(x);
  const isMat = (A) => Array.isArray(A) && A.length > 0 && isArr(A[0]);
  const range = (n) => Array.from({ length: n }, (_, i) => i);

  function fail(fn, msg) {
    throw new Error(`Num.${fn}: ${msg}`);
  }

  function needMat(A, fn) {
    if (!isMat(A) || A[0].length === 0) fail(fn, "expects a non-empty matrix (an array of row arrays)");
    const n = A[0].length;
    for (const r of A) if (r.length !== n) fail(fn, "rows have different lengths");
    return [A.length, n];
  }

  function zeros(m, n) {
    if (n === undefined) return new Array(m).fill(0);
    return Array.from({ length: m }, () => new Array(n).fill(0));
  }

  function ones(m, n) {
    if (n === undefined) return new Array(m).fill(1);
    return Array.from({ length: m }, () => new Array(n).fill(1));
  }

  function eye(n) {
    const I = zeros(n, n);
    for (let i = 0; i < n; i++) I[i][i] = 1;
    return I;
  }

  function diag(v) {
    if (isMat(v)) {
      const k = Math.min(v.length, v[0].length);
      return range(k).map((i) => v[i][i]);
    }
    const D = zeros(v.length, v.length);
    for (let i = 0; i < v.length; i++) D[i][i] = v[i];
    return D;
  }

  function linspace(a, b, n) {
    if (n === 1) return [a];
    return range(n).map((i) => (i === n - 1 ? b : a + ((b - a) * i) / (n - 1)));
  }

  function clone(A) {
    if (!isArr(A)) return A;
    return Array.from(A, (r) => (isArr(r) ? Array.from(r) : r));
  }

  function shape(A) {
    return isMat(A) ? [A.length, A[0].length] : [A.length];
  }

  function transpose(A) {
    if (!isMat(A)) return Array.from(A, (x) => [x]);
    const m = A.length, n = A[0].length, T = zeros(n, m);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
    return T;
  }

  function elementwise(A, B, f, fn) {
    if (isArr(A)) {
      if (!isArr(B) || A.length !== B.length) fail(fn, "shapes do not match");
      return Array.from(A, (a, i) => elementwise(a, B[i], f, fn));
    }
    if (isArr(B)) fail(fn, "shapes do not match");
    return f(A, B);
  }

  function add(...args) {
    let R = args[0];
    for (let k = 1; k < args.length; k++) R = elementwise(R, args[k], (a, b) => a + b, "add");
    return R;
  }

  const sub = (A, B) => elementwise(A, B, (a, b) => a - b, "sub");

  function scale(A, k) {
    return isArr(A) ? Array.from(A, (a) => scale(a, k)) : A * k;
  }

  function lincomb(coefs, vectors) {
    if (coefs.length !== vectors.length) fail("lincomb", "needs one coefficient per vector");
    let out = zeros(vectors[0].length);
    for (let i = 0; i < coefs.length; i++) out = add(out, scale(vectors[i], coefs[i]));
    return out;
  }

  function hstack(...blocks) {
    const mats = blocks.map((B) => (isMat(B) ? B : transpose(B)));
    const m = mats[0].length;
    for (const B of mats) if (B.length !== m) fail("hstack", "blocks have different row counts");
    return range(m).map((i) => [].concat(...mats.map((B) => Array.from(B[i]))));
  }

  function vstack(...blocks) {
    const mats = blocks.map((B) => (isMat(B) ? B : [Array.from(B)]));
    const n = mats[0][0].length;
    for (const B of mats) if (B[0].length !== n) fail("vstack", "blocks have different column counts");
    return [].concat(...mats.map((B) => clone(B)));
  }

  const row = (A, i) => Array.from(A[i]);
  const col = (A, j) => A.map((r) => r[j]);

  function submatrix(A, rows, cols) {
    const R = rows || range(A.length);
    const C = cols || range(A[0].length);
    return R.map((i) => C.map((j) => A[i][j]));
  }

  function allclose(A, B, tol) {
    tol = tol === undefined ? 1e-9 : tol;
    if (isArr(A) || isArr(B)) {
      if (!isArr(A) || !isArr(B) || A.length !== B.length) return false;
      for (let i = 0; i < A.length; i++) if (!allclose(A[i], B[i], tol)) return false;
      return true;
    }
    if (A === B) return true;
    return Math.abs(A - B) <= tol * (1 + Math.abs(B));
  }

  function maxAbs(A) {
    let s = 0;
    for (const r of A) {
      if (isArr(r)) {
        for (const x of r) if (Math.abs(x) > s) s = Math.abs(x);
      } else if (Math.abs(r) > s) s = Math.abs(r);
    }
    return s;
  }

  // The power of two 2^e <= max|a_ij| < 2^(e+1), or 1 for a zero or non-finite matrix.
  // Dividing a matrix by it is exact, and it keeps the squares and products of entries that
  // qr, eigSym, and svd form away from overflow (|a| > 1e77) and underflow (|a| < 1e-77).
  function pow2Scale(A) {
    const s = maxAbs(A);
    if (!(s > 0) || s === Infinity) return 1;
    return Math.pow(2, Math.floor(Math.log2(s)));
  }
  const divideBy = (A, s) => A.map((r) => Array.from(r, (x) => x / s));

  // =====================================================================================
  // Products and norms
  // =====================================================================================

  function dot(a, b) {
    if (a.length !== b.length) fail("dot", `lengths ${a.length} and ${b.length} differ`);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function matvec(A, x) {
    const n = A[0].length;
    if (x.length !== n) fail("matvec", `matrix has ${n} columns, vector has ${x.length} entries`);
    return A.map((r) => {
      let s = 0;
      for (let j = 0; j < n; j++) s += r[j] * x[j];
      return s;
    });
  }

  function vecmat(x, A) {
    const m = A.length, n = A[0].length;
    if (x.length !== m) fail("matmul", `vector has ${x.length} entries, matrix has ${m} rows`);
    const out = zeros(n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j] += x[i] * A[i][j];
    return out;
  }

  function mm(A, B) {
    const aM = isMat(A), bM = isMat(B);
    if (aM && bM) {
      const m = A.length, k = A[0].length, n = B[0].length;
      if (B.length !== k) fail("matmul", `shapes ${m}x${k} and ${B.length}x${n} do not match`);
      const C = zeros(m, n);
      for (let i = 0; i < m; i++) {
        const Ai = A[i], Ci = C[i];
        for (let p = 0; p < k; p++) {
          const a = Ai[p], Bp = B[p];
          for (let j = 0; j < n; j++) Ci[j] += a * Bp[j];
        }
      }
      return C;
    }
    if (aM) return matvec(A, B);
    if (bM) return vecmat(A, B);
    return dot(A, B);
  }

  function matmul(...args) {
    let R = args[0];
    for (let i = 1; i < args.length; i++) R = mm(R, args[i]);
    return R;
  }

  const outer = (a, b) => Array.from(a, (x) => Array.from(b, (y) => x * y));

  function cross(a, b) {
    if (a.length === 2) return a[0] * b[1] - a[1] * b[0];
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }

  // Euclidean norm, scaled by the largest entry so that tiny or huge entries do not
  // underflow or overflow when squared.
  function norm2(x) {
    let s = 0;
    for (const v of x) {
      const a = Math.abs(v);
      if (a > s) s = a;
      else if (a !== a) return NaN;
    }
    if (s === 0 || s === Infinity) return s;
    let t = 0;
    for (const v of x) {
      const r = v / s;
      t += r * r;
    }
    return s * Math.sqrt(t);
  }

  function norm(x, p) {
    if (isMat(x)) {
      if (p === undefined || p === "fro") return norm2([].concat(...x.map((r) => Array.from(r))));
      if (p === 2) return svdvals(x)[0];
      if (p === 1) return Math.max(...range(x[0].length).map((j) => x.reduce((s, r) => s + Math.abs(r[j]), 0)));
      if (p === Infinity || p === "inf") return Math.max(...x.map((r) => r.reduce((s, v) => s + Math.abs(v), 0)));
      fail("norm", `unknown matrix norm ${p}`);
    }
    if (p === undefined || p === 2) return norm2(x);
    if (p === 1) {
      let s = 0;
      for (const v of x) s += Math.abs(v);
      return s;
    }
    if (p === Infinity || p === "inf") {
      let s = 0;
      for (const v of x) s = Math.max(s, Math.abs(v));
      return s;
    }
    if (typeof p === "number" && p > 0) {
      let s = 0;
      for (const v of x) s += Math.pow(Math.abs(v), p);
      return Math.pow(s, 1 / p);
    }
    fail("norm", `unknown vector norm ${p}`);
  }

  function normalize(v) {
    const n = norm2(v);
    return n > 0 ? Array.from(v, (x) => x / n) : Array.from(v);
  }

  function trace(A) {
    let s = 0;
    for (let i = 0; i < Math.min(A.length, A[0].length); i++) s += A[i][i];
    return s;
  }

  // =====================================================================================
  // Linear systems: LU with partial pivoting, Cholesky
  // =====================================================================================

  function lu(A) {
    const [m, n] = needMat(A, "lu");
    if (m !== n) fail("lu", "needs a square matrix");
    const U = clone(A), L = eye(n), perm = range(n);
    let sign = 1;
    for (let k = 0; k < n; k++) {
      // Pivot: the largest |entry| on or below the diagonal in column k (first one on ties).
      let p = k, best = Math.abs(U[k][k]);
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(U[i][k]) > best) {
          best = Math.abs(U[i][k]);
          p = i;
        }
      }
      if (p !== k) {
        [U[k], U[p]] = [U[p], U[k]];
        [perm[k], perm[p]] = [perm[p], perm[k]];
        for (let j = 0; j < k; j++) [L[k][j], L[p][j]] = [L[p][j], L[k][j]];
        sign = -sign;
      }
      const piv = U[k][k];
      if (piv === 0) continue; // the whole column below is zero: nothing to eliminate
      for (let i = k + 1; i < n; i++) {
        const f = U[i][k] / piv;
        L[i][k] = f;
        U[i][k] = 0;
        if (f !== 0) for (let j = k + 1; j < n; j++) U[i][j] -= f * U[k][j];
      }
    }
    const P = zeros(n, n);
    for (let i = 0; i < n; i++) P[i][perm[i]] = 1;
    return { L, U, P, perm, sign };
  }

  function luSolveVec(F, b) {
    const { L, U, perm } = F, n = L.length;
    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[perm[i]];
      for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
      y[i] = s;
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < n; j++) s -= U[i][j] * x[j];
      x[i] = s / U[i][i];
    }
    return x;
  }

  function solve(A, b) {
    const F = lu(A), n = A.length;
    const tiny = n * EPS * maxAbs(A);
    for (let i = 0; i < n; i++) if (!(Math.abs(F.U[i][i]) > tiny)) return null;
    if (isMat(b)) {
      if (b.length !== n) fail("solve", "right-hand side has the wrong number of rows");
      return transpose(transpose(b).map((c) => luSolveVec(F, c)));
    }
    if (b.length !== n) fail("solve", `matrix is ${n}x${n}, right-hand side has ${b.length} entries`);
    return luSolveVec(F, b);
  }

  const inv = (A) => solve(A, eye(A.length));

  function det(A) {
    const F = lu(A);
    let d = F.sign;
    for (let i = 0; i < A.length; i++) d *= F.U[i][i];
    return d;
  }

  function chol(A) {
    const [m, n] = needMat(A, "chol");
    if (m !== n) fail("chol", "needs a square matrix");
    const L = zeros(n, n);
    for (let j = 0; j < n; j++) {
      let s = A[j][j];
      for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
      if (!(s > 0)) return null;
      const d = Math.sqrt(s);
      L[j][j] = d;
      for (let i = j + 1; i < n; i++) {
        let t = A[i][j];
        for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
        L[i][j] = t / d;
      }
    }
    return L;
  }

  function cholSolve(L, b) {
    const n = L.length, y = new Array(n), x = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  // =====================================================================================
  // Orthogonalization: Householder QR, Gram-Schmidt, projectors
  // =====================================================================================

  function qr(A, mode) {
    const [m, n] = needMat(A, "qr");
    const full = mode === "full";
    if (mode !== undefined && mode !== "full" && mode !== "thin") fail("qr", "mode is 'thin' or 'full'");
    const k = Math.min(m, n);
    const sc = pow2Scale(A); // A = sc (A / sc) exactly; R is scaled back at the end
    const R = divideBy(A, sc), Q = eye(m);
    for (let j = 0; j < Math.min(m - 1, n); j++) {
      // Reflect x = R[j:, j] onto -sign(x0) |x| e1 (the sign choice avoids cancellation).
      const v = range(m - j).map((i) => R[j + i][j]);
      const nx = norm2(v);
      if (nx === 0) continue;
      const alpha = v[0] > 0 ? -nx : nx;
      v[0] -= alpha;
      const vv = dot(v, v);
      if (vv === 0) continue;
      for (let c = j; c < n; c++) {
        let t = 0;
        for (let i = 0; i < v.length; i++) t += v[i] * R[j + i][c];
        t = (2 * t) / vv;
        for (let i = 0; i < v.length; i++) R[j + i][c] -= t * v[i];
      }
      for (let r = 0; r < m; r++) {
        let t = 0;
        for (let i = 0; i < v.length; i++) t += Q[r][j + i] * v[i];
        t = (2 * t) / vv;
        for (let i = 0; i < v.length; i++) Q[r][j + i] -= t * v[i];
      }
      R[j][j] = alpha;
      for (let i = j + 1; i < m; i++) R[i][j] = 0;
    }
    for (let i = 0; i < k; i++) {
      if (R[i][i] < 0) {
        for (let c = 0; c < n; c++) R[i][c] = -R[i][c];
        for (let r = 0; r < m; r++) Q[r][i] = -Q[r][i];
      }
    }
    if (sc !== 1) for (const r of R) for (let c = 0; c < n; c++) r[c] *= sc;
    if (full) return { Q, R };
    return { Q: Q.map((r) => r.slice(0, k)), R: R.slice(0, k) };
  }

  // Orthonormal basis (rows) of the orthogonal complement of span(B), where the rows of B are
  // orthonormal vectors of length m.
  function complement(B, m) {
    if (B.length === 0) return eye(m);
    const { Q } = qr(transpose(B), "full");
    return range(m - B.length).map((i) => col(Q, B.length + i));
  }

  function gramSchmidt(vectors, opts) {
    const tol = opts && opts.tol !== undefined ? opts.tol : 1e-10;
    const Q = [], steps = [], coefs = [];
    for (let j = 0; j < vectors.length; j++) {
      const v = Array.from(vectors[j]);
      let w = v.slice();
      const c = new Array(Q.length).fill(0);
      for (let pass = 0; pass < 2; pass++) {
        for (let i = 0; i < Q.length; i++) {
          const a = dot(Q[i], w);
          c[i] += a;
          w = w.map((x, t) => x - a * Q[i][t]);
        }
      }
      const nw = norm2(w), nv = norm2(v);
      const dependent = !(nw > tol * nv) || nv === 0;
      const projections = c.map((coef, i) => ({ onto: i, coef, vector: scale(Q[i], coef) }));
      const q = dependent ? null : w.map((x) => x / nw);
      steps.push({ index: j, v, projections, w, norm: nw, q, dependent });
      if (!dependent) {
        c.push(nw);
        Q.push(q);
      }
      coefs.push(c);
    }
    const r = Q.length;
    const R = zeros(r, vectors.length);
    coefs.forEach((c, j) => c.forEach((x, i) => (R[i][j] = x)));
    return { Q, R, rank: r, steps };
  }

  function projector(A, tol) {
    const B = colspace(A, tol);
    const m = A.length, P = zeros(m, m);
    for (const u of B) for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) P[i][j] += u[i] * u[j];
    return P;
  }

  function project(A, b, tol) {
    const B = colspace(A, tol);
    let p = zeros(A.length);
    for (const u of B) p = add(p, scale(u, dot(u, b)));
    return p;
  }

  // =====================================================================================
  // Symmetric eigenproblem: cyclic Jacobi
  // =====================================================================================

  // Make the entry of largest magnitude positive (the first one among near ties).
  function largestIndex(v) {
    let k = 0;
    for (let i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[k]) + 1e-12) k = i;
    return k;
  }
  const signFix = (v) => (v[largestIndex(v)] < 0 ? v.map((x) => -x) : v);

  function eigSym(A) {
    const [m, n] = needMat(A, "eigSym");
    if (m !== n) fail("eigSym", "needs a square matrix");
    const sc = pow2Scale(A); // rotate A / sc, then scale the eigenvalues back
    const a = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) a[i][j] = 0.5 * (A[i][j] / sc + A[j][i] / sc);
    const V = eye(n);
    const floor = EPS * 1e-3 * norm(a);
    for (let sweep = 0; sweep < 60; sweep++) {
      let rotated = false;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p][q];
          if (Math.abs(apq) <= floor || Math.abs(apq) <= EPS * Math.sqrt(Math.abs(a[p][p] * a[q][q]))) {
            a[p][q] = a[q][p] = 0;
            continue;
          }
          rotated = true;
          // Rotation in the (p, q) plane that zeroes a_pq: cot(2 phi) = theta, t = tan(phi).
          const theta = (a[q][q] - a[p][p]) / (2 * apq);
          const t = Math.abs(theta) > 1e150
            ? 1 / (2 * theta)
            : (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            if (k === p || k === q) continue;
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = a[p][k] = c * akp - s * akq;
            a[k][q] = a[q][k] = s * akp + c * akq;
          }
          a[p][p] -= t * apq;
          a[q][q] += t * apq;
          a[p][q] = a[q][p] = 0;
          for (let k = 0; k < n; k++) {
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
          }
        }
      }
      if (!rotated) break;
    }
    const order = range(n).sort((i, j) => a[j][j] - a[i][i]);
    const values = order.map((i) => a[i][i] * sc);
    const vectors = order.map((i) => signFix(col(V, i)));
    return { values, vectors, V: transpose(vectors) };
  }

  // =====================================================================================
  // Singular value decomposition: one-sided Jacobi (Hestenes)
  // =====================================================================================

  function svd(A, opts) {
    const [m, n] = needMat(A, "svd");
    const full = !!(opts && opts.full);
    // Factor A / sc (exact) so that the sums of squares in svdTall neither overflow nor
    // underflow, then scale the singular values back.
    const sc = pow2Scale(A);
    let r;
    if (m < n) {
      // Work on the tall matrix A^T = U' S V'^T, so A = V' S U'^T.
      const t = svdTall(transpose(divideBy(A, sc)), full);
      r = { U: t.V, S: t.S, V: t.U };
    } else r = svdTall(divideBy(A, sc), full);
    if (sc !== 1) r.S = r.S.map((x) => x * sc);
    return fixSvdSigns(r);
  }

  // One-sided Jacobi on a tall (m >= n) matrix with entries of order 1; no sign convention.
  function svdTall(A, full) {
    const m = A.length, n = A[0].length;
    // Rotate pairs of columns of W = A V until every pair is orthogonal; then
    // sigma_i = |w_i|, u_i = w_i / sigma_i. W[j] holds column j, Vc[j] column j of V.
    const W = transpose(A), Vc = eye(n);
    const small = EPS * norm(A); // columns shorter than this are numerically zero
    for (let sweep = 0; sweep < 60; sweep++) {
      let rotated = false;
      for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < n; j++) {
          const wi = W[i], wj = W[j];
          let alpha = 0, beta = 0, gamma = 0;
          for (let k = 0; k < m; k++) {
            alpha += wi[k] * wi[k];
            beta += wj[k] * wj[k];
            gamma += wi[k] * wj[k];
          }
          if (Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
          if (Math.min(alpha, beta) <= small * small) continue;
          rotated = true;
          const zeta = (beta - alpha) / (2 * gamma);
          const t = Math.abs(zeta) > 1e150
            ? 1 / (2 * zeta)
            : (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (let k = 0; k < m; k++) {
            const x = wi[k], y = wj[k];
            wi[k] = c * x - s * y;
            wj[k] = s * x + c * y;
          }
          const vi = Vc[i], vj = Vc[j];
          for (let k = 0; k < n; k++) {
            const x = vi[k], y = vj[k];
            vi[k] = c * x - s * y;
            vj[k] = s * x + c * y;
          }
        }
      }
      if (!rotated) break;
    }
    const sig = W.map(norm2);
    const order = range(n).sort((a, b) => sig[b] - sig[a]);
    const S = order.map((i) => sig[i]);
    const Vcols = order.map((i) => Vc[i]);
    // Left singular vectors of (numerically) zero singular values are noise; replace them
    // by an orthonormal completion of the others.
    const cut = Math.max(m, n) * EPS * S[0];
    let r = 0;
    while (r < n && S[r] > cut) r++;
    const Ucols = range(r).map((i) => W[order[i]].map((x) => x / S[i]));
    const want = full ? m : n;
    if (Ucols.length < want) {
      const C = complement(Ucols, m);
      for (let i = 0; Ucols.length < want; i++) Ucols.push(C[i]);
    }
    return { U: transpose(Ucols), S, V: transpose(Vcols) };
  }

  function fixSvdSigns(res) {
    const { U, S, V } = res;
    const k = S.length;
    const flipCol = (M, j) => M.forEach((r) => (r[j] = -r[j]));
    for (let j = 0; j < V[0].length; j++) {
      const v = col(V, j);
      if (v[largestIndex(v)] < 0) {
        flipCol(V, j);
        if (j < k) flipCol(U, j);
      }
    }
    for (let j = k; j < U[0].length; j++) {
      const u = col(U, j);
      if (u[largestIndex(u)] < 0) flipCol(U, j);
    }
    return res;
  }

  const svdvals = (A) => svd(A).S;
  const defaultTol = (S, m, n) => Math.max(m, n) * EPS * (S[0] || 0);

  function rankFrom(S, m, n, tol) {
    const t = tol === undefined || tol === null ? defaultTol(S, m, n) : tol;
    return S.filter((s) => s > t).length;
  }

  function rank(A, tol) {
    const [m, n] = needMat(A, "rank");
    return rankFrom(svdvals(A), m, n, tol);
  }

  function cond(A) {
    const [m, n] = needMat(A, "cond");
    const S = svdvals(A), k = S.length;
    if (rankFrom(S, m, n) < k) return Infinity;
    return S[0] / S[k - 1];
  }

  function pinv(A, tol) {
    const [m, n] = needMat(A, "pinv");
    const { U, S, V } = svd(A);
    const r = rankFrom(S, m, n, tol);
    const P = zeros(n, m);
    for (let t = 0; t < r; t++) {
      for (let i = 0; i < n; i++) {
        const f = V[i][t] / S[t];
        for (let j = 0; j < m; j++) P[i][j] += f * U[j][t];
      }
    }
    return P;
  }

  function lstsq(A, b, tol) {
    const [m, n] = needMat(A, "lstsq");
    if (b.length !== m) fail("lstsq", `matrix has ${m} rows, b has ${b.length} entries`);
    const { U, S, V } = svd(A);
    const r = rankFrom(S, m, n, tol);
    const x = zeros(n);
    for (let t = 0; t < r; t++) {
      let c = 0;
      for (let j = 0; j < m; j++) c += U[j][t] * b[j];
      c /= S[t];
      for (let i = 0; i < n; i++) x[i] += c * V[i][t];
    }
    const residual = norm2(sub(matvec(A, x), b));
    return { x, residual, rank: r, S };
  }

  function nullspace(A, tol) {
    const [m, n] = needMat(A, "nullspace");
    const { S, V } = svd(A, { full: true });
    const r = rankFrom(S, m, n, tol);
    return range(n - r).map((i) => col(V, r + i));
  }

  function rowspace(A, tol) {
    const [m, n] = needMat(A, "rowspace");
    const { S, V } = svd(A);
    return range(rankFrom(S, m, n, tol)).map((i) => col(V, i));
  }

  function colspace(A, tol) {
    const [m, n] = needMat(A, "colspace");
    const { S, U } = svd(A);
    return range(rankFrom(S, m, n, tol)).map((i) => col(U, i));
  }

  function leftNullspace(A, tol) {
    const [m, n] = needMat(A, "leftNullspace");
    const { S, U } = svd(A, { full: true });
    const r = rankFrom(S, m, n, tol);
    return range(m - r).map((i) => col(U, r + i));
  }

  // =====================================================================================
  // Formatting (used by rref's step log, and exported)
  // =====================================================================================

  // Best rational approximation p/q with q <= maxDen, by continued fractions; null when no
  // such fraction is within 1e-9 max(1, |x|).
  function ratApprox(x, maxDen) {
    if (!isFinite(x)) return null;
    const sgn = x < 0 ? -1 : 1, y = Math.abs(x), close = 1e-9 * Math.max(1, y);
    let h0 = 0, h1 = 1, k0 = 1, k1 = 0, z = y;
    for (let it = 0; it < 64; it++) {
      const a = Math.floor(z);
      const h2 = a * h1 + h0, k2 = a * k1 + k0;
      if (k2 > maxDen) break;
      h0 = h1; h1 = h2; k0 = k1; k1 = k2;
      if (Math.abs(y - h1 / k1) <= close) return [sgn * h1, k1];
      const f = z - a;
      if (f < 1e-15) break;
      z = 1 / f;
    }
    return null;
  }

  function sig4(x) {
    if (x === 0) return "0";
    return String(Number(x.toPrecision(4)));
  }

  function frac(x, maxDen) {
    const r = ratApprox(x, maxDen || 1000);
    if (!r) return sig4(x);
    if (r[0] === 0) return "0";
    return r[1] === 1 ? String(r[0]) : `${r[0]}/${r[1]}`;
  }

  function fixedStr(x, d) {
    if (x === Infinity) return "∞";
    if (x === -Infinity) return "-∞";
    if (x !== x) return "NaN";
    const s = x.toFixed(d);
    return /^-0\.?0*$/.test(s) ? s.slice(1) : s;
  }

  function fmt(x, d) {
    d = d === undefined ? 3 : d;
    if (!isArr(x)) return fixedStr(x, d);
    if (!isMat(x)) return "[" + Array.from(x, (v) => fixedStr(v, d)).join(", ") + "]";
    const cells = x.map((r) => Array.from(r, (v) => fixedStr(v, d)));
    const w = Math.max(...cells.map((r) => Math.max(...r.map((s) => s.length))));
    return cells.map((r) => "[ " + r.map((s) => s.padStart(w)).join("  ") + " ]").join("\n");
  }

  function texNum(x, digits) {
    if (!isFinite(x)) return x !== x ? "\\text{NaN}" : x > 0 ? "\\infty" : "-\\infty";
    if (digits !== undefined) return fixedStr(x, digits);
    const r = ratApprox(x, 1000);
    if (!r) return sig4(x).replace(/e\+?(-?\d+)$/, " \\times 10^{$1}");
    if (r[1] === 1) return String(r[0]);
    return `${r[0] < 0 ? "-" : ""}\\tfrac{${Math.abs(r[0])}}{${r[1]}}`;
  }

  function tex(A, opts) {
    const digits = opts && opts.digits;
    if (!isArr(A)) return texNum(A, digits);
    const M = isMat(A) ? A : transpose(A);
    const body = M.map((r) => Array.from(r, (v) => texNum(v, digits)).join(" & ")).join(" \\\\ ");
    return `\\begin{bmatrix} ${body} \\end{bmatrix}`;
  }

  // =====================================================================================
  // Row reduction with a step log
  // =====================================================================================

  function rref(A, opts) {
    const [m, n] = needMat(A, "rref");
    opts = opts || {};
    const pivotRule = opts.pivot || "first";
    const tol = opts.tol !== undefined ? opts.tol : 1e-10 * maxAbs(A);
    const M = clone(A);
    const steps = [], pivots = [];
    const clean = () => {
      for (const r of M) for (let j = 0; j < n; j++) if (Math.abs(r[j]) <= tol) r[j] = 0;
    };
    // "(1/2)" for fractions, "3" for integers, "0.3333" otherwise; mag is |k|.
    const coefText = (mag) => {
      const s = frac(mag);
      return s.includes("/") ? `(${s})` : s;
    };
    const coefTex = (mag) => texNum(mag);
    const log = (step) => {
      step.matrix = clone(M);
      steps.push(step);
    };
    clean();
    let r = 0;
    for (let c = 0; c < n && r < m; c++) {
      let p = -1;
      for (let i = r; i < m; i++) {
        if (Math.abs(M[i][c]) <= tol) continue;
        if (p < 0 || (pivotRule === "max" && Math.abs(M[i][c]) > Math.abs(M[p][c]))) p = i;
        if (pivotRule !== "max") break;
      }
      if (p < 0) continue;
      if (p !== r) {
        [M[r], M[p]] = [M[p], M[r]];
        log({ op: "swap", i: r, j: p, pivot: [r, c],
              text: `R${r + 1} ↔ R${p + 1}`, tex: `R_{${r + 1}} \\leftrightarrow R_{${p + 1}}` });
      }
      const piv = M[r][c];
      if (piv !== 1) {
        const k = 1 / piv;
        M[r] = M[r].map((x) => x / piv);
        M[r][c] = 1;
        clean();
        const neg = k < 0 ? "−" : "", negT = k < 0 ? "-" : "";
        const mag = Math.abs(k);
        const t = mag === 1 ? "" : coefText(mag) + " ", tt = mag === 1 ? "" : coefTex(mag) + " ";
        log({ op: "scale", i: r, k, pivot: [r, c],
              text: `R${r + 1} ← ${neg}${t}R${r + 1}`, tex: `R_{${r + 1}} \\leftarrow ${negT}${tt}R_{${r + 1}}` });
      }
      for (let i = 0; i < m; i++) {
        if (i === r) continue;
        const f = M[i][c];
        if (f === 0) continue;
        M[i] = M[i].map((x, j) => x - f * M[r][j]);
        M[i][c] = 0;
        clean();
        const k = -f, sgn = k < 0 ? "−" : "+", sgnT = k < 0 ? "-" : "+";
        const mag = Math.abs(k);
        const t = mag === 1 ? "" : coefText(mag) + " ", tt = mag === 1 ? "" : coefTex(mag) + " ";
        log({ op: "add", i, j: r, k, pivot: [r, c],
              text: `R${i + 1} ← R${i + 1} ${sgn} ${t}R${r + 1}`,
              tex: `R_{${i + 1}} \\leftarrow R_{${i + 1}} ${sgnT} ${tt}R_{${r + 1}}` });
      }
      pivots.push(c);
      r++;
    }
    return { R: M, pivots, rank: pivots.length, steps };
  }

  // =====================================================================================
  // Random numbers: mulberry32 + Box-Muller
  // =====================================================================================

  function rng(seed) {
    let a = (seed === undefined ? 1 : seed) | 0;
    let spare = null;
    function uniform(lo, hi) {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      return lo === undefined ? u : lo + (hi - lo) * u;
    }
    function normal(mu, sigma) {
      let z;
      if (spare !== null) {
        z = spare;
        spare = null;
      } else {
        const u1 = 1 - uniform(), u2 = uniform();
        const r = Math.sqrt(-2 * Math.log(u1));
        z = r * Math.cos(2 * Math.PI * u2);
        spare = r * Math.sin(2 * Math.PI * u2);
      }
      return (mu || 0) + (sigma === undefined ? 1 : sigma) * z;
    }
    const int = (k) => Math.floor(uniform() * k);
    const normalVec = (n) => range(n).map(() => normal());
    const uniformVec = (n, lo, hi) => range(n).map(() => uniform(lo === undefined ? 0 : lo, hi === undefined ? 1 : hi));
    function sphere(n) {
      if (!(n >= 1)) fail("rng.sphere", "needs a dimension n >= 1");
      for (;;) {
        const v = normalVec(n), s = norm2(v);
        if (s > 1e-12) return v.map((x) => x / s);
      }
    }
    function mvnormal(mean, cov) {
      const [m, n] = needMat(cov, "rng.mvnormal");
      if (m !== n || n !== mean.length) fail("rng.mvnormal", `mean has ${mean.length} entries, cov is ${m}x${n}`);
      let L = chol(cov);
      if (!L) {
        // Eigenvalues within rounding of zero are zero, so that the draws of a singular cov
        // stay in its range (sqrt would turn a 1e-16 rounding error into a 1e-8 offset).
        const { values, V } = eigSym(cov);
        const cut = n * EPS * Math.max(...values.map(Math.abs));
        const root = values.map((w) => (w > cut ? Math.sqrt(w) : 0));
        L = V.map((r) => r.map((x, j) => x * root[j]));
      }
      return add(mean, matvec(L, normalVec(mean.length)));
    }
    function shuffle(arr) {
      const out = Array.from(arr);
      for (let i = out.length - 1; i > 0; i--) {
        const j = int(i + 1);
        [out[i], out[j]] = [out[j], out[i]];
      }
      return out;
    }
    return { uniform, normal, int, normalVec, uniformVec, sphere, mvnormal, shuffle };
  }

  // =====================================================================================
  // Linear programming: dense two-phase simplex, Bland's rule
  // =====================================================================================

  function normBounds(bounds, n) {
    const lo = (v) => (v === null || v === undefined ? -Infinity : v);
    const hi = (v) => (v === null || v === undefined ? Infinity : v);
    if (bounds === undefined || bounds === null) return range(n).map(() => [0, Infinity]);
    const perVar = isArr(bounds[0]) || isArr(bounds[1]) || (bounds.length !== 2 && bounds.length === n);
    if (!perVar) return range(n).map(() => [lo(bounds[0]), hi(bounds[1])]);
    if (bounds.length !== n) fail("lp", `bounds has ${bounds.length} entries for ${n} variables`);
    return bounds.map((b) => (b === null || b === undefined ? [-Infinity, Infinity] : [lo(b[0]), hi(b[1])]));
  }

  function lp(prob, opts) {
    const tol = (opts && opts.tol) || 1e-9;
    const c = prob.c, n = c.length;
    const Aub = prob.Aub || [], bub = prob.bub || [], Aeq = prob.Aeq || [], beq = prob.beq || [];
    if (Aub.length !== bub.length) fail("lp", "Aub and bub have different lengths");
    if (Aeq.length !== beq.length) fail("lp", "Aeq and beq have different lengths");
    for (const r of Aub.concat(Aeq)) if (r.length !== n) fail("lp", `a constraint row has ${r.length} entries for ${n} variables`);
    const B = normBounds(prob.bounds, n);
    const empty = (status, fun) => ({ status, x: null, fun, duals: null, iterations: 0 });
    if (B.some(([l, h]) => l > h)) return empty("infeasible", Infinity);

    // Substitute x = x0 + sum_k s_k y_k e_{j_k} with y >= 0.
    const x0 = zeros(n), cols = [], capRows = [];
    for (let j = 0; j < n; j++) {
      const [l, h] = B[j];
      if (l > -Infinity) {
        x0[j] = l;
        cols.push({ j, s: 1 });
        if (h < Infinity) capRows.push({ k: cols.length - 1, cap: h - l, j });
      } else if (h < Infinity) {
        x0[j] = h;
        cols.push({ j, s: -1 });
      } else {
        cols.push({ j, s: 1 }, { j, s: -1 });
      }
    }
    const N = cols.length;
    const rows = [];
    Aub.forEach((a, i) => rows.push({ kind: "ub", idx: i, a: cols.map(({ j, s }) => a[j] * s), b: bub[i] - dot(a, x0) }));
    capRows.forEach(({ k, cap, j }) => {
      const a = zeros(N);
      a[k] = 1;
      rows.push({ kind: "cap", idx: j, a, b: cap });
    });
    Aeq.forEach((a, i) => rows.push({ kind: "eq", idx: i, a: cols.map(({ j, s }) => a[j] * s), b: beq[i] - dot(a, x0) }));

    // Tableau columns: y (N), one slack per ub/cap row, one artificial per row that has no
    // usable slack. Every row starts with a unit column in the basis (init[i]).
    const M = rows.length;
    const nSlack = rows.filter((r) => r.kind !== "eq").length;
    const flip = rows.map((r) => (r.b < 0 ? -1 : 1));
    const needArt = rows.map((r, i) => r.kind === "eq" || flip[i] < 0);
    const nArt = needArt.filter(Boolean).length;
    const artStart = N + nSlack, W = N + nSlack + nArt + 1, RHS = W - 1;
    const T = [], basis = [], init = [];
    let si = 0, ai = 0;
    rows.forEach((r, i) => {
      const t = new Array(W).fill(0);
      for (let k = 0; k < N; k++) t[k] = flip[i] * r.a[k];
      t[RHS] = flip[i] * r.b;
      if (r.kind !== "eq") t[N + si++] = flip[i];
      if (needArt[i]) {
        t[artStart + ai] = 1;
        basis.push(artStart + ai++);
      } else basis.push(N + si - 1);
      init.push(basis[i]);
      T.push(t);
    });
    const rowOf = range(M); // tableau row -> original row (rows can be dropped as redundant)

    let iterations = 0;
    const maxIt = (opts && opts.maxIter) || 50000;
    function pivot(z, r, q) {
      const Tr = T[r], pv = Tr[q];
      for (let j = 0; j < W; j++) Tr[j] /= pv;
      Tr[q] = 1;
      for (let i = 0; i < T.length; i++) {
        if (i === r) continue;
        const f = T[i][q];
        if (f === 0) continue;
        const Ti = T[i];
        for (let j = 0; j < W; j++) Ti[j] -= f * Tr[j];
        Ti[q] = 0;
      }
      const f = z[q];
      if (f !== 0) {
        for (let j = 0; j < W; j++) z[j] -= f * Tr[j];
        z[q] = 0;
      }
      basis[r] = q;
    }
    // z holds reduced costs d_j = c_j - c_B^T B^{-1} a_j and, at RHS, -(objective).
    function simplex(z, allowed) {
      for (;;) {
        if (iterations >= maxIt) return "iteration_limit";
        let q = -1;
        for (let j = 0; j < RHS; j++) {
          if (allowed(j) && z[j] < -tol) {
            q = j;
            break;
          }
        }
        if (q < 0) return "optimal";
        let r = -1, best = Infinity;
        for (let i = 0; i < T.length; i++) {
          const a = T[i][q];
          if (!(a > tol)) continue;
          const ratio = T[i][RHS] / a;
          const tie = 1e-12 * (1 + Math.abs(best));
          if (r < 0 || ratio < best - tie) {
            r = i;
            best = ratio;
          } else if (ratio <= best + tie && basis[i] < basis[r]) {
            r = i;
            best = Math.min(best, ratio);
          }
        }
        if (r < 0) return "unbounded";
        pivot(z, r, q);
        iterations++;
      }
    }
    const reducedCosts = (cost) => {
      const z = new Array(W).fill(0);
      for (let j = 0; j < RHS; j++) z[j] = cost[j];
      T.forEach((t, i) => {
        const cb = cost[basis[i]];
        if (cb !== 0) for (let j = 0; j < W; j++) z[j] -= cb * t[j];
      });
      return z;
    };

    // Phase 1: minimize the sum of the artificials.
    if (nArt > 0) {
      const cost1 = range(RHS).map((j) => (j >= artStart ? 1 : 0));
      const z1 = reducedCosts(cost1);
      const st = simplex(z1, () => true);
      if (st === "iteration_limit") return empty(st, NaN);
      const scaleB = Math.max(1, ...rows.map((r) => Math.abs(r.b)));
      if (-z1[RHS] > tol * scaleB) return empty("infeasible", Infinity);
      // Drive artificials (all at zero now) out of the basis on the largest available entry
      // (the row's right-hand side is zero, so any sign keeps feasibility); drop redundant rows.
      for (let i = T.length - 1; i >= 0; i--) {
        if (basis[i] < artStart) continue;
        T[i][RHS] = 0;
        let q = -1, best = tol;
        for (let j = 0; j < artStart; j++) {
          if (Math.abs(T[i][j]) > best) {
            q = j;
            best = Math.abs(T[i][j]);
          }
        }
        if (q >= 0) pivot(z1, i, q);
        else {
          T.splice(i, 1);
          basis.splice(i, 1);
          rowOf.splice(i, 1);
        }
      }
    }

    // Phase 2: the real objective; artificial columns may not re-enter.
    const cost2 = range(RHS).map((j) => (j < N ? c[cols[j].j] * cols[j].s : 0));
    const z2 = reducedCosts(cost2);
    const st = simplex(z2, (j) => j < artStart);
    if (st === "unbounded") return { ...empty("unbounded", -Infinity), iterations };
    if (st === "iteration_limit") return { ...empty(st, NaN), iterations };

    const y = zeros(RHS);
    T.forEach((t, i) => (y[basis[i]] = t[RHS]));
    const x = x0.slice();
    cols.forEach(({ j, s }, k) => (x[j] += s * y[k]));
    // Duals: y'_i = -z2[init_i] for the (flipped) row i; the original row's is flip * y'.
    const rowDual = zeros(M);
    rowOf.forEach((i) => (rowDual[i] = -flip[i] * z2[init[i]]));
    const duals = { ub: zeros(Aub.length), eq: zeros(Aeq.length), lower: zeros(n), upper: zeros(n) };
    rows.forEach((r, i) => {
      if (r.kind === "ub") duals.ub[r.idx] = rowDual[i];
      else if (r.kind === "eq") duals.eq[r.idx] = rowDual[i];
      else duals.upper[r.idx] = rowDual[i];
    });
    cols.forEach(({ j }, k) => {
      const [l, h] = B[j];
      if (l > -Infinity) duals.lower[j] = z2[k];
      else if (h < Infinity) duals.upper[j] = -z2[k];
    });
    return { status: "optimal", x, fun: dot(c, x), duals, iterations };
  }

  function feasible(prob) {
    let n = prob.n;
    if (n === undefined) {
      if (prob.Aub && prob.Aub.length) n = prob.Aub[0].length;
      else if (prob.Aeq && prob.Aeq.length) n = prob.Aeq[0].length;
      else if (prob.bounds && isArr(prob.bounds[0])) n = prob.bounds.length;
      else fail("feasible", "cannot tell the number of variables; pass n");
    }
    const r = lp({ ...prob, c: zeros(n) });
    return { feasible: r.status === "optimal", x: r.x, status: r.status };
  }

  // =====================================================================================
  // Quadratic programming
  // =====================================================================================

  function eqQP(H, g, A, b) {
    const [n, nH] = needMat(H, "eqQP"), m = A ? A.length : 0;
    g = g || zeros(n);
    if (n !== nH || g.length !== n) fail("eqQP", `H must be n x n and g of length n (H is ${n}x${nH}, g has ${g.length} entries)`);
    if (m && (!b || b.length !== m)) fail("eqQP", `A has ${m} rows, b has ${b ? b.length : 0} entries`);
    for (let i = 0; i < m; i++) if (A[i].length !== n) fail("eqQP", `row ${i} of A has ${A[i].length} entries for ${n} variables`);
    const K = zeros(n + m, n + m), rhs = zeros(n + m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) K[i][j] = H[i][j];
      rhs[i] = -g[i];
    }
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) K[n + i][j] = K[j][n + i] = A[i][j];
      rhs[n + i] = b[i];
    }
    let sol = solve(K, rhs), singular = false;
    if (!sol) {
      sol = lstsq(K, rhs).x;
      singular = true;
    }
    return { x: sol.slice(0, n), nu: sol.slice(n), singular };
  }

  function qp(prob, opts) {
    const tol = (opts && opts.tol) || 1e-9;
    const [n, nH] = needMat(prob.H, "qp");
    if (n !== nH) fail("qp", "H must be square");
    const H = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) H[i][j] = 0.5 * (prob.H[i][j] + prob.H[j][i]);
    const g = prob.g || zeros(n);
    const Aeq = prob.Aeq || [], beq = prob.beq || [], Aub = prob.Aub || [], bub = prob.bub || [];
    const mE = Aeq.length, mI = Aub.length;
    if (g.length !== n) fail("qp", `H is ${n}x${n}, g has ${g.length} entries`);
    if (beq.length !== mE || bub.length !== mI) fail("qp", "Aeq, beq (or Aub, bub) have different lengths");
    for (const r of Aeq.concat(Aub)) if (r.length !== n) fail("qp", `a constraint row has ${r.length} entries for ${n} variables`);
    const fval = (x) => 0.5 * dot(x, matvec(H, x)) + dot(g, x);
    let iterations = 0;
    const done = (status, x, lamEq, lamUb, activeUb) => ({
      status, x, fun: x ? fval(x) : status === "unbounded" ? -Infinity : status === "infeasible" ? Infinity : NaN,
      lamEq: lamEq || null, lamUb: lamUb || null, activeUb: activeUb || null, iterations,
    });

    // Convexity: H must be positive semidefinite.
    const ev = eigSym(H).values;
    const hmax = Math.max(Math.abs(ev[0]), Math.abs(ev[n - 1]));
    if (ev[n - 1] < -1e-10 * hmax) return done("nonconvex", null);

    // Feasibility: LP phase 1 with free variables.
    const start = lp({ c: zeros(n), Aub, bub, Aeq, beq, bounds: [null, null] });
    if (start.status !== "optimal") return done("infeasible", null);

    // Unboundedness: a recession direction d = Z^T u (Z = null(H)) with Aeq d = 0,
    // Aub d <= 0 and g^T d < 0. Normalize with -1 <= u <= 1 and ask an LP.
    const Z = nullspace(H);
    if (Z.length) {
      const Zt = transpose(Z);
      const ray = lp({
        c: matvec(Z, g),
        Aeq: mE ? matmul(Aeq, Zt) : [], beq: zeros(mE),
        Aub: mI ? matmul(Aub, Zt) : [], bub: zeros(mI),
        bounds: [-1, 1],
      });
      if (ray.status === "optimal" && ray.fun < -tol * Math.max(1, norm2(g))) return done("unbounded", null);
    }

    // Primal active-set method (Nocedal & Wright, Algorithm 16.3).
    let x = start.x.slice();
    const work = []; // working set: {type: 'eq' | 'ub', i}
    const basisRows = []; // orthonormal basis of the working normals, for independence tests
    const tryAdd = (a) => {
      let w = Array.from(a);
      for (let pass = 0; pass < 2; pass++) for (const q of basisRows) w = sub(w, scale(q, dot(q, w)));
      const nw = norm2(w);
      if (nw <= 1e-10 * norm2(a)) return false;
      basisRows.push(w.map((v) => v / nw));
      return true;
    };
    for (let i = 0; i < mE; i++) if (tryAdd(Aeq[i])) work.push({ type: "eq", i });
    for (let i = 0; i < mI; i++) {
      if (Math.abs(dot(Aub[i], x) - bub[i]) <= tol * (1 + Math.abs(bub[i])) && tryAdd(Aub[i])) work.push({ type: "ub", i });
    }
    const rowOf = (w) => (w.type === "eq" ? Aeq[w.i] : Aub[w.i]);
    const inWork = (i) => work.some((w) => w.type === "ub" && w.i === i);
    const maxIt = (opts && opts.maxIter) || 50 * (n + mE + mI) + 100;

    for (; iterations < maxIt; iterations++) {
      const gk = add(matvec(H, x), g);
      const AW = work.map(rowOf);
      // Step p minimizes (1/2) p^T H p + gk^T p on {A_W p = 0}: p = Zw^T u with
      // (Zw H Zw^T) u = -Zw gk. When that system is inconsistent (zero curvature with
      // descent), p is a descent direction along which the objective is linear.
      const Zw = AW.length ? nullspace(AW) : eye(n);
      let p = zeros(n), isRay = false;
      if (Zw.length) {
        const Zt = transpose(Zw);
        const Hr = matmul(Zw, H, Zt), gr = matvec(Zw, gk);
        const u = lstsq(Hr, scale(gr, -1));
        if (u.residual <= 1e-9 * (1 + norm2(gr))) p = matvec(Zt, u.x);
        else {
          const Nr = nullspace(Hr);
          let d = zeros(gr.length);
          for (const v of Nr) d = sub(d, scale(v, dot(v, gr)));
          p = matvec(Zt, d);
          isRay = true;
        }
      }
      const pn = norm2(p);
      if (!isRay && pn <= 1e-12 * (1 + norm2(x))) {
        // x minimizes on the working set: check the signs of the inequality multipliers.
        const lam = AW.length ? lstsq(transpose(AW), scale(gk, -1)).x : [];
        let worst = -1, wv = -tol * Math.max(1, norm2(gk));
        work.forEach((w, k) => {
          if (w.type === "ub" && lam[k] < wv) {
            wv = lam[k];
            worst = k;
          }
        });
        if (worst < 0) {
          const lamEq = zeros(mE), lamUb = zeros(mI);
          work.forEach((w, k) => (w.type === "eq" ? (lamEq[w.i] = lam[k]) : (lamUb[w.i] = lam[k])));
          const activeUb = work.filter((w) => w.type === "ub").map((w) => w.i).sort((a, b) => a - b);
          return done("optimal", x, lamEq, lamUb, activeUb);
        }
        work.splice(worst, 1);
        continue;
      }
      // Longest feasible step along p (at most 1 for a Newton step).
      let alpha = isRay ? Infinity : 1, block = -1;
      for (let i = 0; i < mI; i++) {
        if (inWork(i)) continue;
        const ap = dot(Aub[i], p);
        if (ap <= 1e-12 * norm2(Aub[i]) * pn) continue;
        const step = (bub[i] - dot(Aub[i], x)) / ap;
        if (step < alpha) {
          alpha = Math.max(step, 0);
          block = i;
        }
      }
      if (alpha === Infinity) return done("unbounded", null);
      x = add(x, scale(p, alpha));
      if (block >= 0) work.push({ type: "ub", i: block });
    }
    return done("iteration_limit", x);
  }

  // =====================================================================================
  // Nonnegative least squares (Lawson-Hanson)
  // =====================================================================================

  function nnls(A, b, opts) {
    const [m, n] = needMat(A, "nnls");
    if (b.length !== m) fail("nnls", `matrix has ${m} rows, b has ${b.length} entries`);
    const tol = (opts && opts.tol) || 10 * EPS * norm(A, 1) * Math.max(m, n);
    const maxIt = (opts && opts.maxIter) || 3 * n + 30;
    const inP = new Array(n).fill(false);
    let x = zeros(n), iterations = 0;
    const At = transpose(A);
    const gradient = () => matvec(At, sub(b, matvec(A, x)));
    let w = gradient();
    while (iterations < maxIt) {
      let t = -1, wt = tol;
      for (let j = 0; j < n; j++) {
        if (!inP[j] && w[j] > wt) {
          wt = w[j];
          t = j;
        }
      }
      if (t < 0) break;
      inP[t] = true;
      for (;;) {
        iterations++;
        const idx = range(n).filter((j) => inP[j]);
        if (!idx.length) break; // rounding sent every passive coordinate to zero; x = 0
        const zs = lstsq(submatrix(A, null, idx), b).x;
        const z = zeros(n);
        idx.forEach((j, k) => (z[j] = zs[k]));
        if (idx.every((j) => z[j] > 0) || iterations >= maxIt) {
          x = z.map((v) => Math.max(v, 0));
          break;
        }
        // Move from x toward z until the first coordinate in P reaches zero.
        let alpha = Infinity;
        for (const j of idx) if (z[j] <= 0) alpha = Math.min(alpha, x[j] / (x[j] - z[j]));
        x = x.map((v, j) => v + alpha * (z[j] - v));
        const floor = 1e-14 * Math.max(1, maxAbs(x));
        for (const j of idx) {
          if (x[j] <= floor) {
            inP[j] = false;
            x[j] = 0;
          }
        }
      }
      w = gradient();
    }
    return { x, residual: norm2(sub(matvec(A, x), b)), iterations };
  }

  // =====================================================================================
  // Derivatives by central differences
  // =====================================================================================

  const H1 = Math.cbrt(EPS), H2 = Math.pow(EPS, 0.25);
  const stepFor = (h, xi, c) => (h === undefined || h === null ? c * Math.max(1, Math.abs(xi)) : h);

  function derivFD(f, x, h) {
    const s = stepFor(h, x, H1);
    return (f(x + s) - f(x - s)) / (2 * s);
  }

  function gradFD(f, x, h) {
    return Array.from(x, (xi, i) => {
      const s = stepFor(h, xi, H1);
      const xp = Array.from(x), xm = Array.from(x);
      xp[i] += s;
      xm[i] -= s;
      return (f(xp) - f(xm)) / (2 * s);
    });
  }

  function jacFD(F, x, h) {
    const n = x.length;
    const cols = range(n).map((j) => {
      const s = stepFor(h, x[j], H1);
      const xp = Array.from(x), xm = Array.from(x);
      xp[j] += s;
      xm[j] -= s;
      const fp = F(xp), fm = F(xm);
      return Array.from(fp, (v, i) => (v - fm[i]) / (2 * s));
    });
    return transpose(cols);
  }

  function hessFD(f, x, h) {
    const n = x.length, Hm = zeros(n, n), f0 = f(Array.from(x));
    const s = Array.from(x, (xi) => stepFor(h, xi, H2));
    const at = (i, di, j, dj) => {
      const y = Array.from(x);
      y[i] += di;
      if (j >= 0) y[j] += dj;
      return f(y);
    };
    for (let i = 0; i < n; i++) {
      Hm[i][i] = (at(i, s[i], -1) - 2 * f0 + at(i, -s[i], -1)) / (s[i] * s[i]);
      for (let j = i + 1; j < n; j++) {
        const v = (at(i, s[i], j, s[j]) - at(i, s[i], j, -s[j]) - at(i, -s[i], j, s[j]) + at(i, -s[i], j, -s[j])) / (4 * s[i] * s[j]);
        Hm[i][j] = Hm[j][i] = v;
      }
    }
    return Hm;
  }

  // =====================================================================================

  const Num = {
    EPS,
    // construction and elementwise
    zeros, ones, eye, diag, linspace, clone, shape, transpose, add, sub, scale, lincomb,
    hstack, vstack, row, col, submatrix, allclose,
    // products and norms
    matmul, matvec, dot, outer, cross, norm, normalize, trace,
    // linear systems
    lu, solve, inv, det, chol, cholSolve,
    // orthogonalization
    qr, gramSchmidt, projector, project,
    // eigen, SVD, subspaces
    eigSym, svd, svdvals, rank, cond, pinv, lstsq, nullspace, rowspace, colspace, leftNullspace,
    // row reduction
    rref,
    // random
    rng,
    // optimization
    lp, feasible, qp, eqQP, nnls,
    // derivatives
    derivFD, gradFD, jacFD, hessFD,
    // formatting
    fmt, frac, tex,
  };
  root.Num = Num;
})(typeof window !== "undefined" ? window : globalThis);
