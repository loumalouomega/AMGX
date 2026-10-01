# AMGX Roadmap: AMGCL Feature Port & Modernization

This document catalogs features pending implementation in the NVIDIA AMGX backend, ported from AMGCL or designed to achieve algorithmic parity with it. It lists **what is not built**. Nothing here duplicates shipped functionality; where a feature partially exists, the shipped half is identified and the architectural gap is defined explicitly.

Effort key:

- **S** = Days (self-contained kernel or wrapper)
- **M** = 1–2 weeks (multi-component module or algorithmic extension)
- **L** = 1 month (core subsystem redesign, new hierarchy pipeline)
- **XL** = Multi-month research-grade project (novel multi-GPU numerical pipeline)

Every item specifies a **probe** (a reproducible test or failure threshold demonstrating the gap and verifying the fix).

---

## How this file works

- **Sections are strictly ordered by engineering dependency.** Each section defines its architectural scope; items do not move up due to novelty or standalone interest.
- **Defect-shaped items belong in a *Correctness Debts* section (§1).** None is currently open; the first silent failure, NaN leak, memory corruption, or convergence stall opens §1 and renumbers subsequent sections.
- **Closed items are removed entirely**, not struck through.
- **Verification status:** Items estimated without direct source inspection state `(verify first)` alongside their targeted probe.
- **[Non-goals and decisions taken](#non-goals-and-decisions-taken)** record definitive architectural choices with rationale to eliminate recurrent debate.

---

## 1. Port Missing AMGCL Features to NVIDIA AMGX

### Scope & Architectural Boundary

AMGX natively supports classical AMG (Ruge-Stüben, PMIS/HMIS, aggressive coarsening), standard smoothers (Jacobi, block/CF Jacobi, Gauss-Seidel, Multicolor GS/DILU, ILU($k$), Chebyshev/KPZ polynomials), Krylov accelerators (CG, PCG, BiCGStab, GMRES, IDR($s$)), block/BSR operations, MPI rank consolidation with CUDA IPC, and setup structure reuse. **None of these are porting targets.**

The primary architectural gap in AMGX is that its aggregation hierarchy relies on **plain, unsmoothed aggregation with binary transfer operators** ($P \in \{0, 1\}^{n \times m}$), creating coarse spaces that fail to preserve low-energy error modes on elliptic PDEs with high anisotropy, jump coefficients, or vector fields (e.g., linear elasticity). 

The goal of this section is to bring AMGCL's production-grade **Smoothed Aggregation (SA)**, **Near-Nullspace Tentative Prolongation**, **Prolongation Filtering**, and **Composite Physics Preconditioners** into AMGX's high-throughput GPU pipeline.

```
       [Strength-of-Connection Filter] (P0.1)
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Plain Aggregates] (P0.2)   [BSR Pointwise] (P0.3)
       │                           │
       └─────────────┬─────────────┘
                     ▼
        [Batched QR / Nullspace] (P0.4) ─── [Rigid-Body Mode Gen] (P0.5)
                     │
                     ▼
       [Spectral Radius Est.] (P0.6)
                     │
                     ▼
       [Filtered-Jacobi Smoothing] (P0.7)
                     │
                     ▼
     [Prolongation Truncation] (P0.8) ◄── (Prevents SpGEMM memory explosion)
                     │
                     ▼
    [Galerkin Coarse SpGEMM: PᵀAP] (P0.9)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[Scalar SA Pipeline]     [BSR as_scalar] (P0.11)
         │                       │
         └───────────┬───────────┘
                     ▼
    [Distributed MPI Smoothing] (P4.2)
```

---

### P0: Aggregation, Prolongation, and Coarsening Engine

#### P0.1 Symmetric Strength-of-Connection Filter with Diagonal Lumping

- **Effort:** **M**
- **Gap:** AMGX's aggregation algorithms (`SIZE_2`, `SIZE_4`, `SIZE_8`) operate on unweighted or naively thresholded adjacency graphs. AMGCL applies a scaled symmetric filtering criterion that lumps discarded weak connections back into the matrix diagonal to preserve row sums (zero-energy modes).
- **Mathematical Specification:**
  A connection between degree of freedom $i$ and $j$ ($i \ne j$) is strong if:
  $$\frac{a_{ij}^2}{a_{ii} a_{jj}} > \varepsilon_{\text{strong}} \quad (\text{default: } \varepsilon_{\text{strong}} = 0.08)$$
  Weak edges ($a_{ij}^2 \le \varepsilon_{\text{strong}} a_{ii} a_{jj}$) are purged from the sparsity graph of $A^F$. To preserve row sums for the filtered matrix $A^F$:
  $$A^F_{ii} = A_{ii} + \sum_{j \in \text{weak}(i)} A_{ij}$$
  Per-level decay: $\varepsilon_{\text{strong}}^{(l+1)} = \max(\varepsilon_{\text{strong}}^{(l)} \times 0.5, \, 10^{-4})$ to prevent early stagnation on deep coarse levels.
- **AMGX Target:** `src/aggregation/strength/` registered as `SYMMETRIC_SCALED`.
- **AMGCL Reference:** `amgcl/detail/scaled_galerkin.hpp`, `amgcl/coarsening/plain_aggregates.hpp`.
- **Probe:** Run 3D Poisson with strong anisotropy ($\kappa_x = 1, \kappa_y = 1, \kappa_z = 10^{-3}$) on a $128 \times 128 \times 32$ grid. AMGX default aggregation degrades to $\rho > 0.88$ per V-cycle. The probe passes when `SYMMETRIC_SCALED` creates non-isotropic 1D line aggregates along the $z$-direction and achieves asymptotic convergence factor $\rho \le 0.18$.

#### P0.2 GPU Parallel Aggregate Extraction

- **Effort:** **S**
- **Gap:** AMGCL uses a sequential greedy MIS-2 sweep on the CPU. AMGX has high-speed parallel matching routines (`MATCHING`, `THRUST`), but they do not emit an explicit aggregate-to-node index mapping compatible with multi-vector prolongation.
- **Specification:** Adapt AMGX's parallel maximal independent set / maximum weight matching kernels to output an aggregate index array:
  $$\text{agg\_id}[i] \in [0, N_{\text{aggregates}}-1] \quad \forall i \in [0, N-1]$$
  Ensure every unaggregated vertex is assigned to an adjacent aggregate via a parallel boundary-attraction kernel (one GPU pass per unassigned vertex).
- **AMGX Target:** `src/aggregation/selectors/`.
- **Probe:** Verify parallel aggregation of a 10M row Laplacian on a single H100 in $< 15\text{ ms}$; zero unassigned nodes; aggregate sizes bounded within $[2, 16]$.

#### P0.3 Pointwise Aggregation for Block/BSR Matrices

- **Effort:** **M**
- **Gap:** Aggregating each equation of a multi-variable PDE separately destroys block coupling. AMGCL aggregates on the structural graph of grid nodes (pointwise), not individual DOFs.
- **Specification:** For BSR systems with block dimension $B \in [2, 6]$, construct a scalar adjacency graph $G_{\text{point}} = (V_{\text{point}}, E_{\text{point}})$ where:
  $$(I, J) \in E_{\text{point}} \iff \exists i \in \text{dofs}(I), j \in \text{dofs}(J) \text{ s.t. } \|A_{ij}\|_{F}^2 > \varepsilon_{\text{strong}}^2 \|A_{ii}\|_F \|A_{jj}\|_F$$
  Run P0.2 on $G_{\text{point}}$ and expand the resulting aggregate mapping to all DOFs: $\text{agg\_id}_{\text{dof}}(I \cdot B + k) = \text{agg\_id}_{\text{point}}(I) \cdot B + k$.
- **AMGCL Reference:** `amgcl/coarsening/pointwise_aggregates.hpp`.
- **Probe:** Solve 3D Navier-Stokes velocity-pressure coupled BSR system ($B=4$). Verify all components of velocity and pressure at physical node $I$ share identical aggregate assignments.

#### P0.4 Batched In-Warp QR for Near-Nullspace Tentative Prolongation

- **Effort:** **L**
- **Gap:** AMGX builds binary prolongators $\tilde{P}_{ij} \in \{0, 1\}$. It lacks an API to supply near-nullspace modes (e.g., rigid-body translations and rotations) and cannot perform local QR factorizations to construct an orthonormal tentative prolongator $\tilde{P}$.
- **Mathematical Specification:**
  Given $K$ user-defined near-nullspace vectors arranged as $B \in \mathbb{R}^{N \times K}$ (where $K \ll N$, e.g., $K=6$ for 3D elasticity). For aggregate $i$ with $m_i$ local rows, extract the submatrix $B_i \in \mathbb{R}^{m_i \times K}$.
  Compute the thin QR factorization locally on GPU:
  $$B_i = Q_i R_i, \quad Q_i \in \mathbb{R}^{m_i \times K}, \quad R_i \in \mathbb{R}^{K \times K}$$
  Define the tentative prolongator $\tilde{P}$ by setting block row $i$ to $Q_i$. Coarse nullspace candidate for the next level:
  $$B_c^{(i)} = R_i$$
- **GPU Architecture Implementation:**
  Do **not** call `cuSOLVER` (kernel launch overhead kills performance for $10^5$ tiny matrices). Implement an in-warp Modified Gram-Schmidt (MGS) or Householder reflector kernel:
  - 1 warp per aggregate for $m_i \le 32, K \le 6$.
  - Keep $B_i$ entirely in registers and shuffle across lanes.
- **AMGX C API Addition:**
  
  ```c
  AMGX_RC AMGX_matrix_set_nullspace(AMGX_matrix_handle mtx, int num_vectors, const double **vectors);
  ```
- **Probe:** Pass 6 analytical rigid-body modes into a cantilever beam problem ($10^6$ DOFs). Prove that $\tilde{P}^T \tilde{P} = I_K$ to machine precision ($\| \tilde{P}^T \tilde{P} - I_K \|_2 < 10^{-14}$) and that coarse nullspace candidate satisfies $\tilde{P} B_c = B$.

#### P0.5 Rigid-Body Mode Generator Utility

- **Effort:** **S**
- **Gap:** Users must manually construct spatial coordinate rotation modes.
- **Specification:** C API and internal utility taking nodal coordinates $(x_i, y_i, z_i)$ and producing:
  - 2D ($K=3$): Translations $(1, 0), (0, 1)$; Rotation $(-y_i, x_i)$.
  - 3D ($K=6$): Translations $(1,0,0), (0,1,0), (0,0,1)$; Rotations $(0, -z_i, y_i), (z_i, 0, -x_i), (-y_i, x_i, 0)$.
- **AMGCL Reference:** `amgcl/coarsening/rigid_body_modes.hpp`.
- **Probe:** Unit test checking mutual orthogonality of generated modes on a regular grid and verifying rigid-body strain energy $\mathbf{u}^T K \mathbf{u} < 10^{-10}$ with a free-floating stiffness matrix.

#### P0.6 Spectral Radius Estimator ($\rho(D^{-1} A^F)$)

- **Effort:** **S**
- **Gap:** Smoothed aggregation requires an accurate upper bound of the maximum eigenvalue of the diagonally preconditioned filtered matrix to scale Jacobi prolongation smoothing without divergence.
- **Specification:**
  - `power_iters = 0`: Fast Gershgorin circle bound:
    $$\lambda_{\max} \le \max_i \left( \frac{1}{|A_{ii}|} \sum_{j} |A_{ij}^F| \right)$$
  - `power_iters > 0` (default: 4): Batched power iteration with Rayleigh quotient on GPU:
    $$x_{k+1} = D^{-1} A^F x_k, \quad \rho \approx \frac{x_{k+1}^T x_k}{x_k^T x_k}$$
  - Calculate relaxation parameter:
    $$\omega = \text{relax} \times \frac{4}{3 \, \rho(D^{-1} A^F)} \quad (\text{default: relax} = 0.67)$$
- **AMGX Configuration Flags:** `estimate_spectral_radius=1`, `power_iters=4`, `relax=0.67`.
- **Probe:** Match analytical $\lambda_{\max} = 4.0$ on 1D discrete Laplacian within $1\%$ error using 4 power iterations.

#### P0.7 Filtered-Jacobi Prolongator Smoothing

- **Effort:** **L**
- **Gap:** AMGX prolongators are piecewise constant. High-frequency error components are not damped across level transfers, degrading convergence on nonsmooth problems.
- **Specification:** Smooth the tentative prolongator $\tilde{P}$ using the filtered matrix $A^F$:
  $$P = \left( I - \omega D^{-1} A^F \right) \tilde{P}$$
  Implementation must use a custom fused Sparse-Matrix times Tall-Skinny-Matrix kernel or AMGX SpGEMM (`A_smoother * P_tilde`).
- **AMGX Target:** Register selector `SMOOTHED_AGGREGATION` in `src/aggregation/`.
- **AMGCL Reference:** `amgcl/coarsening/smoothed_aggregation.hpp`.
- **Probe:** On a 2D biharmonic equation $\Delta^2 u = f$, standard AMGX unsmoothed aggregation diverges or requires $> 150$ iterations. Smoothed aggregation must converge within 25 iterations.

#### P0.8 Prolongator Sparsity Truncation (Anti-Dilation Gate)

- **Effort:** **M**
- **Gap:** **Critical omission in basic SA.** Applying $(I - \omega D^{-1} A^F) \tilde{P}$ triples stencil widths. Without truncation, coarse operators $A_c = P^T A P$ rapidly become dense, exhausting GPU memory by level 3.
- **Specification:** Implement post-smoothing drop tolerance:
  $$P_{ij} = \begin{cases} 0 & \text{if } |P_{ij}| < \varepsilon_{\text{drop}} \max_k |P_{ik}| \quad (\text{default: } \varepsilon_{\text{drop}} = 0.02) \\ P_{ij} & \text{otherwise} \end{cases}$$
  Followed by column-wise rescaling to ensure the near-nullspace property is strictly preserved:
  $$\sum_j P_{ij} B_{c, j} = B_i$$
- **Probe:** Profile operator complexity on a 3D 27-point stencil ($64^3$). Truncation must keep operator complexity below $1.35\times$ while unsmoothed/untruncated exceeds $2.5\times$, preserving total GPU VRAM consumption under 4 GB.

#### P0.9 Optimized Triple Product Galerkin Coarse Operator ($A_c = P^T A P$)

- **Effort:** **M**
- **Gap:** AMGX currently performs $P^T A P$ assuming binary (boolean) $P$, which reduces to row/column index summation. Smoothed $P$ has floating-point entries, requiring a true floating-point triple matrix product.
- **Specification:** Integrate a two-stage SpGEMM:
  1. $R = P^T A$ via cuSPARSE `cusparseSpGEMM` or internal AMGX hash-based SpGEMM.
  2. $A_c = R P$.
     Provide memory workspace estimation to prevent out-of-memory crashes on large GPU allocations.
- **Probe:** Zero difference in numerical values between computed $A_c$ and host reference `scipy.sparse` computation; execution time under 40 ms on 1M unknowns on an A100.

#### P0.10 Scaled Galerkin / Over-Interpolation

- **Effort:** **S**
- **Gap:** When unsmoothed aggregation must be retained for low setup cost, coarse grid corrections tend to over-correct high frequencies.
- **Specification:** Introduce an empirical damping factor $\alpha$ directly into the coarse operator:
  $$A_c = \frac{1}{\alpha} P^T A P \quad (\text{default: } \alpha = 1.5)$$
  Expose parameter `over_interp` in AMGX configuration files.
- **AMGCL Reference:** `amgcl/detail/scaled_galerkin.hpp`.
- **Probe:** On 3D isotropic Poisson, verify that setting `over_interp=1.5` reduces V-cycle count of unsmoothed aggregation by at least $25\%$.

#### P0.11 `as_scalar` Adapter for Coupled Multi-Physics

- **Effort:** **M**
- **Gap:** Many simulations provide coupled PDE matrices in scalar CSR format rather than structured BSR, but naive scalar aggregation mixes physical fields arbitrarily.
- **Specification:** Provide a logical wrapper that reindexes a scalar CSR matrix of interleaved DOFs (e.g., $u_1, v_1, p_1, u_2, v_2, p_2, \dots$) into a virtual block matrix with block dimension $B$, runs pointwise aggregation (P0.3), builds tentative and smoothed prolongators, and unpacks the resulting operators back into standard scalar CSR format.
- **Probe:** Solve an unstructured 3D Stokes system supplied as a pure scalar CSR; verify convergence rate is identical to explicit BSR input within $2\%$ runtime difference.

---

### P1: Advanced Smoothers and Relaxation

#### P1.1 Sparse Approximate Inverse: SPAI-0 and SPAI-1

- **Effort:** **M** (SPAI-0: **S**, SPAI-1: **M**)
- **Gap:** AMGX relies on Jacobi, Gauss-Seidel, or ILU($k$). Gauss-Seidel and ILU are sequential by nature and require matrix coloring, which degrades smoothing quality on highly unstructured grids. SPAI smoothers are inherently parallel and require no coloring.
- **Mathematical Specification:**
  Compute an explicit sparse matrix $M \approx A^{-1}$ minimizing $\|AM - I\|_F^2$.
  - **SPAI-0 (Static diagonal sparsity pattern):**
    $$M = \operatorname{diag}(m_i), \quad m_i = \frac{A_{ii}}{\sum_j A_{ij}^2}$$
    Completely closed-form, single-kernel GPU implementation.
  - **SPAI-1 (Sparsity pattern of $M$ matches the sparsity pattern of $A$):**
    For each row $i$, let $\mathcal{J}_i = \{j \mid A_{ij} \ne 0\}$. Solve an independent tiny dense least-squares problem:
    $$\min_{m_i} \| A(\mathcal{J}_i, \mathcal{J}_i) m_i - e_i \|_2^2$$
    Execute via batched small Givens QR in CUDA shared memory.
- **AMGX Target:** Register smoothers `SPAI0` and `SPAI1` in `src/smoothers/`.
- **AMGCL Reference:** `amgcl/relaxation/spai0.hpp`, `amgcl/relaxation/spai1.hpp`.
- **Probe:** Run on an advection-diffusion matrix with high Péclet number ($Pe = 10^4$) where Multicolor Gauss-Seidel stalls; verify SPAI-1 reduces residual by 1 order of magnitude per smoothing pass without requiring graph coloring.

#### P1.2 Dual-Threshold Incomplete LU (ILUT) and Power-Pattern ILU (ILUP)

- **Effort:** **L**
- **Gap:** AMGX's `ILU0` and `ILUK` retain fixed static structural patterns based on graph distance $k$. They fail on matrices with dynamic recirculation or localized boundary layers where fill-in must follow numerical magnitudes.
- **Specification:**
  - **ILUT (Threshold-based drop rule):** Drop entries if $|l_{ij}| < \tau |a_{ii}|$ or $|u_{ij}| < \tau |a_{ii}|$, while strictly limiting the maximum nonzeros per row to $p$. Implement via GPU parallel merge-sort or quickselect per row.
  - **ILUP (Symbolic matrix power pattern):** Compute sparsity pattern from the structural matrix power $A^p$, then compute numerical factorizations into that pre-allocated pattern on GPU.
- **AMGCL Reference:** `amgcl/relaxation/ilut.hpp`, `amgcl/relaxation/ilup.hpp`.
- **Probe:** Solve anisotropic convection-diffusion problem where standard `ILU0` fails to factorize or diverges; show stable factorization and convergence with $\tau = 10^{-3}, p = 30$.

---

### P2: Krylov Accelerators

#### P2.1 BiCGStab(L) with Polynomial Stabilization

- **Effort:** **M**
- **Gap:** AMGX provides standard `BICGSTAB` ($L=1$). Standard BiCGStab exhibits severe residual oscillations and stagnation when the system matrix has complex eigenvalues with large imaginary parts (e.g., high-frequency Helmholtz or non-Hermitian wave propagation).
- **Mathematical Specification:**
  Generalize the scalar stabilization parameter $\omega$ of BiCGStab to a degree-$L$ polynomial ($L \in [2, 8]$). Maintain a Krylov subspace $\operatorname{span}\{r, Ar, A^2 r, \dots, A^L r\}$ and minimize the residual over the subspace every $L$ steps using a small dense $L \times L$ least-squares system.
- **AMGX Target:** Register solver `BICGSTABL` with configuration parameter `bicgstab_l` (default: 2 or 4).
- **AMGCL Reference:** `amgcl/solver/bicgstabl.hpp`.
- **Probe:** High-wavenumber Helmholtz problem ($\kappa = 50$). Prove `BICGSTABL` ($L=4$) converges monotonically without the erratic spikes or stagnation observed in AMGX's default `BICGSTAB`.

#### P2.2 Augmented Restarted GMRES (LGMRES)

- **Effort:** **M**
- **Gap:** Restarted GMRES($m$) suffers from alternating convergence stalls when error vectors cycle through invariant subspaces. AMGCL includes LGMRES, which preserves approximations of the error vector across restart boundaries.
- **Specification:**
  At each restart of GMRES($m$), add $k$ error vectors $z_j = x_m - x_0$ from the previous $k$ restart cycles into the Krylov basis alongside standard Krylov vectors $v_1, \dots, v_{m-k}$.
- **AMGX Target:** Register solver `LGMRES` with parameters `gmres_m` (default: 30) and `lgmres_k` (default: 3).
- **AMGCL Reference:** `amgcl/solver/lgmres.hpp`.
- **Probe:** Test on a recirculating cavity flow matrix where `GMRES(30)` exhibits stagnation cycles; verify `LGMRES(30, 3)` breaks the limit cycle and converges in $< 3$ restarts.

---

### P3: Composite Physics Preconditioners

#### P3.1 Constrained Pressure Residual (CPR) and Dynamic Row-Sum (CPR-DRS)

- **Effort:** **L**
- **Gap:** Reservoir simulation and multiphase flow produce highly ill-conditioned coupled systems with distinct elliptic (pressure) and hyperbolic (saturation) regimes. AMGX lacks a native two-stage physics-split solver.
- **Mathematical Specification:**
  Given the blocked system:
  $$\begin{bmatrix} A_{pp} & A_{ps} \\ A_{sp} & A_{ss} \end{bmatrix} \begin{bmatrix} u_p \\ u_s \end{bmatrix} = \begin{bmatrix} f_p \\ f_s \end{bmatrix}$$
  1. **Decoupling Stage (DRS):** Compute a row-scaling transformation $F$ to eliminate weak cross-coupling terms:
     $$F = \begin{bmatrix} I & -\operatorname{diag}(A_{ps}) \operatorname{diag}(A_{ss})^{-1} \\ 0 & I \end{bmatrix}$$
  2. **True Pressure Extraction:** Form the pressure system $A_p = R A F^T$, where $R$ extracts the pressure equations.
  3. **Two-Stage Cycle:**
     - Stage 1: Solve elliptic pressure part $A_p \Delta u_p = r_p$ with AMGX Smoothed Aggregation AMG.
     - Stage 2: Correct full residual with global ILU(0) or Block-Jacobi smoother: $u \leftarrow u + M^{-1}(f - A u)$.
- **Configuration Parameters:** `cpr_block_size`, `eps_dd`, `eps_ps`, `drs_weight`.
- **AMGCL Reference:** `amgcl/preconditioner/cpr.hpp`, `amgcl/preconditioner/cpr_drs.hpp`.
- **Probe:** Run SPE10 Benchmark Model 2 reservoir dataset. AMGX single-stage AMG fails or takes $> 500$ iterations. CPR-DRS must converge within 35 outer FGMRES iterations.

#### P3.2 Schur Complement Pressure Correction (Navier-Stokes / Stokes)

- **Effort:** **L**
- **Gap:** Incompressible flow produces saddle-point systems with zero pressure diagonal blocks, rendering standard AMG coarsening mathematically invalid.
- **Specification:**
  Given:
  $$\begin{bmatrix} A & B_1^T \\ B_2 & -C \end{bmatrix} \begin{bmatrix} u \\ p \end{bmatrix} = \begin{bmatrix} f \\ g \end{bmatrix}$$
  Construct an approximate Schur complement matrix explicitly on GPU:
  $$\hat{S} = C + B_2 \left[\operatorname{diag}(A)\right]^{-1} B_1^T$$
  Implement block-triangular preconditioning:
  1. Solve $\hat{S} \Delta p = B_2 A^{-1} f - g$ using Pressure AMG.
  2. Solve $A \Delta u = f - B_1^T \Delta p$ using Velocity AMG/Smoother.
- **AMGCL Reference:** `amgcl/preconditioner/schur_pressure_correction.hpp`.
- **Probe:** 3D Lid-driven cavity Stokes problem ($C = 0$). Standard AMGX errors out on zero diagonal; Schur preconditioner converges within 20 iterations.

#### P3.3 Coarse-Space Deflated Preconditioner

- **Effort:** **M**
- **Gap:** When a matrix has a small set of isolated extremely small eigenvalues, standard multigrid smoothers and coarse grids cannot damp them, causing flat convergence tails.
- **Specification:**
  Project out a deflation subspace $V \in \mathbb{R}^{n \times k}$ (e.g., rigid-body modes or user vectors):
  $$P_{\text{def}} = I - A V (V^T A V)^{-1} V^T$$
  Apply standard AMG to the deflated operator, evaluating the coarse $k \times k$ projection directly on GPU using dense LU factorization.
- **AMGCL Reference:** `amgcl/deflated_solver.hpp`.
- **Probe:** Poisson problem with $100$ disjoint subdomains without Dirichlet grounding (singular Neumann problem with 100 zero eigenvalues). Deflated solver converges cleanly; un-deflated AMG diverges.

---

### P4: Distributed and Multi-GPU Scaling

#### P4.1 Graph-Partitioned Distributed Coarsening (ParMETIS / PT-Scotch Wrappers)

- **Effort:** **L**
- **Gap:** AMGX relies on a geometric or rank-based consolidation pass via CUDA IPC on deep levels. It lacks a graph-partitioning-based repartitioning pass for unstructured distributed meshes, causing extreme communication bottlenecks on deep coarse levels across thousands of GPUs.
- **Specification:** Integrate external graph repartitioning calls (ParMETIS / PT-Scotch) at deep multigrid levels: when the number of unknowns per MPI rank falls below 2,000, trigger distributed graph repartitioning, gather the coarse matrix onto a reduced set of active ranks, and create an MPI communicator subset.
- **Probe:** Weak scaling test up to 256 GPUs on an unstructured tetrahedral mesh. Prove setup and solve time scalability efficiency $\ge 75\%$ from 8 to 256 GPUs on level 4 and deeper.

#### P4.2 Distributed Smoothed Aggregation

- **Effort:** **L**
- **Dependencies:** Requires P0.1, P0.4, P0.6, P0.7.
- **Specification:** Extend smoothed aggregation to MPI distributed matrices (`DistributedMatrix`):
  - Distributed boundary exchange for halo-aggregate tentative prolongation.
  - Parallel communication of off-rank near-nullspace orthonormalization vectors.
  - Two-tier parallel distributed SpGEMM for $P^T A P$ across MPI ranks with overlapping GPU communication (NVLink/NCCL).
- **Probe:** Multi-GPU 3D Elasticity with 6 rigid-body modes across 16 GPUs (4 nodes $\times$ 4 H100s). Check that iteration counts are identical ($\pm 2\%$) to single-GPU execution with identical physical parameters.

---

### P5: Adapters, Reordering, and Setup Reuse

#### P5.1 Reverse Cuthill-McKee (RCM) Bandwidth Minimization Adapter

- **Effort:** **M**
- **Gap:** AMGX provides matrix coloring routines, but lacks bandwidth-reducing reordering. RCM substantially improves GPU L2 cache hit rates for sparse matrix-vector multiplication (SpMV) and incomplete factorizations.
- **Specification:** Implement a fast GPU-accelerated Reverse Cuthill-McKee permutation kernel. Reorder matrix rows and columns before hierarchy construction; automatically un-permute solution vectors upon return.
- **AMGCL Reference:** `amgcl/adapter/reorder.hpp`.
- **Probe:** On a high-aspect-ratio serpentine mesh, profile SpMV with NVIDIA Nsight Compute. RCM reordering must show $\ge 25\%$ reduction in L2 cache misses and a minimum $15\%$ throughput increase in AMG smoothing cycles.

#### P5.2 Hierarchy Numerical Rebuild vs. Structural Reuse Parity Check

- **Effort:** **M** `(verify first)`
- **Gap:** In non-linear Newton-Raphson or transient time-stepping problems, sparsity patterns remain invariant while matrix coefficients change every step. AMGCL has an ultra-fast `rebuild()` path that updates numerical values of $A_c$ and smoother factors without recomputing graph aggregation. AMGX has `structure_reuse_levels`, but its exact boundary must be benchmarked.
- **Probe:** Profile 50 consecutive non-linear iterations. If AMGX spends $> 20\%$ of hierarchy update time in symbolic structure analysis or reallocation, implement a strict numeric-only update path (`AMGX_matrix_replace_coefficients` pipeline).

---

## 2. Implementation Execution Plan

The work must proceed in four tightly gated stages. Each stage ends with an empirical validation milestone that must be met before advancing.

```
Stage 1: Core Scalar Smoothed Aggregation Engine
  [P0.1 Strength] ──► [P0.2 Aggregates] ──► [P0.6 Spectral Radius] ──► [P0.7 Filtered-Jacobi] ──► [P0.8 Truncation] ──► [P0.9 Galerkin]
  Milestone: 3D Anisotropic Diffusion convergence factor ρ ≤ 0.18 on a single GPU.

Stage 2: Vector PDEs & Block Elasticity Subsystem
  [P0.3 BSR Aggs] ──► [P0.4 Batched QR] ──► [P0.5 Rigid-Body Modes] ──► [P0.10 Over-Interp] ──► [P0.11 as_scalar]
  Milestone: 3D Cantilever Beam Elasticity (6 rigid-body modes) converges in < 30 iterations.

Stage 3: Advanced Smoothers & Composite Preconditioners
  [P1.1 SPAI-0/1] ──► [P1.2 ILUT/ILUP] ──► [P2.1 BiCGStab(L)] ──► [P2.2 LGMRES] ──► [P3.1 CPR-DRS] ──► [P3.2 Schur] ──► [P3.3 Deflated]
  Milestone: SPE10 Benchmark Model 2 converges in < 35 iterations; Stokes lid-driven cavity converges in < 20 iterations.

Stage 4: Distributed Scaling, Cache Optimization, and Production Hardening
  [P4.1 ParMETIS] ──► [P4.2 Distributed SA] ──► [P5.1 RCM Reorder] ──► [P5.2 Numeric Rebuild]
  Milestone: Weak scaling across 64+ GPUs maintains ≥ 75% parallel efficiency on unstructured meshes.
```

---

## 3. Configuration & Parameter Mapping Reference

To maintain ergonomic parity with AMGCL while integrating cleanly into AMGX, all newly exposed parameters follow the AMGX JSON/string configuration schema:

| Component       | AMGCL Parameter | AMGX Proposed Parameter    | Type    | Default | Description                                     |
| --------------- | --------------- | -------------------------- | ------- | ------- | ----------------------------------------------- |
| **Strength**    | `eps_strong`    | `aggregation_eps_strong`   | `float` | `0.08`  | Threshold for symmetric strength-of-connection  |
| **Strength**    | N/A             | `aggregation_eps_decay`    | `float` | `0.5`   | Per-level multiplier for strength threshold     |
| **Smoothing**   | `relax`         | `sa_relaxation_factor`     | `float` | `0.67`  | Jacobi prolongator relaxation multiplier        |
| **Smoothing**   | `power_iters`   | `sa_spectral_radius_iters` | `int`   | `4`     | Power iteration steps for $\rho(D^{-1}A^F)$     |
| **Truncation**  | `eps_drop`      | `sa_prolongator_drop_tol`  | `float` | `0.02`  | Drop threshold for smoothed prolongator entries |
| **Over-Interp** | `over_interp`   | `aggregation_over_interp`  | `float` | `1.5`   | Galerkin coarse scaling factor $\alpha$         |
| **Krylov**      | `L`             | `bicgstab_l`               | `int`   | `2`     | Polynomial order for BiCGStab(L)                |
| **Krylov**      | `k`             | `lgmres_augmentation_k`    | `int`   | `3`     | Number of augmentation vectors for LGMRES       |
| **CPR**         | `eps_dd`        | `cpr_diag_dominant_tol`    | `float` | `0.2`   | Diagonal dominance threshold for DRS            |
| **CPR**         | `block_size`    | `cpr_block_dim`            | `int`   | `3`     | Dimension of coupled system block               |

---

## 4. Non-Goals and Decisions Taken

1. **No Sequential AMGCL Fallback Kernels on GPU:**
   
   - *Decision:* Do not port AMGCL's CPU-based sequential loops (such as greedy MIS coarsening or row-by-row CPU ILUT) directly into CUDA device code.
   - *Reason:* Serial loops on GPU suffer from warp divergence and thread underutilization. All coarsening algorithms in AMGX must be massively parallel (matching-based, parallel independent sets, or in-warp shared-memory collectives).

2. **No Porting of AMGCL OpenCL, Vulkan, or VexCL Backends:**
   
   - *Decision:* Ignore all non-CUDA backends in AMGCL.
   - *Reason:* AMGX is dedicated exclusively to the NVIDIA hardware ecosystem (CUDA, NVLink, NCCL, Tensor Cores). Introducing hardware abstraction layers would degrade execution efficiency and complicate maintenance.

3. **No Duplication of IDR($s$) Solver:**
   
   - *Decision:* AMGCL’s `idrs` solver is excluded from porting.
   - *Reason:* AMGX already features an optimized, native multi-GPU implementation of `IDR` and `IDRMSYNC` supporting arbitrary `subspace_dim_s`.

4. **Rejection of Pure-CPU Preconditioner Setups with Device-Only Solves:**
   
   - *Decision:* The setup phase (coarsening, QR, SpGEMM) must execute entirely on the GPU.
   - *Reason:* Host-to-device transfers of multi-level matrix hierarchies for multi-million-row systems dominate execution time, erasing any GPU acceleration advantage during the solve phase.

5. **No Adoption of Deprecated Subdomain Deflation:**
   
   - *Decision:* Do not port AMGCL's legacy `amgcl::mpi::subdomain_deflation`.
   - *Reason:* Subdomain deflation exhibits sub-optimal algorithmic scaling compared to true hierarchical multigrid. Efforts are focused entirely on distributed smoothed aggregation (P4.2) and graph-repartitioned consolidation (P4.1).

6. **Preservation of Permissive Licensing Attributions:**
   
   - *Decision:* Retain full copyright notices from AMGCL (MIT, Denis Demidov) in any ported or adapted headers, housed alongside AMGX's BSD-3-Clause license. Maintain full provenance tracking in `LICENSE.txt`.

7. **Strict Enforcement of CUDA Graph Safety:**
   
   - *Decision:* Any new solver or smoother module added must contain an allocation-free execution path during the solve phase.
   - *Reason:* In-flight dynamic memory allocations (`cudaMalloc`) or asynchronous host-device synchronizations (`cudaMemcpy` with host polling) break CUDA Graph capture, which is essential for low-latency inference and transient simulation loops.
