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
- **Defect-shaped items belong in a *Correctness Debts* section (§1).** Opened 2026-10-28 by C1; defects fixed in the change that found them are recorded under [Non-goals and decisions taken](#non-goals-and-decisions-taken) instead.
- **Closed items are removed entirely**, not struck through.
- **Verification status:** Items estimated without direct source inspection state `(verify first)` alongside their targeted probe.
- **[Non-goals and decisions taken](#non-goals-and-decisions-taken)** record definitive architectural choices with rationale to eliminate recurrent debate.

---

## 1. Correctness Debts

No open debts. C1 (PBICGSTAB false convergence) and C2 (SA on BSR with multicolor smoothers) are closed in decision 14.

---

## 2. Port Missing AMGCL Features to NVIDIA AMGX

### Scope & Architectural Boundary

AMGX natively supports classical AMG (Ruge-Stüben, PMIS/HMIS, aggressive coarsening), standard smoothers (Jacobi, block/CF Jacobi, Gauss-Seidel, Multicolor GS/DILU, ILU($k$), Chebyshev/KPZ polynomials), Krylov accelerators (CG, PCG, BiCGStab, GMRES, IDR($s$)), block/BSR operations, MPI rank consolidation with CUDA IPC, and setup structure reuse. **None of these are porting targets.**

AMGX's `AGGREGATION` level uses **plain, unsmoothed aggregation with binary transfer operators** ($P \in \{0, 1\}^{n \times m}$). The `SMOOTHED_AGGREGATION` level now provides AMGCL's smoothed aggregation, on one GPU and distributed with rank-local aggregates (decision 15):
- symmetric strength filter
- pointwise aggregation for BSR or interleaved scalar systems
- near-nullspace QR prolongation (`AMGX_matrix_set_nullspace`, `AMGX_generate_rigid_body_modes`)
- filtered-Jacobi smoothing with nullspace-preserving truncation
- floating-point Galerkin product

Over-interpolation is available on `AGGREGATION`. What remains of the distributed version is its multi-node validation (P4.2).

Coloring-free smoothers are also available (scalar, single GPU, setup on the GPU):
- `SPAI0` (also BSR) and `SPAI1`
- `ILUP` and `ILUT`, factorized with parallel ParILU/ParILUT sweeps, with Jacobi or exact cuSPARSE SpSV triangular solves

The goal of this section is to bring the remaining AMGCL components into AMGX's GPU pipeline: SA performance verification, advanced smoothers, Krylov accelerators and **Composite Physics Preconditioners**.

```
   [SMOOTHED_AGGREGATION level, single GPU and MPI]  (shipped)
                     │
                     ▼
    [H100 / A100 Performance Verification] (P0.12)
                     │
                     ▼
       [16-GPU Distributed SA Validation] (P4.2)
```

---

### P0: Aggregation, Prolongation, and Coarsening Engine

#### P0.12 Data-Center GPU Performance Verification of the SA Setup

- **Effort:** **S**
- **Gap:** The SA setup kernels are functionally verified (unit tests `SA*` in `src/tests/smoothed_aggregation.cu`), but their absolute timing targets were set for data-center GPUs. They have only been measured on an RTX 2000 Ada laptop GPU (8 GB).
  - The setup phases carry NVTX ranges (`sa_strength`, `sa_mis2`, `sa_filtered_operator`, `sa_spectral_radius`, `sa_tentative`, `sa_smooth_p`, `sa_truncate`).
  - Optimized on RTX 2000 Ada (4.1M-row 7-point Laplacian, iterations unchanged): SA setup 0.443 → 0.33–0.35 s.
    - Truncation 98 → 12 ms: a fused row-sum truncation (`SAProlongation::truncateRowSum`) replaces `Truncate::truncateByFactor`.
    - Prolongator smoothing 49 → 25 ms: a row-merge path for single-entry tentative rows replaces the SpGEMM.
  - What is left over all levels: MIS(2) 52–55 ms (wall, with its per-round convergence checks), strength 7.5 ms, strength + MIS kernels about 40 ms. Checking MIS convergence every 4 rounds instead of every round saved under 1 ms and was not kept.
- **Specification:** Profile `strongConnectionsKernel`, the `MIS2Aggregates` kernels and `CSR_Multiply::csr_galerkin_product` on the target hardware. Optimize the thread-per-row kernels (strength, MIS rounds, nullspace truncation) to warp-per-row variants if they dominate.
- **Probe:**
  - Strength plus MIS(2) aggregation of a 10M-row 7-point Laplacian in $< 15\text{ ms}$ on one H100. On RTX 2000 Ada, 4.1M rows take about 40 ms of kernel time over all levels.
  - $A_c = P^T A P$ for 1M unknowns in $< 40\text{ ms}$ on an A100.

---

### P1: Advanced Smoothers and Relaxation

#### P1.3 ParILUT Step Throughput

- **Effort:** **M**
- **Gap:** The redesign of decision 14 makes the ILUP setup 4–9× faster and its resetup 12× faster. ParILUT keeps the iteration counts and is 3× faster at 20 sweeps (0.688 → 0.229 s on the 65k-row probe), but a step still costs 4–20 ms as the pattern fills in.
  - Profile at 20 sweeps: the candidate key sort 30%, per-entry rank selection 16%, candidate generation (thread per row) 9%, candidate values 8%.
  - Iterations by sweep count (unchanged by the redesign):

  | Sweeps | 3 | 5 | 8 | 10 | 15 | 20 | AMGCL exact |
  |---|---|---|---|---|---|---|---|
  | Iterations | 37 | 23 | 15 | 12 | 10 | 9 | 9 |
  | Setup (s) | 0.04 | 0.05 | 0.06 | 0.09 | 0.16 | 0.23 | 0.07 |

- **Tried and reverted (2026-10-29, RTX 2000 Ada):**
  - Per-row shared-memory steps (one CTA per row: candidate gather, bitonic de-duplication, fused values and selection) instead of the global key sort and rank selection. 0.264 s at 20 sweeps against 0.233 s for the global path, after strided ownership and float |v| keys in shared memory. Occupancy is bounded by the per-row shared memory, and most rows are short.
  - Fewer pattern-selection steps than sweeps (`ilut_pattern_steps`, values-only sweeps in between): 3 steps → 33 iterations (0.040 s), 5 → 21 (0.047 s), 8 → 14 (0.062 s). That is no better than plain sweeps at equal setup time.
  - The probe needs about 20 selection steps on this matrix; at 4–20 ms per step the 0.07 s target is out of reach for ParILUT here.
- **Specification:**
  - An asynchronous Gauss-Seidel-style sweep variant behind `determinism_flag=0`, which needs fewer steps.
  - A better starting pattern (e.g. ILUP(1) values) to cut the number of selection steps.
- **Probe:** ILUT on the 65k-row anisotropic convection-diffusion probe reaches AMGCL's GMRES(50) iteration count (9) with setup at most AMGCL's serial time (0.07 s) on RTX 2000 Ada.

---

### P2: Krylov Accelerators

All P2 items are closed (decisions 11 and 14).

---

### P3: Composite Physics Preconditioners

All P3 items are closed; design choices and AMGCL-baseline measurements are in decision 12.

---

### P4: Distributed and Multi-GPU Scaling

#### P4.1 Graph-Partitioned Distributed Coarsening (ParMETIS / PT-Scotch Wrappers)

- **Effort:** **L**
- **Gap:** The graph-partitioned consolidation of `AGGREGATION` levels is implemented and verified on one GPU with up to 8 ranks (decision 15): `matrix_consolidation_partitioner=SCOTCH` groups neighbouring ranks with PT-Scotch when the rows per rank fall below `matrix_consolidation_lower_threshold` (reference config `src/configs/PCG_AGGREGATION_CONSOLIDATION.json`, threshold 2000). What is missing is the scaling evidence, which needs a cluster.
  - Not in scope: the CLASSICAL glue path and SA levels (consolidation stops SA with an error).
- **Probe:** Weak scaling test up to 256 GPUs on an unstructured tetrahedral mesh. Prove setup and solve time scalability efficiency $\ge 75\%$ from 8 to 256 GPUs on level 4 and deeper.

#### P4.2 Distributed Smoothed Aggregation

- **Effort:** **L**
- **Gap:** Distributed `SMOOTHED_AGGREGATION` is implemented (rank-local aggregates, halo tentative rows, the classical explicit-$P$ coarse path; decision 15) and verified with 1–4 MPI ranks sharing one GPU (`examples/amgx_mpi_sa_elasticity`).
  - Iterations, 1/2/3/4 ranks, exact coarse solve, true residual $< 10^{-8}$ in every run:
    - 7-point Poisson 64³: 20 / 20 / 19 / 20.
    - Clamped elasticity, 46,875 DOFs, 6 rigid-body modes: 33 / 30 / 32 / 33.
  - The ±2% criterion is met at 4 ranks. At 2 and 3 ranks elasticity takes 1–3 *fewer* iterations, since the aggregates depend on the partition.
  - Not done: the 16-GPU run, and overlapping the halo exchanges of the Galerkin product with computation (the classical path's exchanges are synchronous).
- **Probe:** Multi-GPU 3D Elasticity with 6 rigid-body modes across 16 GPUs (4 nodes $\times$ 4 H100s). Check that iteration counts are identical ($\pm 2\%$) to single-GPU execution with identical physical parameters.

---

### P5: Adapters, Reordering, and Setup Reuse

#### P5.1 RCM L2-Miss Measurement

- **Effort:** **S**
- **Gap:** The RCM reordering is implemented (`reordering=RCM`, decision 13). Its throughput probe is met on unstructured numberings. The Nsight Compute L2-miss half of the probe is not measured yet: GPU performance counters are blocked on the development machine (`ERR_NVGPUCTRPERM`; on WSL, NVIDIA Control Panel → Developer → Manage GPU Performance Counters).
- **Probe:**
  - Mesh: the 16.9M-cell serpentine of decision 13, randomly numbered.
  - Metrics: `ncu --metrics lts__t_sectors_srcunit_tex_lookup_miss.sum,lts__t_sectors_srcunit_tex.sum` on the SpMV (`csrmv`) and smoother kernels.
  - Threshold: $\ge 25\%$ fewer L2 misses with `reordering=RCM`.

---

## 3. Implementation Execution Plan

The work must proceed in four tightly gated stages. Each stage ends with an empirical validation milestone that must be met before advancing.

```
Stage 1: Core Scalar Smoothed Aggregation Engine                      [done, 2026-10-28]
  Milestone (AMGCL baseline): SA-PCG iteration counts at most 1.3x those of AMGCL CG+SA with the same smoother.
  Met on the anisotropic and isotropic 3D Poisson problems (22 vs 17 and 20 vs 17 with Jacobi, 15 vs 13 and 14 vs 13 with GS).

Stage 2: Vector PDEs & Block Elasticity Subsystem                      [done, 2026-10-28]
  Milestone (AMGCL baseline): clamped 3D elasticity with 6 rigid-body modes converges at AMGCL parity or better.
  PCG iterations, AMGX vs AMGCL CG+SA:
    15k DOFs:  19 (BSR and scalar) vs 40
    107k DOFs: 48 (BSR) / 32 (scalar) vs 54
    812k DOFs: 54 (BSR and scalar)
  Remaining: [P0.12 H100/A100 timing]

Stage 3: Advanced Smoothers & Composite Preconditioners
  [P1.1 SPAI-0/1, done] ──► [P1.2 ILUT/ILUP, done; P1.3 ParILUT throughput open] ──► [P2.1 BiCGStab(L), done] ──► [P2.2 LGMRES, done] ──► [P3.1 CPR-DRS, done] ──► [P3.2 Schur, done] ──► [P3.3 Deflated, done]
  Milestone (AMGCL baseline, decision 12):
    SPE10 Model 2 (2.24M unknowns): CPR-DRS + FGMRES(50) in 85 iterations vs AMGCL 156. The absolute < 35 is missed
      (85 iterations; 32 on Upper Ness alone). Single-stage AMG and ILU(0) do not converge in 500.
    3D lid-driven cavity Stokes, MAC: Schur pressure correction in 19 (32^3) and 20 (48^3) iterations. Met.
  Remaining: [P1.3 ParILUT step throughput]

Stage 4: Distributed Scaling, Cache Optimization, and Production Hardening
  [P4.1 PT-Scotch consolidation, done; scaling probe open] ──► [P4.2 Distributed SA, done; 16-GPU probe open] ──► [P5.1 RCM Reorder, done; L2 measurement open] ──► [P5.2 Numeric Rebuild, done]
  Milestone: Weak scaling across 64+ GPUs maintains ≥ 75% parallel efficiency on unstructured meshes.
  P5 measured (decision 13): RCM 2.6-2.8x faster CG / AMG-PCG iterations on a randomly numbered 16.9M-cell
    serpentine (no change on lexicographic numbering); coefficient-only resetup 49% (SA) / 37% (classical) faster.
  P4 measured on one GPU with up to 8 MPI ranks (decision 15): distributed SA within one iteration of single GPU
    (Poisson, elasticity); multi-rank consolidation fixed and graph-partitioned. Scaling probes need a cluster.
```

---

## 4. Configuration & Parameter Mapping Reference

To maintain ergonomic parity with AMGCL while integrating cleanly into AMGX, all newly exposed parameters follow the AMGX JSON/string configuration schema:

| Component       | AMGCL Parameter | AMGX Proposed Parameter    | Type    | Default | Description                                     |
| --------------- | --------------- | -------------------------- | ------- | ------- | ----------------------------------------------- |
| **Strength**    | `eps_strong`    | `aggregation_eps_strong`   | `float` | `0.08`  | Threshold for symmetric strength-of-connection  |
| **Strength**    | N/A             | `aggregation_eps_decay`    | `float` | `0.5`   | Per-level multiplier for strength threshold     |
| **Pointwise**   | `block_size`    | `aggregation_block_size`   | `int`   | `1`     | Interleaved unknowns per node of a scalar fine matrix (BSR uses its block size) |
| **Smoothing**   | `relax`         | `sa_relaxation_factor`     | `float` | `0.67`  | Jacobi prolongator relaxation multiplier        |
| **Smoothing**   | `power_iters`   | `sa_spectral_radius_iters` | `int`   | `4`     | Power iteration steps for $\rho(D^{-1}A^F)$     |
| **Truncation**  | `eps_drop`      | `sa_prolongator_drop_tol`  | `float` | `0.02`  | Drop threshold for smoothed prolongator entries |
| **Over-Interp** | `over_interp`   | `aggregation_over_interp`  | `float` | `1.0`   | Galerkin coarse scaling factor $\alpha$ (AMGCL uses 1.5; 1.0 keeps existing `AGGREGATION` configs unchanged) |
| **ILUT**        | `tau`           | `ilut_drop_tol`            | `float` | `1e-2`  | Drop entries not above $\tau \sum_j |a_{ij}| / \mathrm{nnz}_{\mathrm{off}}(a_i)$ |
| **ILUT**        | `p`             | `ilut_fill_factor`         | `float` | `2`     | Keep at most $p \cdot$ nnz of the lower/upper part of $a_i$ per factor row |
| **ILUP**        | `k`             | `ilup_fill_level`          | `int`   | `1`     | ILU on the pattern of $A^{k+1}$ (0 = ILU(0))    |
| **ILU**         | N/A             | `ilu_factor_sweeps`        | `int`   | `-1`    | ParILU/ParILUT fixed-point sweeps; -1 = variant default (ILUP 30, ILUT 10) |
| **ILU**         | `solve.iters`   | `ilu_solve_iters`          | `int`   | `2`     | Jacobi iterations per triangular solve          |
| **ILU**         | `solve.damping` | `ilu_solve_damping`        | `float` | `0.72`  | Damping of the Jacobi triangular solves         |
| **ILU**         | N/A             | `ilu_solve_method`         | `enum`  | `JACOBI`| `JACOBI` (approximate, graph-safe) or `CUSPARSE` (exact SpSV) |
| **Krylov**      | `L`             | `bicgstab_l`               | `int`   | `2`     | Polynomial order for BiCGStab(L)                |
| **Krylov**      | `k`             | `lgmres_augmentation_k`    | `int`   | `3`     | Augmentation vectors of LGMRES; the basis is `gmres_n_restart` + K, as AMGCL's $M + K$ |
| **Aggregation** | N/A (greedy order) | `aggregation_mis_priority` | `enum` | `HASH` | SA MIS(2) seed priority; `INDEX` gives AMGCL's greedy seeds |
| **CPR**         | `eps_dd`        | `cpr_diag_dominant_tol`    | `float` | `0.2`   | Diagonal dominance threshold for DRS            |
| **CPR**         | `eps_ps`        | `cpr_pressure_coupling_tol`| `float` | `0.02`  | Pressure-coupling threshold for DRS             |
| **CPR**         | `weights`       | `cpr_drs_weights`          | `string`| empty   | Comma-separated per-component DRS weights (AMGCL takes one per row) |
| **CPR**         | `block_size`    | `cpr_block_dim`            | `int`   | `3`     | Unknowns per cell of a scalar matrix, pressure first (BSR uses its block size) |
| **CPR**         | N/A (`cpr` vs `cpr_drs`) | `cpr_decoupling`  | `enum`  | `DRS`   | `DRS` or `QUASI_IMPES` restriction weights      |
| **CPR**         | `pprecond`      | `cpr_pressure_solver`      | solver  | `AMG`   | Solver of the pressure matrix $A_{pp}$          |
| **CPR**         | `sprecond`      | `cpr_global_smoother`      | solver  | `MULTICOLOR_DILU` | Global first stage on the full system  |
| **Schur**       | `pmask`         | `schur_pmask_pattern` / `AMGX_matrix_set_row_markers` | `string` | empty | `%start:stride`, `>start`, `<end` (JSON) or `stride.S.T`, `tail.S`, `head.E` |
| **Schur**       | `type`          | `schur_variant`            | `int`   | `1`     | 1 pressure correction (U, P, U), 2 block triangular (P, U) |
| **Schur**       | `adjust_p`      | `schur_adjust_p`           | `int`   | `2`     | Pressure matrix $K_{pp}$, $K_{pp} - \mathrm{diag}(K_{pu} M K_{up})$, $K_{pp} - K_{pu} M K_{up}$ |
| **Schur**       | `simplec_dia`   | `schur_simplec_dia`        | `int`   | `1`     | $M = 1/\sum_j |K_{uu,ij}|$ (1) or $\mathrm{diag}(K_{uu})^{-1}$ (0) |
| **Schur**       | `usolver` / `psolver` | `schur_velocity_solver` / `schur_pressure_solver` | solver | `AMG` | Velocity and pressure block solvers |
| **Deflation**   | `vec`, `nvec`   | `AMGX_matrix_set_deflation_vectors` | API | — | Deflation vectors $Z$ (column per vector) |
| **Deflation**   | N/A             | `deflation_use_nullspace`  | `int`   | `0`     | Deflate the near-nullspace of `AMGX_matrix_set_nullspace` instead |
| **Reorder**     | `adapter::reorder` | `reordering`            | `enum`  | `NONE`  | `RCM`: reverse Cuthill-McKee of the system before setup (top-level scope, single GPU) |
| **Rebuild**     | `allow_rebuild` | `numeric_rebuild`          | `int`   | `1`     | With `structure_reuse_levels != 0`, recompute only the values of $AP$ and $RAP$ (SA and CLASSICAL) |
| **Consolidation** | N/A (`mpi::repartition::scotch`) | `matrix_consolidation_partitioner` | `enum` | `RANK` | Rank groups of a consolidated `AGGREGATION` level: `RANK` (rank $i$ into $i \bmod$ groups) or `SCOTCH` (PT-Scotch partition of the rank graph) |
| **Coarse solve** | N/A            | `exact_coarse_solve` (existing) | `int` | `0`   | Distributed `DENSE_LU_SOLVER` on the whole coarse matrix instead of block Jacobi; now also for SA (set with `algorithm` in the coarse solver's scope) |

---

## 5. Non-Goals and Decisions Taken

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

8. **Smoothed Aggregation Probes Use AMGCL as the Baseline:**
   
   - *Decision:* The Stage 1/2 probes were restated as parity with AMGCL 1.4.3 (Kratos copy), run on the same matrices with the same smoother class. The absolute thresholds in the original text were dropped where AMGCL itself does not meet them:
     - V(1,1) factor $\rho \le 0.18$: AMGCL gets 0.52–0.68.
     - Biharmonic in $\le 25$ iterations: AMGCL needs 112–1000. It needs a linear-mode nullspace.
     - Power iteration within 1% in 4 steps: impossible as the spectral gap tends to 0.
     - Untruncated operator complexity above $2.5\times$: measured 1.04–1.46.
     - Over-interpolation saving $\ge 25\%$: AMGCL saves 20–22%, AMGX 8–18%. A standalone V-cycle with $\alpha = 1.5$ diverges in both.
   - *Reason:* A threshold that the reference implementation misses measures the problem, not the port.

9. **Smoothed Aggregation Design Choices:**
   
   - **Strength criterion:** AMGCL's code form, $a_{ij}^2 > \varepsilon^2 a_{ii} a_{jj}$, applied on the point matrix (max $|a_{ij}|$ per block, not the Frobenius norm) for systems. When a level has no strong connection (e.g. 27-point stencils, $1/26 < 0.08$), $\varepsilon$ keeps decaying instead of ending the hierarchy.
   - **Aggregation:** a parallel deterministic distance-2 MIS on the strong graph replaces AMGCL's serial greedy pass.
   - **Coarse levels are scalar CSR.** BSR fine matrices are expanded internally (`as_scalar`), and later levels inherit the unknowns per node, i.e. the nullspace size.
   - **Truncation:** for one constant vector it uses row-sum rescaling (`Truncate`). For block or nullspace prolongators it applies the minimum-norm correction of the kept entries that preserves $(P B_c)_i$, and keeps the row whole when that is impossible.
   - **Rigid-body modes** are centred and orthonormalized exactly. AMGCL's Gram-Schmidt assumes unit-norm translations, which they are not.
   - **`aggregation_over_interp` defaults to 1.0,** so existing `AGGREGATION` configs are unchanged.

10. **SPAI / ILU Smoother Design Choices and AMGCL-Baseline Probes:**
    
    - *Measured (P1.1),* 2D upwind convection-diffusion, global Pe = $10^4$, $128^2$ unknowns. AMGX values are exactly AMGCL's:
      - Per-sweep Richardson factor: SPAI0 0.9968, SPAI1 0.9947.
      - GMRES(50) iterations: SPAI0 347, SPAI1 294.
      - Multicolor GS stalls at 1.02 per sweep.
      - As SA V(1,1) smoothers in GMRES, AMGX does better: SPAI1 13 iterations (AMGCL 29) and SPAI0 18 (AMGCL diverges).
    - *Measured (P1.2),* anisotropic convection-diffusion ($\varepsilon_x = 10^{-3}$), $256^2$ unknowns, GMRES(50) iterations:
      - ILU(0): 299, the same as AMGCL once converged.
      - ILUP(1): 149, the same as AMGCL.
      - ILUT ($\tau = 10^{-3}$, $p = 15$, exact solves): 9, the same as AMGCL, after 20 sweeps (23 after the default 5; see P1.3).
    - *Restated probes.* The original thresholds were replaced by these parity checks:
      - "SPAI-1 reduces the residual tenfold per pass": AMGCL achieves 0.9947, not 0.1.
      - "ILU0 fails to factorize": it factorizes but needs about 300 iterations against ILUT's 9.
    - **SPAI1** solves the normal equations $(A_I A_I^T) m = A(I, i)$ per row in shared memory (warp-per-row Cholesky), with the entries taken from $C = A A^T$. This is the same minimizer as AMGCL's QR. Rows longer than 64 entries fall back to SPAI0.
    - **ILU factorization** uses parallel fixed-point sweeps (Chow–Patel ParILU; Anzt–Chow–Dongarra ParILUT) instead of cuSPARSE `csrilu02`, which is deprecated in CUDA 13.1. ILUT uses AMGCL's dropping rule, $|v| > \tau \sum_j |a_{ij}| / \mathrm{nnz}_{\mathrm{off}}$, plus the $p\cdot$nnz cap per part.
    - **Triangular solves.** `JACOBI` (AMGCL's GPU scheme) is allocation-free and graph-safe, but weak on strongly convective problems: 168 GMRES iterations against 7–9 exact. `CUSPARSE` (SpSV, analysis at setup) is exact.
    - **Damping.** The smoothers honour `relaxation_factor` (AMGX default 0.9). AMGCL parity runs set it to 1.

11. **Krylov Accelerators (P2): Design Choices, Complex Fixes and AMGCL-Baseline Probes:**
    
    - **Solvers.** `BICGSTABL` (`bicgstab_l`, default 2) and `LGMRES` (`lgmres_augmentation_k`, default 3) are right preconditioned as in AMGCL and run in real and complex modes.
      - `BICGSTABL`: one iteration is one BiCG step, as AMGCL counts. It uses AMGCL's default convex (minimal-residual) polynomial. The `delta` reliable update is not ported; instead the true residual is checked when the recursive one claims convergence (see C1).
      - `LGMRES`: the basis has `gmres_n_restart` Krylov plus `lgmres_augmentation_k` augmentation columns, AMGCL's $M + K$ (corrected in decision 14; the first version used `gmres_n_restart` as the total).
    - **Complex arithmetic defects fixed in the same change.** Complex GMRES did not converge on non-Hermitian systems because of three defects:
      - The device `amgx::dotc` called the unconjugated `cublasZdotu` (`src/blas.cu`). It now uses `Cublas::dotc`, matching the host path.
      - GMRES orthogonalized with $\langle V_{i+1}, V_k \rangle$ instead of $V_k^H V_{i+1}$.
      - The complex Givens rotation used a real sine that ignored the phase of $h_{ii}$. A corrected version now lives in `include/solvers/givens_rotation.h`, shared by GMRES and LGMRES.

      Separately, `cuDoubleComplex` × `double`/`float` scaled only the real part (`include/amgx_types/math.h`).
    - *Measured (P2.1),* complex Helmholtz $\kappa = 50$ with absorbing boundary, $128^2$. Iterations unpreconditioned / Jacobi, AMGX vs AMGCL:

      | Solver | AMGX | AMGCL |
      |---|---|---|
      | BiCGStab | 1994 / 2042 (BiCGStab(1)) | breaks down (zero $\omega$) |
      | BiCGStab(2) | 1267 / 1200 | 1244 / 1199 |
      | BiCGStab(4) | 1001 / 940 | 1300 / 1240 |
      | GMRES(30) | 1569 / 1512 | 1590 / 1591 |
      | LGMRES(30,3) | 1641 / 1698 | 1648 / 1606 |

      Centred convection, cell Péclet number about 4:
      - BiCGStab(2) and BiCGStab(4) take 383–415 iterations to a true $10^{-8}$, against 1477 for BiCGStab(1).
      - AMGCL's BiCGStab breaks down, and its BiCGStab(L) takes 275–284 iterations measured on the recursive residual.
      - GMRES(30) takes 438 in both libraries.
    - *Restated P2.1 probe.* "Monotone convergence of BiCGStab(L)" is not a property of the BiCGStab family in either library. With $L = 4$ the residual still rises 400–450 times on Helmholtz, though it overshoots its running minimum by more than 10× only 4–8 times. The probe became parity with AMGCL plus robustness where BiCGStab breaks down.
    - *Measured (P2.2),* recirculating flow (`c = Pe·(sin πx cos πy, −cos πx sin πy)`), $128^2$:
      - At Pe = 2000, unpreconditioned or Jacobi, GMRES(30) stagnates at about $10^{-1}$ in both libraries.
      - LGMRES(30,3) breaks the stagnation, to $4.8 \times 10^{-5}$ / $4.2 \times 10^{-4}$ after 5000 iterations (AMGCL: $2.5 \times 10^{-4}$ / $4.35 \times 10^{-4}$), but does not converge.
      - With ILU(0) at Pe = 2000, LGMRES cuts the iteration count: GMRES(20) 2458 → LGMRES(20,3) 847 (AMGCL 2496 → 847, identical once the basis sizes agree, decision 14).
      - "Converges in < 3 restarts" is not met by either library.

12. **Composite Preconditioners (P3): Design Choices and AMGCL-Baseline Probes:**
    
    - **Solvers.** All three are real-mode and single-GPU. Setup runs on the device through `include/composite/submatrix.h` (masked extraction, COO→CSR sort/reduce, sparse add, SpGEMM), and the apply paths allocate nothing.
      - `CPR`: AMGCL `cpr`/`cpr_drs` apply order. The global smoother acts on the full system, then the weighted pressure residual is restricted, solved and injected into component 0.
      - `SCHUR_PRESSURE_CORRECTION`: AMGCL types 1 and 2.
      - `DEFLATION`: AMGCL's `deflated_solver` apply, $x = Mb$, $x \mathrel{+}= Z E^{-1} Z^T (b - Ax)$. $E = Z^T A Z$ is LU-factorized with cuSOLVER at setup.
    - **Deviations from AMGCL:**
      - **Schur pressure solve:** it works on the *assembled* matrix. AMGX's `Solver::setup` needs a `Matrix`, so AMGCL's matrix-free $S$ (`approx_schur = false`) is not ported. The AMGCL baseline runs with `approx_schur = true`; its default gave the same 90 iterations.
      - **`schur_adjust_p = 1`** inserts the diagonal correction where $K_{pp}$ has no diagonal entry. AMGCL only updates existing diagonals, so with $C = 0$ its Jacobi pressure solve has zero pivots (no convergence in 300 iterations).
      - **`schur_adjust_p = 0` with $C = 0$:** now warns at setup.
      - **Deflation initial guess:** AMGCL projects $x_0$; a preconditioner cannot.
      - **CPR `active_rows`** is not ported.
      - **`cpr_drs_weights`** are per cell component; AMGCL's are per row.
      - **The Schur pressure mask** comes from `AMGX_matrix_set_row_markers` or from `schur_pmask_pattern`. AMGCL's `%`, `>` and `<` forms work in JSON. Config strings only accept `[A-Za-z0-9=_.+-]`, so they use `stride.S.T`, `tail.S` and `head.E`.
    - *Measured (P3.1),* SPE10 Model 2 (60×220×85, from the SPE CSP files):
      - **Problem:** a 2-phase oil-water fully implicit TPFA Jacobian with unknowns $(p, S_w)$, 2.24M unknowns. It is built at a five-spot pressure field and an upwinded flood front with $\Delta t$ = 30 days. The right-hand side is a water rate change at the central injector.
      - **Solver:** FGMRES(50) to $10^{-6}$.
      - **AMGX:** CPR-DRS with classical AMG pressure and ILU(0) (30 sweeps, exact solves) takes **85 iterations** (2.4 s on RTX 2000 Ada). With an SA pressure solver it takes 188 (AMGCL with the same scalar ILU0: 205, decision 14). Quasi-IMPES stalls.
      - **AMGCL:** `cpr_drs` with SA and block ILU0 takes **156** (25 min CPU). On the 5- and 35-layer subsets, AMGCL `cpr` (quasi-IMPES) needs 247–300.
      - Single-stage AMGX AMG and ILU(0) do not converge in 500 iterations.
      - The original "< 35" is missed on the full model (it is met on Upper Ness alone, 32). It is restated as AMGCL parity, which is met.
      - Rows with $\phi < 10^{-3}$ (7% of Tarbert, 25% of Upper Ness) make rhs = 1 tests unsolvable in both libraries. Hence the physical right-hand side.
    - *Measured (P3.2),* 3D lid-driven cavity, MAC Stokes, $C = 0$, one pressure pinned, FGMRES(50) to $10^{-6}$:
      - **`schur_adjust_p = 2`** (the roadmap's $\hat S = B\,\mathrm{diag}(A)^{-1} B^T$) with SA AMG on $\hat S$ takes 85 iterations ($32^3$) and 130 ($48^3$). AMGCL takes 90–92. A more exact pressure solve makes it worse (187), because $\hat S$ behaves like a scaled Laplacian, whereas steady Stokes needs a mass-matrix-like approximation.
      - **`schur_adjust_p = 1`** (diagonal) with Jacobi takes **19 ($32^3$) and 20 ($48^3$)**, mesh-independent. `src/configs/FGMRES_SCHUR_STOKES.json` uses it.
      - Plain AMGX AMG on the full system converges in 145. It does not "error out" as the original text assumed.
    - *Measured (P3.3):*
      - **The original probe is ill-posed.** With 100 disjoint Neumann subdomains the indicator vectors span $\ker A$, so $E = Z^T A Z = 0$ cannot be factorized (setup reports it).
      - **Restated problem:** 100 boxes (5×5×4, 80×80×64 = 410k unknowns) coupled by $10^{-6}$, Dirichlet on one face, $Z$ = box indicators.
      - **With Jacobi:** PCG drops from 442 to **89** iterations with deflation (AMGCL `deflated_solver`: 442 to 88, which is parity). FGMRES(50) drops from not converging in 2000 to 117.
      - **SA AMG** already handles the weak coupling: 17 iterations with or without deflation (AMGCL 13 / 13). The "un-deflated AMG diverges" claim does not hold.

13. **Reordering and Numeric Rebuild (P5): Design Choices and Measurements:**
    
    - **RCM** (`include/reorder/rcm.h`, `src/reorder/rcm.cu`). It runs on the symmetrized off-diagonal pattern of the (block) matrix.
      - Each connected component starts from a George-Liu pseudo-peripheral node.
      - Cuthill-McKee levels are ordered by (first parent position, degree, index), then the order is reversed.
      - AMGCL's `cuthill_mckee` (Kratos copy) starts from node 0 (which its `node > 0` test never expands), takes neighbours in column order and never reverses, so it is plain CM. The bandwidths are not comparable one to one.
      - Levels are inherently sequential. A persistent single-block kernel sweeps them and bitonic-sorts each level of up to 1024 nodes in shared memory, with no host round trip per level. Larger levels fall back to device sorts. The diameter-66k serpentine below is reordered in 2.1–2.9 s (16.9M rows), against 20–36 s with one host round trip per level.
    - **Placement:** `reordering=RCM` applies to the top-level `AMG_Solver`.
      - The user's matrix is left untouched (`AMGX_matrix_replace_coefficients` writes into it in its own order). A permuted copy and an nnz map are kept, so a resetup only gathers values.
      - b and x are permuted around the solve; norms, histories and status are unchanged.
      - The nullspace, deflation vectors and Schur row markers follow the permutation.
      - Distributed matrices are skipped with a warning.
    - *Measured (P5.1),* serpentine voxel channel (16×16 cross-section, 32 legs, cells stretched 8:1, 16.9M unknowns), RTX 2000 Ada:
      - Bandwidth goes from 1,056,512 (lexicographic over the bounding box) or 16.9M (random) to 448.
      - CG iterations (SpMV-bound): random numbering 61.4 → 21.9 ms (2.8×).
      - PCG + SA (Jacobi, 2+2 sweeps), 8.5M unknowns: random numbering 145.8 → 55.4 ms per iteration (2.6×).
      - Lexicographic numbering: no gain (−2% to +3%). The bounding-box order already has the channel's locality.
      - Problems up to about 2M unknowns show no effect, since their vectors fit in the 32 MB L2.
      - Iteration counts stay within ±1.
      - The L2-miss counters are P5.1.
    - *Measured (P5.2),* 1M-row 7-point diffusion with coefficients changing every step, 20 `replace_coefficients` + `resetup` steps:
      - **Structural reuse already exists:** with `structure_reuse_levels=-1` a resetup costs 26 ms (SA, Jacobi) or 147 ms (classical PMIS/D2), against 84 / 328 ms for a full rebuild.
      - **Profile:** NVTX ranges `spgemm_symbolic`/`spgemm_numeric`/`smoother_setup`/`coarse_solver_setup`/`ilu_factorize` with nsys put the symbolic SpGEMM phase, including its host synchronizations, at about 60% (SA) and 66% (classical) of a resetup. That is above the 20% threshold.
      - **Implemented:** `NumericGalerkin` (`include/numeric_galerkin.h`).
        - At the first build with structure reuse it keeps the sorted $AP$ and $RAP$ patterns.
        - On reuse levels it recomputes only their values. Thread groups per row are sized like `csr_galerkin_product` and accumulate with shared/global atomics; there is no symbolic pass, allocation or host sync.
        - SA resetup goes from 26.7 to **13.7 ms (−49%)**, classical from 146.7 to **93.1 ms (−37%)**, with identical iteration counts.
        - The first setup costs more because of the pattern sort: 0.14 → 0.16 s (SA), 0.39 → 0.52 s (classical).
        - `numeric_rebuild=0` restores the old path.
      - **AMGCL reference (CPU, same sequence):** `rebuild()` takes 229 ms against 395 ms for a full setup.
      - **Not changed:**
        - Smoother and coarse-solver setup stay under 2% of a resetup with Jacobi/DILU.
        - With ILUP the resetup is dominated by the factorization itself (P1.3).
        - P and R stay those of the first matrix on reuse levels (SA: the smoothing is not redone), as in AMGCL's `rebuild()`.

14. **Correctness Debts and Open Items (C1, C2, P0.13, P1.3, P2.3):**
    
    - **C1 (fixed).** `PBICGSTAB` and `BICGSTAB` confirm every convergence claim with the true residual $b - Ax$ and restart the recurrences from it otherwise (residual replacement).
      - Both exits are covered: the half-step exit on $\|s\|$ and the end of the iteration.
      - `BICGSTAB` also tested the norm of $s$ instead of $r$ at the end of the step, and skipped restoring the matrix view.
      - On the C1 probe all variants now stop at a true $8.2 \times 10^{-9}$ after 1507 iterations; before, they reported SUCCESS at $1.06 \times 10^{-2}$. `KrylovP2Solvers` checks this.
    - **C2 (not a defect).** The probe matrix was indefinite: its diagonal blocks $d I + 0.01 d (J - I)$ make $x = (u, -u, 0)$ a negative direction. `MULTICOLOR_GS` was also run without `symmetric_GS`, which is not valid in PCG.
      - On an SPD version, SA converges in 12 (Jacobi), 9 (`MULTICOLOR_DILU`) and 8 (symmetric `MULTICOLOR_GS`) iterations. AGGREGATION takes 21 / 13 / 13.
      - `compute-sanitizer` is clean.
      - The `max_levels=1` "kernel launch error" was a 275 GB dense coarse matrix. `DENSE_LU_SOLVER` now reports it as `AMGX_ERR_NO_MEMORY` with the size and the free memory.
    - **P0.13 (closed).** Against a like-for-like AMGCL baseline (`cpr_drs` with scalar ILU0, as the AMGX run uses), full SPE10 takes 205 iterations in AMGCL and 188 in AMGX's SA pressure solver. The earlier 156 used AMGCL's block ILU0.
      - The aggregation density was not the cause. `aggregation_mis_priority=INDEX` (the lexicographically first MIS(2), i.e. AMGCL's greedy seeds; the unit test checks it against a host greedy pass) matches AMGCL's level-1 size: 14288 vs 14316 rows, against 12314 with HASH.
      - With INDEX the SPE10 iterations are mixed: 45 → 50 (5 layers), 361 → 172 (Upper Ness), 188 → 274 (full model).
      - INDEX converges better on the P0 Poisson probes: aniso7 64³ 23 → 16 (AMGCL 17), poisson7 64³ 20 → 17 (AMGCL 17), aniso7 128³ 27 → 18. But its round count grows with the dependency chains of the index order, so setup is 1.2–1.6× at 64³ and 2.6–4.4× at 128³. HASH stays the default.
    - **P1.3 (redesigned; ParILUT throughput open).** The factorization now keeps a persistent row-sorted pattern `ParILUFactor` (`include/solvers/parilu.h`).
      - **Sweeps:** each sweep is one deterministic thread-per-entry kernel (binary searches in sorted rows; no SpGEMM, sort, allocation or sync).
      - **ILUP** keeps its pattern and refactors values only when A's pattern is unchanged; the cuSPARSE solve then refreshes in place (`cusparseSpSV_updateMatrix`).
      - **ParILUT steps** generate candidates with one bit-limited radix sort and select with per-entry ranks.
      - **Results** (iterations unchanged everywhere):
        - ILUP(0)/(1) setup inside SA (1M rows): 0.77 → 0.20 s and 9.7 → 1.07 s.
        - ILUP resetup with `structure_reuse_levels=-1`: 521 → 43 ms and 6.9 → 0.61 s.
        - CPR setup on full SPE10 (ILU(0), 30 sweeps): 5.8 → 0.91 s.
        - ParILUT (65k probe, 20 sweeps): 0.688 → 0.229 s.
      - **Sweep defaults** are now per variant (`ilu_factor_sweeps=-1`): ILUP 30, because extra ILUP sweeps cost almost nothing (+7 ms for 60 on 65k rows) and SPE10 needs 30; ILUT 10 (12 iterations for 0.09 s).
    - **P2.3 (not a defect).** AMGCL's LGMRES basis holds $M + K$ columns; AMGX used `gmres_n_restart` as the total. AMGX LGMRES(20,3) was therefore AMGCL's (17,3), which stalls identically in AMGCL: 0.918 after 3000 iterations, and (13,3) / (7,3) match AMGX (16,3) / (10,3) at 0.964 / 0.995.
      - With the basis now `gmres_n_restart` + K, LGMRES(20,3) with ILU(0) at Pe = 5000 takes **874** iterations (AMGCL 874, the same residual $9.09 \times 10^{-9}$); (30,3) takes 660 (AMGCL 660); Pe = 2000 takes 847 (AMGCL 847).
      - The inexact 5-sweep ILU(0) was not the cause: 30 and 60 sweeps stall the same way under the old sizes.

15. **Distributed SA and Coarse-Level Consolidation (P4.2, P4.1): Design Choices and Measurements:**
    
    - **Rank-local aggregates (P4.2).** MIS(2) and its assignment passes ignore halo columns, as the `AGGREGATION` path does, so each aggregate lives on one rank. The setup's decisions are global so that every rank takes them together: the ε retry when no strong connection exists, and the end of the hierarchy (the level reports its fine size when the global aggregate count is 0).
      - Strength uses the halo diagonal (one exchange). Pointwise aggregation keeps halo couplings out of the point matrix and gives them the scalar criterion afterwards.
      - The spectral radius uses the distributed SpMV, global reductions, and a start vector hashed on global row ids, so it does not depend on the partition.
    - **Prolongator.** The tentative rows of the halo nodes are exchanged at a fixed width (global columns, values, and the coarse nullspace rows of their columns). $P_{tent}$ is extended to the FULL fine rows with the halo coarse columns after the owned ones. Smoothing and truncation are local, and P's manager comes from its global columns (`initialize_manager_from_global_col_indices`), as in distance-2 classical interpolation.
      - With $K > 1$ coarse unknowns per aggregate, each P row is padded with explicit zeros to whole aggregates. Truncation drops entries one by one, so otherwise the unknowns of an aggregate get different patterns in $RAP$, and the interior/boundary renumbering of the coarse level separates them. The coarse setup asserts that each node's unknowns stay consecutive.
    - **Coarse level.** The classical explicit-$P$ path step for step: `exchange_halo_rows_P`, R = Pᵀ over the owned rows with a stub manager, `csr_galerkin_product`, `exchange_RAP_ext`, compression of unused halo columns, `renumberMatrixOneRing`, `renumber_P_R`, `createOneRingHaloRows`. The user and coarse nullspaces follow the renumberings. Transfers use the masked products of the classical path.
      - **Scope.** Scalar fine matrices, including interleaved systems with `aggregation_block_size`; distributed BSR input fails with a message. `NumericGalerkin` is off when distributed, while structure reuse works (resetup 0.65 → 0.24 s on 4 ranks). SA levels are not consolidated.
    - **Coarse solve.** The distributed `DENSE_LU_SOLVER` is block Jacobi over the ranks unless `exact_coarse_solve=1`, which was gated to CLASSICAL. It now also accepts SMOOTHED_AGGREGATION, whose coarse levels carry the same `local_to_global_map`. The flag and `algorithm` are read in the coarse solver's own scope (see `src/configs/PCG_SA_ELASTICITY.json`).
      - This is what makes iterations independent of the rank count. With block Jacobi, elasticity took 33 / 41 / 50 / 54 iterations on 1–4 ranks; with the exact solve it takes 33 / 30 / 32 / 33. Poisson 64³ goes from 20 / 21 / 21 / 23 to 20 / 20 / 19 / 20.
      - AMGX also stops coarsening when the *smallest* rank falls below `min_coarse_rows`/`dense_lu_num_rows`; `use_sum_stopping_criteria=1` makes the hierarchy the same as on one GPU.
    - *Measured (P4.2), one RTX 2000 Ada, ranks sharing it (`mpirun --oversubscribe`), `examples/amgx_mpi_sa_elasticity` with the true residual:* Poisson 64³ 20 / 20 / 19 / 20 iterations on 1–4 ranks. Clamped elasticity 24³ elements (46,875 DOFs): 33 / 30 / 32 / 33 with the 6 rigid-body modes, 59 / 58 / 56 / 59 without them (pointwise, 3 unknowns per node).
    - **Consolidation fixes (P4.1).** Three defects kept multi-root consolidation of `AGGREGATION` levels from ever running:
      - `computeDestinationPartitions` did not store its rank → root map with `wantNeighbors=false` (the only caller's setting). Every rank merged onto rank 0 whatever `matrix_consolidation_upper_threshold` asked for.
      - The consolidated level's `B2L_rings` had one entry too many and were sized from the level's B2L copy, which had just been cleared. The next consolidation crashed in the halo exchange.
      - The consolidated level never initialized its B2L offsets (`initialize_B2L_maps_offsets`), so the first solve with neighbours left crashed in `gather_B2L_v2`.
    - **SCOTCH partitioner** (`include/distributed/repartition.h`). The ranks are graph vertices, weighted by the rows they bring to the consolidated level; their neighbours are edges, weighted by the halo sent plus received. `SCOTCH_dgraphPart` splits the graph into the groups the thresholds ask for, and every rank merges into the lowest rank of its group, so rank 0 stays active and its output covers the hierarchy.
      - It fails over to RANK if PT-Scotch is absent (`CMAKE_NO_PTSCOTCH`, or not found) or the rank graph is not symmetric.
      - PT-Scotch rather than ParMETIS: the user's choice, with a free licence and an MPI-native API.
    - *Measured (P4.1), AGGREGATION + PCG, consolidation 600/700–1300:*
      - **Groups.** SCOTCH groups neighbouring ranks where RANK's $i \bmod g$ does not:

        | Rank layout | RANK groups | SCOTCH groups |
        |---|---|---|
        | 4×1×1 strip | {0,2} {1,3} | {0,1} {2,3} |
        | 4×2×1 | x even / odd | x ∈ {0,1} / {2,3} |
        | 8-rank chain | {0,2,4,6} {1,3,5,7} | {0–3} {4–7} |
        | 2×2×2 (both connected) | x halves | z halves |

      - **Iterations.** Consolidation raises the iterations, as single-root consolidation already did. Without it: 22–27. Single root: 24–32. RANK multi-root: 23–29. SCOTCH: 24–32. The true residuals agree with the tolerance in every run. On one GPU the partitioner cannot show its communication benefit; that is the P4.1 scaling probe.
      - **12-row `examples/matrix.mtx` on 4 ranks, stalling at 1.19: not a defect.** The last rank's 3×3 diagonal block of this nonsymmetric, indefinite matrix is singular. The single-level AMG is then a block Jacobi with an undefined block. Unpreconditioned FGMRES(5) stalls identically on 1, 2 and 4 ranks, so the distributed operator is consistent.
