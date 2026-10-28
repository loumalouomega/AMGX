# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AmgX: NVIDIA's GPU-accelerated algebraic multigrid / Krylov solver library (CUDA C++, exposed through a C API in `include/amgx_c.h`). This fork (`dev` branch) is working toward porting AMGCL features (smoothed aggregation, near-nullspace prolongation, CPR, etc.); see `doc/roadmap.md`.

## Build

Requires CMake ≥ 3.18, CUDA Toolkit ≥ 12.0, optionally MPI. CUDA ≥ 13.1 forces C++17. In-source builds are rejected.

```bash
cd build
bash configure.sh                   # Release, MPI, Ninja, archs "86;89;90;100;120"; configures AND builds+installs (not executable, run via bash)
bash configure.sh --debug --no-mpi --arch "89"
cmake --build . --parallel 16       # incremental rebuild after configuring
cmake --build . --target amgxsh     # just the shared lib
```

- `build/configure.sh` (and `configure.bat` on Windows) runs `cmake` then `cmake --build . --target install`. The default `CMAKE_INSTALL_PREFIX` is `..` relative to the build dir, i.e. it installs into the repo root's `lib/` and `include/`.
- Use `CMAKE_CUDA_ARCHITECTURES`; the old `CUDA_ARCH` flag is deprecated and translated. CMake default if unset is `90;100;120`. Restricting to one arch speeds up builds a lot — the library is a large template-instantiation build.
- Other flags: `CMAKE_NO_MPI`, `AMGX_NO_RPATH`, `MKL_ROOT_DIR`/`MAGMA_ROOT_DIR` (some eigensolvers), build types `Debug;Release;Profile;RelWithTraces`.
- Outputs: `libamgx.a` + `libamgxsh.so` (both built from the same `amgx_libs` object library; all `src/**/*.cu` except `src/tests/` and `utest.cu` are globbed in), example binaries in `build/examples/`, `build/src/amgx_tests_launcher`, and `src/configs/` copied to `build/configs/`.
- nvcc runs with `--Werror cross-execution-space-call` and `--extended-lambda`. Thrust/CUB are wrapped in namespace `amgx` (`-DTHRUST_CUB_WRAPPED_NAMESPACE=amgx`), so write `amgx::thrust::...` / use `include/thrust_wrapper.h`, not bare `thrust::`. CCCL 3 tuple shims live in `include/basic_types.h`.

## Running

```bash
# from build/
examples/amgx_capi -m ../examples/matrix.mtx -c ../src/configs/FGMRES_AGGREGATION.json
mpirun -n 2 examples/amgx_mpi_capi -m ../examples/matrix.mtx -c ../src/configs/FGMRES_AGGREGATION.json
```

Solver behaviour is driven entirely by JSON configs (`"config_version": 2`, nested `solver`/`preconditioner` objects each with a `scope`). `src/configs/` holds the reference set.

## Tests

`amgx_tests_launcher` (`src/utest.cu`) is a custom framework, not CTest:

```bash
build/src/amgx_tests_launcher --help                 # lists registered tests
build/src/amgx_tests_launcher --mode dDDI TestName   # single test, single mode
build/src/amgx_tests_launcher --key KEY --data /path/to/test_data
```

**Gotcha:** the ~60 test sources in `src/tests/*.cu` are *not* currently compiled — `src/CMakeLists.txt` only puts `testframework.cu test_utils.cu unit_test.cu` (plus `tests/smoothed_aggregation.cu`, the `SA*` tests, `tests/smoothers_p1.cu`, the SPAI/ILU tests, `tests/krylov_p2.cu`, the BiCGStab(L)/LGMRES/complex tests, `tests/composite_p3.cu`, the CPR/Schur/deflation tests, and `tests/reorder_rebuild_p5.cu`, the RCM and numeric-rebuild tests). The launcher only runs real modes (`--mode` accepts h/d DDI, DFI, FFI): exercise complex code with explicit `TemplateMode<AMGX_mode_dZZI>` types or through the C API in `tests_all`. To run a test, add its file to `tests_all`. Tests driving `AMG_Solver` directly can misbehave; prefer the C API for end-to-end solves (see `SAEndToEnd`), and set `main:monitor_residual=1` (without it Krylov solvers run to `max_iters`, can hit NaN after exact convergence, and still report success). A solver allocated from a local `AMG_Config` keeps a reference to it, so the config must outlive the solver. Many tests also expect external matrices under `--data` (default `../../test_data/`, not in the repo). `ci/test.sh` / `ci/run.sh` (HPCCM containers, old CUDA images) are WIP and tolerate test failures.

Tests are written as `DECLARE_UNITTEST_BEGIN(Name); void run() {...} DECLARE_UNITTEST_END(Name);` followed by explicit per-mode instances, e.g. `Name<TemplateMode<AMGX_mode_dDDI>::Type> Name_instance_mode_dDDI;`. Assertions: `UNITTEST_ASSERT_*` from `include/unit_test.h`.

## Architecture

**Modes / templating.** Everything is templated on a `TemplateConfig` encoding memory space + vector/matrix value types + index type. Modes are named like `dDDI` (device, double vec, double mat, int index), `dDFI`, `dFFI`, `hDDI`, and complex `dZZI` etc. (`include/amgx_config.h`, `include/basic_types.h`). Every templated class ends its `.cu` file with explicit instantiation:

```cpp
#define AMGX_CASE_LINE(CASE) template class Foo<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)        // + AMGX_FORCOMPLEX_BUILDS where supported
#undef AMGX_CASE_LINE
```

Many classes are partially specialized on host vs. device (`TemplateConfig<AMGX_host, ...>` / `<AMGX_device, ...>`) with the GPU kernels in the device specialization.

**C API → C++.** `src/amgx_c.cu` implements the public `AMGX_*` functions; it maps opaque handles to C++ objects and dispatches on the runtime `AMGX_Mode` via switch statements generated from the same `AMGX_FORALL_BUILDS` macros. `include/amgx_c.h` and `include/amgx_config.h` are the only installed headers.

**Factories + registration (central wiring point: `src/core.cu`).** Solvers, smoothers, AMG level types, cycles, aggregation/classical selectors, interpolators, coarse-A generators, convergence criteria, matrix colorings, scalers, etc. are all created through string-keyed factories (`SolverFactory<T>::registerFactory("PCG", new PCG_SolverFactory<T>)`). `core.cu` also holds `registerParameters()`, where every config parameter is declared with `AMG_Config::registerParameter<T>(name, description, default[, allowed values])`. Adding a new algorithm means: implement class + its Factory, explicit-instantiate it, register the factory in `core.cu`, and register any new parameters in `registerParameters()` there.

**Solver composition.** `Solver<T>` (`include/solvers/solver.h`) is the common base for Krylov methods, smoothers and AMG alike, so any solver can be a preconditioner, smoother or coarse solver of another, each reading parameters from its own config `scope`. `AlgebraicMultigrid_Solver` wraps `AMG` (`src/amg.cu`), which builds a hierarchy of `AMG_Level`s; the level type is `CLASSICAL` (`src/classical/`: strength, selectors, interpolators) `AGGREGATION` (`src/aggregation/`: selectors, coarseAgenerators — plain unsmoothed aggregation with implicit binary P/R), or `SMOOTHED_AGGREGATION` (`src/aggregation/sa_amg_level.cu` + `strength/`, `selectors/mis2_aggregates.cu`, `pointwise.cu`, `sa_prolongation.cu`: explicit scalar P/R, pointwise aggregation for BSR or `aggregation_block_size`, near-nullspace from `AMGX_matrix_set_nullspace`, MIS(2) seed order `aggregation_mis_priority` HASH (default) or INDEX (AMGCL's greedy order, costlier setup); single GPU only, coarse levels always scalar). Coloring-free smoothers `SPAI0`/`SPAI1` (`src/solvers/spai_solver.cu`) and `ILUT`/`ILUP` (`src/solvers/ilu_smoothers.cu`, factorized by ParILU/ParILUT sweeps in `src/solvers/parilu.cu` through a persistent `ParILUFactor` that refactors ILUP values only on an unchanged pattern, `ilu_factor_sweeps=-1` = ILUP 30 / ILUT 10, triangular solves by Jacobi iterations or cuSPARSE SpSV via `ilu_solve_method`) are scalar, single-GPU, GPU-setup only. Krylov solvers `BICGSTABL` (`bicgstab_l`) and `LGMRES` (`lgmres_augmentation_k`; basis `gmres_n_restart` Krylov + K augmentation columns, AMGCL's M + K) run in real and complex modes; `BICGSTAB`/`PBICGSTAB`/`BICGSTABL` confirm convergence with the true residual (residual replacement). Composite preconditioners `CPR` (`src/solvers/cpr_solver.cu`, `cpr_decoupling` DRS/QUASI_IMPES, inner `cpr_pressure_solver`/`cpr_global_smoother`), `SCHUR_PRESSURE_CORRECTION` (`src/solvers/schur_solver.cu`, pressure unknowns from `AMGX_matrix_set_row_markers` or `schur_pmask_pattern`, inner `schur_velocity_solver`/`schur_pressure_solver`) and `DEFLATION` (`src/solvers/deflation_solver.cu`, vectors from `AMGX_matrix_set_deflation_vectors` or the nullspace) are real-mode, single-GPU, with device setup helpers in `src/composite/submatrix.cu`. A new solver-valued parameter must be added to `m_solver_list` in `src/amg_config.cu` (else `name(scope)=...` is rejected) and to the recursion guard in `SolverFactory::allocate` (`src/solvers/solver.cu`). Config strings accept only `[A-Za-z0-9=_.+-]` in values; JSON values are unrestricted. `thrust_wrapper::copy<AMGX_device>` from a `std::vector` fails (D->D on host memory); use `amgx::thrust::copy`. Mixed-precision dDFI SpMV is unsupported for CUDA >= 10.1. `DENSE_LU_SOLVER` on a too-large coarsest level fails with `AMGX_ERR_NO_MEMORY` (dense size vs free memory); PCG needs SPD systems and symmetric smoothers (`symmetric_GS=1`) — a non-converging PCG probe is first a probe question. Gotchas: `axmb(A,x,b,r)` computes `r = b - A x` despite its name; `types::util<T>::invert(x)` is `-x`; `dot(A,a,b)`/`dotc` conjugate the first argument. Cycles live in `src/cycles/`.

**Reordering and setup reuse.** `reordering=RCM` (top-level scope) makes `AMG_Solver` (`src/amg_solver.cu`) build a permuted copy of the user's matrix with `reorder::RCM` (`src/reorder/rcm.cu`) and permute b/x around the solve; the user's matrix is never modified, resetup only re-gathers values through an nnz map. With `structure_reuse_levels != 0`, SA and CLASSICAL levels keep the sorted AP/RAP patterns in `NumericGalerkin` (`src/numeric_galerkin.cu`) and recompute only values on reuse levels (`numeric_rebuild=0` restores `csr_galerkin_product`). Setup phases carry NVTX ranges (`spgemm_symbolic`, `spgemm_numeric`, `galerkin_build`, `galerkin_numeric_refill`, `smoother_setup`, `coarse_solver_setup`, `ilu_factorize`): profile with `nsys profile --trace=cuda,nvtx` and `nsys stats --report nvtx_sum`. `levelProfile` tic/toc is a no-op outside the (broken) Profile build.

**Data structures.** `Matrix<T>` (`include/matrix.h`) is block-CSR with optional separate diagonal; `Vector<T>` wraps thrust-style storage with a custom pooled allocator (`global_thread_handle`, memory pool params in `core.cu`). `include/cusp/` is a vendored, modified CUSP.

**Distributed.** `src/distributed/` + `include/distributed/`: `DistributedManager` owns partition/halo info per matrix, `DistributedComms` (MPI host-buffer or GPU-direct) does exchanges, with consolidation of coarse levels across ranks. Compiled in only when `AMGX_WITH_MPI` is defined.

## Roadmap conventions (`doc/roadmap.md`)

When working on roadmap items, the document's rules apply:
- Sections are ordered by engineering dependency; closed items are deleted, not struck through; defects go in a "Correctness Debts" §1 that is opened on first occurrence.
- Each item has a *probe* (reproducible test/threshold) — implement and use it to verify.
- Non-goals: no serial CPU-style loops ported to device code (coarsening must be massively parallel); CUDA only, no other backends; setup must run fully on GPU; new solvers/smoothers must have an allocation-free, sync-free solve path (CUDA Graph safety); keep AMGCL MIT attribution headers in ported code.
- New parameters follow the naming in the roadmap's "Configuration & Parameter Mapping Reference" table.
