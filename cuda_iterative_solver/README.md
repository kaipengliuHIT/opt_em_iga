# CUDA Iterative Solver Sandbox

This folder is a separate sandbox for a future pure C++/CUDA sparse iterative
solver backend. It is intentionally decoupled from the current MFEM/IGA driver
so that the paper-critical CPU auxiliary-space experiments can continue while
the GPU path matures.

## Intended role

The near-term solver target is a complex CSR Krylov backend for large 3D
Maxwell demo problems:

- CSR sparse matrix storage on device.
- GMRES or FGMRES iteration.
- cuSPARSE SpMV.
- cuBLAS vector operations.
- Optional right preconditioner hook for diagonal, block diagonal, or later
  auxiliary-space preconditioners.

This is different from Hypre AMS comparison. The AMS baseline should remain a
CPU experiment unless the exact hypre/MFEM stack being used proves that AMS has
a production-quality GPU path for the required Nedelec setup.

## External solver candidates

- AmgX: useful as a CUDA sparse solver/preconditioner baseline, especially if
  the matrix can be exported in CSR.
- Ginkgo: modern C++ sparse linear algebra with CUDA/HIP/OpenMP executors and
  Krylov solvers.
- cuSPARSE/cuBLAS/cuSOLVER: low-level building blocks for an in-repo solver.

The recommended development path is:

1. Export one assembled MFEM complex system to CSR.
2. Reproduce the CPU GMRES residual history on CPU with the same CSR data.
3. Move SpMV and vector kernels to CUDA.
4. Add preconditioner hooks only after the unpreconditioned GPU GMRES is
   numerically consistent.

## Current local note

`nvcc` was not found on the current PATH during inspection. The repo has CUDA
include/library paths in the Makefile, but this sandbox should not be wired into
the default build until the local CUDA compiler/toolkit path is confirmed.
