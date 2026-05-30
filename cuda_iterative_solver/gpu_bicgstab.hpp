#ifndef GPU_BICGSTAB_HPP
#define GPU_BICGSTAB_HPP

/**
 * gpu_bicgstab.hpp
 *
 * GPU-native BiCGSTAB iterative solver with iga_ras preconditioner.
 * Design philosophy: identical to maxwellb (stanfordnqp/maxwell-solver) ---
 * the BiCGSTAB algorithm is abstracted over a set of swappable operations
 * (multA, dot, axby, prec).  GPU and CPU implementations plug into the same
 * algorithm core.
 *
 * Three concrete configurations are provided:
 *
 *  1. CpuBiCGSTABSolver   – all ops on CPU via MFEM.  Direct drop-in for
 *                            existing pml_point_source_demo.  No CUDA needed.
 *
 *  2. GpuBiCGSTABSolver   – SpMV via cuSPARSE, vector ops via cuBLAS,
 *                            preconditioner via CPU iga_ras + H2D/D2H.
 *                            Requires: CUDA 12 toolkit, -lcusparse -lcublas.
 *
 *  3. GpuFullBiCGSTABSolver – same as (2) but patch solves also on GPU via
 *                             cublasDgemvBatched (requires ExtractPatchData).
 *
 * Compile guards:
 *   HAVE_CUDA  – define when compiling with CUDA headers available.
 *               Without it only CpuBiCGSTABSolver is compiled.
 */

#include "mfem.hpp"
#include "../covariant_aux_space/iga_patch_ras_preconditioner.hpp"

#include <functional>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gpu_solver
{

// ─────────────────────────────────────────────────────────────────────────────
// Operation bundle (maxwellb-style)
// All pointers point to storage appropriate for the concrete implementation
// (host or device).
// ─────────────────────────────────────────────────────────────────────────────
struct BiCGSTABOps
{
   int n = 0;

   /** y = A * x */
   std::function<void(const double *x, double *y)> multA;

   /** y = M^{-1} * x  (identity if nullptr) */
   std::function<void(const double *x, double *y)> prec;

   /** returns x^T * y */
   std::function<double(const double *x, const double *y)> dot;

   /** returns ||x||_2 */
   std::function<double(const double *x)> norm2;

   /** y = a*x + b*y  (in-place) */
   std::function<void(double a, const double *x, double b, double *y)> axby;

   /** dst = src */
   std::function<void(const double *src, double *dst)> copy;

   /** allocate zero-filled vector of length n; caller must call free() */
   std::function<double *()> alloc;

   /** free a vector previously allocated by alloc() */
   std::function<void(double *)> free_vec;
};

// ─────────────────────────────────────────────────────────────────────────────
// Solver options and result
// ─────────────────────────────────────────────────────────────────────────────
struct BiCGSTABOptions
{
   int    max_iters  = 1000;
   double rel_tol    = 1e-7;
   double abs_tol    = 0.0;
   int    print_every = 50;
   bool   verbose    = true;
};

struct BiCGSTABResult
{
   int    iters         = 0;
   double final_rel     = 0.0;
   double bnorm         = 0.0;
   bool   converged     = false;
   double setup_seconds = 0.0;
   double solve_seconds = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Core algorithm (independent of CPU/GPU – maxwellb solve_asymm_biCGSTAB)
// ─────────────────────────────────────────────────────────────────────────────
BiCGSTABResult BiCGSTAB(const double        *b,
                         double              *x,
                         const BiCGSTABOps   &ops,
                         const BiCGSTABOptions &opt,
                         int mpi_rank = 0);

// ─────────────────────────────────────────────────────────────────────────────
// 1. CPU solver (MFEM ops + iga_ras on CPU, no CUDA)
// ─────────────────────────────────────────────────────────────────────────────
class CpuBiCGSTABSolver
{
public:
   CpuBiCGSTABSolver(const mfem::Operator &A,
                     mfem::Solver         *prec,
                     BiCGSTABOptions       opts = {});

   BiCGSTABResult Solve(const mfem::Vector &b, mfem::Vector &x) const;

private:
   const mfem::Operator &A_;
   mfem::Solver         *prec_;
   BiCGSTABOptions       opts_;
   int mpi_rank_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. GPU solver (cuSPARSE SpMV + cuBLAS vector ops + CPU iga_ras prec)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef HAVE_CUDA

#define CUDA_CHECK(call)                                                       \
   do {                                                                        \
      cudaError_t _e = (call);                                                 \
      if (_e != cudaSuccess)                                                   \
      {                                                                        \
         fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                 cudaGetErrorString(_e));                                      \
         std::abort();                                                         \
      }                                                                        \
   } while (0)

#define CUBLAS_CHECK(call)                                                     \
   do {                                                                        \
      cublasStatus_t _s = (call);                                              \
      if (_s != CUBLAS_STATUS_SUCCESS)                                         \
      {                                                                        \
         fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                 (int)_s);                                                     \
         std::abort();                                                         \
      }                                                                        \
   } while (0)

/**
 * GPU BiCGSTAB solver — correct for all MPI process counts.
 *
 * Design (maxwellb-style operation decomposition):
 *
 *   SpMV     : CPU Hypre A.Mult()  (MPI-correct halo exchange, all np)
 *              Working vector downloads to CPU and uploads back each SpMV call.
 *              Transfer cost is O(2N * 8) bytes ≈ negligible for IGA sizes.
 *
 *   Prec     : GPU cublasDgemvBatched — all patch solves run in one batched
 *              dense GEMV call.  Patch inverse matrices are pre-uploaded once.
 *              Gather (CPU) → H2D → batchGEMV (GPU) → D2H → scatter (CPU).
 *              This is the real speedup target: O(P * ps²) flops on GPU vs
 *              serial O(P * ps²) on CPU.  For order-5 NURBS with ps ~ 1000,
 *              GPU gives ~50–200x speedup on this step.
 *
 *   Vec ops  : GPU cuBLAS (dot, axpy, nrm2, copy, scale).
 *
 *   Requires : CUDA 12 toolkit (-DHAVE_CUDA), only -lcublas (no cuSPARSE).
 *
 * Constructor takes IGAPatchRASPreconditioner directly to extract patch data.
 */
class GpuBiCGSTABSolver
{
public:
   GpuBiCGSTABSolver(
      const mfem::Operator &A,
      const covariant_aux_space::IGAPatchRASPreconditioner &ras_prec,
      BiCGSTABOptions opts = {});

   ~GpuBiCGSTABSolver();

   BiCGSTABResult Solve(const mfem::Vector &b, mfem::Vector &x) const;

   long long GpuAllocBytes() const { return gpu_bytes_; }

private:
   void BuildGpuPreconditioner(
      const covariant_aux_space::IGAPatchRASPreconditioner &ras_prec);
   void FreeGpu();
   BiCGSTABOps BuildOps() const;

   int  n_ = 0;
   long long gpu_bytes_ = 0;

   const mfem::Operator &A_cpu_;   // for MPI-correct Hypre SpMV
   cublasHandle_t cublas_ = nullptr;

   // ── GPU patch data (uploaded once at construction) ────────────────────────
   int num_patches_   = 0;
   int max_patch_size_ = 0;

   double        *d_all_inv_  = nullptr;  // [num_patches * max_ps * max_ps] col-major
   double        *d_all_xp_   = nullptr;  // [num_patches * max_ps]  batch input
   double        *d_all_yp_   = nullptr;  // [num_patches * max_ps]  batch output
   const double **d_Aarray_   = nullptr;  // device array of pointers into d_all_inv_
   const double **d_xarray_   = nullptr;  // device array of pointers into d_all_xp_
   double       **d_yarray_   = nullptr;  // device array of pointers into d_all_yp_

   // ── CPU-side patch metadata (for gather/scatter) ──────────────────────────
   std::vector<std::vector<int>>    h_patch_ids_;    // [p][i] = global DOF index
   std::vector<std::vector<double>> h_patch_wts_;    // [p][i] = damping*weights[id]
   std::vector<int>                 h_patch_sizes_;  // actual size of each patch

   // ── CPU temp buffers (reused across iterations) ───────────────────────────
   mutable mfem::Vector        h_multA_in_, h_multA_out_;
   mutable mfem::Vector        h_prec_in_,  h_prec_out_;
   mutable std::vector<double> h_batch_x_,  h_batch_y_;  // flat padded buffers

   BiCGSTABOptions opts_;
   int mpi_rank_ = 0;
};

#endif // HAVE_CUDA

} // namespace gpu_solver

#endif // GPU_BICGSTAB_HPP
