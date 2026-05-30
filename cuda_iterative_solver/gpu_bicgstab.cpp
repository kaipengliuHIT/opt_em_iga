#include "gpu_bicgstab.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mpi.h>

namespace gpu_solver
{

// ═════════════════════════════════════════════════════════════════════════════
// Core BiCGSTAB algorithm  (maxwellb solve_asymm_biCGSTAB, C++ translation)
//
//  The algorithm operates purely through the BiCGSTABOps function bundle.
//  It knows nothing about CPU vs GPU – that distinction lives in the ops.
//
//  Reference: Van der Vorst (1992), "Bi-CGSTAB: A Fast and Smoothly
//  Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear
//  Systems", SIAM J. Sci. Stat. Comput., 13(2), 631–644.
// ═════════════════════════════════════════════════════════════════════════════
BiCGSTABResult BiCGSTAB(const double         *b,
                         double               *x,
                         const BiCGSTABOps    &ops,
                         const BiCGSTABOptions &opt,
                         int                   mpi_rank)
{
   BiCGSTABResult res;
   res.bnorm = ops.norm2(b);

   const double tol = (res.bnorm > 0.0) ?
                      opt.rel_tol * res.bnorm : opt.abs_tol;

   // Allocate workspace vectors
   double *r     = ops.alloc();  // current residual
   double *rhat  = ops.alloc();  // shadow residual (fixed)
   double *p     = ops.alloc();  // search direction
   double *phat  = ops.alloc();  // M^{-1} p
   double *v     = ops.alloc();  // A * phat
   double *s     = ops.alloc();  // r - alpha*v
   double *shat  = ops.alloc();  // M^{-1} s
   double *t     = ops.alloc();  // A * shat

   auto tick = std::chrono::high_resolution_clock::now;
   auto t0   = tick();

   // r = b - A*x
   ops.multA(x, r);
   ops.axby(1.0, b, -1.0, r);   // r = b - A*x  ← r = 1*b + (-1)*r

   // rhat = r0  (fixed shadow residual)
   ops.copy(r, rhat);

   double rho_prev = 1.0, alpha = 1.0, omega = 1.0;
   // p = 0, v = 0 (already zero from alloc)

   for (int k = 0; k < opt.max_iters; k++)
   {
      const double r_norm = ops.norm2(r);
      const double r_rel  = (res.bnorm > 0.0) ? r_norm / res.bnorm : r_norm;

      if (mpi_rank == 0 && opt.verbose &&
          (k == 0 || (k + 1) % opt.print_every == 0))
      {
         std::cout << "[bicgstab] iter=" << (k + 1)
                   << "  ||r||/||b||=" << r_rel << std::endl;
      }

      if (r_norm < tol || (opt.abs_tol > 0.0 && r_norm < opt.abs_tol))
      {
         res.iters      = k;
         res.final_rel  = r_rel;
         res.converged  = true;
         break;
      }

      // ── rho = (rhat, r) ──────────────────────────────────────────────────
      const double rho = ops.dot(rhat, r);
      if (std::abs(rho) < 1e-300)
      {
         if (mpi_rank == 0) { std::cerr << "[bicgstab] breakdown: rho=0\n"; }
         res.iters = k; res.final_rel = r_rel; res.converged = false;
         break;
      }

      // ── β = (rho / rho_prev) * (alpha / omega) ───────────────────────────
      const double beta = (rho / rho_prev) * (alpha / omega);

      // ── p = r + β*(p - ω*v)  ─────────────────────────────────────────────
      ops.axby(-omega, v, 1.0, p);        // p = -ω*v + 1*p  (in-place)
      ops.axby(1.0, r, beta, p);          // p = 1*r + β*p   (in-place)

      // ── phat = M^{-1} p ──────────────────────────────────────────────────
      if (ops.prec) { ops.prec(p, phat); }
      else          { ops.copy(p, phat); }

      // ── v = A * phat ──────────────────────────────────────────────────────
      ops.multA(phat, v);

      // ── alpha = rho / (rhat, v) ───────────────────────────────────────────
      const double rhatv = ops.dot(rhat, v);
      if (std::abs(rhatv) < 1e-300)
      {
         if (mpi_rank == 0) { std::cerr << "[bicgstab] breakdown: rhat·v=0\n"; }
         res.iters = k; res.final_rel = r_rel; res.converged = false;
         break;
      }
      alpha = rho / rhatv;

      // ── s = r - alpha*v ───────────────────────────────────────────────────
      ops.copy(r, s);                     // s = r
      ops.axby(-alpha, v, 1.0, s);       // s = -α*v + s

      const double s_norm = ops.norm2(s);
      if (s_norm < tol)
      {
         // Early termination: x = x + alpha*phat
         ops.axby(alpha, phat, 1.0, x);
         res.iters     = k + 1;
         res.final_rel = (res.bnorm > 0.0) ? s_norm / res.bnorm : s_norm;
         res.converged = true;
         break;
      }

      // ── shat = M^{-1} s ──────────────────────────────────────────────────
      if (ops.prec) { ops.prec(s, shat); }
      else          { ops.copy(s, shat); }

      // ── t = A * shat ──────────────────────────────────────────────────────
      ops.multA(shat, t);

      // ── omega = (t, s) / (t, t) ───────────────────────────────────────────
      const double tt = ops.dot(t, t);
      omega = (std::abs(tt) > 1e-300) ? (ops.dot(t, s) / tt) : 0.0;

      // ── x = x + alpha*phat + omega*shat ───────────────────────────────────
      ops.axby(alpha, phat, 1.0, x);     // x += α*phat
      ops.axby(omega, shat, 1.0, x);     // x += ω*shat

      // ── r = s - omega*t ───────────────────────────────────────────────────
      ops.copy(s, r);
      ops.axby(-omega, t, 1.0, r);       // r = s - ω*t

      rho_prev = rho;

      if (k == opt.max_iters - 1)
      {
         res.iters     = k + 1;
         res.final_rel = ops.norm2(r) / (res.bnorm > 0.0 ? res.bnorm : 1.0);
         res.converged = false;
      }
   }

   res.solve_seconds = std::chrono::duration<double>(tick() - t0).count();

   ops.free_vec(r);
   ops.free_vec(rhat);
   ops.free_vec(p);
   ops.free_vec(phat);
   ops.free_vec(v);
   ops.free_vec(s);
   ops.free_vec(shat);
   ops.free_vec(t);

   return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. CPU BiCGSTAB (MFEM ops, no CUDA dependency)
// ═════════════════════════════════════════════════════════════════════════════
CpuBiCGSTABSolver::CpuBiCGSTABSolver(const mfem::Operator &A,
                                       mfem::Solver         *prec,
                                       BiCGSTABOptions       opts)
   : A_(A), prec_(prec), opts_(opts)
{
   mpi_rank_ = mfem::Mpi::WorldRank();
}

BiCGSTABResult CpuBiCGSTABSolver::Solve(const mfem::Vector &b,
                                          mfem::Vector       &x) const
{
   const int n = b.Size();
   MFEM_VERIFY(n == A_.Height(), "CpuBiCGSTAB: b size != A height");
   x.SetSize(n);

   static mfem::Vector tmp_A(n), tmp_M(n);
   tmp_A.SetSize(n);
   tmp_M.SetSize(n);

   BiCGSTABOps ops;
   ops.n = n;

   ops.multA = [&](const double *xp, double *yp)
   {
      mfem::Vector xv(const_cast<double *>(xp), n);
      mfem::Vector yv(yp, n);
      A_.Mult(xv, yv);
   };

   if (prec_)
   {
      ops.prec = [&](const double *xp, double *yp)
      {
         mfem::Vector xv(const_cast<double *>(xp), n);
         mfem::Vector yv(yp, n);
         prec_->Mult(xv, yv);
      };
   }

   ops.dot = [&](const double *xp, const double *yp) -> double
   {
      double local = 0.0;
      for (int i = 0; i < n; i++) { local += xp[i] * yp[i]; }
      double global = local;
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return global;
   };

   ops.norm2 = [&](const double *xp) -> double
   {
      return std::sqrt(ops.dot(xp, xp));
   };

   ops.axby = [&](double a, const double *xp, double b_coef, double *yp)
   {
      for (int i = 0; i < n; i++) { yp[i] = a * xp[i] + b_coef * yp[i]; }
   };

   ops.copy = [&](const double *src, double *dst)
   {
      std::memcpy(dst, src, n * sizeof(double));
   };

   ops.alloc = [&]() -> double *
   {
      double *p = new double[n]();  // zero-init
      return p;
   };

   ops.free_vec = [](double *p) { delete[] p; };

   return BiCGSTAB(b.GetData(), x.GetData(), ops, opts_, mpi_rank_);
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. GPU BiCGSTAB
//    SpMV : CPU Hypre A.Mult  (MPI-correct, all np)
//    Prec : GPU cublasDgemvBatched  (batch patch solve, main speedup)
//    Vecs : GPU cuBLAS
// ═════════════════════════════════════════════════════════════════════════════
#ifdef HAVE_CUDA

GpuBiCGSTABSolver::GpuBiCGSTABSolver(
   const mfem::Operator &A,
   const covariant_aux_space::IGAPatchRASPreconditioner &ras_prec,
   BiCGSTABOptions opts)
   : A_cpu_(A), opts_(opts)
{
   mpi_rank_ = mfem::Mpi::WorldRank();
   n_ = A.Height();

   CUBLAS_CHECK(cublasCreate(&cublas_));

   h_multA_in_.SetSize(n_);
   h_multA_out_.SetSize(n_);
   h_prec_in_.SetSize(n_);
   h_prec_out_.SetSize(n_);

   BuildGpuPreconditioner(ras_prec);
}

GpuBiCGSTABSolver::~GpuBiCGSTABSolver()
{
   FreeGpu();
   if (cublas_) { cublasDestroy(cublas_); }
}

void GpuBiCGSTABSolver::BuildGpuPreconditioner(
   const covariant_aux_space::IGAPatchRASPreconditioner &ras_prec)
{
   using PD = covariant_aux_space::IGAPatchRASPreconditioner::PatchData;
   PD pd = ras_prec.ExtractPatchData();

   num_patches_    = pd.num_patches;
   max_patch_size_ = pd.max_patch_size;
   const int P  = num_patches_;
   const int ps = max_patch_size_;

   if (mpi_rank_ == 0)
   {
      std::cout << "[gpu_bicgstab] patches=" << P
                << "  max_patch_size=" << ps
                << "  damping=" << pd.damping << "\n";
   }

   // ── Build CPU patch metadata ──────────────────────────────────────────────
   h_patch_ids_.resize(P);
   h_patch_wts_.resize(P);
   h_patch_sizes_.resize(P);
   for (int p = 0; p < P; p++)
   {
      const int m = static_cast<int>(pd.ids[p].size());
      h_patch_sizes_[p] = m;
      h_patch_ids_[p]   = pd.ids[p];
      h_patch_wts_[p].resize(m);
      for (int i = 0; i < m; i++)
      {
         const int id = pd.ids[p][i];
         h_patch_wts_[p][i] = pd.damping * pd.weights[id];
      }
   }

   // ── Pack patch inverses into padded column-major host array ───────────────
   // Layout: inv_p is ps×ps col-major; patch p starts at p * ps * ps
   // For patch of actual size m < ps, entries outside [0..m-1, 0..m-1] = 0
   std::vector<double> h_all_inv((long long)P * ps * ps, 0.0);
   for (int p = 0; p < P; p++)
   {
      const int m = h_patch_sizes_[p];
      const mfem::DenseMatrix &inv = pd.inv_mats[p];  // m×m col-major
      for (int j = 0; j < m; j++)
      {
         for (int i = 0; i < m; i++)
         {
            h_all_inv[(long long)p * ps * ps + j * ps + i] = inv(i, j);
         }
      }
   }

   // ── Upload patch inverses to GPU ──────────────────────────────────────────
   const long long inv_bytes  = (long long)P * ps * ps * sizeof(double);
   const long long vec_bytes  = (long long)P * ps       * sizeof(double);
   const long long ptr_bytes  = (long long)P             * sizeof(void *);

   CUDA_CHECK(cudaMalloc(&d_all_inv_, inv_bytes));
   CUDA_CHECK(cudaMalloc(&d_all_xp_,  vec_bytes));
   CUDA_CHECK(cudaMalloc(&d_all_yp_,  vec_bytes));
   CUDA_CHECK(cudaMemcpy(d_all_inv_, h_all_inv.data(), inv_bytes,
                         cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemset(d_all_xp_, 0, vec_bytes));
   CUDA_CHECK(cudaMemset(d_all_yp_, 0, vec_bytes));

   // ── Build device pointer arrays for cublasDgemvBatched ───────────────────
   std::vector<const double *> h_Aarray(P), h_xarray(P);
   std::vector<double *>       h_yarray(P);
   for (int p = 0; p < P; p++)
   {
      h_Aarray[p] = d_all_inv_ + (long long)p * ps * ps;
      h_xarray[p] = d_all_xp_  + (long long)p * ps;
      h_yarray[p] = d_all_yp_  + (long long)p * ps;
   }

   CUDA_CHECK(cudaMalloc(&d_Aarray_, ptr_bytes));
   CUDA_CHECK(cudaMalloc(&d_xarray_, ptr_bytes));
   CUDA_CHECK(cudaMalloc(&d_yarray_, ptr_bytes));
   CUDA_CHECK(cudaMemcpy(d_Aarray_, h_Aarray.data(), ptr_bytes,
                         cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_xarray_, h_xarray.data(), ptr_bytes,
                         cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_yarray_, h_yarray.data(), ptr_bytes,
                         cudaMemcpyHostToDevice));

   gpu_bytes_ = inv_bytes + 2 * vec_bytes + 3 * ptr_bytes;

   // CPU flat batch buffers
   h_batch_x_.assign((long long)P * ps, 0.0);
   h_batch_y_.assign((long long)P * ps, 0.0);

   if (mpi_rank_ == 0)
   {
      std::cout << "[gpu_bicgstab] patch_inv gpu_MB="
                << gpu_bytes_ / (1024 * 1024) << "\n";
   }
}

void GpuBiCGSTABSolver::FreeGpu()
{
   if (d_all_inv_) { cudaFree(d_all_inv_); d_all_inv_ = nullptr; }
   if (d_all_xp_)  { cudaFree(d_all_xp_);  d_all_xp_  = nullptr; }
   if (d_all_yp_)  { cudaFree(d_all_yp_);  d_all_yp_  = nullptr; }
   if (d_Aarray_)  { cudaFree(d_Aarray_);  d_Aarray_  = nullptr; }
   if (d_xarray_)  { cudaFree(d_xarray_);  d_xarray_  = nullptr; }
   if (d_yarray_)  { cudaFree(d_yarray_);  d_yarray_  = nullptr; }
}

BiCGSTABOps GpuBiCGSTABSolver::BuildOps() const
{
   BiCGSTABOps ops;
   ops.n = n_;

   // ── multA: CPU Hypre A.Mult — MPI-correct for all np ─────────────────────
   // Pattern: D2H → CPU A.Mult → H2D
   ops.multA = [this](const double *x_dev, double *y_dev)
   {
      CUDA_CHECK(cudaMemcpy(h_multA_in_.GetData(), x_dev,
                            n_ * sizeof(double), cudaMemcpyDeviceToHost));
      A_cpu_.Mult(h_multA_in_, h_multA_out_);
      CUDA_CHECK(cudaMemcpy(y_dev, h_multA_out_.GetData(),
                            n_ * sizeof(double), cudaMemcpyHostToDevice));
   };

   // ── prec: GPU cublasDgemvBatched patch solve ───────────────────────────────
   // Pattern: D2H → CPU gather → H2D → batchGEMV → D2H → CPU scatter → H2D
   ops.prec = [this](const double *x_dev, double *y_dev)
   {
      const int P  = num_patches_;
      const int ps = max_patch_size_;

      // 1. Download x from GPU
      CUDA_CHECK(cudaMemcpy(h_prec_in_.GetData(), x_dev,
                            n_ * sizeof(double), cudaMemcpyDeviceToHost));

      // 2. CPU gather: fill padded batch input (zero-pad between)
      std::fill(h_batch_x_.begin(), h_batch_x_.end(), 0.0);
      for (int p = 0; p < P; p++)
      {
         const int m = h_patch_sizes_[p];
         for (int i = 0; i < m; i++)
         {
            h_batch_x_[(long long)p * ps + i] =
               h_prec_in_[h_patch_ids_[p][i]];
         }
      }

      // 3. Upload batch x to GPU
      CUDA_CHECK(cudaMemcpy(d_all_xp_, h_batch_x_.data(),
                            (long long)P * ps * sizeof(double),
                            cudaMemcpyHostToDevice));

      // 4. GPU: cublasDgemvBatched  y_p = A_p^{-1} * x_p  (all patches at once)
      const double one = 1.0, zero = 0.0;
      CUBLAS_CHECK(cublasDgemvBatched(
         cublas_,
         CUBLAS_OP_N,
         ps, ps,
         &one,
         d_Aarray_, ps,
         d_xarray_, 1,
         &zero,
         d_yarray_, 1,
         P));

      // 5. Download batch y from GPU
      CUDA_CHECK(cudaMemcpy(h_batch_y_.data(), d_all_yp_,
                            (long long)P * ps * sizeof(double),
                            cudaMemcpyDeviceToHost));

      // 6. CPU scatter-add with damping weights
      h_prec_out_ = 0.0;
      for (int p = 0; p < P; p++)
      {
         const int m = h_patch_sizes_[p];
         for (int i = 0; i < m; i++)
         {
            h_prec_out_[h_patch_ids_[p][i]] +=
               h_patch_wts_[p][i] * h_batch_y_[(long long)p * ps + i];
         }
      }

      // 7. Upload result to GPU
      CUDA_CHECK(cudaMemcpy(y_dev, h_prec_out_.GetData(),
                            n_ * sizeof(double), cudaMemcpyHostToDevice));
   };

   // ── dot: cuBLAS Ddot + MPI_Allreduce (global inner product) ───────────────
   ops.dot = [this](const double *x_dev, const double *y_dev) -> double
   {
      double local_result = 0.0;
      CUBLAS_CHECK(cublasDdot(cublas_, n_, x_dev, 1, y_dev, 1, &local_result));
      double global_result = local_result;
      MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      return global_result;
   };

   // ── norm2: local dot + MPI_Allreduce, then sqrt ───────────────────────────
   ops.norm2 = [this](const double *x_dev) -> double
   {
      double local_dot = 0.0;
      CUBLAS_CHECK(cublasDdot(cublas_, n_, x_dev, 1, x_dev, 1, &local_dot));
      double global_dot = local_dot;
      MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      return std::sqrt(global_dot);
   };

   // ── axby: y = a*x + b*y  via cuBLAS Dscal + Daxpy ────────────────────────
   ops.axby = [this](double a, const double *x_dev, double b_coef, double *y_dev)
   {
      CUBLAS_CHECK(cublasDscal(cublas_, n_, &b_coef, y_dev, 1));
      CUBLAS_CHECK(cublasDaxpy(cublas_, n_, &a, x_dev, 1, y_dev, 1));
   };

   // ── copy: cudaMemcpy D2D ──────────────────────────────────────────────────
   ops.copy = [this](const double *src, double *dst)
   {
      CUDA_CHECK(cudaMemcpy(dst, src, n_ * sizeof(double),
                            cudaMemcpyDeviceToDevice));
   };

   // ── alloc: zero-filled device vector ─────────────────────────────────────
   ops.alloc = [this]() -> double *
   {
      double *p = nullptr;
      CUDA_CHECK(cudaMalloc(&p, n_ * sizeof(double)));
      CUDA_CHECK(cudaMemset(p, 0, n_ * sizeof(double)));
      return p;
   };

   ops.free_vec = [](double *p) { cudaFree(p); };

   return ops;
}

BiCGSTABResult GpuBiCGSTABSolver::Solve(const mfem::Vector &b,
                                          mfem::Vector       &x) const
{
   MFEM_VERIFY(b.Size() == n_, "GpuBiCGSTAB: b size mismatch");
   x.SetSize(n_);

   // Upload b and x to device
   double *d_b = nullptr, *d_x = nullptr;
   CUDA_CHECK(cudaMalloc(&d_b, n_ * sizeof(double)));
   CUDA_CHECK(cudaMalloc(&d_x, n_ * sizeof(double)));
   CUDA_CHECK(cudaMemcpy(d_b, b.GetData(), n_ * sizeof(double),
                         cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_x, x.GetData(), n_ * sizeof(double),
                         cudaMemcpyHostToDevice));

   BiCGSTABOps ops = BuildOps();
   BiCGSTABResult res = BiCGSTAB(d_b, d_x, ops, opts_, mpi_rank_);

   // Download solution
   CUDA_CHECK(cudaMemcpy(x.GetData(), d_x, n_ * sizeof(double),
                         cudaMemcpyDeviceToHost));
   cudaFree(d_b);
   cudaFree(d_x);

   return res;
}

#endif // HAVE_CUDA

} // namespace gpu_solver
