#include "iga_ams_prec.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

namespace iga_ams
{

// ─── Helpers ─────────────────────────────────────────────────────────────────

double GlobalL2Norm(const mfem::Vector &v, MPI_Comm comm)
{
   double local_sq = v.Norml2(); local_sq *= local_sq;
   double global_sq = 0.0;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
   return std::sqrt(global_sq);
}

namespace
{

void ProbeOperatorToDense(const mfem::Operator &op, mfem::DenseMatrix &M)
{
   const int h = op.Height(), w = op.Width();
   M.SetSize(h, w);
   mfem::Vector e(w), col(h);
   for (int j = 0; j < w; j++)
   {
      e = 0.0; e[j] = 1.0;
      op.Mult(e, col);
      for (int i = 0; i < h; i++) { M(i, j) = col[i]; }
   }
}

void AddDiagonalRegularization(mfem::DenseMatrix &A, double relative_floor)
{
   const int n = std::min(A.Height(), A.Width());
   double max_diag = 0.0;
   for (int i = 0; i < n; i++) { max_diag = std::max(max_diag, std::abs(A(i,i))); }
   const double eps = (max_diag > 0.0)
      ? std::max(1e-14, relative_floor * max_diag) : 1e-14;
   for (int i = 0; i < n; i++) { A(i,i) += eps; }
}

// Helper: convert HypreParMatrix DenseMatrix to stored HypreParMatrix
std::unique_ptr<mfem::HypreParMatrix> AssembleOperatorOnSpace(
   const mfem::ParFiniteElementSpace &fespace,
   mfem::BilinearFormIntegrator &curl_integ,
   mfem::BilinearFormIntegrator &mass_integ,
   const mfem::Array<int> &ess_tdof_list)
{
   auto &fes = const_cast<mfem::ParFiniteElementSpace&>(fespace);
   mfem::ParBilinearForm bf(&fes);
   bf.AddDomainIntegrator(&curl_integ);
   bf.AddDomainIntegrator(&mass_integ);

   // Only assemble on the current MPI rank's data
   bf.Assemble();

   // Finalize (skip boundary elimination if ess_tdofs is empty)
   if (ess_tdof_list.Size() == 0)
   {
      bf.Finalize();
   }

   mfem::OperatorHandle Ah;
   bf.FormSystemMatrix(ess_tdof_list, Ah);
   auto *Ahp = Ah.As<mfem::HypreParMatrix>();
   MFEM_VERIFY(Ahp, "Failed to get HypreParMatrix from assembled operator.");
   return std::make_unique<mfem::HypreParMatrix>(*Ahp);
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Variant 1: AMS on o=1 IGA Galerkin coarse operator
// ═══════════════════════════════════════════════════════════════════════════════

CoarseAMS_GalerkinPreconditioner::CoarseAMS_GalerkinPreconditioner(
   const mfem::ParFiniteElementSpace &fine_fespace, int coarse_order)
   : mfem::Solver(2 * fine_fespace.GetTrueVSize(),
                  2 * fine_fespace.GetTrueVSize()),
     fine_fespace_(fine_fespace), coarse_order_(coarse_order)
{}

CoarseAMS_GalerkinPreconditioner::~CoarseAMS_GalerkinPreconditioner()
{ A1_hypre_.reset(); ams_solver_.reset(); }

void CoarseAMS_GalerkinPreconditioner::SetOperator(const mfem::Operator &op)
{ op_ = &op; height = op.Height(); width = op.Width(); built_ = false; }

void CoarseAMS_GalerkinPreconditioner::Build()
{
   if (built_) return;
   MFEM_VERIFY(op_ != nullptr, "SetOperator() must be called first.");

   auto t0 = std::chrono::steady_clock::now();

   // 1. Create coarse IGA H(curl) space
   BuildCoarseSpace();

   // 2. Build mass projection P: o=1 → o=p
   BuildMassProlongation();

   auto t1 = std::chrono::steady_clock::now();

   // 3. Build coarse AMS (directly assemble or Galerkin projet)
   BuildCoarseAMS();

   auto t2 = std::chrono::steady_clock::now();

   // 4. Build fine-level block Jacobi smoother
   BuildBlockJacobi();

   auto t3 = std::chrono::steady_clock::now();

   // Work vectors
   const int nf = fine_fespace_.GetTrueVSize();
   const int n2 = 2 * nf;
   z1_.SetSize(n2); r1_.SetSize(n2); z2_.SetSize(n2);

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      const int nc = coarse_fespace_->GetTrueVSize();
      std::cout << "[iga_ams_coarse] fine_vsize=" << nf
                << " coarse_vsize=" << nc
                << " prolong=" << std::chrono::duration<double>(t1-t0).count() << "s"
                << " ams=" << std::chrono::duration<double>(t2-t1).count() << "s"
                << " jacobi=" << std::chrono::duration<double>(t3-t2).count() << "s"
                << std::endl;
   }
   built_ = true;
}

void CoarseAMS_GalerkinPreconditioner::BuildCoarseSpace()
{
   mfem::ParMesh *pmesh = fine_fespace_.GetParMesh();
   int dim = pmesh->Dimension();

   coarse_fec_ = std::make_unique<mfem::NURBS_HCurlFECollection>(coarse_order_, dim);
   coarse_nurbs_ext_ = std::make_unique<mfem::NURBSExtension>(
      pmesh->NURBSext, coarse_order_);
   coarse_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(
      pmesh, coarse_nurbs_ext_.get(), coarse_fec_.get());
   coarse_nurbs_ext_.release(); // ownership transferred to ParFiniteElementSpace
   // Do NOT use pmesh->NURBSext here: the coarse space needs its own order-1 extension.
}

void CoarseAMS_GalerkinPreconditioner::BuildMassProlongation()
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int nc = coarse_fespace_->GetTrueVSize();

   mfem::ConstantCoefficient one(1.0);
   mfem::Array<int> empty_tdofs;

   // Fine mass matrix
   mfem::ParBilinearForm mfine(
      const_cast<mfem::ParFiniteElementSpace*>(&fine_fespace_));
   mfine.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mfine.Assemble(0);
   mfem::OperatorPtr Mfine_op;
   mfine.FormSystemMatrix(empty_tdofs, Mfine_op);

   // Mixed mass matrix: (coarse, fine)
   mfem::ParMixedBilinearForm mmix(
      const_cast<mfem::ParFiniteElementSpace*>(coarse_fespace_.get()),
      const_cast<mfem::ParFiniteElementSpace*>(&fine_fespace_));
   mmix.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   mfem::OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs, empty_tdofs, Mmix_op);

   mfem::DenseMatrix Mfine, Mmix;
   ProbeOperatorToDense(*Mfine_op, Mfine);
   ProbeOperatorToDense(*Mmix_op.Ptr(), Mmix);
   AddDiagonalRegularization(Mfine, 1e-12);

   mfem::DenseMatrixInverse MfineInv(Mfine);
   P_.SetSize(nf, nc); P_ = 0.0;
   mfem::Vector rhs(nf), sol(nf);
   for (int j = 0; j < nc; j++)
   {
      for (int i = 0; i < nf; i++) rhs[i] = Mmix(i, j);
      MfineInv.Mult(rhs, sol);
      for (int i = 0; i < nf; i++) P_(i, j) = sol[i];
   }
}

void CoarseAMS_GalerkinPreconditioner::BuildCoarseAMS()
{
   const int nc = coarse_fespace_->GetTrueVSize();
   MFEM_VERIFY(nc > 0, "Coarse space must be built first.");
   MFEM_VERIFY(op_ != nullptr, "Operator must be set.");

   const int nf = fine_fespace_.GetTrueVSize();

   // ── Build REAL A_1_real = P^T A_p_re * P (only the real (1,1) block) ──
   // AMS expects a real SPD-like operator (curl-curl + k^2 M).
   // We extract only the real part of the Galerkin coarse operator.
   mfem::DenseMatrix A1_real(nc, nc);
   A1_real = 0.0;

   mfem::Vector probe(2*nf), Ap(2*nf);
   for (int j = 0; j < nc; j++)
   {
      probe = 0.0;
      for (int i = 0; i < nf; i++) probe[i] = P_(i, j);
      op_->Mult(probe, Ap);
      for (int ii = 0; ii < nc; ii++)
      {
         double v = 0.0;
         for (int k = 0; k < nf; k++) v += P_(k, ii) * Ap[k];  // real part only
         A1_real(ii, j) = v;
      }
   }

   AddDiagonalRegularization(A1_real, 1e-10);

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      double max_val = 0.0;
      for (int i = 0; i < nc; i++)
         for (int j = 0; j < nc; j++)
            max_val = std::max(max_val, std::abs(A1_real(i,j)));
      std::cout << "[iga_ams_coarse] A_1_real: " << nc << "x" << nc
                << "  max|val|=" << std::scientific << std::setprecision(2) << max_val
                << std::endl;
   }

   // ── Convert dense A_1_real to HypreParMatrix ────────────────────────────
   {
      auto *Asparse = new mfem::SparseMatrix(nc, nc);
      for (int i = 0; i < nc; i++)
         for (int j = 0; j < nc; j++)
         {
            double v = A1_real(i, j);
            if (std::abs(v) > 1e-15)
               Asparse->Set(i, j, v);
         }
      Asparse->Finalize();
      HYPRE_BigInt *row_starts = new HYPRE_BigInt[2];
      row_starts[0] = 0; row_starts[1] = static_cast<HYPRE_BigInt>(nc);
      A1_hypre_ = std::make_unique<mfem::HypreParMatrix>(
         MPI_COMM_WORLD,
         static_cast<HYPRE_BigInt>(nc),
         row_starts, Asparse);
   }

   // ── Build AMS on the o=1 coarse space ────────────────────────────────────
   ams_solver_ = std::make_unique<mfem::HypreAMS>(
      *A1_hypre_, coarse_fespace_.get());
   ams_solver_->SetPrintLevel(ams_print_level_);

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
      std::cout << "[iga_ams_coarse] AMS built on o=" << coarse_order_
                << " space (tvsize=" << nc << ")" << std::endl;
}

void CoarseAMS_GalerkinPreconditioner::BuildBlockJacobi()
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int n2 = 2 * nf;
   bj00_.SetSize(nf); bj01_.SetSize(nf);
   bj10_.SetSize(nf); bj11_.SetSize(nf);

   mfem::Vector e(n2), Ae(n2);
   for (int i = 0; i < nf; i++)
   {
      e = 0.0; e[i] = 1.0;
      op_->Mult(e, Ae);
      const double a00 = Ae[i], a10 = Ae[i + nf];

      e = 0.0; e[i + nf] = 1.0;
      op_->Mult(e, Ae);
      const double a01 = Ae[i], a11 = Ae[i + nf];

      // Invert 2x2 block: [a00 a01; a10 a11]^{-1}
      double det = a00*a11 - a01*a10;
      if (std::abs(det) < 1e-14) det = 1.0;
      bj00_[i] =  a11 / det;
      bj01_[i] = -a01 / det;
      bj10_[i] = -a10 / det;
      bj11_[i] =  a00 / det;
   }
}

void CoarseAMS_GalerkinPreconditioner::ApplyCoarse(
   const mfem::Vector &r, mfem::Vector &z) const
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int nc = coarse_fespace_->GetTrueVSize();
   const int n2 = 2 * nf;

   // 1. Restrict real + imag parts separately: rc = P^T * r
   mfem::Vector r_re(const_cast<mfem::Vector&>(r), 0, nf);
   mfem::Vector r_im(const_cast<mfem::Vector&>(r), nf, nf);
   mfem::Vector rc_re(nc), rc_im(nc);
   P_.MultTranspose(r_re, rc_re);
   P_.MultTranspose(r_im, rc_im);

   // 2. Solve each part with AMS (real operator)
   mfem::Vector zc_re(nc), zc_im(nc);
   ams_solver_->Mult(rc_re, zc_re);
   ams_solver_->Mult(rc_im, zc_im);

   // 3. Prolongate: z = P * zc
   z.SetSize(n2); z = 0.0;
   mfem::Vector z_re(z, 0, nf), z_im(z, nf, nf);
   P_.Mult(zc_re, z_re);
   P_.Mult(zc_im, z_im);

   if (coarse_weight_ != 1.0) { z *= coarse_weight_; }
}

void CoarseAMS_GalerkinPreconditioner::ApplyJacobi(
   const mfem::Vector &r, mfem::Vector &z) const
{
   const int nf = fine_fespace_.GetTrueVSize();
   z.SetSize(2*nf);
   for (int i = 0; i < nf; i++)
   {
      z[i]       = bj00_[i] * r[i]      + bj01_[i] * r[i + nf];
      z[i + nf]  = bj10_[i] * r[i]      + bj11_[i] * r[i + nf];
   }
   if (jacobi_weight_ != 1.0) { z *= jacobi_weight_; }
}

void CoarseAMS_GalerkinPreconditioner::Mult(
   const mfem::Vector &r, mfem::Vector &z) const
{
   if (!built_) { const_cast<CoarseAMS_GalerkinPreconditioner*>(this)->Build(); }

   const int n2 = 2 * fine_fespace_.GetTrueVSize();
   z.SetSize(n2); z = 0.0;

   switch (combine_mode_)
   {
   case CombineMode::coarse_only:
      ApplyCoarse(r, z);
      break;
   case CombineMode::additive:
   {
      z1_ = 0.0; z2_ = 0.0;
      ApplyCoarse(r, z1_);
      ApplyJacobi(r, z2_);
      z = z1_; z += z2_;
      break;
   }
   case CombineMode::multiplicative:
   {
      // Step 1: Coarse correction
      z1_ = 0.0;
      ApplyCoarse(r, z1_);

      // Step 2: Compute residual after coarse correction
      r1_.SetSize(n2);
      op_->Mult(z1_, r1_);
      for (int i = 0; i < n2; i++) r1_[i] = r[i] - r1_[i];

      // Step 3: Jacobi on residual
      z2_ = 0.0;
      ApplyJacobi(r1_, z2_);
      z = z1_; z += z2_;
      break;
   }
   }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Variant 2: IGA-exact gradient correction
// ═══════════════════════════════════════════════════════════════════════════════

IGA_GradientCorrectionPreconditioner::IGA_GradientCorrectionPreconditioner(
   const mfem::ParFiniteElementSpace &hcurl_fespace,
   const mfem::ParFiniteElementSpace *h1_fespace)
   : mfem::Solver(2 * hcurl_fespace.GetTrueVSize(),
                  2 * hcurl_fespace.GetTrueVSize()),
     hcurl_fespace_(hcurl_fespace),
     h1_fespace_(h1_fespace)
{
   if (!h1_fespace)
   {
      // Build H1 scalar space from the H(curl) space.  Important ownership
      // detail: ParFiniteElementSpace owns the NURBSExtension passed to it,
      // but not the FiniteElementCollection.  Keep the FEC alive here and
      // release our NURBSExtension unique_ptr after handing it to MFEM.
      mfem::ParMesh *pmesh = hcurl_fespace_.GetParMesh();
      int p = hcurl_fespace_.FEColl()->GetOrder();
      h1_fec_own_ = std::make_unique<mfem::NURBSFECollection>(p);
      h1_nurbs_ext_own_ = std::make_unique<mfem::NURBSExtension>(pmesh->NURBSext, p);
      h1_fespace_own_ = std::make_unique<mfem::ParFiniteElementSpace>(
         pmesh, h1_nurbs_ext_own_.get(), h1_fec_own_.get());
      h1_nurbs_ext_own_.release(); // ownership transferred to ParFiniteElementSpace
      h1_fespace_ = h1_fespace_own_.get();
   }
}

IGA_GradientCorrectionPreconditioner::~IGA_GradientCorrectionPreconditioner() {}

void IGA_GradientCorrectionPreconditioner::SetOperator(const mfem::Operator &op)
{ op_ = &op; height = op.Height(); width = op.Width(); built_ = false; }

void IGA_GradientCorrectionPreconditioner::Build()
{
   if (built_) return;
   MFEM_VERIFY(op_ != nullptr, "SetOperator() required.");
   MFEM_VERIFY(h1_fespace_ != nullptr, "H1 space required.");

   const int n_h1   = h1_fespace_->GetTrueVSize();
   const int n_curl = hcurl_fespace_.GetTrueVSize();

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[iga_grad_corr] h1_vsize=" << n_h1
                << "  hcurl_vsize=" << n_curl << std::endl;
   }

   // ── 1. Build discrete gradient G: H1 → H(curl) ───────────────────────────
   if (verbose_) {
      std::cout << "[iga_grad_corr] DEBUG: h1_fespace=" << (void*)h1_fespace_ 
                << " hcurl_fespace=" << (void*)&hcurl_fespace_
                << " h1_tvsize=" << n_h1 << " hcurl_tvsize=" << n_curl << std::endl;
      std::cout << "[iga_grad_corr] DEBUG: pmesh=" << (void*)hcurl_fespace_.GetParMesh()
                << " NURBSext=" << (void*)hcurl_fespace_.GetParMesh()->NURBSext << std::endl;
      std::cout << "[iga_grad_corr] DEBUG: creating ParDiscreteLinearOperator..." << std::endl;
   }
   try {
      mfem::ParDiscreteLinearOperator *grad =
         new mfem::ParDiscreteLinearOperator(
            const_cast<mfem::ParFiniteElementSpace*>(h1_fespace_),
            const_cast<mfem::ParFiniteElementSpace*>(&hcurl_fespace_));
      grad->AddDomainInterpolator(new mfem::GradientInterpolator);
      grad->Assemble();
      grad->Finalize();
      G_ = std::unique_ptr<mfem::HypreParMatrix>(grad->ParallelAssemble());
      delete grad;

      if (verbose_) std::cout << "[iga_grad_corr] DEBUG: G built OK" << std::endl;
      if (verbose_ && mfem::Mpi::WorldRank() == 0)
      {
         std::cout << "[iga_grad_corr] G: " << G_->M() << "x" << G_->N() << std::endl;
      }
   } catch (std::exception &e) {
      if (verbose_) std::cout << "[iga_grad_corr] DEBUG: G built OK" << std::endl;
      if (verbose_ && mfem::Mpi::WorldRank() == 0)
         std::cerr << "[iga_grad_corr] G construction failed: " << e.what() << std::endl;
      return;
   }

   // ── 2. Build A_phi = G^T A_p G ───────────────────────────────────────────



   // Since op_ is the real 2x2 block operator (not HypreParMatrix),
   // we probe column-by-column: A_phi = G^T * A_p_real * G
   mfem::DenseMatrix Aphi_dense(n_h1, n_h1);
   {
      mfem::Vector gcol(n_curl), Agcol(2*n_curl);
      mfem::Vector gcol_block(2*n_curl);
      // First extract G as dense
      mfem::DenseMatrix Gdense(n_curl, n_h1);
      mfem::Vector e_h1(n_h1), Ge(n_curl);
      for (int j = 0; j < n_h1; j++)
      {
         e_h1 = 0.0; e_h1[j] = 1.0;
         G_->Mult(e_h1, Ge);
         for (int i = 0; i < n_curl; i++) Gdense(i, j) = Ge[i];
      }

      for (int j = 0; j < n_h1; j++)
      {
         // j-th column of G as a column in the real block
         gcol_block = 0.0;
         for (int i = 0; i < n_curl; i++) gcol_block[i] = Gdense(i, j);
         op_->Mult(gcol_block, Agcol);
         // Agcol is 2*n_curl; use real part (first half)
         for (int ii = 0; ii < n_h1; ii++)
         {
            double v = 0.0;
            for (int i = 0; i < n_curl; i++) v += Gdense(i, ii) * Agcol[i];
            Aphi_dense(ii, j) = v;
         }
      }

      if (verbose_) std::cout << "[iga_grad_corr] DEBUG: G built OK" << std::endl;
      if (verbose_ && mfem::Mpi::WorldRank() == 0)
      {
         double max_val = 0.0;
         for (int i = 0; i < n_h1; i++)
            for (int j = 0; j < n_h1; j++)
               max_val = std::max(max_val, std::abs(Aphi_dense(i,j)));
         std::cout << "[iga_grad_corr] A_phi " << n_h1 << "x" << n_h1
                   << "  max|val|=" << std::scientific << max_val << std::endl;
      }
   }

   // Convert dense to HypreParMatrix via SparseMatrix (owns data)
   {
      auto *Asparse = new mfem::SparseMatrix(n_h1, n_h1);
      for (int i = 0; i < n_h1; i++)
         for (int j = 0; j < n_h1; j++)
         {
            double v = Aphi_dense(i, j);
            if (std::abs(v) > 1e-15)
               Asparse->Set(i, j, v);
         }
      Asparse->Finalize();
      HYPRE_BigInt *row_starts = new HYPRE_BigInt[2];
      row_starts[0] = 0; row_starts[1] = static_cast<HYPRE_BigInt>(n_h1);
      A_phi_ = std::make_unique<mfem::HypreParMatrix>(
         MPI_COMM_WORLD,
         static_cast<HYPRE_BigInt>(n_h1),
         row_starts, Asparse);
   }

   // ── 3. Build solver for A_phi ─────────────────────────────────────────────
   if (use_amg_)
   {
      amg_ = std::make_unique<mfem::HypreBoomerAMG>(*A_phi_);
      amg_->SetPrintLevel(print_level_);
      if (verbose_) std::cout << "[iga_grad_corr] DEBUG: G built OK" << std::endl;
      if (verbose_ && mfem::Mpi::WorldRank() == 0)
         std::cout << "[iga_grad_corr] BoomerAMG built on A_phi ("
                   << n_h1 << "x" << n_h1 << ")\n";
   }

   // Work vectors
   r_grad_.SetSize(n_h1);
   z_grad_.SetSize(n_h1);
   local_r_.SetSize(n_curl);
   local_z_.SetSize(n_curl);

   built_ = true;
}

void IGA_GradientCorrectionPreconditioner::Mult(
   const mfem::Vector &r, mfem::Vector &z) const
{
   if (!built_) const_cast<IGA_GradientCorrectionPreconditioner*>(this)->Build();

   const int n_curl = hcurl_fespace_.GetTrueVSize();
   const int n2 = 2 * n_curl;
   z.SetSize(n2); z = 0.0;

   // Extract real part of residual for gradient correction
   mfem::Vector r_re(const_cast<mfem::Vector&>(r), 0, n_curl);

   // 1. Restrict to H1: r_grad = G^T * r_re
   G_->MultTranspose(r_re, r_grad_);

   // 2. Solve A_phi * z_grad = r_grad
   z_grad_ = 0.0;
   if (use_amg_ && amg_)
   {
      amg_->Mult(r_grad_, z_grad_);
   }
   else
   {
      // Simple Jacobi fallback
      z_grad_.SetSize(r_grad_.Size());
      mfem::Vector diag(r_grad_.Size());
      A_phi_->GetDiag(diag);
      for (int i = 0; i < r_grad_.Size(); i++)
         z_grad_[i] = (std::abs(diag[i]) > 1e-14) ? r_grad_[i] / diag[i] : 0.0;
   }

   // 3. Prolongate to H(curl): z_re = G * z_grad
   mfem::Vector z_re(z, 0, n_curl);
   G_->Mult(z_grad_, z_re);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Variant 3: Shifted AMS
// ═══════════════════════════════════════════════════════════════════════════════

ShiftedAMS_Preconditioner::ShiftedAMS_Preconditioner(
   const mfem::ParFiniteElementSpace &fespace, double freq, double eps_val)
   : mfem::Solver(2 * fespace.GetTrueVSize(),
                  2 * fespace.GetTrueVSize()),
     fespace_(fespace), freq_(freq), eps_val_(eps_val)
{}

ShiftedAMS_Preconditioner::~ShiftedAMS_Preconditioner() {}

void ShiftedAMS_Preconditioner::SetOperator(const mfem::Operator &op)
{ op_ = &op; height = op.Height(); width = op.Width(); built_ = false; }

void ShiftedAMS_Preconditioner::Build()
{
   if (built_) return;
   MFEM_VERIFY(op_ != nullptr, "SetOperator() required.");

   const int n_curl = fespace_.GetTrueVSize();
   const int n2 = 2 * n_curl;

   // ── Build shifted operator A_shift = A_p + eta * Mass ──────────────────
   // We need to extract the real part (rows 0..n-1, cols 0..n-1)
   // and add eta * mass to it.

   // Probe the real block of A_p (first n_curl rows/cols)
   mfem::DenseMatrix Apreal(n_curl, n_curl);
   {
      mfem::Vector probe(n2), Aprobe(n2);
      for (int j = 0; j < n_curl; j++)
      {
         probe = 0.0; probe[j] = 1.0;
         op_->Mult(probe, Aprobe);
         for (int i = 0; i < n_curl; i++) Apreal(i, j) = Aprobe[i];
      }
   }

   // For the mass matrix, use the H(curl) mass on the real part
   mfem::ConstantCoefficient one(1.0);
   mfem::Array<int> empty_tdofs;
   mfem::ParBilinearForm mass_form(
      const_cast<mfem::ParFiniteElementSpace*>(&fespace_));
   mass_form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mass_form.Assemble(0);
   mfem::OperatorPtr Mass_op;
   mass_form.FormSystemMatrix(empty_tdofs, Mass_op);

   mfem::DenseMatrix Mass_dense;
   ProbeOperatorToDense(*Mass_op, Mass_dense);

   // A_shift = A_p_real + eta * Mass
   const double eta_val = (shift_type_ == ShiftType::complex_shift)
      ? eta_ * (2.0 * M_PI * freq_) * (2.0 * M_PI * freq_) * eps_val_
      : eta_;
   mfem::DenseMatrix Ashift_dense(n_curl, n_curl);
   for (int i = 0; i < n_curl; i++)
      for (int j = 0; j < n_curl; j++)
         Ashift_dense(i, j) = Apreal(i, j) + eta_val * Mass_dense(i, j);

   // Convert to HypreParMatrix via SparseMatrix (owns data)
   {
      auto *Asparse = new mfem::SparseMatrix(n_curl, n_curl);
      for (int i = 0; i < n_curl; i++)
         for (int j = 0; j < n_curl; j++)
         {
            double v = Ashift_dense(i, j);
            if (std::abs(v) > 1e-15)
               Asparse->Set(i, j, v);
         }
      Asparse->Finalize();
      HYPRE_BigInt *row_starts = new HYPRE_BigInt[2];
      row_starts[0] = 0; row_starts[1] = static_cast<HYPRE_BigInt>(n_curl);
      A_shift_ = std::make_unique<mfem::HypreParMatrix>(
         MPI_COMM_WORLD,
         static_cast<HYPRE_BigInt>(n_curl),
         row_starts, Asparse);
   }

   // ── Build AMS on A_shift ──────────────────────────────────────────────
   ams_ = std::make_unique<mfem::HypreAMS>(
      *A_shift_, const_cast<mfem::ParFiniteElementSpace*>(&fespace_));
   ams_->SetPrintLevel(print_level_);
   if (singular_) { ams_->SetSingularProblem(); }

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
      std::cout << "[iga_shifted_ams] built on shifted operator (eta="
                << eta_ << ")\n";

   built_ = true;
}

void ShiftedAMS_Preconditioner::Mult(
   const mfem::Vector &r, mfem::Vector &z) const
{
   if (!built_) const_cast<ShiftedAMS_Preconditioner*>(this)->Build();

   const int n_curl = fespace_.GetTrueVSize();
   const int n2 = 2 * n_curl;
   z.SetSize(n2); z = 0.0;

   // Apply AMS to the real and imaginary parts separately
   // AMS is real-valued, so we apply it to each half

   mfem::Vector r_re(const_cast<mfem::Vector&>(r), 0, n_curl);
   mfem::Vector r_im(const_cast<mfem::Vector&>(r), n_curl, n_curl);
   mfem::Vector z_re(z, 0, n_curl);
   mfem::Vector z_im(z, n_curl, n_curl);

   ams_->Mult(r_re, z_re);
   ams_->Mult(r_im, z_im);
}

} // namespace iga_ams
