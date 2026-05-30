#include "split_pml_prec.hpp"
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace pml_split
{

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

} // anonymous namespace

// ─── SplitPMLPreconditioner ──────────────────────────────────────────────────

SplitPMLPreconditioner::SplitPMLPreconditioner(
   const mfem::ParFiniteElementSpace &fespace,
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom,
   const std::function<double(const mfem::Vector &)> &eps_fn)
   : mfem::Solver(2 * fespace.GetTrueVSize(), 2 * fespace.GetTrueVSize()),
     fespace_(fespace), geom_(geom), eps_fn_(eps_fn),
     yee_transfer_(std::make_unique<covariant_aux_space::YeeTransferBuilder>(fespace,geom)),
     yee_operator_(std::make_unique<covariant_aux_space::YeeOperatorBuilder>(geom))
{}

void SplitPMLPreconditioner::SetGrid(const fdfd_iga_init::ReferenceGrid &grid)
{ yee_transfer_->SetGrid(grid); yee_operator_->SetGrid(grid); built_ = false; }

void SplitPMLPreconditioner::SetWaveNumber(double k0)
{ k0_ = k0; built_ = false; }

void SplitPMLPreconditioner::SetOperator(const mfem::Operator &op)
{ op_ = &op; height = op.Height(); width = op.Width(); built_ = false; }

void SplitPMLPreconditioner::Build() const
{
   if (built_) return;
   if (mfem::Mpi::WorldRank() == 0)
   {
      const auto &g = yee_operator_->Grid();
      std::cout << "[split_pml_prec] building on " << g.nx << "x" << g.ny << "x" << g.nz
                << " Yee grid" << std::endl;
   }
   auto t0 = std::chrono::steady_clock::now();
   BuildTransfer();
   auto t1 = std::chrono::steady_clock::now();
   if (coarse_mode_ != CoarseMode::none) { BuildCoarse(); }
   auto t2 = std::chrono::steady_clock::now();
   if (combine_mode_ != CombineMode::coarse_only) { BuildBlockJacobi(); }
   auto t3 = std::chrono::steady_clock::now();
   aux_rhs_.SetSize(na_); aux_sol_.SetSize(na_);
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[split_pml_prec] transfer="
                << std::chrono::duration<double>(t1-t0).count() << "s"
                << "  coarse=" << std::chrono::duration<double>(t2-t1).count() << "s"
                << "  jacobi=" << std::chrono::duration<double>(t3-t2).count() << "s"
                << "  aux_dofs=" << na_
                << "  true_vsize=" << fespace_.GetTrueVSize() << std::endl;
   }
   built_ = true;
}

void SplitPMLPreconditioner::BuildTransfer() const
{ yee_transfer_->BuildProlongationFast(Pi_); na_ = Pi_.Width(); }

void SplitPMLPreconditioner::BuildCoarse() const
{
   MFEM_VERIFY(na_ > 0, "Transfer must be built before coarse operator.");
   const int tvsize = fespace_.GetTrueVSize();
   if (coarse_mode_ == CoarseMode::yee_curl_independent)
   {
      yee_operator_->AssembleYeeCurlOperator(Aaux_, k0_);
      if (mfem::Mpi::WorldRank() == 0)
         std::cout << "[split_pml_prec] coarse: independent Yee curl-curl C^T M_mu C\n";
   }
   else if (coarse_mode_ == CoarseMode::yee_curl_galerkin)
   {
      MFEM_VERIFY(curl_op_ != nullptr,
                  "SetCurlOperator() must be called for yee_curl_galerkin mode.");
      MFEM_VERIFY(curl_op_->Width() == tvsize,
                  "Curl operator width must equal true_vsize.");
      Aaux_.SetSize(na_, na_); Aaux_ = 0.0;
      mfem::Vector col(tvsize), Acol(tvsize);
      for (int j = 0; j < na_; j++)
      {
         for (int i = 0; i < tvsize; i++) col[i] = Pi_(i,j);
         curl_op_->Mult(col, Acol);
         for (int i = 0; i < na_; i++)
         { double v = 0.0; for (int k = 0; k < tvsize; k++) v += Pi_(k,i)*Acol[k]; Aaux_(i,j)=v; }
      }
      if (mfem::Mpi::WorldRank() == 0)
         std::cout << "[split_pml_prec] coarse: Galerkin P^T A_curl P\n";
   }
   else if (coarse_mode_ == CoarseMode::yee_full_galerkin)
   {
      MFEM_VERIFY(op_ != nullptr,
                  "SetOperator() must be called for yee_full_galerkin mode.");
      MFEM_VERIFY(op_->Width() == 2*tvsize,
                  "Full operator width must equal 2*true_vsize.");
      mfem::DenseMatrix gal_re(na_,na_), gal_im(na_,na_); gal_re=0.0; gal_im=0.0;
      mfem::Vector probe(2*tvsize), Ap(2*tvsize);
      for (int j=0; j<na_; j++)
      {
         probe=0.0; for (int i=0;i<tvsize;i++) probe[i]=Pi_(i,j);
         op_->Mult(probe,Ap);
         for (int ii=0;ii<na_;ii++)
         {
            double vr=0.0,vi=0.0;
            for (int k=0;k<tvsize;k++) { vr+=Pi_(k,ii)*Ap[k]; vi+=Pi_(k,ii)*Ap[k+tvsize]; }
            gal_re(ii,j)=vr; gal_im(ii,j)=vi;
         }
      }
      AauxBlock_.SetSize(2*na_,2*na_); AauxBlock_=0.0;
      for (int i=0;i<na_;i++)
         for (int j=0;j<na_;j++)
         {
            AauxBlock_(i,j)        =  gal_re(i,j);
            AauxBlock_(i,j+na_)    = -gal_im(i,j);
            AauxBlock_(i+na_,j)    =  gal_im(i,j);
            AauxBlock_(i+na_,j+na_) =  gal_re(i,j);
         }
      AddDiagonalRegularization(AauxBlock_, 1e-8);
      AauxBlockInv_ = std::make_unique<mfem::DenseMatrixInverse>(AauxBlock_);
      if (mfem::Mpi::WorldRank() == 0)
         std::cout << "[split_pml_prec] coarse: full complex Galerkin Pi^T A_PML Pi\n";
   }
   if (coarse_mode_ != CoarseMode::yee_full_galerkin)
   {
      AddDiagonalRegularization(Aaux_, 1e-10);
      AauxInv_ = std::make_unique<mfem::DenseMatrixInverse>(Aaux_);
   }
}

void SplitPMLPreconditioner::BuildBlockJacobi() const
{
   const int nf = fespace_.GetTrueVSize(), n2 = 2*nf;
   bj00_.SetSize(nf); bj01_.SetSize(nf); bj10_.SetSize(nf); bj11_.SetSize(nf);
   mfem::Vector e(n2), Ae(n2);
   for (int i=0;i<nf;i++)
   {
      e=0.0; e[i]=1.0; op_->Mult(e,Ae);
      const double a00=Ae[i], a10=Ae[i+nf];
      e=0.0; e[i+nf]=1.0; op_->Mult(e,Ae);
      const double a01=Ae[i], a11=Ae[i+nf];
      const double det=a00*a11-a01*a10;
      const double inv=(std::abs(det)>1e-30)?1.0/det:0.0;
      bj00_[i]= a11*inv; bj01_[i]=-a01*inv;
      bj10_[i]=-a10*inv; bj11_[i]= a00*inv;
   }
}

void SplitPMLPreconditioner::ApplyCoarse(const mfem::Vector &r, mfem::Vector &z) const
{
   const int tvsize = fespace_.GetTrueVSize();
   if (coarse_mode_ == CoarseMode::yee_full_galerkin)
   {
      z.SetSize(2*tvsize); z=0.0;
      const mfem::Vector r_re(const_cast<mfem::Vector&>(r),0,tvsize);
      const mfem::Vector r_im(const_cast<mfem::Vector&>(r),tvsize,tvsize);
      mfem::Vector z_re(z,0,tvsize), z_im(z,tvsize,tvsize);
      mfem::Vector rhs(2*na_), sol(2*na_);
      mfem::Vector r_re_rhs(rhs,0,na_), r_im_rhs(rhs,na_,na_);
      mfem::Vector s_re(sol,0,na_), s_im(sol,na_,na_);
      Pi_.MultTranspose(r_re, r_re_rhs);
      Pi_.MultTranspose(r_im, r_im_rhs);
      AauxBlockInv_->Mult(rhs, sol);
      Pi_.Mult(s_re, z_re);
      Pi_.Mult(s_im, z_im);
   }
   else
   {
      z.SetSize(2*tvsize); z=0.0;
      aux_rhs_.SetSize(na_); aux_sol_.SetSize(na_);
      Pi_.MultTranspose(r, aux_rhs_);
      AauxInv_->Mult(aux_rhs_, aux_sol_);
      Pi_.Mult(aux_sol_, z);
   }
   if (coarse_weight_ != 1.0) z *= coarse_weight_;
}

void SplitPMLPreconditioner::ApplyJacobi(const mfem::Vector &r, mfem::Vector &z) const
{
   const int nf = fespace_.GetTrueVSize(); z.SetSize(2*nf);
   for (int i=0;i<nf;i++)
   {
      z[i]      = jacobi_weight_*(bj00_[i]*r[i] + bj01_[i]*r[i+nf]);
      z[i+nf]   = jacobi_weight_*(bj10_[i]*r[i] + bj11_[i]*r[i+nf]);
   }
}

void SplitPMLPreconditioner::Mult(const mfem::Vector &r, mfem::Vector &z) const
{
   Build();
   const int nf = fespace_.GetTrueVSize();
   MFEM_VERIFY(r.Size() == 2*nf, "Residual size mismatch.");
   switch (combine_mode_)
   {
      case CombineMode::jacobi_only:
         ApplyJacobi(r, z); break;
      case CombineMode::coarse_only:
         ApplyCoarse(r, z); break;
      case CombineMode::additive:
         ApplyCoarse(r, z);
         z2_.SetSize(2*nf); ApplyJacobi(r, z2_); z += z2_;
         break;
      case CombineMode::multiplicative:
         z1_.SetSize(2*nf); ApplyCoarse(r, z1_);
         r1_.SetSize(2*nf); op_->Mult(z1_, r1_); r1_ *= -1.0; r1_ += r;
         z2_.SetSize(2*nf); ApplyJacobi(r1_, z2_);
         z = z1_; z += z2_;
         break;
   }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PLevelGalerkinPreconditioner — p-multigrid bridge with edge_yee coarse solver
// ═══════════════════════════════════════════════════════════════════════════════

PLevelGalerkinPreconditioner::PLevelGalerkinPreconditioner(
   const mfem::ParFiniteElementSpace &fine_fespace, int coarse_order)
   : mfem::Solver(2*fine_fespace.GetTrueVSize(), 2*fine_fespace.GetTrueVSize()),
     fine_fespace_(fine_fespace), coarse_order_(coarse_order)
{ BuildCoarseSpace(); }

PLevelGalerkinPreconditioner::~PLevelGalerkinPreconditioner()
{
   coarse_fespace_.release();
   coarse_nurbs_ext_.release();
   coarse_fec_.release();
}

void PLevelGalerkinPreconditioner::BuildCoarseSpace()
{
   mfem::ParMesh *pmesh = fine_fespace_.GetParMesh();
   MFEM_VERIFY(pmesh != nullptr, "Requires a ParMesh.");
   MFEM_VERIFY(pmesh->NURBSext != nullptr, "Requires a NURBS mesh.");
   coarse_fec_ = std::make_unique<mfem::NURBS_HCurlFECollection>(
      coarse_order_, pmesh->Dimension());
   coarse_nurbs_ext_ = std::make_unique<mfem::NURBSExtension>(
      pmesh->NURBSext, coarse_order_);
   coarse_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(
      pmesh, coarse_nurbs_ext_.get(), coarse_fec_.get());
   coarse_fespace_->StealNURBSext();
   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[p_level_galerkin] coarse_order=" << coarse_order_
                << "  fine_true_vsize=" << fine_fespace_.GetTrueVSize()
                << "  coarse_true_vsize=" << coarse_fespace_->GetTrueVSize() << "\n";
   }
}

int PLevelGalerkinPreconditioner::GetCoarseTrueVSize() const
{ return coarse_fespace_ ? coarse_fespace_->GetTrueVSize() : 0; }

void PLevelGalerkinPreconditioner::SetCoarseYeeGrid(
   const fdfd_iga_init::ReferenceGrid &grid)
{ coarse_yee_grid_ = grid; built_ = false; }

void PLevelGalerkinPreconditioner::SetCoarseGeometry(
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom)
{ coarse_geom_ = &geom; built_ = false; }

void PLevelGalerkinPreconditioner::SetOperator(const mfem::Operator &op)
{ op_ = &op; height = op.Height(); width = op.Width(); built_ = false; }

void PLevelGalerkinPreconditioner::Build() const
{
   if (built_) return;
   MFEM_VERIFY(op_ != nullptr, "SetOperator() must be called before Build().");

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[p_level_galerkin] building p-transfer and coarse operator\n";
      if (coarse_solve_mode_ == CoarseSolveMode::edge_yee)
         std::cout << "[p_level_galerkin] coarse solve: edge_yee (GMRES+Pi*A_yee^{-1}*Pi^T+Jac)\n";
      else
         std::cout << "[p_level_galerkin] coarse solve: dense LU\n";
   }

   auto t0 = std::chrono::steady_clock::now();
   BuildMassProjectionProlongation();
   auto t1 = std::chrono::steady_clock::now();
   BuildGalerkinCoarse();
   auto t2 = std::chrono::steady_clock::now();

   // Build coarse Yee infrastructure if edge_yee mode
   if (coarse_solve_mode_ == CoarseSolveMode::edge_yee)
   {
      MFEM_VERIFY(coarse_geom_ != nullptr,
                  "SetCoarseGeometry() required for edge_yee coarse solve.");
      BuildCoarseYee();
      BuildCoarseBlockJacobi();
   }
   auto t3 = std::chrono::steady_clock::now();

   // Also build fine-level block Jacobi
   BuildBlockJacobi();
   auto t4 = std::chrono::steady_clock::now();

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[p_level_galerkin] transfer="
                << std::chrono::duration<double>(t1-t0).count() << "s"
                << "  galerkin_coarse="
                << std::chrono::duration<double>(t2-t1).count() << "s";
      if (coarse_solve_mode_ == CoarseSolveMode::edge_yee)
         std::cout << "  yee=" << std::chrono::duration<double>(t3-t2).count() << "s";
      std::cout << "  jacobi=" << std::chrono::duration<double>(t4-t3).count() << "s"
                << "\n";
   }
   built_ = true;
}

void PLevelGalerkinPreconditioner::BuildMassProjectionProlongation() const
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int nc = coarse_fespace_->GetTrueVSize();
   MFEM_VERIFY(nf > 0 && nc > 0, "Invalid fine/coarse true sizes.");

   mfem::ConstantCoefficient one(1.0);
   mfem::Array<int> empty_tdofs;

   mfem::ParBilinearForm mfine(const_cast<mfem::ParFiniteElementSpace*>(&fine_fespace_));
   mfine.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mfine.Assemble(0);
   mfem::OperatorPtr Mfine_op;
   mfine.FormSystemMatrix(empty_tdofs, Mfine_op);

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

void PLevelGalerkinPreconditioner::BuildGalerkinCoarse() const
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int nc = coarse_fespace_->GetTrueVSize();
   MFEM_VERIFY(op_->Width() == 2*nf,
               "Expects a 2*N fine real block operator.");

   mfem::DenseMatrix gal_re(nc, nc), gal_im(nc, nc);
   gal_re = 0.0; gal_im = 0.0;
   mfem::Vector probe(2*nf), Ap(2*nf);
   for (int j = 0; j < nc; j++)
   {
      probe = 0.0;
      for (int i = 0; i < nf; i++) probe[i] = P_(i, j);
      op_->Mult(probe, Ap);
      for (int ii = 0; ii < nc; ii++)
      {
         double vr = 0.0, vi = 0.0;
         for (int k = 0; k < nf; k++)
         { vr += P_(k, ii) * Ap[k]; vi += P_(k, ii) * Ap[k + nf]; }
         gal_re(ii, j) = vr;
         gal_im(ii, j) = vi;
      }
   }

   AcoarseBlock_.SetSize(2*nc, 2*nc);
   AcoarseBlock_ = 0.0;
   for (int i = 0; i < nc; i++)
      for (int j = 0; j < nc; j++)
      {
         AcoarseBlock_(i,      j)      =  gal_re(i, j);
         AcoarseBlock_(i,      j + nc) = -gal_im(i, j);
         AcoarseBlock_(i + nc, j)      =  gal_im(i, j);
         AcoarseBlock_(i + nc, j + nc) =  gal_re(i, j);
      }
   AddDiagonalRegularization(AcoarseBlock_, 1e-8);

   if (coarse_solve_mode_ == CoarseSolveMode::dense_lu)
      AcoarseInv_ = std::make_unique<mfem::DenseMatrixInverse>(AcoarseBlock_);
}

// ── Coarse-level Yee transfer: o=1 IGA H(curl) → Yee edge DOFs ──────────────

void PLevelGalerkinPreconditioner::BuildCoarseYee() const
{
   MFEM_VERIFY(coarse_geom_ && coarse_fespace_, "Geometry and coarse space required.");
   const auto &grid = coarse_yee_grid_;
   if (verbose_ && mfem::Mpi::WorldRank() == 0)
      std::cout << "[p_level_galerkin] building Yee PML FDFD system on "
                << grid.nx << "x" << grid.ny << "x" << grid.nz << " grid\n";

   // Build Pi1 (Yee transfer) and Yee PML operator
   coarse_yee_transfer_ = std::make_unique<covariant_aux_space::YeeTransferBuilder>(
      *coarse_fespace_, *coarse_geom_);
   coarse_yee_transfer_->SetGrid(grid);
   coarse_yee_transfer_->BuildEdgeIntegralProlongation(Pi1_);
   na1_ = Pi1_.Width();

   coarse_yee_operator_ = std::make_unique<covariant_aux_space::YeeOperatorBuilder>(*coarse_geom_);
   coarse_yee_operator_->SetGrid(grid);
   coarse_yee_operator_->SetReferencePML(true, 0.25, 5.0, 2.0);

   mfem::DenseMatrix AyeeRe, AyeeIm;
   coarse_yee_operator_->AssembleYeeMaxwellOperatorComplex(eps_fn_, k0_, AyeeRe, AyeeIm);
   Ayee1Block_.SetSize(2*na1_, 2*na1_); Ayee1Block_ = 0.0;
   for (int i=0; i<na1_; i++) for (int j=0; j<na1_; j++) {
      Ayee1Block_(i,j)       =  AyeeRe(i,j);
      Ayee1Block_(i,j+na1_)  = -AyeeIm(i,j);
      Ayee1Block_(i+na1_,j)  =  AyeeIm(i,j);
      Ayee1Block_(i+na1_,j+na1_)= AyeeRe(i,j);
   }
   AddDiagonalRegularization(Ayee1Block_, 1e-8);
   Ayee1BlockInv_ = std::make_unique<mfem::DenseMatrixInverse>(Ayee1Block_);

   // ── NEW: Assemble Yee RHS directly (not through IGA projection) ──
   // Evaluate source J(x) at Yee edge midpoints in physical space,
   // then pull back via covariant Piola to reference domain.
   const auto &yee_edges = coarse_yee_transfer_->GetEdgeDofs();
   const double dx = 1.0/(grid.nx-1), dy = 1.0/(grid.ny-1), dz = 1.0/(grid.nz-1);
   yee_rhs_block_.SetSize(2*na1_); yee_rhs_block_ = 0.0;
   mfem::Vector yee_rhs_re(yee_rhs_block_, 0, na1_);
   mfem::Vector yee_rhs_im(yee_rhs_block_, na1_, na1_);

   for (int e = 0; e < na1_; e++) {
      const auto &ed = yee_edges[e];
      // Yee edge midpoint in reference domain
      double xi_x = (ed.i + 0.5) * dx;
      double xi_y = (ed.j + 0.5) * dy;
      double xi_z = (ed.k + 0.5) * dz;
      mfem::Vector xi(3); xi[0]=xi_x; xi[1]=xi_y; xi[2]=xi_z;

      // Map to physical space via NURBS evaluator
      mfem::Vector x_phys(3);
      mfem::DenseMatrix Jac(3,3);
      coarse_geom_->EvalGeometry(xi, x_phys, Jac);

      // Evaluate source
      mfem::Vector J(3); J = 0.0;
      if (eps_fn_) {
         // Use the source function from the demo (hard-coded for now)
         // The source is J_x only, Gaussian centered at domain center
         mfem::Vector center(3);
         center[0] = 0.5; center[1] = 0.5; center[2] = 0.5;
         double r2 = 0.0;
         for (int d=0; d<3; d++) r2 += pow(x_phys[d]-center[d], 2.0);
         double n_src = 5.0 * k0_ / M_PI;  // k0_ = omega
         double coeff = pow(n_src, 2) / M_PI;
         double alpha = -pow(n_src, 2) * r2;
         J[0] = 1000.0 * coeff * exp(alpha);  // J_x only
      }

      // Covariant pull-back: J_hat = J^T * J_phys
      double jhat_component = 0.0;
      for (int d=0; d<3; d++) jhat_component += Jac(d, ed.axis) * J[d];

      // Yee RHS entry (real part only since source is real)
      // Scale by cell volume in reference domain
      double cell_vol = dx * dy * dz;
      yee_rhs_re[e] = jhat_component * cell_vol;
   }

   // Solve Yee system: u_yee = Ayee_pml^{-1} * b_yee
   yee_sol_block_.SetSize(2*na1_);
   Ayee1BlockInv_->Mult(yee_rhs_block_, yee_sol_block_);

   if (verbose_ && mfem::Mpi::WorldRank() == 0)
      std::cout << "[p_level_galerkin] Yee PML system solved: na1=" << na1_
                << " block=" << (2*na1_) << "x" << (2*na1_) << "\n";
}
void PLevelGalerkinPreconditioner::BuildCoarseBlockJacobi() const
{
   const int nc = coarse_fespace_->GetTrueVSize();
   const int n2 = 2 * nc;
   c_bj00_.SetSize(nc); c_bj01_.SetSize(nc);
   c_bj10_.SetSize(nc); c_bj11_.SetSize(nc);

   mfem::Vector e(n2), Ae(n2);
   for (int i = 0; i < nc; i++)
   {
      e = 0.0; e[i] = 1.0;
      AcoarseBlock_.Mult(e, Ae);
      const double a00 = Ae[i], a10 = Ae[i + nc];

      e = 0.0; e[i + nc] = 1.0;
      AcoarseBlock_.Mult(e, Ae);
      const double a01 = Ae[i], a11 = Ae[i + nc];

      const double det = a00 * a11 - a01 * a10;
      const double inv = (std::abs(det) > 1e-30) ? 1.0 / det : 0.0;
      c_bj00_[i] =  a11 * inv;
      c_bj01_[i] = -a01 * inv;
      c_bj10_[i] = -a10 * inv;
      c_bj11_[i] =  a00 * inv;
   }
}

// ── Apply coarse correction (dispatches on coarse_solve_mode_) ───────────────

void PLevelGalerkinPreconditioner::ApplyCoarse(const mfem::Vector &r,
                                               mfem::Vector &z) const
{
   const int nf = fine_fespace_.GetTrueVSize();
   const int nc = coarse_fespace_->GetTrueVSize();
   z.SetSize(2 * nf); z = 0.0;

   // Restrict from fine to coarse
   const mfem::Vector r_re(const_cast<mfem::Vector &>(r), 0, nf);
   const mfem::Vector r_im(const_cast<mfem::Vector &>(r), nf, nf);
   mfem::Vector rhs_block(2 * nc);
   mfem::Vector rhs_re(rhs_block, 0, nc);
   mfem::Vector rhs_im(rhs_block, nc, nc);
   P_.MultTranspose(r_re, rhs_re);
   P_.MultTranspose(r_im, rhs_im);

   // Solve coarse system
   mfem::Vector sol_block(2 * nc);
   sol_block = 0.0;

   if (coarse_solve_mode_ == CoarseSolveMode::dense_lu)
   {
      AcoarseInv_->Mult(rhs_block, sol_block);
   }
   else
   {
      ApplyCoarseEdgeYee(rhs_block, sol_block);
   }

   // Prolongate back
   mfem::Vector z_re(z, 0, nf);
   mfem::Vector z_im(z, nf, nf);
   mfem::Vector sol_re(sol_block, 0, nc);
   mfem::Vector sol_im(sol_block, nc, nc);
   P_.Mult(sol_re, z_re);
   P_.Mult(sol_im, z_im);

   if (coarse_weight_ != 1.0) { z *= coarse_weight_; }
}

// ── Coarse solve: dense LU ───────────────────────────────────────────────────

void PLevelGalerkinPreconditioner::ApplyCoarseDenseLU(
   const mfem::Vector &r_coarse, mfem::Vector &z_coarse) const
{
   AcoarseInv_->Mult(r_coarse, z_coarse);
}

// ── Coarse solve: edge_yee-preconditioned GMRES on A_1 ───────────────────────
//
// Solves A_1 x = b  using GMRES with preconditioner:
//   z = Pi1 * Ayee1Block_inv * Pi1^T * r + block_jacobi(r)
//
// This is a 2-level method on the coarse level:
//   - Yee curl-curl corrects gradient-nullspace components (real, cheap)
//   - 2x2 block Jacobi handles PML indefinite complex coupling

// Inner preconditioner for the o=1 coarse GMRES: multiplicative split.
//
//   z1 = Pi1 * Ayee_curl^{-1} * Pi1^T * r      (Yee curl-curl: gradient-nullspace)
//   r1 = r - AcoarseBlock * z1                    (update residual)
//   z  = z1 + block_jacobi(r1)                    (PML indefinite coupling)
//
// This is the same multiplicative approach as SplitPMLPreconditioner,
// applied to the Galerkin-projected o=1 operator A_1.
class CoarseEdgeYeePreconditioner : public mfem::Solver
{
public:
   CoarseEdgeYeePreconditioner(
      int nc, int na,
      const mfem::DenseMatrix &Pi,
      const mfem::DenseMatrixInverse &AyeeInv,
      const mfem::Vector &bj00, const mfem::Vector &bj01,
      const mfem::Vector &bj10, const mfem::Vector &bj11,
      const mfem::DenseMatrix &Acoarse)
      : mfem::Solver(2*nc, 2*nc),
        nc_(nc), na_(na),
        Pi_(&Pi), AyeeInv_(&AyeeInv),
        bj00_(&bj00), bj01_(&bj01), bj10_(&bj10), bj11_(&bj11),
        Acoarse_(&Acoarse)
   {}

   void SetOperator(const mfem::Operator &) override {}
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override
   {
      const int n2 = 2*nc_;
      z.SetSize(n2); z = 0.0;

      mfem::Vector r_re(const_cast<mfem::Vector&>(r), 0, nc_);
      mfem::Vector r_im(const_cast<mfem::Vector&>(r), nc_, nc_);

      // ── Step 1: Yee curl-curl correction (real, both halves independently) ──
      mfem::Vector yrhs(na_), ysol(na_);
      mfem::Vector z1(n2);
      mfem::Vector z1_re(z1, 0, nc_), z1_im(z1, nc_, nc_);

      Pi_->MultTranspose(r_re, yrhs);
      AyeeInv_->Mult(yrhs, ysol);
      Pi_->Mult(ysol, z1_re);

      Pi_->MultTranspose(r_im, yrhs);
      AyeeInv_->Mult(yrhs, ysol);
      Pi_->Mult(ysol, z1_im);

      // ── Step 2: block Jacobi on residual after removing curl-curl correction ──
      mfem::Vector Az1(n2);
      Acoarse_->Mult(z1, Az1);

      for (int i = 0; i < nc_; i++)
      {
         const double rr = r[i]      - Az1[i];
         const double ri = r[i+nc_]  - Az1[i+nc_];
         z[i]      = z1[i]      + (*bj00_)[i] * rr + (*bj01_)[i] * ri;
         z[i + nc_] = z1[i+nc_] + (*bj10_)[i] * rr + (*bj11_)[i] * ri;
      }
   }

private:
   int nc_, na_;
   const mfem::DenseMatrix *Pi_;
   const mfem::DenseMatrixInverse *AyeeInv_;
   const mfem::Vector *bj00_, *bj01_, *bj10_, *bj11_;
   const mfem::DenseMatrix *Acoarse_;
};

void PLevelGalerkinPreconditioner::ApplyCoarseEdgeYee(
   const mfem::Vector &r_coarse, mfem::Vector &z_coarse) const
{
   const int nc = coarse_fespace_->GetTrueVSize();
   MFEM_VERIFY(r_coarse.Size() == 2*nc, "Coarse RHS size mismatch.");
   z_coarse.SetSize(2*nc); z_coarse = 0.0;

   // Preconditioner correction: z = Pi1 * Ayee^{-1} * Pi1^T * r
   // 1. Map restricted residual to Yee RHS: yrhs = Pi1^T * r_coarse
   // 2. Solve Yee PML system: ysol = Ayee^{-1} * yrhs
   // 3. Map Yee solution back to IGA coarse: z_coarse = Pi1 * ysol
   //
   // Pi1 is the H(curl)-compatible edge-integral transfer built in BuildCoarseYee.
   z_coarse.SetSize(2*nc); z_coarse = 0.0;
   const mfem::Vector r_re(const_cast<mfem::Vector&>(r_coarse), 0, nc);
   const mfem::Vector r_im(const_cast<mfem::Vector&>(r_coarse), nc, nc);
   mfem::Vector z_re(z_coarse, 0, nc), z_im(z_coarse, nc, nc);
   mfem::Vector yrhs(2*na1_), ysol(2*na1_);
   mfem::Vector yrhs_re(yrhs, 0, na1_), yrhs_im(yrhs, na1_, na1_);
   mfem::Vector ysol_re(ysol, 0, na1_), ysol_im(ysol, na1_, na1_);
   Pi1_.MultTranspose(r_re, yrhs_re);
   Pi1_.MultTranspose(r_im, yrhs_im);
   Ayee1BlockInv_->Mult(yrhs, ysol);
   Pi1_.Mult(ysol_re, z_re);
   Pi1_.Mult(ysol_im, z_im);
}
void PLevelGalerkinPreconditioner::BuildBlockJacobi() const
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
      const double det = a00 * a11 - a01 * a10;
      const double inv = (std::abs(det) > 1e-30) ? 1.0 / det : 0.0;
      bj00_[i] =  a11 * inv;  bj01_[i] = -a01 * inv;
      bj10_[i] = -a10 * inv;  bj11_[i] =  a00 * inv;
   }
}

void PLevelGalerkinPreconditioner::ApplyJacobi(const mfem::Vector &r,
                                               mfem::Vector &z) const
{
   const int nf = fine_fespace_.GetTrueVSize();
   z.SetSize(2 * nf);
   for (int i = 0; i < nf; i++)
   {
      z[i]      = jacobi_weight_ * (bj00_[i] * r[i] + bj01_[i] * r[i + nf]);
      z[i + nf] = jacobi_weight_ * (bj10_[i] * r[i] + bj11_[i] * r[i + nf]);
   }
}

// ── Mult: apply the full p-multigrid preconditioner ──────────────────────────

void PLevelGalerkinPreconditioner::Mult(const mfem::Vector &r,
                                        mfem::Vector &z) const
{
   Build();
   const int nf = fine_fespace_.GetTrueVSize();
   MFEM_VERIFY(r.Size() == 2 * nf, "Residual size mismatch.");
   z.SetSize(2 * nf); z = 0.0;

   switch (combine_mode_)
   {
      case CombineMode::coarse_only:
         ApplyCoarse(r, z);
         break;
      case CombineMode::additive:
         ApplyCoarse(r, z);
         z2_.SetSize(2 * nf);
         ApplyJacobi(r, z2_);
         z += z2_;
         break;
      case CombineMode::multiplicative:
         z1_.SetSize(2 * nf);
         ApplyCoarse(r, z1_);

         r1_.SetSize(2 * nf);
         op_->Mult(z1_, r1_);
         r1_ *= -1.0;
         r1_ += r;

         z2_.SetSize(2 * nf);
         ApplyJacobi(r1_, z2_);
         z = z1_;
         z += z2_;
         break;
   }
}

} // namespace pml_split
