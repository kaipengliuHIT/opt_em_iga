// ============================================================================
// prec_selector.cpp — Diagnostic-driven adaptive preconditioner selector
// for H(curl)-conforming IGA Maxwell problems (SPD and PML).
//
// 4 candidate preconditioners:
//   1. Jacobi:  B_J^{-1} r = ω D^{-1} r
//   2. Pi_lumped + Jacobi additive:
//      B_P^{-1} r = Π A_c^{-1} Π^T r + ω D^{-1} r
//   3. Jacobi→Pi multiplicative:
//      z_J = ω D^{-1} r,  r1 = r - A z_J,  z_P = Π A_c^{-1} Π^T r1,  z = z_J+z_P
//   4. Pi→Jacobi multiplicative:
//      z_P = Π A_c^{-1} Π^T r,  r1 = r - A z_P,  z_J = ω D^{-1} r1,  z = z_P+z_J
//
// 2 selector schemes:
//   A. One-step residual reduction probe:  ρ = |r0 - A B^{-1} r0| / |r0|
//   B. 10-step warm-up probe: score = log(|r0|/|r10|) / (setup + probe_time)
//
// Usage:
//   SPD:  ./prec_selector -system spd -r 2 -o 2 -f 5 -ao 1
//   PML:  ./prec_selector -system pml -r 2 -o 2 -f 5
// ============================================================================

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <chrono>
#include <cfloat>
#include <complex>
#include <algorithm>
#include <vector>
#include <memory>
using namespace std;
using namespace std::chrono;
using namespace mfem;

// ===========================================================================
// RHS functions
// ===========================================================================
void gaussian_rhs(const Vector &x, Vector &f)
{
   f = 0.0;
   double r2 = 0.0;
   for (int d = 0; d < 3; d++) { r2 += pow(x[d] - 0.5, 2); }
   f[0] = exp(-100.0 * r2);
}

void point_source(const Vector &x, Vector &f);

// ===========================================================================
// Unified CG result
// ===========================================================================
struct CGResult
{
   int    iters      = 0;
   double rel_final  = 1.0;
   double rel_10     = 1.0;
   double rel_50     = 1.0;
   double rel_100    = 1.0;
   bool   converged  = false;
   bool   breakdown  = false;
   double setup_ms   = 0.0;
   double apply_ms   = 0.0;
   int    dim_pi     = 0;
   int    nnz_pi     = 0;
};

// ===========================================================================
// Candidate result for probe
// ===========================================================================
struct CandidateResult
{
   string name;
   int    candidate_id;   // 1=Jacobi, 2=Pi+Jac, 3=Jac→Pi, 4=Pi→Jac
   double omega;
   double one_step_rho    = 1.0;
   double setup_probe_ms  = 0.0;
   double apply_probe_ms  = 0.0;
   // 10-step warm-up
   double rel_10_warmup   = 1.0;
   double warmup_score    = 0.0;
   double warmup_setup_ms = 0.0;
   double warmup_solve_ms = 0.0;
   // Full solve
   CGResult full;
   // Selection flags
   bool selected_one_step = false;
   bool selected_warmup   = false;
};

// ===========================================================================
// SPD: Sparse Pi Preconditioner (additive, from ams_sweep_v3.cpp)
// ===========================================================================
class SparsePiPreconditioner : public Solver
{
public:
   SparsePiPreconditioner(const HypreParMatrix &M_cross,
                          const Vector         &D_inv_M,
                          const Vector         &D_inv_A,
                          const DenseMatrix    &Ac,
                          double                omega)
      : M_cross_(M_cross), D_inv_M_(D_inv_M), D_inv_A_(D_inv_A),
        omega_(omega), n_(M_cross.Height()), no1_(M_cross.Width())
   {
      Ac_reg_ = Ac;
      double mxQ = 0.0;
      for (int i = 0; i < no1_; i++) { mxQ = max(mxQ, abs(Ac_reg_(i, i))); }
      double reg = max(1e-12, mxQ * 1e-10);
      for (int i = 0; i < no1_; i++) { Ac_reg_(i, i) += reg; }
      Ac_inv_ = make_unique<DenseMatrixInverse>(Ac_reg_);
      r_scaled_.SetSize(n_);  r_coarse_.SetSize(no1_);
      z_coarse_.SetSize(no1_); tmp_.SetSize(n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // z = Π A_c^{-1} Π^T r + ω D_A^{-1} r
      for (int i = 0; i < n_; i++) { r_scaled_[i] = r[i] * D_inv_M_[i]; }
      M_cross_.MultTranspose(r_scaled_, r_coarse_);
      z_coarse_ = 0.0;  Ac_inv_->Mult(r_coarse_, z_coarse_);
      M_cross_.Mult(z_coarse_, tmp_);
      for (int i = 0; i < n_; i++) { z[i] = tmp_[i] * D_inv_M_[i]; }
      for (int i = 0; i < n_; i++) { z[i] += omega_ * D_inv_A_[i] * r[i]; }
   }
   void SetOperator(const Operator &op) override { height = op.Height(); width = op.Width(); }
   int CoarseDim() const { return no1_; }
   int NnzCross()  const { return M_cross_.NNZ(); }

private:
   const HypreParMatrix &M_cross_;
   const Vector &D_inv_M_, &D_inv_A_;
   double omega_;
   int n_, no1_;
   DenseMatrix Ac_reg_;
   unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_scaled_, r_coarse_, z_coarse_, tmp_;
};

// ===========================================================================
// SPD: Jacobi→Pi multiplicative
//   z_J = ω D_A^{-1} r,  r1 = r - A z_J,  z_P = Π A_c^{-1} Π^T r1,  z = z_J + z_P
// ===========================================================================
class JacobiToPiPreconditioner : public Solver
{
public:
   JacobiToPiPreconditioner(const Operator         &A_op,
                            const HypreParMatrix   &M_cross,
                            const Vector           &D_inv_M,
                            const Vector           &D_inv_A,
                            const DenseMatrix      &Ac,
                            double                  omega)
      : A_op_(A_op), M_cross_(M_cross), D_inv_M_(D_inv_M), D_inv_A_(D_inv_A),
        omega_(omega), n_(M_cross.Height()), no1_(M_cross.Width())
   {
      Ac_reg_ = Ac;
      double mxQ = 0.0;
      for (int i = 0; i < no1_; i++) { mxQ = max(mxQ, abs(Ac_reg_(i, i))); }
      double reg = max(1e-12, mxQ * 1e-10);
      for (int i = 0; i < no1_; i++) { Ac_reg_(i, i) += reg; }
      Ac_inv_ = make_unique<DenseMatrixInverse>(Ac_reg_);
      r_scaled_.SetSize(n_);  r_coarse_.SetSize(no1_);
      z_coarse_.SetSize(no1_); tmp_.SetSize(n_);  r1_.SetSize(n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // z_J = ω D_A^{-1} r
      for (int i = 0; i < n_; i++) { z[i] = omega_ * D_inv_A_[i] * r[i]; }
      // r1 = r - A z_J
      A_op_.Mult(z, tmp_);
      for (int i = 0; i < n_; i++) { r1_[i] = r[i] - tmp_[i]; }
      // z_P = Π A_c^{-1} Π^T r1
      for (int i = 0; i < n_; i++) { r_scaled_[i] = r1_[i] * D_inv_M_[i]; }
      M_cross_.MultTranspose(r_scaled_, r_coarse_);
      z_coarse_ = 0.0;  Ac_inv_->Mult(r_coarse_, z_coarse_);
      M_cross_.Mult(z_coarse_, tmp_);
      for (int i = 0; i < n_; i++) { z[i] += tmp_[i] * D_inv_M_[i]; }
   }
   void SetOperator(const Operator &op) override { height = op.Height(); width = op.Width(); }
   int CoarseDim() const { return no1_; }
   int NnzCross()  const { return M_cross_.NNZ(); }

private:
   const Operator         &A_op_;
   const HypreParMatrix   &M_cross_;
   const Vector           &D_inv_M_, &D_inv_A_;
   double                  omega_;
   int                     n_, no1_;
   DenseMatrix             Ac_reg_;
   unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_scaled_, r_coarse_, z_coarse_, tmp_, r1_;
};

// ===========================================================================
// SPD: Pi→Jacobi multiplicative
//   z_P = Π A_c^{-1} Π^T r,  r1 = r - A z_P,  z_J = ω D_A^{-1} r1,  z = z_P + z_J
// ===========================================================================
class PiToJacobiPreconditioner : public Solver
{
public:
   PiToJacobiPreconditioner(const Operator         &A_op,
                            const HypreParMatrix   &M_cross,
                            const Vector           &D_inv_M,
                            const Vector           &D_inv_A,
                            const DenseMatrix      &Ac,
                            double                  omega)
      : A_op_(A_op), M_cross_(M_cross), D_inv_M_(D_inv_M), D_inv_A_(D_inv_A),
        omega_(omega), n_(M_cross.Height()), no1_(M_cross.Width())
   {
      Ac_reg_ = Ac;
      double mxQ = 0.0;
      for (int i = 0; i < no1_; i++) { mxQ = max(mxQ, abs(Ac_reg_(i, i))); }
      double reg = max(1e-12, mxQ * 1e-10);
      for (int i = 0; i < no1_; i++) { Ac_reg_(i, i) += reg; }
      Ac_inv_ = make_unique<DenseMatrixInverse>(Ac_reg_);
      r_scaled_.SetSize(n_);  r_coarse_.SetSize(no1_);
      z_coarse_.SetSize(no1_); tmp_.SetSize(n_);  r1_.SetSize(n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // z_P = Π A_c^{-1} Π^T r
      for (int i = 0; i < n_; i++) { r_scaled_[i] = r[i] * D_inv_M_[i]; }
      M_cross_.MultTranspose(r_scaled_, r_coarse_);
      z_coarse_ = 0.0;  Ac_inv_->Mult(r_coarse_, z_coarse_);
      M_cross_.Mult(z_coarse_, tmp_);
      for (int i = 0; i < n_; i++) { z[i] = tmp_[i] * D_inv_M_[i]; }
      // r1 = r - A z_P
      A_op_.Mult(z, tmp_);
      for (int i = 0; i < n_; i++) { r1_[i] = r[i] - tmp_[i]; }
      // z_J = ω D_A^{-1} r1
      for (int i = 0; i < n_; i++) { z[i] += omega_ * D_inv_A_[i] * r1_[i]; }
   }
   void SetOperator(const Operator &op) override { height = op.Height(); width = op.Width(); }
   int CoarseDim() const { return no1_; }
   int NnzCross()  const { return M_cross_.NNZ(); }

private:
   const Operator         &A_op_;
   const HypreParMatrix   &M_cross_;
   const Vector           &D_inv_M_, &D_inv_A_;
   double                  omega_;
   int                     n_, no1_;
   DenseMatrix             Ac_reg_;
   unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_scaled_, r_coarse_, z_coarse_, tmp_, r1_;
};

// ===========================================================================
// SPD: Form Ac = Π^T A Π via sparse column-by-column probing
// ===========================================================================
DenseMatrix FormCoarseOperator(const HypreParMatrix &A,
                               const HypreParMatrix &M_cross,
                               const Vector         &D_inv,
                               int ncurl, int no1)
{
   DenseMatrix Ac(no1, no1);  Ac = 0.0;
   Vector eJ(no1), pj(ncurl), Apj(ncurl), Apj_s(ncurl), Ac_col(no1);
   for (int j = 0; j < no1; j++)
   {
      eJ = 0.0; eJ[j] = 1.0;
      pj = 0.0; M_cross.Mult(eJ, pj);
      for (int i = 0; i < ncurl; i++) { pj[i] *= D_inv[i]; }
      Apj = 0.0; A.Mult(pj, Apj);
      for (int i = 0; i < ncurl; i++) { Apj_s[i] = Apj[i] * D_inv[i]; }
      Ac_col = 0.0; M_cross.MultTranspose(Apj_s, Ac_col);
      for (int k = 0; k < no1; k++) { Ac(k, j) = Ac_col[k]; }
   }
   for (int i = 0; i < no1; i++)
      for (int j = i + 1; j < no1; j++)
      { double avg = 0.5 * (Ac(i, j) + Ac(j, i)); Ac(i, j) = Ac(j, i) = avg; }
   return Ac;
}

// ===========================================================================
// SPD CG solvers
// ===========================================================================
CGResult run_cg(const Operator &A_op, const Vector &B, Solver *prec,
                double tol = 1e-8, int max_iter = 500)
{
   CGResult res;
   int N = A_op.Height();
   Vector X(N);  X = 0.0;
   Vector R(B.Size());  R = B;
   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector Z(N), P(N), AP(N);
   if (prec) { prec->Mult(R, Z); }
   else      { Z = R; }
   P = Z;
   double rz = R * Z;
   if (rz <= 0) { res.breakdown = true; return res; }

   double rel_res = 1.0;
   int iter;
   auto ta0 = steady_clock::now();
   for (iter = 1; iter <= max_iter; iter++)
   {
      A_op.Mult(P, AP);
      double pap = P * AP;
      if (pap <= 0 || !isfinite(pap)) { res.breakdown = true; break; }
      double alpha = rz / pap;
      X.Add(alpha, P);
      R.Add(-alpha, AP);
      rel_res = sqrt(R * R) / bnorm;
      if (iter == 10)  { res.rel_10  = rel_res; }
      if (iter == 50)  { res.rel_50  = rel_res; }
      if (iter == 100) { res.rel_100 = rel_res; }
      if (rel_res <= tol) { break; }
      if (prec) { prec->Mult(R, Z); }
      else      { Z = R; }
      double rz_new = R * Z;
      if (rz_new <= 0 || !isfinite(rz_new)) { res.breakdown = true; break; }
      double beta = rz_new / rz;
      for (int i = 0; i < N; i++) { P[i] = Z[i] + beta * P[i]; }
      rz = rz_new;
   }
   res.apply_ms = duration<double>(steady_clock::now() - ta0).count() * 1000.0;
   res.iters = min(iter, max_iter);
   res.rel_final = rel_res;
   res.converged = (rel_res <= tol);
   return res;
}

// ===========================================================================
// SPD: one-step residual probe
//   ρ = |r0 - A B^{-1} r0| / |r0|
// ===========================================================================
CandidateResult probe_one_step_spd(const Operator &A_op, const Vector &r0,
                                    Solver &prec, const string &name,
                                    int cand_id, double omega)
{
   CandidateResult cr;
   cr.name = name;
   cr.candidate_id = cand_id;
   cr.omega = omega;

   int N = A_op.Height();
   double bnorm = sqrt(r0 * r0);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector z(N), Az(N);
   auto tp0 = steady_clock::now();
   prec.Mult(r0, z);
   cr.apply_probe_ms = duration<double>(steady_clock::now() - tp0).count() * 1000.0;

   A_op.Mult(z, Az);
   double rho_num = 0.0;
   for (int i = 0; i < N; i++) { double d = r0[i] - Az[i]; rho_num += d * d; }
   cr.one_step_rho = sqrt(rho_num) / bnorm;

   return cr;
}

// ===========================================================================
// SPD: 10-step warm-up probe
//   score = log(|r0|/|r10|) / (setup_time + probe_solve_time)
// ===========================================================================
CandidateResult probe_warmup_spd(const Operator &A_op, const Vector &B,
                                  Solver &prec, const string &name,
                                  int cand_id, double omega,
                                  double setup_ms)
{
   CandidateResult cr;
   cr.name = name;
   cr.candidate_id = cand_id;
   cr.omega = omega;
   cr.warmup_setup_ms = setup_ms;

   int N = A_op.Height();
   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector X(N);  X = 0.0;
   Vector R(B.Size());  R = B;
   Vector Z(N), P(N), AP(N);
   double rz = 0.0;

   auto tp0 = steady_clock::now();

   for (int iter = 1; iter <= 10; iter++)
   {
      prec.Mult(R, Z);
      double rz_new = R * Z;
      if (rz_new <= 0 || !isfinite(rz_new)) { break; }
      if (iter == 1) { P = Z; }
      else { double beta = rz_new / rz; for (int i = 0; i < N; i++) P[i] = Z[i] + beta * P[i]; }
      A_op.Mult(P, AP);
      double pap = P * AP;
      if (pap <= 0 || !isfinite(pap)) { break; }
      double alpha = rz_new / pap;
      X.Add(alpha, P);
      R.Add(-alpha, AP);
      rz = rz_new;
   }

   cr.warmup_solve_ms = duration<double>(steady_clock::now() - tp0).count() * 1000.0;
   cr.rel_10_warmup = sqrt(R * R) / bnorm;
   double log_ratio = log(1.0 / max(cr.rel_10_warmup, 1e-30));
   cr.warmup_score = log_ratio / max(cr.warmup_setup_ms + cr.warmup_solve_ms, 1e-6);

   return cr;
}

// ===========================================================================
// SPD: run all candidates + selectors
// ===========================================================================
void run_spd_case(int myid, const char *mesh_file, int ref, int order,
                  int aux_order, double freq, ostream &out)
{
   int dim = 3;
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref; l++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->NURBSext = nullptr; delete mesh;

   auto *fec      = new NURBS_HCurlFECollection(order, dim);
   auto *nurbsExt = new NURBSExtension(pmesh->NURBSext, order);
   auto *fespace  = new ParFiniteElementSpace(pmesh, nurbsExt, fec);
   int   ncurl    = fespace->GetTrueVSize();

   auto *aux_fec  = new NURBS_HCurlFECollection(aux_order, dim);
   auto *aux_ext  = new NURBSExtension(pmesh->NURBSext, aux_order);
   auto *aux_fes  = new ParFiniteElementSpace(pmesh, aux_ext, aux_fec);
   int   no1      = aux_fes->GetTrueVSize();

   if (myid == 0)
   {
      out << "\n══════════════════════════════════════════════════════════\n";
      out << " Case: SPD  r=" << ref << "  o=" << order
          << "  f=" << (int)freq << "  aux_o=" << aux_order
          << "  N=" << ncurl << "  dim(Pi)=" << no1 << "\n";
      out << "══════════════════════════════════════════════════════════\n";
   }

   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   double omega_val = 2.0 * M_PI * freq;
   double k2 = omega_val * omega_val;
   Array<int> attr_dom(pmesh->attributes.Max());
   attr_dom = 0; attr_dom[0] = 1;
   ConstantCoefficient muinv_c(1.0), k2_c(k2), one(1.0);
   RestrictedCoefficient muinv_r(muinv_c, attr_dom);
   RestrictedCoefficient k2_r(k2_c, attr_dom);

   auto t_assemble_start = steady_clock::now();
   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   a.Assemble(0); a.Finalize(0);
   OperatorPtr A_op; a.FormSystemMatrix(ess_tdofs, A_op);
   auto *Ah = A_op.As<HypreParMatrix>();

   ParLinearForm b_form(fespace);
   VectorFunctionCoefficient src_coeff(dim, gaussian_rhs);
   b_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b_form.Assemble();
   Vector B_vec(ncurl); b_form.ParallelAssemble(B_vec);
   for (int i = 0; i < ess_tdofs.Size(); i++) { B_vec[ess_tdofs[i]] = 0.0; }

   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0); mform.Finalize(0);
   OperatorPtr M_op_ptr; mform.FormSystemMatrix(ess_tdofs, M_op_ptr);
   auto *M_curl = M_op_ptr.As<HypreParMatrix>();

   ParMixedBilinearForm mmix(aux_fes, fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0); mmix.Finalize(0);
   Array<int> empty_tdofs;
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs, empty_tdofs, Mmix_op);
   auto *M_cross = Mmix_op.As<HypreParMatrix>();

   double t_assemble = duration<double>(steady_clock::now() - t_assemble_start).count();

   Vector D_inv_M(ncurl); M_curl->GetDiag(D_inv_M);
   for (int i = 0; i < ncurl; i++) { D_inv_M[i] = 1.0 / max(1e-14, abs(D_inv_M[i])); }
   Vector D_inv_A(ncurl); Ah->GetDiag(D_inv_A);
   for (int i = 0; i < ncurl; i++) { D_inv_A[i] = 1.0 / max(1e-14, abs(D_inv_A[i])); }

   auto t_ac0 = steady_clock::now();
   DenseMatrix Ac = FormCoarseOperator(*Ah, *M_cross, D_inv_M, ncurl, no1);
   double t_ac = duration<double>(steady_clock::now() - t_ac0).count();

   if (myid == 0)
   {
      out << "[SPD] assemble=" << fixed << setprecision(2) << t_assemble
          << "s  form_Ac=" << t_ac << "s  nnz(M_cross)=" << M_cross->NNZ()
          << "  Ac=" << no1 << "x" << no1 << endl;
   }

   // Build all candidate preconditioners
   vector<CandidateResult> candidates;
   double omega_list[] = {0.5, 0.7, 1.0};

   // Candidate 1: Jacobi (no Pi setup needed)
   for (double w : omega_list)
   {
      ostringstream nm;
      nm << "Jacobi ω=" << fixed << setprecision(1) << w;
      // Jacobi is just diagonal scaling, implement inline
      class JacobiPrec : public Solver
      {
      public:
         JacobiPrec(const Vector &D_inv, double w) : D_inv_(D_inv), w_(w) {}
         void Mult(const Vector &r, Vector &z) const override
         { for (int i = 0; i < D_inv_.Size(); i++) z[i] = w_ * D_inv_[i] * r[i]; }
         void SetOperator(const Operator &op) override { height = op.Height(); width = op.Width(); }
      private:
         const Vector &D_inv_; double w_;
      };
      JacobiPrec jac(D_inv_A, w);
      jac.SetOperator(*Ah);

      // One-step probe
      auto cr = probe_one_step_spd(*Ah, B_vec, jac, nm.str(), 1, w);
      // 10-step warm-up (merge, don't overwrite one-step result)
      auto cr_w = probe_warmup_spd(*Ah, B_vec, jac, nm.str(), 1, w, 0.0);
      cr.rel_10_warmup   = cr_w.rel_10_warmup;
      cr.warmup_score    = cr_w.warmup_score;
      cr.warmup_setup_ms = cr_w.warmup_setup_ms;
      cr.warmup_solve_ms = cr_w.warmup_solve_ms;
      candidates.push_back(cr);
   }

   // Candidates 2-4: Pi-based (share Ac setup)
   double setup_pi_ms = t_ac * 1000.0;

   auto t_prec0 = steady_clock::now();
   SparsePiPreconditioner prec_pi_add(*M_cross, D_inv_M, D_inv_A, Ac, 1.0);
   JacobiToPiPreconditioner prec_jac2pi(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, 1.0);
   PiToJacobiPreconditioner prec_pi2jac(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, 1.0);

   for (double w : omega_list)
   {
      // Candidate 2: Pi_lumped + Jacobi additive
      {
         SparsePiPreconditioner prec(*M_cross, D_inv_M, D_inv_A, Ac, w);
         prec.SetOperator(*Ah);
         ostringstream nm;
         nm << "Pi_lumped+Jac ω=" << fixed << setprecision(1) << w;
         auto cr = probe_one_step_spd(*Ah, B_vec, prec, nm.str(), 2, w);
         cr.setup_probe_ms = setup_pi_ms;
         auto cr_w = probe_warmup_spd(*Ah, B_vec, prec, nm.str(), 2, w, setup_pi_ms);
         cr.rel_10_warmup   = cr_w.rel_10_warmup;
         cr.warmup_score    = cr_w.warmup_score;
         cr.warmup_setup_ms = cr_w.warmup_setup_ms;
         cr.warmup_solve_ms = cr_w.warmup_solve_ms;
         cr.setup_probe_ms = setup_pi_ms;
         candidates.push_back(cr);
      }
      // Candidate 3: Jacobi→Pi multiplicative
      {
         JacobiToPiPreconditioner prec(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, w);
         prec.SetOperator(*Ah);
         ostringstream nm;
         nm << "Jac→Pi ω=" << fixed << setprecision(1) << w;
         auto cr = probe_one_step_spd(*Ah, B_vec, prec, nm.str(), 3, w);
         cr.setup_probe_ms = setup_pi_ms;
         auto cr_w = probe_warmup_spd(*Ah, B_vec, prec, nm.str(), 3, w, setup_pi_ms);
         cr.rel_10_warmup   = cr_w.rel_10_warmup;
         cr.warmup_score    = cr_w.warmup_score;
         cr.warmup_setup_ms = cr_w.warmup_setup_ms;
         cr.warmup_solve_ms = cr_w.warmup_solve_ms;
         cr.setup_probe_ms = setup_pi_ms;
         candidates.push_back(cr);
      }
      // Candidate 4: Pi→Jacobi multiplicative
      {
         PiToJacobiPreconditioner prec(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, w);
         prec.SetOperator(*Ah);
         ostringstream nm;
         nm << "Pi→Jac ω=" << fixed << setprecision(1) << w;
         auto cr = probe_one_step_spd(*Ah, B_vec, prec, nm.str(), 4, w);
         cr.setup_probe_ms = setup_pi_ms;
         auto cr_w = probe_warmup_spd(*Ah, B_vec, prec, nm.str(), 4, w, setup_pi_ms);
         cr.rel_10_warmup   = cr_w.rel_10_warmup;
         cr.warmup_score    = cr_w.warmup_score;
         cr.warmup_setup_ms = cr_w.warmup_setup_ms;
         cr.warmup_solve_ms = cr_w.warmup_solve_ms;
         cr.setup_probe_ms = setup_pi_ms;
         candidates.push_back(cr);
      }
   }

   // Select winners
   int best_one_step = 0, best_warmup = 0;
   double min_rho = 1e30, max_score = -1e30;
   for (int i = 0; i < (int)candidates.size(); i++)
   {
      if (candidates[i].one_step_rho < min_rho)
      { min_rho = candidates[i].one_step_rho; best_one_step = i; }
      if (candidates[i].warmup_score > max_score)
      { max_score = candidates[i].warmup_score; best_warmup = i; }
   }
   candidates[best_one_step].selected_one_step = true;
   candidates[best_warmup].selected_warmup = true;

   if (myid == 0)
   {
      // Print probe results table
      out << "\n┌─ Probe Results "
          << string(90, '─') << "┐\n";
      out << "│ " << left << setw(32) << "Candidate" << " "
          << right << setw(6) << "ω" << " "
          << setw(10) << "1-step ρ" << " "
          << setw(12) << "10-step |r|" << " "
          << setw(10) << "score" << " "
          << setw(12) << "setup(ms)" << " "
          << setw(10) << "probe(ms)" << " "
          << setw(10) << "Sel-1s" << " "
          << setw(10) << "Sel-10s" << " │\n";
      out << "├─" << string(32, '─') << "─"
          << string(7, '─') << "─"
          << string(11, '─') << "─"
          << string(13, '─') << "─"
          << string(11, '─') << "─"
          << string(13, '─') << "─"
          << string(11, '─') << "─"
          << string(11, '─') << "─"
          << string(11, '─') << "┤\n";

      for (auto &cr : candidates)
      {
         out << "│ " << left << setw(32) << cr.name << " "
             << fixed << setprecision(1) << right << setw(6) << cr.omega << " "
             << scientific << setprecision(3) << setw(10) << cr.one_step_rho << " "
             << scientific << setprecision(3) << setw(12) << cr.rel_10_warmup << " "
             << scientific << setprecision(3) << setw(10) << cr.warmup_score << " "
             << fixed << setprecision(1) << setw(12) << cr.setup_probe_ms << " "
             << fixed << setprecision(1) << setw(10) << cr.apply_probe_ms << " "
             << setw(10) << (cr.selected_one_step ? " ★YES" : "     ")
             << setw(10) << (cr.selected_warmup   ? " ★YES" : "     ") << " │\n";
      }
      out << "└─" << string(32, '─') << "─"
          << string(7, '─') << "─"
          << string(11, '─') << "─"
          << string(13, '─') << "─"
          << string(11, '─') << "─"
          << string(13, '─') << "─"
          << string(11, '─') << "─"
          << string(11, '─') << "─"
          << string(11, '─') << "┘\n";

      // Run full solves for all candidates
      out << "\n┌─ Full Solve Comparison "
          << string(100, '─') << "┐\n";
      out << "│ " << left << setw(32) << "Method" << " "
          << right << setw(5) << "Iters" << " "
          << setw(9) << "rel(10)" << " "
          << setw(9) << "rel(50)" << " "
          << setw(9) << "rel(100)" << " "
          << setw(9) << "rel(end)" << " "
          << setw(6) << "Break" << " "
          << setw(8) << "Setup" << " "
          << setw(8) << "Apply" << " "
          << setw(12) << "Sel-1s?" << " "
          << setw(12) << "Sel-10s?" << " │\n";
      out << "├─" << string(32, '─') << "─"
          << string(6, '─') << "─"
          << string(10, '─') << "─"
          << string(10, '─') << "─"
          << string(10, '─') << "─"
          << string(10, '─') << "─"
          << string(7, '─') << "─"
          << string(9, '─') << "─"
          << string(9, '─') << "─"
          << string(13, '─') << "─"
          << string(13, '─') << "┤\n";

      // None
      { auto r = run_cg(*Ah, B_vec, nullptr); auto fmt=[&](double v){if(v<1.0)out<<scientific<<setprecision(2)<<setw(9)<<v;else out<<"    —     ";}; out << "│ " << left << setw(32) << "none" << " " << right << setw(5) << r.iters << " "; fmt(r.rel_10); out<<" "; fmt(r.rel_50); out<<" "; fmt(r.rel_100); out<<" "; out<<scientific<<setprecision(2)<<setw(9)<<r.rel_final<<" "<<setw(6)<<(r.breakdown?"BRK":" ")<<" "<<fixed<<setprecision(0)<<setw(7)<<0<<"ms"<<" "<<setw(7)<<r.apply_ms<<"ms"<<" "<<setw(12)<<" "<<" "<<setw(12)<<" │\n"; }

      // Full solves for all candidates
      for (auto &cr : candidates)
      {
         unique_ptr<Solver> prec_ptr;
         if (cr.candidate_id == 1)
         {
            class JacobiPrec2 : public Solver {
            public:
               JacobiPrec2(const Vector &D, double w) : D_(D), w_(w) {}
               void Mult(const Vector &r, Vector &z) const override
               { for (int i = 0; i < D_.Size(); i++) z[i] = w_ * D_[i] * r[i]; }
               void SetOperator(const Operator &op) override { height=op.Height(); width=op.Width(); }
            private: const Vector &D_; double w_;
            };
            prec_ptr = make_unique<JacobiPrec2>(D_inv_A, cr.omega);
         }
         else if (cr.candidate_id == 2)
            prec_ptr = make_unique<SparsePiPreconditioner>(*M_cross, D_inv_M, D_inv_A, Ac, cr.omega);
         else if (cr.candidate_id == 3)
            prec_ptr = make_unique<JacobiToPiPreconditioner>(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, cr.omega);
         else if (cr.candidate_id == 4)
            prec_ptr = make_unique<PiToJacobiPreconditioner>(*Ah, *M_cross, D_inv_M, D_inv_A, Ac, cr.omega);
         prec_ptr->SetOperator(*Ah);
         cr.full = run_cg(*Ah, B_vec, prec_ptr.get());
         cr.full.dim_pi = (cr.candidate_id >= 2) ? no1 : 0;
         cr.full.nnz_pi = (cr.candidate_id >= 2) ? M_cross->NNZ() : 0;
         auto fmt=[&](double v){if(v<1.0)out<<scientific<<setprecision(2)<<setw(9)<<v;else out<<"    —     ";};
         out << "│ " << left << setw(32) << cr.name << " " << right << setw(5) << cr.full.iters << " ";
         fmt(cr.full.rel_10); out<<" "; fmt(cr.full.rel_50); out<<" "; fmt(cr.full.rel_100); out<<" ";
         out << scientific << setprecision(2) << setw(9) << cr.full.rel_final << " "
             << setw(6) << (cr.full.breakdown?"BRK":" ") << " "
             << fixed << setprecision(0) << setw(7) << cr.setup_probe_ms << "ms" << " "
             << setw(7) << cr.full.apply_ms << "ms" << " "
             << setw(12) << (cr.selected_one_step ? "  ★1-STEP" : " ")
             << " " << setw(12) << (cr.selected_warmup   ? "  ★WARMUP" : " ") << " │\n";
      }
      out << "└─" << string(32,'─') << "─" << string(6,'─') << "─" << string(10,'─') << "─"
          << string(10,'─') << "─" << string(10,'─') << "─" << string(10,'─') << "─"
          << string(7,'─') << "─" << string(9,'─') << "─" << string(9,'─') << "─"
          << string(13,'─') << "─" << string(13,'─') << "┘\n";
      out.flush();

      // Summary
      out << "\n★★★ SELECTOR DECISION ★★★\n";
      out << "  One-step probe selects: " << candidates[best_one_step].name << " (ρ="
          << scientific << setprecision(3) << candidates[best_one_step].one_step_rho << ")\n";
      out << "  10-step warmup selects: " << candidates[best_warmup].name << " (score="
          << scientific << setprecision(3) << candidates[best_warmup].warmup_score << ")\n";
      out << "  Best one-step iters=" << candidates[best_one_step].full.iters
          << "  Best warmup iters=" << candidates[best_warmup].full.iters << "\n\n";
   }

   // NOTE: FE spaces are not explicitly deleted to avoid dangling
   // references in HypreParMatrix objects still held by OperatorHandle.
   // The OS will reclaim memory at process exit.
   delete pmesh;
}// ===========================================================================
// PML class (from pml_pi_prec_test.cpp)
// ===========================================================================
class PML
{
private:
   Mesh *mesh;
   int dim;
   Array2D<real_t> length;
   Array2D<real_t> comp_dom_bdr;
   Array2D<real_t> dom_bdr;
   Array<int> elems;
   void SetBoundaries();
public:
   PML(Mesh *mesh_, Array2D<real_t> length_);
   Array2D<real_t> GetCompDomainBdr() { return comp_dom_bdr; }
   Array2D<real_t> GetDomainBdr()      { return dom_bdr; }
   Array<int> *GetMarkedPMLElements()  { return &elems; }
   void SetAttributes(ParMesh *mesh_);
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs);
};

class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML *pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int dim,
                            void (*F)(const Vector &, PML *, Vector &),
                            PML *pml_)
      : VectorCoefficient(dim), pml(pml_), Function(F) {}
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      real_t x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K);
   }
};

template <typename T> T pow2(const T &x) { return x * x; }

static real_t g_omega_pml = 1.0, g_mu_pml = 1.0, g_epsilon_pml = 1.0;

void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D);

// ===========================================================================
// Complex Pi Preconditioner (additive, from pml_pi_prec_test.cpp)
// ===========================================================================
class ComplexPiPreconditioner : public Solver
{
public:
   ComplexPiPreconditioner(const HypreParMatrix &M_cross,
                           const Vector         &D_inv_M,
                           const DenseMatrix    &Ac_r,
                           const DenseMatrix    &Ac_i,
                           const Vector         &D_r,
                           const Vector         &D_i,
                           double                omega)
      : M_cross_(M_cross), D_inv_M_(D_inv_M),
        Ac_r_(Ac_r), Ac_i_(Ac_i),
        D_r_(D_r), D_i_(D_i), omega_(omega),
        n_(M_cross.Height()), no1_(M_cross.Width())
   { BuildCoarseSolver(); AllocWork(); }

   void Mult(const Vector &r, Vector &z) const override
   {
      for (int i = 0; i < n_; i++)
      { r_scaled_r_(i) = r(i) * D_inv_M_(i); r_scaled_i_(i) = r(i + n_) * D_inv_M_(i); }
      M_cross_.MultTranspose(r_scaled_r_, rc_r_);
      M_cross_.MultTranspose(r_scaled_i_, rc_i_);
      rc_i_.Neg();
      for (int i = 0; i < no1_; i++)
      { r_coarse_full_(i) = rc_r_(i); r_coarse_full_(i + no1_) = rc_i_(i); }
      z_coarse_full_ = 0.0; Ac_inv_->Mult(r_coarse_full_, z_coarse_full_);
      for (int i = 0; i < no1_; i++)
      { zc_r_(i) = z_coarse_full_(i); zc_i_(i) = z_coarse_full_(i + no1_); }
      M_cross_.Mult(zc_r_, tmp_r_); M_cross_.Mult(zc_i_, tmp_i_);
      for (int i = 0; i < n_; i++)
      { z(i) = tmp_r_(i) * D_inv_M_(i); z(i + n_) = tmp_i_(i) * D_inv_M_(i); }
      for (int i = 0; i < n_; i++)
      {
         double inv_den = 1.0 / (D_r_[i]*D_r_[i] + D_i_[i]*D_i_[i] + 1e-30);
         double pr = D_r_[i] * inv_den, pi = D_i_[i] * inv_den;
         z[i]      += omega_ * (pr * r[i]      + pi * r[i + n_]);
         z[i + n_] += omega_ * (pr * r[i + n_] - pi * r[i]);
      }
   }
   void SetOperator(const Operator &op) override { height=op.Height(); width=op.Width(); }
   int CoarseDim() const { return no1_; }
   int NnzCross()  const { return M_cross_.NNZ(); }

private:
   const HypreParMatrix &M_cross_;
   const Vector &D_inv_M_;
   const DenseMatrix &Ac_r_, &Ac_i_;
   const Vector &D_r_, &D_i_;
   double omega_;
   int n_, no1_;
   DenseMatrix Ac_full_;
   unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_scaled_r_, r_scaled_i_, rc_r_, rc_i_, zc_r_, zc_i_;
   mutable Vector r_coarse_full_, z_coarse_full_, tmp_r_, tmp_i_;

   void BuildCoarseSolver()
   {
      int Nc = 2 * no1_;
      Ac_full_.SetSize(Nc, Nc); Ac_full_ = 0.0;
      for (int i = 0; i < no1_; i++)
         for (int j = 0; j < no1_; j++)
         {
            double ar = Ac_r_(i,j), ai = Ac_i_(i,j);
            Ac_full_(i,j)=ar; Ac_full_(i,j+no1_)=-ai;
            Ac_full_(i+no1_,j)=ai; Ac_full_(i+no1_,j+no1_)=ar;
         }
      double mx=0.0;
      for (int i=0;i<Nc;i++) mx=max(mx,abs(Ac_full_(i,i)));
      double reg=max(1e-12,mx*1e-10);
      for (int i=0;i<Nc;i++) Ac_full_(i,i)+=reg;
      Ac_inv_=make_unique<DenseMatrixInverse>(Ac_full_);
   }
   void AllocWork()
   {
      r_scaled_r_.SetSize(n_); r_scaled_i_.SetSize(n_);
      rc_r_.SetSize(no1_); rc_i_.SetSize(no1_);
      zc_r_.SetSize(no1_); zc_i_.SetSize(no1_);
      r_coarse_full_.SetSize(2*no1_); z_coarse_full_.SetSize(2*no1_);
      tmp_r_.SetSize(n_); tmp_i_.SetSize(n_);
   }
};

// ===========================================================================
// Complex Jacobi (2×2 block inversion per node)
// ===========================================================================
class ComplexJacobi : public Solver
{
public:
   ComplexJacobi(const Vector &Dr, const Vector &Di, double w)
      : Dr_(Dr), Di_(Di), w_(w) {}
   void Mult(const Vector &r, Vector &z) const override
   {
      int n = Dr_.Size();
      for (int i = 0; i < n; i++)
      {
         double inv = 1.0 / (Dr_[i]*Dr_[i] + Di_[i]*Di_[i] + 1e-30);
         double pr = Dr_[i] * inv * w_, pi = Di_[i] * inv * w_;
         z[i]     = pr * r[i]     + pi * r[i + n];
         z[i + n] = pr * r[i + n] - pi * r[i];
      }
   }
   void SetOperator(const Operator &op) override { height=op.Height(); width=op.Width(); }
private:
   const Vector &Dr_, &Di_; double w_;
};

// ===========================================================================
// Complex Jacobi→Pi multiplicative
//   z_J = ω D^{-1} r, r1 = r - A z_J, z_P = Π A_c^{-1} Π^T r1, z = z_J + z_P
// ===========================================================================
class ComplexJacobiToPiPreconditioner : public Solver
{
public:
   ComplexJacobiToPiPreconditioner(const Operator       &A_op,
                                   const HypreParMatrix &M_cross,
                                   const Vector         &D_inv_M,
                                   const DenseMatrix    &Ac_r,
                                   const DenseMatrix    &Ac_i,
                                   const Vector         &D_r,
                                   const Vector         &D_i,
                                   double                omega)
      : A_op_(A_op), M_cross_(M_cross), D_inv_M_(D_inv_M),
        Ac_r_(Ac_r), Ac_i_(Ac_i), D_r_(D_r), D_i_(D_i),
        omega_(omega), n_(M_cross.Height()), no1_(M_cross.Width())
   {
      // Build 2no1×2no1 coarse solver
      int Nc = 2*no1_;
      Ac_full_.SetSize(Nc,Nc); Ac_full_=0.0;
      for (int i=0;i<no1_;i++)
         for (int j=0;j<no1_;j++)
         { double ar=Ac_r_(i,j),ai=Ac_i_(i,j);
           Ac_full_(i,j)=ar; Ac_full_(i,j+no1_)=-ai;
           Ac_full_(i+no1_,j)=ai; Ac_full_(i+no1_,j+no1_)=ar; }
      double mx=0.0; for(int i=0;i<Nc;i++)mx=max(mx,abs(Ac_full_(i,i)));
      double reg=max(1e-12,mx*1e-10); for(int i=0;i<Nc;i++)Ac_full_(i,i)+=reg;
      Ac_inv_=make_unique<DenseMatrixInverse>(Ac_full_);
      r_sr_.SetSize(n_); r_si_.SetSize(n_); rcr_.SetSize(no1_); rci_.SetSize(no1_);
      zcr_.SetSize(no1_); zci_.SetSize(no1_);
      rcf_.SetSize(2*no1_); zcf_.SetSize(2*no1_);
      tr_.SetSize(n_); ti_.SetSize(n_); r1_.SetSize(2*n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // z_J = ω complex Jacobi(r)
      for (int i=0;i<n_;i++)
      {
         double inv=1.0/(D_r_[i]*D_r_[i]+D_i_[i]*D_i_[i]+1e-30);
         double pr=D_r_[i]*inv*omega_, pi=D_i_[i]*inv*omega_;
         z[i]=pr*r[i]+pi*r[i+n_]; z[i+n_]=pr*r[i+n_]-pi*r[i];
      }
      // r1 = r - A z_J
      A_op_.Mult(z, r1_);
      for (int i=0;i<2*n_;i++) r1_[i]=r[i]-r1_[i];
      // z_P = Π A_c^{-1} Π^T r1
      for (int i=0;i<n_;i++){r_sr_(i)=r1_[i]*D_inv_M_(i);r_si_(i)=r1_[i+n_]*D_inv_M_(i);}
      M_cross_.MultTranspose(r_sr_,rcr_); M_cross_.MultTranspose(r_si_,rci_); rci_.Neg();
      for(int i=0;i<no1_;i++){rcf_(i)=rcr_(i);rcf_(i+no1_)=rci_(i);}
      zcf_=0.0; Ac_inv_->Mult(rcf_,zcf_);
      for(int i=0;i<no1_;i++){zcr_(i)=zcf_(i);zci_(i)=zcf_(i+no1_);}
      M_cross_.Mult(zcr_,tr_); M_cross_.Mult(zci_,ti_);
      for(int i=0;i<n_;i++){z[i]+=tr_(i)*D_inv_M_(i); z[i+n_]+=ti_(i)*D_inv_M_(i);}
   }
   void SetOperator(const Operator &op) override { height=op.Height(); width=op.Width(); }
private:
   const Operator &A_op_; const HypreParMatrix &M_cross_; const Vector &D_inv_M_;
   const DenseMatrix &Ac_r_,&Ac_i_; const Vector &D_r_,&D_i_; double omega_; int n_,no1_;
   DenseMatrix Ac_full_; unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_sr_,r_si_,rcr_,rci_,zcr_,zci_,rcf_,zcf_,tr_,ti_,r1_;
};

// ===========================================================================
// Complex Pi→Jacobi multiplicative
// ===========================================================================
class ComplexPiToJacobiPreconditioner : public Solver
{
public:
   ComplexPiToJacobiPreconditioner(const Operator       &A_op,
                                   const HypreParMatrix &M_cross,
                                   const Vector         &D_inv_M,
                                   const DenseMatrix    &Ac_r,
                                   const DenseMatrix    &Ac_i,
                                   const Vector         &D_r,
                                   const Vector         &D_i,
                                   double                omega)
      : A_op_(A_op), M_cross_(M_cross), D_inv_M_(D_inv_M),
        Ac_r_(Ac_r), Ac_i_(Ac_i), D_r_(D_r), D_i_(D_i),
        omega_(omega), n_(M_cross.Height()), no1_(M_cross.Width())
   {
      int Nc = 2*no1_;
      Ac_full_.SetSize(Nc,Nc); Ac_full_=0.0;
      for (int i=0;i<no1_;i++)
         for (int j=0;j<no1_;j++)
         { double ar=Ac_r_(i,j),ai=Ac_i_(i,j);
           Ac_full_(i,j)=ar; Ac_full_(i,j+no1_)=-ai;
           Ac_full_(i+no1_,j)=ai; Ac_full_(i+no1_,j+no1_)=ar; }
      double mx=0.0; for(int i=0;i<Nc;i++)mx=max(mx,abs(Ac_full_(i,i)));
      double reg=max(1e-12,mx*1e-10); for(int i=0;i<Nc;i++)Ac_full_(i,i)+=reg;
      Ac_inv_=make_unique<DenseMatrixInverse>(Ac_full_);
      r_sr_.SetSize(n_); r_si_.SetSize(n_); rcr_.SetSize(no1_); rci_.SetSize(no1_);
      zcr_.SetSize(no1_); zci_.SetSize(no1_);
      rcf_.SetSize(2*no1_); zcf_.SetSize(2*no1_);
      tr_.SetSize(n_); ti_.SetSize(n_); r1_.SetSize(2*n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // z_P = Π A_c^{-1} Π^T r
      for (int i=0;i<n_;i++){r_sr_(i)=r[i]*D_inv_M_(i);r_si_(i)=r[i+n_]*D_inv_M_(i);}
      M_cross_.MultTranspose(r_sr_,rcr_); M_cross_.MultTranspose(r_si_,rci_); rci_.Neg();
      for(int i=0;i<no1_;i++){rcf_(i)=rcr_(i);rcf_(i+no1_)=rci_(i);}
      zcf_=0.0; Ac_inv_->Mult(rcf_,zcf_);
      for(int i=0;i<no1_;i++){zcr_(i)=zcf_(i);zci_(i)=zcf_(i+no1_);}
      M_cross_.Mult(zcr_,tr_); M_cross_.Mult(zci_,ti_);
      for(int i=0;i<n_;i++){z[i]=tr_(i)*D_inv_M_(i); z[i+n_]=ti_(i)*D_inv_M_(i);}
      // r1 = r - A z_P
      A_op_.Mult(z, r1_);
      for (int i=0;i<2*n_;i++) r1_[i]=r[i]-r1_[i];
      // z_J = ω complex Jacobi(r1)
      for (int i=0;i<n_;i++)
      {
         double inv=1.0/(D_r_[i]*D_r_[i]+D_i_[i]*D_i_[i]+1e-30);
         double pr=D_r_[i]*inv*omega_, pi=D_i_[i]*inv*omega_;
         z[i]+=pr*r1_[i]+pi*r1_[i+n_]; z[i+n_]+=pr*r1_[i+n_]-pi*r1_[i];
      }
   }
   void SetOperator(const Operator &op) override { height=op.Height(); width=op.Width(); }
private:
   const Operator &A_op_; const HypreParMatrix &M_cross_; const Vector &D_inv_M_;
   const DenseMatrix &Ac_r_,&Ac_i_; const Vector &D_r_,&D_i_; double omega_; int n_,no1_;
   DenseMatrix Ac_full_; unique_ptr<DenseMatrixInverse> Ac_inv_;
   mutable Vector r_sr_,r_si_,rcr_,rci_,zcr_,zci_,rcf_,zcf_,tr_,ti_,r1_;
};

// ===========================================================================
// Form complex coarse operator (from pml_pi_prec_test.cpp)
// ===========================================================================
void FormComplexCoarseOperator(const HypreParMatrix &A_r,
                               const HypreParMatrix &A_i,
                               const HypreParMatrix &M_cross,
                               const Vector         &D_inv,
                               int n, int no1,
                               DenseMatrix &Ac_r, DenseMatrix &Ac_i)
{
   Ac_r.SetSize(no1,no1); Ac_i.SetSize(no1,no1); Ac_r=0.0; Ac_i=0.0;
   Vector eJ(no1), pj(n), Apj_r(n), Apj_i(n), Apj_s_r(n), Apj_s_i(n);
   Vector Ac_col_r(no1), Ac_col_i(no1);
   for (int j=0;j<no1;j++)
   {
      eJ=0.0; eJ[j]=1.0; pj=0.0; M_cross.Mult(eJ,pj);
      for(int i=0;i<n;i++)pj[i]*=D_inv[i];
      Apj_r=0.0; A_r.Mult(pj,Apj_r); Apj_i=0.0; A_i.Mult(pj,Apj_i);
      for(int i=0;i<n;i++){Apj_s_r[i]=Apj_r[i]*D_inv[i]; Apj_s_i[i]=Apj_i[i]*D_inv[i];}
      Ac_col_r=0.0; M_cross.MultTranspose(Apj_s_r,Ac_col_r);
      Ac_col_i=0.0; M_cross.MultTranspose(Apj_s_i,Ac_col_i);
      for(int k=0;k<no1;k++){Ac_r(k,j)=Ac_col_r[k]; Ac_i(k,j)=Ac_col_i[k];}
   }
   for(int i=0;i<no1;i++)
   {
      for(int j=i+1;j<no1;j++)
      { double ar=0.5*(Ac_r(i,j)+Ac_r(j,i)), ai=0.5*(Ac_i(i,j)+Ac_i(j,i));
        Ac_r(i,j)=Ac_r(j,i)=ar; Ac_i(i,j)=Ac_i(j,i)=ai; }
      Ac_i(i,i)=0.0;
   }
}

// ===========================================================================
// GMRES result
// ===========================================================================
struct GMRESResult
{
   int iters=0; double rel_final=1.0,rel_10=1.0,rel_50=1.0,rel_100=1.0;
   bool converged=false,breakdown=false,nan_detected=false;
   double setup_ms=0.0,total_ms=0.0; int dim_pi=0,nnz_pi=0;
};

// ===========================================================================
// Right-preconditioned GMRES for complex block system
// ===========================================================================
GMRESResult run_gmres(const Operator &A_op, const Vector &B, Solver *prec,
                      int restart=200, int max_iter=5000, double tol=1e-8)
{
   GMRESResult res; int N=A_op.Height();
   double bnorm=sqrt(B*B); if(bnorm<1e-30)bnorm=1.0;
   int m=restart;
   DenseMatrix H(m+1,m), V(N,m+1);
   Vector g(m+1), cs(m), sn(m), y_(m), w(N), Av(N);
   Vector X(N); X=0.0;
   Vector R(B.Size()); R=B;
   double beta=sqrt(R*R), rel_res=beta/bnorm;
   int total_iter=0;
   auto t_start=steady_clock::now();
   while(total_iter<max_iter && rel_res>tol)
   {
      for(int i=0;i<N;i++)V(i,0)=R[i]/beta; g=0.0; g[0]=beta;
      int j;
      for(j=0;j<m && total_iter<max_iter;j++,total_iter++)
      {
         if(prec){Vector vj(V.GetData()+j*N,N); prec->Mult(vj,w);}
         else {for(int i=0;i<N;i++)w[i]=V(i,j);}
         A_op.Mult(w,Av);
         for(int i=0;i<=j;i++){Vector vi(V.GetData()+i*N,N); H(i,j)=Av*vi; Av.Add(-H(i,j),vi);}
         H(j+1,j)=sqrt(Av*Av); if(H(j+1,j)<1e-30)H(j+1,j)=0.0;
         if(H(j+1,j)>0){for(int i=0;i<N;i++)V(i,j+1)=Av[i]/H(j+1,j);}
         for(int i=0;i<j;i++){double h1=cs[i]*H(i,j)+sn[i]*H(i+1,j),h2=-sn[i]*H(i,j)+cs[i]*H(i+1,j);H(i,j)=h1;H(i+1,j)=h2;}
         double hjj=H(j,j),hjp1j=H(j+1,j),rho=sqrt(hjj*hjj+hjp1j*hjp1j);
         if(rho<1e-30){res.breakdown=true;break;}
         cs[j]=hjj/rho; sn[j]=hjp1j/rho; H(j,j)=rho; H(j+1,j)=0.0;
         double g1=cs[j]*g[j]+sn[j]*g[j+1],g2=-sn[j]*g[j]+cs[j]*g[j+1];
         g[j]=g1; g[j+1]=g2;
         rel_res=abs(g[j+1])/bnorm;
         if(total_iter==9)res.rel_10=rel_res;
         if(total_iter==49)res.rel_50=rel_res;
         if(total_iter==99)res.rel_100=rel_res;
         if(rel_res<=tol){j++;break;}
      }
      for(int i=j-1;i>=0;i--){y_[i]=g[i]; for(int k=i+1;k<j;k++)y_[i]-=H(i,k)*y_[k]; y_[i]/=H(i,i);}
      for(int k=0;k<j;k++){Vector vk(V.GetData()+k*N,N); if(prec)prec->Mult(vk,w); else w=vk; X.Add(y_[k],w);}
      A_op.Mult(X,Av); for(int i=0;i<N;i++)R[i]=B[i]-Av[i];
      beta=sqrt(R*R); rel_res=beta/bnorm;
      if(!isfinite(rel_res)){res.nan_detected=true;break;}
   }
   res.total_ms=duration<double>(steady_clock::now()-t_start).count()*1000.0;
   res.iters=total_iter; res.rel_final=rel_res; res.converged=(rel_res<=tol);
   return res;
}

// ===========================================================================
// PML Candidate result
// ===========================================================================
struct PMLCandidateResult
{
   string name; int candidate_id; double omega;
   double one_step_rho=1.0; double setup_probe_ms=0.0; double apply_probe_ms=0.0;
   double rel_10_warmup=1.0; double warmup_score=0.0; double warmup_setup_ms=0.0; double warmup_solve_ms=0.0;
   GMRESResult full;
   bool selected_one_step=false, selected_warmup=false;
};

// ===========================================================================
// PML one-step probe
// ===========================================================================
PMLCandidateResult probe_one_step_pml(const Operator &A_op, const Vector &r0,
                                       Solver &prec, const string &name,
                                       int cand_id, double omega)
{
   PMLCandidateResult cr; cr.name=name; cr.candidate_id=cand_id; cr.omega=omega;
   int N=A_op.Height();
   double bnorm=sqrt(r0*r0); if(bnorm<1e-30)bnorm=1.0;
   Vector z(N), Az(N);
   auto tp0=steady_clock::now();
   prec.Mult(r0,z);
   cr.apply_probe_ms=duration<double>(steady_clock::now()-tp0).count()*1000.0;
   A_op.Mult(z,Az);
   double rho_num=0.0; for(int i=0;i<N;i++){double d=r0[i]-Az[i];rho_num+=d*d;}
   cr.one_step_rho=sqrt(rho_num)/bnorm;
   return cr;
}

// ===========================================================================
// PML 10-step warm-up probe (FGMRES)
// ===========================================================================
PMLCandidateResult probe_warmup_pml(const Operator &A_op, const Vector &B,
                                     Solver &prec, const string &name,
                                     int cand_id, double omega, double setup_ms)
{
   PMLCandidateResult cr; cr.name=name; cr.candidate_id=cand_id; cr.omega=omega;
   cr.warmup_setup_ms=setup_ms;
   int N=A_op.Height(), m=10;
   double bnorm=sqrt(B*B); if(bnorm<1e-30)bnorm=1.0;
   DenseMatrix H(m+1,m), V_(N,m+1);
   Vector g_(m+1), cs(m), sn(m), w(N), Av(N), y_(m);
   Vector X(N); X=0.0; Vector R(B.Size()); R=B;
   double beta=sqrt(R*R);
   int steps=0, j_done=0;
   auto tp0=steady_clock::now();
   for (int cycle=0;cycle<1&&steps<10;cycle++)
   {
      for(int i=0;i<N;i++)V_(i,0)=R[i]/beta; g_=0.0; g_[0]=beta;
      int j;
      for(j=0;j<m&&steps<10;j++,steps++)
      {
         {Vector vj(V_.GetData()+j*N,N); prec.Mult(vj,w);}
         A_op.Mult(w,Av);
         for(int i=0;i<=j;i++){Vector vi(V_.GetData()+i*N,N);H(i,j)=Av*vi;Av.Add(-H(i,j),vi);}
         H(j+1,j)=sqrt(Av*Av); if(H(j+1,j)<1e-30)H(j+1,j)=0.0;
         if(H(j+1,j)>0){for(int i=0;i<N;i++)V_(i,j+1)=Av[i]/H(j+1,j);}
         for(int i=0;i<j;i++){double h1=cs[i]*H(i,j)+sn[i]*H(i+1,j),h2=-sn[i]*H(i,j)+cs[i]*H(i+1,j);H(i,j)=h1;H(i+1,j)=h2;}
         double rho=sqrt(H(j,j)*H(j,j)+H(j+1,j)*H(j+1,j));
         if(rho<1e-30)break;
         cs[j]=H(j,j)/rho; sn[j]=H(j+1,j)/rho; H(j,j)=rho; H(j+1,j)=0.0;
         g_[j+1]=-sn[j]*g_[j]+cs[j]*g_[j+1]; g_[j]=cs[j]*g_[j]+sn[j]*g_[j+1];
      }
      // Back-solve and update X, then recompute residual (was missing!)
      j_done=j;
      for(int i=j_done-1;i>=0;i--){y_[i]=g_[i];for(int k=i+1;k<j_done;k++)y_[i]-=H(i,k)*y_[k];y_[i]/=H(i,i);}
      for(int k=0;k<j_done;k++){Vector vk(V_.GetData()+k*N,N);prec.Mult(vk,w);X.Add(y_[k],w);}
      A_op.Mult(X,Av); for(int i=0;i<N;i++)R[i]=B[i]-Av[i];
      beta=sqrt(R*R);
   }
   cr.warmup_solve_ms=duration<double>(steady_clock::now()-tp0).count()*1000.0;
   cr.rel_10_warmup=sqrt(R*R)/bnorm;
   double lr=log(1.0/max(cr.rel_10_warmup,1e-30));
   cr.warmup_score=lr/max(cr.warmup_setup_ms+cr.warmup_solve_ms,1e-6);
   return cr;
}

// ===========================================================================
// PML: run all candidates + selectors
// ===========================================================================
void run_pml_case(int myid, const char *mesh_file, int ref, int order,
                  int aux_order, double freq, ostream &out)
{
   Device device("cpu"); int dim=3;
   Mesh *serial_mesh=new Mesh(mesh_file,1,1);
   Array2D<real_t> pml_length(dim,2); pml_length=0.25;
   PML *pml=new PML(serial_mesh,pml_length);
   real_t omega_val=real_t(2.0*M_PI)*freq, mu=1.0, epsilon=1.0;
   g_omega_pml=omega_val; g_mu_pml=mu; g_epsilon_pml=epsilon;
   for(int l=0;l<ref;l++)serial_mesh->UniformRefinement();
   ParMesh *pmesh=new ParMesh(MPI_COMM_WORLD,*serial_mesh);
   delete serial_mesh; pml->SetAttributes(pmesh);
   auto *fec=new NURBS_HCurlFECollection(order,dim);
   auto *nurbsExt=new NURBSExtension(pmesh->NURBSext,order);
   auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int ncurl=fespace->GetTrueVSize();
   auto *aux_fec=new NURBS_HCurlFECollection(aux_order,dim);
   auto *aux_ext=new NURBSExtension(pmesh->NURBSext,aux_order);
   auto *aux_fes=new ParFiniteElementSpace(pmesh,aux_ext,aux_fec);
   int no1=aux_fes->GetTrueVSize();
   if(myid==0){out<<"\n══════════════════════════════════════════════════════════\n"; out<<" Case: PML  r="<<ref<<"  o="<<order<<"  f="<<(int)freq<<"  aux_o="<<aux_order<<"  N="<<ncurl<<"  dim(Pi)="<<no1<<"\n"; out<<"══════════════════════════════════════════════════════════\n";}
   Array<int> ess_tdofs,ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);
   ComplexOperator::Convention conv=ComplexOperator::HERMITIAN;
   VectorFunctionCoefficient f_src(dim,point_source);
   ParComplexLinearForm b_form(fespace,conv);
   b_form.AddDomainIntegrator(NULL,new VectorFEDomainLFIntegrator(f_src));
   b_form.Vector::operator=(0.0); b_form.Assemble();
   ParComplexGridFunction x_gf(fespace); x_gf=0.0;
   Array<int> attr,attrPML;
   if(pmesh->attributes.Size()){attr.SetSize(pmesh->attributes.Max());attrPML.SetSize(pmesh->attributes.Max());attr=0;attr[0]=1;attrPML=0;if(pmesh->attributes.Max()>1)attrPML[1]=1;}
   ConstantCoefficient muinv_c(1.0/mu),omeg_c(-pow2(omega_val)*epsilon);
   RestrictedCoefficient restr_muinv(muinv_c,attr),restr_omeg(omeg_c,attr);
   ParSesquilinearForm a(fespace,conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);
   const int cdim=(dim==2)?1:dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re,pml),pml_c1_Im(cdim,detJ_inv_JT_J_Im,pml);
   ScalarVectorProductCoefficient c1_Re(muinv_c,pml_c1_Re),c1_Im(muinv_c,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML),restr_c1_Im(c1_Im,attrPML);
   PMLDiagMatrixCoefficient pml_c2_Re(dim,detJ_JT_J_inv_Re,pml),pml_c2_Im(dim,detJ_JT_J_inv_Im,pml);
   ScalarVectorProductCoefficient c2_Re(omeg_c,pml_c2_Re),c2_Im(omeg_c,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML),restr_c2_Im(c2_Im,attrPML);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),new VectorFEMassIntegrator(restr_c2_Im));
   auto t_as0=steady_clock::now(); a.Assemble(0);
   OperatorPtr A_op_all; Vector B_all,X_all;
   a.FormLinearSystem(ess_tdofs,x_gf,b_form,A_op_all,X_all,B_all);
   double t_assemble=duration<double>(steady_clock::now()-t_as0).count();
   HypreParMatrix *A_r_ptr=nullptr,*A_i_ptr=nullptr;
   auto *A_blk=dynamic_cast<BlockOperator*>(A_op_all.Ptr());
   if(A_blk){A_r_ptr=dynamic_cast<HypreParMatrix*>(&A_blk->GetBlock(0,0));A_i_ptr=dynamic_cast<HypreParMatrix*>(&A_blk->GetBlock(1,0));}
   else{auto *A_cpx=dynamic_cast<ComplexHypreParMatrix*>(A_op_all.Ptr());if(A_cpx){A_r_ptr=&A_cpx->real();A_i_ptr=&A_cpx->imag();}}
   MFEM_VERIFY(A_r_ptr&&A_i_ptr,"Cannot extract real/imag blocks");
   ConstantCoefficient one(1.0);
   ParBilinearForm mform(fespace); mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0);mform.Finalize(0); OperatorPtr M_op; mform.FormSystemMatrix(ess_tdofs,M_op);
   auto *M_curl=M_op.As<HypreParMatrix>();
   ParMixedBilinearForm mmix(aux_fes,fespace); mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);mmix.Finalize(0);
   Array<int> empty_tdofs; OperatorHandle Mmix_h;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_h);
   auto *M_cross=Mmix_h.As<HypreParMatrix>();
   Vector D_inv_M(ncurl); M_curl->GetDiag(D_inv_M);
   for(int i=0;i<ncurl;i++)D_inv_M[i]=1.0/max(1e-14,abs(D_inv_M[i]));
   Vector D_r(ncurl),D_i(ncurl); A_r_ptr->GetDiag(D_r); A_i_ptr->GetDiag(D_i);
   auto t_ac0=steady_clock::now(); DenseMatrix Ac_r,Ac_i;
   FormComplexCoarseOperator(*A_r_ptr,*A_i_ptr,*M_cross,D_inv_M,ncurl,no1,Ac_r,Ac_i);
   double t_ac=duration<double>(steady_clock::now()-t_ac0).count();
   if(myid==0)out<<"[PML] assemble="<<fixed<<setprecision(2)<<t_assemble<<"s  form_Ac="<<t_ac<<"s  nnz(M_cross)="<<M_cross->NNZ()<<"  coarse="<<(2*no1)<<"x"<<(2*no1)<<endl;
   vector<PMLCandidateResult> candidates;
   double omega_list[]={0.5,0.7,1.0};
   int N2=A_op_all->Height();
   // Candidate 1: Complex Jacobi
   for(double w:omega_list){ostringstream nm;nm<<"Jacobi(cpx) ω="<<fixed<<setprecision(1)<<w; ComplexJacobi jac(D_r,D_i,w); jac.SetOperator(*A_op_all); auto cr=probe_one_step_pml(*A_op_all,B_all,jac,nm.str(),1,w); auto cr_w=probe_warmup_pml(*A_op_all,B_all,jac,nm.str(),1,w,0.0); cr.rel_10_warmup=cr_w.rel_10_warmup; cr.warmup_score=cr_w.warmup_score; cr.warmup_setup_ms=cr_w.warmup_setup_ms; cr.warmup_solve_ms=cr_w.warmup_solve_ms; candidates.push_back(cr);}
   double setup_pi_ms=t_ac*1000.0;
   // Candidates 2-4
   for(double w:omega_list){
      {ComplexPiPreconditioner prec(*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,w);prec.SetOperator(*A_op_all);ostringstream nm;nm<<"Pi_lumped+Jac ω="<<fixed<<setprecision(1)<<w;auto cr=probe_one_step_pml(*A_op_all,B_all,prec,nm.str(),2,w);cr.setup_probe_ms=setup_pi_ms;auto cr_w=probe_warmup_pml(*A_op_all,B_all,prec,nm.str(),2,w,setup_pi_ms);cr.rel_10_warmup=cr_w.rel_10_warmup;cr.warmup_score=cr_w.warmup_score;cr.warmup_setup_ms=cr_w.warmup_setup_ms;cr.warmup_solve_ms=cr_w.warmup_solve_ms;candidates.push_back(cr);}
      {ComplexJacobiToPiPreconditioner prec(*A_op_all,*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,w);prec.SetOperator(*A_op_all);ostringstream nm;nm<<"Jac→Pi ω="<<fixed<<setprecision(1)<<w;auto cr=probe_one_step_pml(*A_op_all,B_all,prec,nm.str(),3,w);cr.setup_probe_ms=setup_pi_ms;auto cr_w=probe_warmup_pml(*A_op_all,B_all,prec,nm.str(),3,w,setup_pi_ms);cr.rel_10_warmup=cr_w.rel_10_warmup;cr.warmup_score=cr_w.warmup_score;cr.warmup_setup_ms=cr_w.warmup_setup_ms;cr.warmup_solve_ms=cr_w.warmup_solve_ms;candidates.push_back(cr);}
      {ComplexPiToJacobiPreconditioner prec(*A_op_all,*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,w);prec.SetOperator(*A_op_all);ostringstream nm;nm<<"Pi→Jac ω="<<fixed<<setprecision(1)<<w;auto cr=probe_one_step_pml(*A_op_all,B_all,prec,nm.str(),4,w);cr.setup_probe_ms=setup_pi_ms;auto cr_w=probe_warmup_pml(*A_op_all,B_all,prec,nm.str(),4,w,setup_pi_ms);cr.rel_10_warmup=cr_w.rel_10_warmup;cr.warmup_score=cr_w.warmup_score;cr.warmup_setup_ms=cr_w.warmup_setup_ms;cr.warmup_solve_ms=cr_w.warmup_solve_ms;candidates.push_back(cr);}
   }
   int best_one_step=0,best_warmup=0; double min_rho=1e30,max_score=-1e30;
   for(int i=0;i<(int)candidates.size();i++){if(candidates[i].one_step_rho<min_rho){min_rho=candidates[i].one_step_rho;best_one_step=i;} if(candidates[i].warmup_score>max_score){max_score=candidates[i].warmup_score;best_warmup=i;}}
   candidates[best_one_step].selected_one_step=true; candidates[best_warmup].selected_warmup=true;
   if(myid==0){
      out<<"\n┌─ Probe Results "<<string(90,'─')<<"┐\n"; out<<"│ "<<left<<setw(32)<<"Candidate"<<" "<<right<<setw(6)<<"ω"<<" "<<setw(10)<<"1-step ρ"<<" "<<setw(12)<<"10-step |r|"<<" "<<setw(10)<<"score"<<" "<<setw(12)<<"setup(ms)"<<" "<<setw(10)<<"probe(ms)"<<" "<<setw(10)<<"Sel-1s"<<" "<<setw(10)<<"Sel-10s"<<" │\n"; out<<"├─"<<string(32,'─')<<"─"<<string(7,'─')<<"─"<<string(11,'─')<<"─"<<string(13,'─')<<"─"<<string(11,'─')<<"─"<<string(13,'─')<<"─"<<string(11,'─')<<"─"<<string(11,'─')<<"─"<<string(11,'─')<<"┤\n";
      for(auto&cr:candidates){out<<"│ "<<left<<setw(32)<<cr.name<<" "<<fixed<<setprecision(1)<<right<<setw(6)<<cr.omega<<" "<<scientific<<setprecision(3)<<setw(10)<<cr.one_step_rho<<" "<<scientific<<setprecision(3)<<setw(12)<<cr.rel_10_warmup<<" "<<scientific<<setprecision(3)<<setw(10)<<cr.warmup_score<<" "<<fixed<<setprecision(1)<<setw(12)<<cr.setup_probe_ms<<" "<<fixed<<setprecision(1)<<setw(10)<<cr.apply_probe_ms<<" "<<setw(10)<<(cr.selected_one_step?" ★YES":"     ")<<setw(10)<<(cr.selected_warmup?" ★YES":"     ")<<" │\n";}
      out<<"└─"<<string(32,'─')<<"─"<<string(7,'─')<<"─"<<string(11,'─')<<"─"<<string(13,'─')<<"─"<<string(11,'─')<<"─"<<string(13,'─')<<"─"<<string(11,'─')<<"─"<<string(11,'─')<<"─"<<string(11,'─')<<"┘\n";
      out<<"\n┌─ Full Solve Comparison "<<string(100,'─')<<"┐\n"; out<<"│ "<<left<<setw(32)<<"Method"<<" "<<right<<setw(5)<<"Iters"<<" "<<setw(9)<<"rel(10)"<<" "<<setw(9)<<"rel(50)"<<" "<<setw(9)<<"rel(100)"<<" "<<setw(9)<<"rel(end)"<<" "<<setw(6)<<"BrkNan"<<" "<<setw(8)<<"Setup"<<" "<<setw(8)<<"Total"<<" "<<setw(12)<<"Sel-1s?"<<" "<<setw(12)<<"Sel-10s?"<<" │\n"; out<<"├─"<<string(32,'─')<<"─"<<string(6,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(7,'─')<<"─"<<string(9,'─')<<"─"<<string(9,'─')<<"─"<<string(13,'─')<<"─"<<string(13,'─')<<"┤\n";
      {auto r=run_gmres(*A_op_all,B_all,nullptr,200,1000,1e-8);auto fmt=[&](double v){if(v<1.0)out<<scientific<<setprecision(2)<<setw(9)<<v;else out<<"    —     ";};out<<"│ "<<left<<setw(32)<<"none"<<" "<<right<<setw(5)<<r.iters<<" ";fmt(r.rel_10);out<<" ";fmt(r.rel_50);out<<" ";fmt(r.rel_100);out<<" ";out<<scientific<<setprecision(2)<<setw(9)<<r.rel_final<<" "<<setw(6)<<(r.nan_detected?"NaN":r.breakdown?"BRK":" ")<<" "<<fixed<<setprecision(0)<<setw(7)<<0<<"ms"<<" "<<setw(7)<<r.total_ms<<"ms"<<" "<<setw(12)<<" "<<" "<<setw(12)<<" │\n";}
      for(auto&cr:candidates){unique_ptr<Solver> pp;if(cr.candidate_id==1)pp=make_unique<ComplexJacobi>(D_r,D_i,cr.omega);else if(cr.candidate_id==2)pp=make_unique<ComplexPiPreconditioner>(*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,cr.omega);else if(cr.candidate_id==3)pp=make_unique<ComplexJacobiToPiPreconditioner>(*A_op_all,*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,cr.omega);else if(cr.candidate_id==4)pp=make_unique<ComplexPiToJacobiPreconditioner>(*A_op_all,*M_cross,D_inv_M,Ac_r,Ac_i,D_r,D_i,cr.omega);pp->SetOperator(*A_op_all);cr.full=run_gmres(*A_op_all,B_all,pp.get(),200,1000,1e-8);cr.full.dim_pi=(cr.candidate_id>=2)?no1:0;cr.full.nnz_pi=(cr.candidate_id>=2)?M_cross->NNZ():0;auto fmt=[&](double v){if(v<1.0)out<<scientific<<setprecision(2)<<setw(9)<<v;else out<<"    —     ";};out<<"│ "<<left<<setw(32)<<cr.name<<" "<<right<<setw(5)<<cr.full.iters<<" ";fmt(cr.full.rel_10);out<<" ";fmt(cr.full.rel_50);out<<" ";fmt(cr.full.rel_100);out<<" ";out<<scientific<<setprecision(2)<<setw(9)<<cr.full.rel_final<<" "<<setw(6)<<(cr.full.nan_detected?"NaN":cr.full.breakdown?"BRK":" ")<<" "<<fixed<<setprecision(0)<<setw(7)<<cr.setup_probe_ms<<"ms"<<" "<<setw(7)<<cr.full.total_ms<<"ms"<<" "<<setw(12)<<(cr.selected_one_step?"  ★1-STEP":" ")<<" "<<setw(12)<<(cr.selected_warmup?"  ★WARMUP":" ")<<" │\n";}
      out<<"└─"<<string(32,'─')<<"─"<<string(6,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(10,'─')<<"─"<<string(7,'─')<<"─"<<string(9,'─')<<"─"<<string(9,'─')<<"─"<<string(13,'─')<<"─"<<string(13,'─')<<"┘\n";
      out.flush();
      out<<"\n★★★ SELECTOR DECISION ★★★\n";
      out<<"  One-step probe selects: "<<candidates[best_one_step].name<<" (ρ="<<scientific<<setprecision(3)<<candidates[best_one_step].one_step_rho<<")\n";
      out<<"  10-step warmup selects: "<<candidates[best_warmup].name<<" (score="<<scientific<<setprecision(3)<<candidates[best_warmup].warmup_score<<")\n";
      out<<"  Best one-step iters="<<candidates[best_one_step].full.iters<<"  Best warmup iters="<<candidates[best_warmup].full.iters<<"\n\n";
   }
   delete pml;delete pmesh;
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char *argv[])
{
   MPI_Init(&argc,&argv);
   int myid; MPI_Comm_rank(MPI_COMM_WORLD,&myid);
   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   const char *system_type="spd";
   int ref=2, order=2, aux_order=1; double freq=5.0;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh file.");
   args.AddOption(&system_type,"-system","--system","System type: spd or pml.");
   args.AddOption(&ref,"-r","--refine","Refinement levels.");
   args.AddOption(&order,"-o","--order","Spline order (fine).");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.AddOption(&aux_order,"-ao","--aux-order","Auxiliary NURBS order (default 1).");
   args.Parse();
   if(!args.Good()){if(myid==0)args.PrintUsage(cout);MPI_Finalize();return 1;}
   if(myid==0){cout<<"\n╔══════════════════════════════════════════════════╗\n";cout<<"║  Diagnostic-Driven Preconditioner Selector      ║\n";cout<<"║  H(curl) IGA Maxwell — 4 candidates, 2 probes  ║\n";cout<<"╚══════════════════════════════════════════════════╝\n";}
   string sys(system_type);for(auto&c:sys)c=tolower(c);
   if(sys=="spd") run_spd_case(myid,mesh_file,ref,order,aux_order,freq,cout);
   else if(sys=="pml") run_pml_case(myid,mesh_file,ref,order,aux_order,freq,cout);
   else{if(myid==0)cerr<<"Unknown system type: "<<system_type<<endl;MPI_Finalize();return 1;}
   MPI_Finalize(); return 0;
}

// ===========================================================================
// PML implementation
// ===========================================================================
PML::PML(Mesh *mesh_, Array2D<real_t> length_) : mesh(mesh_), length(length_)
{ dim=mesh->Dimension(); SetBoundaries(); }

void PML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim,2); dom_bdr.SetSize(dim,2);
   Vector pmin,pmax; mesh->GetBoundingBox(pmin,pmax);
   for(int i=0;i<dim;i++){dom_bdr(i,0)=pmin(i);dom_bdr(i,1)=pmax(i);comp_dom_bdr(i,0)=dom_bdr(i,0)+length(i,0);comp_dom_bdr(i,1)=dom_bdr(i,1)-length(i,1);}
}

void PML::SetAttributes(ParMesh *mesh_)
{
   for(int i=0;i<mesh_->GetNBE();++i)mesh_->GetBdrElement(i)->SetAttribute(i+1);
   int nrelem=mesh_->GetNE(); elems.SetSize(nrelem);
   for(int i=0;i<nrelem;++i){elems[i]=1;bool in_pml=false;Element*el=mesh_->GetElement(i);Array<int>vertices;el->SetAttribute(1);el->GetVertices(vertices);int nrvert=vertices.Size();for(int iv=0;iv<nrvert;++iv){int vert_idx=vertices[iv];real_t*coords=mesh_->GetVertex(vert_idx);for(int comp=0;comp<dim;++comp){if(coords[comp]>comp_dom_bdr(comp,1)||coords[comp]<comp_dom_bdr(comp,0)){in_pml=true;break;}}}if(in_pml){elems[i]=0;el->SetAttribute(2);}}
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x, vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi=complex<real_t>(0.,1.);
   real_t n_pml=2.0,c_pml=5.0,k=g_omega_pml*sqrt(g_epsilon_pml*g_mu_pml);
   for(int i=0;i<dim;++i){dxs[i]=1.0;if(x(i)>=comp_dom_bdr(i,1)){real_t coeff=n_pml*c_pml/k/pow(length(i,1),n_pml);dxs[i]=1_r+zi*coeff*abs(pow(x(i)-comp_dom_bdr(i,1),n_pml-1_r));}if(x(i)<=comp_dom_bdr(i,0)){real_t coeff=n_pml*c_pml/k/pow(length(i,0),n_pml);dxs[i]=1_r+zi*coeff*abs(pow(x(i)-comp_dom_bdr(i,0),n_pml-1_r));}}
}

void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det(1.0,0.0);pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];for(int i=0;i<D.Size();++i)D(i)=(det/pow2(dxs[i])).real(); }

void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det=1.0;pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];for(int i=0;i<D.Size();++i)D(i)=(det/pow2(dxs[i])).imag(); }

void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det=1.0;pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];for(int i=0;i<D.Size();++i)D(i)=abs(det/pow2(dxs[i])); }

void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det(1.0,0.0);pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];int sz=D.Size();if(sz==2)D=(real_t(1.0)/det).real();else for(int i=0;i<sz;++i)D(i)=(pow2(dxs[i])/det).real(); }

void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det=1.0;pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];int sz=D.Size();if(sz==2)D=(real_t(1.0)/det).imag();else for(int i=0;i<sz;++i)D(i)=(pow2(dxs[i])/det).imag(); }

void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D)
{ vector<complex<real_t>>dxs(D.Size());complex<real_t>det=1.0;pml->StretchFunction(x,dxs);for(int i=0;i<D.Size();++i)det*=dxs[i];int sz=D.Size();if(sz==2)D=abs(real_t(1.0)/det);else for(int i=0;i<sz;++i)D(i)=abs(pow2(dxs[i])/det); }

void point_source(const Vector &x, Vector &f)
{ int dim=x.Size();Vector center(dim);real_t r2=0.0;for(int i=0;i<dim;++i){center(i)=0.5;r2+=pow(x[i]-center[i],2.);}real_t n=5.0*g_omega_pml*sqrt(g_epsilon_pml*g_mu_pml)/M_PI;real_t coeff=pow(n,2)/M_PI;real_t alpha=-pow(n,2)*r2;f=0.0;f[0]=1000.0*coeff*exp(alpha); }
