// ============================================================================
// ams_sweep_v3.cpp — Fully sparse Pi_lumped auxiliary preconditioner
// for IGA H(curl) SPD Maxwell problems.
//
// Preconditioner: B^{-1} r = Pi * Ac^{-1} * Pi^T r + omega * D^{-1} r
//   Pi        = diag(M_curl)^{-1} * M_cross      (implicit, sparse matvec)
//   Pi^T      = M_cross^T * diag(M_curl)^{-1}    (implicit, sparse matvec)
//   Ac        = Pi^T * A * Pi                     (no1×no1, DenseMatrix)
//   D^{-1}    = 1/diag(M_curl)                    (Jacobi smoother diagonal)
//   Coarse    = DenseMatrixInverse(Ac)            (direct, O(no1^3), small)
//   Outer CG  = hand-rolled PCG with sparse A matvecs
//
// Key differences from v1:
//   - NO dense Pi matrix: Pi*vec and Pi^T*vec use HypreParMatrix matvecs
//   - Ac is small (no1 x no1), NOT ncurl x ncurl
//   - Efficient diag(M_curl) via HypreParMatrix::GetDiag(), not O(N²) probing
//   - All fine-level operations are sparse O(nnz) matvecs
//
// Verify: r=2,o=2,f=5 → none≈80, Jacobi≈71, Pi_lumped+Jacobi (ω=1.0)≈42
// ============================================================================

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <chrono>
#include <cfloat>
using namespace std;
using namespace std::chrono;
using namespace mfem;

// ─── Gaussian RHS ──────────────────────────────────────────────────
void gaussian_rhs(const Vector &x, Vector &f)
{
   f = 0.0;
   double r2 = 0.0;
   for (int d = 0; d < 3; d++) { r2 += pow(x[d] - 0.5, 2); }
   f[0] = exp(-100.0 * r2);
}

// ─── CG result struct ──────────────────────────────────────────────
struct CGResult
{
   int    iters       = 0;
   double rel_final   = 1.0;
   double rel_10      = 1.0;
   double rel_20      = 1.0;
   bool   converged   = false;
   bool   breakdown   = false;
   double setup_ms    = 0.0;
   double apply_ms    = 0.0;
   int    nnz_pi      = 0;     // nnz of M_cross (proxy for Pi)
   int    nnz_ac      = 0;     // nonzeros in Ac (approx no1^2 for dense)
   int    dim_pi      = 0;     // no1 (coarse space dimension)
};

// ─── Sparse Pi Preconditioner ──────────────────────────────────────
//
// Applies B^{-1} r = Pi * Ac^{-1} * Pi^T r + omega * D^{-1} r
//
// where Pi = D^{-1} * M_cross (implicit, never materialized)
//
// All matvecs use sparse HypreParMatrix operations.
class SparsePiPreconditioner : public Solver
{
public:
   SparsePiPreconditioner(const HypreParMatrix &M_cross,
                          const Vector         &D_inv_M,   // diag(M_curl)^{-1} for Pi
                          const Vector         &D_inv_A,   // diag(A)^{-1} for Jacobi
                          const DenseMatrix    &Ac,
                          double                omega)
      : M_cross_(M_cross)
      , D_inv_M_(D_inv_M)
      , D_inv_A_(D_inv_A)
      , omega_(omega)
      , ncurl_(M_cross.Height())
      , no1_(M_cross.Width())
   {
      // Build coarse solver: DenseMatrixInverse on regularized Ac
      Ac_reg_ = Ac;
      double mxQ = 0.0;
      for (int i = 0; i < no1_; i++)
      {
         mxQ = max(mxQ, abs(Ac_reg_(i, i)));
      }
      double reg = max(1e-12, mxQ * 1e-10);
      for (int i = 0; i < no1_; i++) { Ac_reg_(i, i) += reg; }
      Ac_inv_ = make_unique<DenseMatrixInverse>(Ac_reg_);

      // Allocate work vectors
      r_scaled_.SetSize(ncurl_);
      r_coarse_.SetSize(no1_);
      z_coarse_.SetSize(no1_);
      tmp_.SetSize(ncurl_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // Step 1: r_coarse = Pi^T * r = M_cross^T * (D_M^{-1} .* r)
      for (int i = 0; i < ncurl_; i++) { r_scaled_[i] = r[i] * D_inv_M_[i]; }
      M_cross_.MultTranspose(r_scaled_, r_coarse_);

      // Step 2: z_coarse = Ac^{-1} * r_coarse  (dense LU, O(no1^2))
      z_coarse_ = 0.0;
      Ac_inv_->Mult(r_coarse_, z_coarse_);

      // Step 3: z = Pi * z_coarse = D_M^{-1} .* (M_cross * z_coarse)
      M_cross_.Mult(z_coarse_, tmp_);
      for (int i = 0; i < ncurl_; i++) { z[i] = tmp_[i] * D_inv_M_[i]; }

      // Step 4: z += omega * D_A^{-1} .* r  (Jacobi smoother on A)
      for (int i = 0; i < ncurl_; i++) { z[i] += omega_ * D_inv_A_[i] * r[i]; }
   }

   void SetOperator(const Operator &op) override
   {
      // No-op: the preconditioner is self-contained.
      // op is the fine operator A, but we don't use it in Mult().
      height = op.Height();
      width  = op.Width();
   }

   int  CoarseDim() const { return no1_; }
   int  NnzCross()  const { return M_cross_.NNZ(); }

private:
   const HypreParMatrix &M_cross_;
   const Vector         &D_inv_M_;   // diag(M_curl)^{-1} for Pi
   const Vector         &D_inv_A_;   // diag(A)^{-1} for Jacobi
   double                omega_;
   int                   ncurl_;
   int                   no1_;

   DenseMatrix                               Ac_reg_;
   unique_ptr<DenseMatrixInverse>            Ac_inv_;

   mutable Vector r_scaled_, r_coarse_, z_coarse_, tmp_;
};

// ─── Form Ac = Pi^T * A * Pi (dense Pi method, v1-compatible) ─────
// Builds Pi as a DenseMatrix (Pi_lumped), then forms QTAQ.
// This is the reference method that v1 uses.
DenseMatrix FormCoarseOperatorDense(const Operator &A_op,
                                    const Operator &M_cross_op,
                                    const Vector   &D_inv,
                                    int             ncurl,
                                    int             no1)
{
   // Build Pi_lumped as DenseMatrix (v1 method)
   DenseMatrix Pi(ncurl, no1);
   Vector e(no1), Mc(ncurl);
   for (int j = 0; j < no1; j++)
   {
      e = 0.0; e[j] = 1.0; Mc = 0.0;
      M_cross_op.Mult(e, Mc);
      for (int i = 0; i < ncurl; i++) { Pi(i, j) = Mc[i] * D_inv[i]; }
   }

   // Build QTAQ = Pi^T * A * Pi (v1 method)
   DenseMatrix QTAQ(no1, no1);
   QTAQ = 0.0;
   Vector pj(ncurl), Apj(ncurl);
   for (int j = 0; j < no1; j++)
   {
      for (int i = 0; i < ncurl; i++) { pj[i] = Pi(i, j); }
      A_op.Mult(pj, Apj);
      for (int k = 0; k < no1; k++)
      {
         double s = 0.0;
         for (int i = 0; i < ncurl; i++) { s += Pi(i, k) * Apj[i]; }
         QTAQ(k, j) = s;
      }
   }
   return QTAQ;
}

// ─── Form Ac = Pi^T * A * Pi (sparse column-by-column probing) ────
//
// Ac(j,k) = (Pi(:,j))^T * A * Pi(:,k)
//
// Column j of Pi:  pj = D^{-1} .* (column j of M_cross)
//
// Ac is formed column-by-column using sparse probing:
//   1. Extract column j of M_cross → scale by D^{-1} → pj = Pi(:,j)
//   2. Apj = A * pj  (sparse)
//   3. Apj_s = D^{-1} .* Apj
//   4. Ac(:,j) = M_cross^T * Apj_s  (sparse)
//
// Complexity: no1 sparse matvecs with A + 2*no1 matvecs with M_cross.
// For no1 ≈ 300-2000 and sparse A, this is fast (seconds).
DenseMatrix FormCoarseOperator(const HypreParMatrix &A,
                               const HypreParMatrix &M_cross,
                               const Vector         &D_inv,
                               int                   ncurl,
                               int                   no1)
{
   DenseMatrix Ac(no1, no1);
   Ac = 0.0;

   Vector eJ(no1);  // unit vector in coarse space
   Vector pj(ncurl), Apj(ncurl), Apj_s(ncurl), Ac_col(no1);

   for (int j = 0; j < no1; j++)
   {
      // 1. pj = Pi(:,j) = D^{-1} .* (M_cross * e_j)
      eJ = 0.0;
      eJ[j] = 1.0;
      pj = 0.0;
      M_cross.Mult(eJ, pj);
      for (int i = 0; i < ncurl; i++) { pj[i] *= D_inv[i]; }

      // 2. Apj = A * pj
      Apj = 0.0;
      A.Mult(pj, Apj);

      // 3. Apj_s = D^{-1} .* Apj
      for (int i = 0; i < ncurl; i++) { Apj_s[i] = Apj[i] * D_inv[i]; }

      // 4. Ac_col = M_cross^T * Apj_s → Ac(:,j)
      Ac_col = 0.0;
      M_cross.MultTranspose(Apj_s, Ac_col);

      for (int k = 0; k < no1; k++) { Ac(k, j) = Ac_col[k]; }
   }

   // Symmetrize to reduce numerical drift (Ac = (Ac + Ac^T)/2)
   for (int i = 0; i < no1; i++)
   {
      for (int j = i + 1; j < no1; j++)
      {
         double avg = 0.5 * (Ac(i, j) + Ac(j, i));
         Ac(i, j) = Ac(j, i) = avg;
      }
   }

   return Ac;
}

// ─── CG: no preconditioner ─────────────────────────────────────────
CGResult run_cg_none(const Operator &A_op, const Vector &B,
                     double tol = 1e-8, int max_iter = 500)
{
   CGResult res;
   int N = A_op.Height();
   res.setup_ms = 0.0;  // no setup

   Vector X(N);
   X = 0.0;
   Vector R(B.Size());
   R = B;
   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector Z(N), P(N), AP(N);
   Z = R;  // no preconditioner
   P = Z;
   double rz = R * Z;
   if (rz <= 0) { res.breakdown = true; return res; }

   double rel_res = 1.0;
   int    iter;
   auto   ta0 = steady_clock::now();
   for (iter = 1; iter <= max_iter; iter++)
   {
      A_op.Mult(P, AP);
      double pap = P * AP;
      if (pap <= 0 || !isfinite(pap)) { res.breakdown = true; break; }
      double alpha = rz / pap;
      X.Add(alpha, P);
      R.Add(-alpha, AP);
      rel_res = sqrt(R * R) / bnorm;
      if (iter == 10) { res.rel_10 = rel_res; }
      if (iter == 20) { res.rel_20 = rel_res; }
      if (rel_res <= tol) { break; }
      Z = R;
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

// ─── CG: Jacobi preconditioner ─────────────────────────────────────
CGResult run_cg_jacobi(const Operator &A_op, const Vector &B,
                       const Vector &D_inv, double omega,
                       double tol = 1e-8, int max_iter = 500)
{
   CGResult res;
   int N = A_op.Height();
   res.setup_ms = 0.0;  // D_inv already computed

   Vector X(N);
   X = 0.0;
   Vector R(B.Size());
   R = B;
   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector Z(N), P(N), AP(N);
   for (int i = 0; i < N; i++) { Z[i] = omega * D_inv[i] * R[i]; }
   P = Z;
   double rz = R * Z;
   if (rz <= 0) { res.breakdown = true; return res; }

   double rel_res = 1.0;
   int    iter;
   auto   ta0 = steady_clock::now();
   for (iter = 1; iter <= max_iter; iter++)
   {
      A_op.Mult(P, AP);
      double pap = P * AP;
      if (pap <= 0 || !isfinite(pap)) { res.breakdown = true; break; }
      double alpha = rz / pap;
      X.Add(alpha, P);
      R.Add(-alpha, AP);
      rel_res = sqrt(R * R) / bnorm;
      if (iter == 10) { res.rel_10 = rel_res; }
      if (iter == 20) { res.rel_20 = rel_res; }
      if (rel_res <= tol) { break; }
      for (int i = 0; i < N; i++) { Z[i] = omega * D_inv[i] * R[i]; }
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

// ─── CG: SparsePi + Jacobi preconditioner ──────────────────────────
CGResult run_cg_sparse_pi(const Operator            &A_op,
                          const Vector              &B,
                          SparsePiPreconditioner    &prec,
                          double                     tol = 1e-8,
                          int                        max_iter = 500)
{
   CGResult res;
   int N = A_op.Height();

   // Setup already done (prec is pre-built)
   res.setup_ms = 0.0;  // measured separately
   res.dim_pi = prec.CoarseDim();
   res.nnz_pi = prec.NnzCross();

   Vector X(N);
   X = 0.0;
   Vector R(B.Size());
   R = B;
   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   Vector Z(N), P(N), AP(N);
   prec.Mult(R, Z);
   for (int i = 0; i < N; i++)
   {
      if (!isfinite(Z[i])) { res.breakdown = true; return res; }
   }
   P = Z;
   double rz = R * Z;
   if (rz <= 0) { res.breakdown = true; return res; }

   double rel_res = 1.0;
   int    iter;
   auto   ta0 = steady_clock::now();
   for (iter = 1; iter <= max_iter; iter++)
   {
      A_op.Mult(P, AP);
      double pap = P * AP;
      if (pap <= 0 || !isfinite(pap)) { res.breakdown = true; break; }
      double alpha = rz / pap;
      X.Add(alpha, P);
      R.Add(-alpha, AP);
      rel_res = sqrt(R * R) / bnorm;
      if (iter == 10) { res.rel_10 = rel_res; }
      if (iter == 20) { res.rel_20 = rel_res; }
      if (rel_res <= tol) { break; }
      prec.Mult(R, Z);
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

// ─── Report helper ─────────────────────────────────────────────────
void report(int myid, const string &label, const CGResult &r,
            int r_val, int o_val, double f_val, int N, ostream &out)
{
   if (myid != 0) { return; }

   // First line: header
   static bool header_printed = false;
   if (!header_printed)
   {
      out << "\n┌───────────────────────────────────────────────────────────────"
          << "─────────────────────────────────────────────┐\n";
      out << "│  Sweep v3 (sparse): r=" << setw(1) << r_val
          << " o=" << setw(1) << o_val
          << " f=" << setw(4) << (int)f_val
          << "  N=" << setw(5) << N
          << "                                                         │\n";
      out << "├────────────────────────────────────────┬──────┬──────┬──────"
          << "┬──────┬──────┬──────┬──────┬──────┬──────┤\n";
      out << "│ Preconditioner                         │ Iters│ r(10)│ r(20)│"
          << " r(end)│Break │Setup │Apply │nnz(Pi)│dim(Pi)│\n";
      out << "├────────────────────────────────────────┼──────┼──────┼──────"
          << "┼──────┼──────┼──────┼──────┼──────┼──────┤\n";
      header_printed = true;
   }

   out << "│ " << left << setw(38) << label << " │ "
       << right << setw(4) << r.iters << " │ ";

   auto fmt_rel = [&](double v)
   {
      if (v < 1.0) { out << scientific << setprecision(2) << setw(8) << v; }
      else { out << "    —    "; }
   };

   fmt_rel(r.rel_10);
   out << " │ ";
   fmt_rel(r.rel_20);
   out << " │ " << scientific << setprecision(2) << setw(8) << r.rel_final;
   out << " │ " << (r.breakdown ? "BRK" : "   ") << " │ ";
   out << fixed << setprecision(0) << setw(5) << r.setup_ms << "│ ";
   out << fixed << setprecision(0) << setw(5) << r.apply_ms << "│ ";
   out << setw(5) << r.nnz_pi << " │";
   out << setw(5) << r.dim_pi << " │\n";
}

// ─── Main ──────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // ── Parse options ───────────────────────────────────────────
   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int    ref          = 2;
   int    order        = 2;
   double freq         = 5.0;
   int    test_hypre   = 0;
   int    lump_only    = 0;  // always lumped in v3; kept for CLI compat
   int    aux_order    = 1;  // order of auxiliary (coarse) space

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Spline order (fine).");
   args.AddOption(&freq, "-f", "--frequency", "Frequency.");
   args.AddOption(&test_hypre, "-hypre", "--hypre", "Test Hypre AMS.");
   args.AddOption(&aux_order, "-ao", "--aux-order",
                  "Auxiliary (coarse) NURBS order (default 1).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      MPI_Finalize();
      return 1;
   }

   int dim = 3;

   // ── Mesh ────────────────────────────────────────────────────
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref; l++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->NURBSext = nullptr;
   delete mesh;

   // ── Fine H(curl) space (order = p) ──────────────────────────
   auto *fec      = new NURBS_HCurlFECollection(order, dim);
   auto *nurbsExt = new NURBSExtension(pmesh->NURBSext, order);
   auto *fespace  = new ParFiniteElementSpace(pmesh, nurbsExt, fec);
   int   ncurl    = fespace->GetTrueVSize();

   // ── Auxiliary H(curl) space (order = aux_order, default 1) ──
   auto *aux_fec  = new NURBS_HCurlFECollection(aux_order, dim);
   auto *aux_ext  = new NURBSExtension(pmesh->NURBSext, aux_order);
   auto *aux_fes  = new ParFiniteElementSpace(pmesh, aux_ext, aux_fec);
   int   no1      = aux_fes->GetTrueVSize();

   if (myid == 0)
   {
      cout << "\n[SparsePi v3] fine order=" << order
           << " (N=" << ncurl << ")"
           << "  aux order=" << aux_order
           << " (N_aux=" << no1 << ")"
           << "  freq=" << freq
           << "  ref=" << ref << endl;
   }

   // ── BCs ─────────────────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   double omega_val = 2.0 * M_PI * freq;
   double k2        = omega_val * omega_val;

   Array<int> attr_dom(pmesh->attributes.Max());
   attr_dom    = 0;
   attr_dom[0] = 1;

   ConstantCoefficient   muinv_c(1.0), k2_c(k2), one(1.0);
   RestrictedCoefficient muinv_r(muinv_c, attr_dom);
   RestrictedCoefficient k2_r(k2_c, attr_dom);

   // ── Assemble A = C^T C + k^2 M ─────────────────────────────
   auto t_assemble_start = steady_clock::now();

   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   a.Assemble(0);
   a.Finalize(0);
   OperatorPtr  A_op;
   a.FormSystemMatrix(ess_tdofs, A_op);
   auto        *Ah = A_op.As<HypreParMatrix>();

   // ── RHS ─────────────────────────────────────────────────────
   ParLinearForm b(fespace);
   VectorFunctionCoefficient src_coeff(dim, gaussian_rhs);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();
   Vector B_vec(ncurl);
   b.ParallelAssemble(B_vec);
   for (int i = 0; i < ess_tdofs.Size(); i++) { B_vec[ess_tdofs[i]] = 0.0; }

   // ── M_curl (fine mass matrix) ───────────────────────────────
   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0);
   mform.Finalize(0);
   OperatorPtr  M_op_ptr;
   mform.FormSystemMatrix(ess_tdofs, M_op_ptr);
   auto        *M_curl = M_op_ptr.As<HypreParMatrix>();

   // ── M_cross: aux × fine mixed mass ──────────────────────────
   ParMixedBilinearForm mmix(aux_fes, fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   mmix.Finalize(0);

   Array<int> empty_tdofs;
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs, empty_tdofs, Mmix_op);
   auto *M_cross = Mmix_op.As<HypreParMatrix>();

   double t_assemble = duration<double>(steady_clock::now()
                                        - t_assemble_start).count();

   // ── Extract D_M = diag(M_curl), D_A = diag(A) ───────────────
   auto t_diag_start = steady_clock::now();

   Vector D_inv_M(ncurl);  // diag(M_curl)^{-1} → for Pi construction
   M_curl->GetDiag(D_inv_M);
   for (int i = 0; i < ncurl; i++)
   {
      D_inv_M[i] = 1.0 / max(1e-14, abs(D_inv_M[i]));
   }

   Vector D_inv_A(ncurl);  // diag(A)^{-1} → for Jacobi smoother
   Ah->GetDiag(D_inv_A);
   for (int i = 0; i < ncurl; i++)
   {
      D_inv_A[i] = 1.0 / max(1e-14, abs(D_inv_A[i]));
   }

   double t_diag = duration<double>(steady_clock::now()
                                    - t_diag_start).count();

   // ── Form Ac = Pi^T * A * Pi ──────────────────────────────────
   auto t_form_ac = steady_clock::now();
   DenseMatrix Ac = FormCoarseOperator(*Ah, *M_cross, D_inv_M, ncurl, no1);
   double    t_ac  = duration<double>(steady_clock::now()
                                      - t_form_ac).count();

   // ── Build preconditioners ────────────────────────────────────
   auto t_prec_setup = steady_clock::now();

   SparsePiPreconditioner prec_pi07(*M_cross, D_inv_M, D_inv_A, Ac, 0.7);
   SparsePiPreconditioner prec_pi10(*M_cross, D_inv_M, D_inv_A, Ac, 1.0);

   double prec_setup_s = duration<double>(steady_clock::now()
                                          - t_prec_setup).count();

   // ── Hypre AMS (optional) ─────────────────────────────────────
   HypreAMS *ams = nullptr;
   if (test_hypre != 0)
   {
      ams = new HypreAMS(*Ah, fespace);
      ams->SetPrintLevel(0);
   }

   // ── Timing summary ───────────────────────────────────────────
   if (myid == 0)
   {
      cout << "[SparsePi v3] assemble=" << fixed << setprecision(2)
           << t_assemble << "s"
           << "  diag=" << setprecision(4) << t_diag << "s"
           << "  form_Ac=" << setprecision(2) << t_ac << "s"
           << "  prec_setup=" << setprecision(4) << prec_setup_s << "s"
           << "  nnz(M_cross)=" << M_cross->NNZ()
           << "  Ac=" << no1 << "x" << no1 << endl;
   }

   // ── Run benchmarks ───────────────────────────────────────────
   // none
   {
      auto r = run_cg_none(*Ah, B_vec);
      report(myid, "none", r, ref, order, freq, ncurl, cout);
   }

   // Jacobi
   for (double w : {0.7, 1.0})
   {
      auto r = run_cg_jacobi(*Ah, B_vec, D_inv_A, w);
      ostringstream os;
      os << "Jacobi ω=" << fixed << setprecision(1) << w;
      report(myid, os.str(), r, ref, order, freq, ncurl, cout);
   }

   // Pi_lumped + Jacobi
   {
      // ω = 0.7
      prec_pi07.SetOperator(*Ah);
      auto r07 = run_cg_sparse_pi(*Ah, B_vec, prec_pi07);
      r07.setup_ms = (t_diag + t_ac + prec_setup_s) * 1000.0;
      ostringstream os07;
      os07 << "Pi_lumped+jac ω=0.7";
      report(myid, os07.str(), r07, ref, order, freq, ncurl, cout);
   }
   {
      // ω = 1.0
      prec_pi10.SetOperator(*Ah);
      auto r10 = run_cg_sparse_pi(*Ah, B_vec, prec_pi10);
      r10.setup_ms = (t_diag + t_ac + prec_setup_s) * 1000.0;
      ostringstream os10;
      os10 << "Pi_lumped+jac ω=1.0";
      report(myid, os10.str(), r10, ref, order, freq, ncurl, cout);
   }

   // Hypre AMS
   if (test_hypre != 0 && ams)
   {
      Vector X(ncurl);
      X = 0.0;
      Vector R(B_vec.Size());
      R      = B_vec;
      double bnorm_ams = sqrt(B_vec * B_vec);

      CGSolver pcg(MPI_COMM_WORLD);
      pcg.SetOperator(*Ah);
      pcg.SetPreconditioner(*ams);
      pcg.SetRelTol(1e-8);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(0);
      pcg.Mult(B_vec, X);

      Vector AX(ncurl);
      Ah->Mult(X, AX);
      Vector Res(ncurl);
      for (int i = 0; i < ncurl; i++) { Res[i] = B_vec[i] - AX[i]; }
      double rel_f = sqrt(Res * Res) / bnorm_ams;

      if (myid == 0)
      {
         cout << "│ " << left << setw(38) << "hypre_ams" << " │ "
              << right << setw(4) << pcg.GetNumIterations()
              << " │     —     │     —    │ "
              << scientific << setprecision(2) << setw(8) << rel_f
              << " │ " << (pcg.GetConverged() ? "   " : "NCV")
              << " │    — │    — │    — │    — │\n";
      }
   }

   if (myid == 0)
   {
      cout << "└────────────────────────────────────────┴──────┴──────┴──────"
           << "┴──────┴──────┴──────┴──────┴──────┴──────┘\n";
      cout.flush();
   }

   // ── Cleanup ──────────────────────────────────────────────────
   if (ams) { delete ams; }
   delete aux_fes;
   delete aux_ext;
   delete aux_fec;
   delete fespace;
   delete nurbsExt;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}
