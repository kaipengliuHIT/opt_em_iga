// ============================================================================
// pml_pi_prec_test.cpp — Sparse Pi_lumped complex preconditioner for
// IGA H(curl) PML Maxwell point-source problems.
//
// References MFEM nurbs-25p for PML setup; uses sparse Pi_lumped v3.
//
// Preconditioner:
//   B^{-1} r = Pi * Ac^{-1} * Pi^T r  +  omega * D^{-1} r
//   Pi        = diag(M)^{-1} * M_cross   (real, sparse)
//   Ac        = Pi^T * A_pml * Pi        (complex, 2no1 x 2no1 dense)
//   D^{-1}    = complex Jacobi           (2x2 block inversion per node)
//
// Outer: right-preconditioned GMRES (fixed coarse → GMRES is fine)
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
using namespace std;
using namespace std::chrono;
using namespace mfem;

// ===========================================================================
// PML class (from MFEM nurbs-25p)
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

// File-scope globals used by PML stretch functions (same pattern as nurbs-25p)
static real_t g_omega = 1.0, g_mu = 1.0, g_epsilon = 1.0;

// PML stretching coefficient functions
void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D);

// Source: Gaussian point source at domain center
void point_source(const Vector &x, Vector &f);

// ===========================================================================
// Complex Pi Preconditioner
// ===========================================================================
//
// Applies B^{-1} on the 2n block vector [rr; ri]:
//   zc = Ac^{-1} * (Pi^T * r)           (coarse correction, complex)
//   z  = Pi * zc + omega * D^{-1} * r   (prolongation + smoother)
//
// Ac = Pi^T * A_pml * Pi  →  stored as two DenseMatrices (Ac_r, Ac_i)
// D^{-1} = complex block Jacobi (2x2 per-node inversion)
//
class ComplexPiPreconditioner : public Solver
{
public:
   ComplexPiPreconditioner(const HypreParMatrix &M_cross,
                           const Vector         &D_inv_M,
                           const DenseMatrix    &Ac_r,
                           const DenseMatrix    &Ac_i,
                           const Vector         &D_r,     // diag(A_pml_real)
                           const Vector         &D_i,     // diag(A_pml_imag)
                           double                omega)
      : M_cross_(M_cross), D_inv_M_(D_inv_M),
        Ac_r_(Ac_r), Ac_i_(Ac_i),
        D_r_(D_r), D_i_(D_i), omega_(omega),
        n_(M_cross.Height()), no1_(M_cross.Width())
   {
      // Build 2no1 x 2no1 coarse solver
      BuildCoarseSolver();
      r_scaled_r_.SetSize(n_);
      r_scaled_i_.SetSize(n_);
      rc_r_.SetSize(no1_);
      rc_i_.SetSize(no1_);
      zc_r_.SetSize(no1_);
      zc_i_.SetSize(no1_);
      r_coarse_full_.SetSize(2 * no1_);
      z_coarse_full_.SetSize(2 * no1_);
      tmp_r_.SetSize(n_);
      tmp_i_.SetSize(n_);
   }

   void Mult(const Vector &r, Vector &z) const override
   {
      // r is [rr; ri], z is [zr; zi], each length n_
      // Step 1: r_coarse = Pi^T * r
      for (int i = 0; i < n_; i++)
      {
         r_scaled_r_(i) = r(i)        * D_inv_M_(i);
         r_scaled_i_(i) = r(i + n_)   * D_inv_M_(i);
      }
      M_cross_.MultTranspose(r_scaled_r_, rc_r_);
      M_cross_.MultTranspose(r_scaled_i_, rc_i_);
      rc_i_.Neg();  // compatibility with block system convention

      // Step 2: z_coarse = Ac^{-1} * r_coarse
      // Build full vector for the dense solver
      for (int i = 0; i < no1_; i++)
      {
         r_coarse_full_(i)        = rc_r_(i);
         r_coarse_full_(i + no1_) = rc_i_(i);
      }
      z_coarse_full_ = 0.0;
      Ac_inv_->Mult(r_coarse_full_, z_coarse_full_);
      // Extract components
      for (int i = 0; i < no1_; i++)
      {
         zc_r_(i) = z_coarse_full_(i);
         zc_i_(i) = z_coarse_full_(i + no1_);
      }

      // Step 3: z = Pi * z_coarse
      M_cross_.Mult(zc_r_, tmp_r_);
      M_cross_.Mult(zc_i_, tmp_i_);
      for (int i = 0; i < n_; i++)
      {
         z(i)       = tmp_r_(i) * D_inv_M_(i);
         z(i + n_)  = tmp_i_(i) * D_inv_M_(i);
      }

      // Step 4: z += omega * D^{-1} * r  (complex Jacobi)
      //  D^{-1} = [d_r, d_i; -d_i, d_r] ./ (d_r² + d_i²)
      for (int i = 0; i < n_; i++)
      {
         double inv_den = 1.0 / (D_r_[i] * D_r_[i] + D_i_[i] * D_i_[i] + 1e-30);
         double pr = D_r_[i] * inv_den;
         double pi = D_i_[i] * inv_den;
         // [zr; zi] += omega * [pr, pi; -pi, pr] * [rr; ri]
         z[i]       += omega_ * (pr * r[i]       + pi * r[i + n_]);
         z[i + n_]  += omega_ * (pr * r[i + n_]  - pi * r[i]);
      }
   }

   void SetOperator(const Operator &op) override
   {
      height = op.Height();
      width  = op.Width();
   }

   int CoarseDim() const { return no1_; }
   int NnzCross()  const { return M_cross_.NNZ(); }

private:
   const HypreParMatrix &M_cross_;
   const Vector         &D_inv_M_;
   const DenseMatrix    &Ac_r_, &Ac_i_;
   const Vector         &D_r_, &D_i_;
   double                omega_;
   int                   n_, no1_;

   // 2no1 x 2no1 coarse solver (LU factorization)
   DenseMatrix                        Ac_full_;
   unique_ptr<DenseMatrixInverse>     Ac_inv_;

   mutable Vector r_scaled_r_, r_scaled_i_;
   mutable Vector rc_r_, rc_i_;       // coarse rhs components
   mutable Vector zc_r_, zc_i_;       // coarse solution components
   mutable Vector r_coarse_full_;     // 2no1-dim for dense solver
   mutable Vector z_coarse_full_;
   mutable Vector tmp_r_, tmp_i_;

   void BuildCoarseSolver()
   {
      // Coarse system: [ Ac_r   -Ac_i ] [zc_r]   [rc_r]
      //                [ Ac_i    Ac_r ] [zc_i] = [rc_i]
      int Nc = 2 * no1_;
      Ac_full_.SetSize(Nc, Nc);
      Ac_full_ = 0.0;

      for (int i = 0; i < no1_; i++)
      {
         for (int j = 0; j < no1_; j++)
         {
            double ar = Ac_r_(i, j);
            double ai = Ac_i_(i, j);
            Ac_full_(i,        j)        =  ar;
            Ac_full_(i,        j + no1_) = -ai;
            Ac_full_(i + no1_, j)        =  ai;
            Ac_full_(i + no1_, j + no1_) =  ar;
         }
      }

      // Regularize diagonal
      double mx = 0.0;
      for (int i = 0; i < Nc; i++) { mx = max(mx, abs(Ac_full_(i, i))); }
      double reg = max(1e-12, mx * 1e-10);
      for (int i = 0; i < Nc; i++) { Ac_full_(i, i) += reg; }

      Ac_inv_ = make_unique<DenseMatrixInverse>(Ac_full_);
   }
};

// ===========================================================================
// Form complex coarse operator: Ac_r = Pi^T * A_r * Pi, Ac_i = Pi^T * A_i * Pi
// via sparse column-by-column probing.
// If shift_eta != 0: uses A_shift_real = A_r + eta * M_curl
// If shift_gamma != 0: uses A_shift_imag = A_i + gamma * M_curl
// ===========================================================================
void FormComplexCoarseOperator(const HypreParMatrix &A_r,
                               const HypreParMatrix &A_i,
                               const HypreParMatrix &M_cross,
                               const Vector         &D_inv,
                               int                   n,
                               int                   no1,
                               DenseMatrix          &Ac_r,
                               DenseMatrix          &Ac_i,
                               double                shift_eta   = 0.0,
                               double                shift_gamma = 0.0,
                               const HypreParMatrix *M_curl      = nullptr)
{
   Ac_r.SetSize(no1, no1);
   Ac_i.SetSize(no1, no1);
   Ac_r = 0.0;
   Ac_i = 0.0;

   bool do_shift = (shift_eta != 0.0 || shift_gamma != 0.0) && M_curl;

   Vector eJ(no1), pj(n), Apj_r(n), Apj_i(n), Apj_s_r(n), Apj_s_i(n);
   Vector Ac_col_r(no1), Ac_col_i(no1);
   Vector Mpj(n);  // extra workspace for shift

   for (int j = 0; j < no1; j++)
   {
      // pj = Pi(:,j) = D_inv .* (M_cross * e_j)
      eJ = 0.0; eJ[j] = 1.0;
      pj = 0.0;
      M_cross.Mult(eJ, pj);
      for (int i = 0; i < n; i++) { pj[i] *= D_inv[i]; }

      // Apj_r = A_r * pj,  Apj_i = A_i * pj
      Apj_r = 0.0; A_r.Mult(pj, Apj_r);
      Apj_i = 0.0; A_i.Mult(pj, Apj_i);

      // Optional shift: Apj += shift * M_curl * pj
      if (do_shift)
      {
         Mpj = 0.0;
         M_curl->Mult(pj, Mpj);
         if (shift_eta != 0.0)   { Apj_r.Add(shift_eta,   Mpj); }
         if (shift_gamma != 0.0) { Apj_i.Add(shift_gamma, Mpj); }
      }

      // Scale by D_inv
      for (int i = 0; i < n; i++)
      {
         Apj_s_r[i] = Apj_r[i] * D_inv[i];
         Apj_s_i[i] = Apj_i[i] * D_inv[i];
      }

      // Ac_col = M_cross^T * Apj_s
      Ac_col_r = 0.0; M_cross.MultTranspose(Apj_s_r, Ac_col_r);
      Ac_col_i = 0.0; M_cross.MultTranspose(Apj_s_i, Ac_col_i);

      for (int k = 0; k < no1; k++)
      {
         Ac_r(k, j) = Ac_col_r[k];
         Ac_i(k, j) = Ac_col_i[k];
      }
   }

   // Symmetrize Ac_r (should be symmetric for Hermitian system)
   // Ac_i should be skew-symmetric for Hermitian
   for (int i = 0; i < no1; i++)
   {
      for (int j = i + 1; j < no1; j++)
      {
         double avg_r = 0.5 * (Ac_r(i, j) + Ac_r(j, i));
         double avg_i = 0.5 * (Ac_i(i, j) + Ac_i(j, i));
         Ac_r(i, j) = Ac_r(j, i) = avg_r;
         Ac_i(i, j) = Ac_i(j, i) = avg_i;
      }
      // Ac_i diagonal: antisymmetrize with zero (should be ~0 for Hermitian)
      Ac_i(i, i) = 0.0;
   }
}

// ===========================================================================
// Hand-rolled right-preconditioned FGMRES for complex block system
// ===========================================================================
struct GMRESResult
{
   int    iters       = 0;
   double rel_final   = 1.0;
   double rel_10      = 1.0;
   double rel_50      = 1.0;
   double rel_100     = 1.0;
   bool   converged   = false;
   bool   breakdown   = false;
   bool   stagnation  = false;
   bool   nan_detected = false;
   double setup_ms    = 0.0;
   double apply_ms    = 0.0;
   double avg_prec_ms = 0.0;
   double total_ms    = 0.0;
   int    nnz_pi      = 0;
   int    dim_pi      = 0;
   vector<double> res_history;
};

GMRESResult run_gmres_complex(const Operator         &A_op,
                              const Vector           &B,
                              Solver                 *prec,
                              int                     restart  = 200,
                              int                     max_iter = 5000,
                              double                  tol      = 1e-8)
{
   GMRESResult res;
   int N = A_op.Height();

   // Preconditioner setup time (measured externally)
   res.dim_pi = (prec && dynamic_cast<ComplexPiPreconditioner*>(prec))
                ? dynamic_cast<ComplexPiPreconditioner*>(prec)->CoarseDim() : 0;
   if (prec && res.dim_pi > 0)
   {
      res.nnz_pi = dynamic_cast<ComplexPiPreconditioner*>(prec)->NnzCross();
   }

   double bnorm = sqrt(B * B);
   if (bnorm < 1e-30) { bnorm = 1.0; }

   // Right-preconditioned GMRES: solve A * B^{-1} * y = b, then x = B^{-1} * y
   // Implementation: Arnoldi with A * (B^{-1} * v) at each step
   // So w = B^{-1} * v, then A * w

   int m = restart;
   DenseMatrix H(m + 1, m);   // Hessenberg matrix
   DenseMatrix V(N, m + 1);   // Arnoldi vectors (columns)
   Vector g(m + 1);            // RHS for least-squares
   Vector cs(m), sn(m);        // Givens rotations
   Vector y_(m), w(N), Av(N);
   Vector X(N);
   X = 0.0;

   Vector R(B.Size());
   R = B;
   double beta = sqrt(R * R);
   double rel_res = beta / bnorm;
   res.res_history.push_back(rel_res);

   int total_iter = 0;
   auto t_start = steady_clock::now();
   double prec_total_ms = 0.0;

   while (total_iter < max_iter && rel_res > tol)
   {
      // First Arnoldi vector: v1 = R / beta
      for (int i = 0; i < N; i++) { V(i, 0) = R[i] / beta; }
      g = 0.0;
      g[0] = beta;

      int j;
      for (j = 0; j < m && total_iter < max_iter; j++, total_iter++)
      {
         // w = B^{-1} * v_j
         if (prec)
         {
            Vector vj(V.GetData() + j * N, N);
            auto tp0 = steady_clock::now();
            prec->Mult(vj, w);
            prec_total_ms += duration<double>(steady_clock::now() - tp0).count() * 1000.0;
         }
         else
         {
            for (int i = 0; i < N; i++) { w[i] = V(i, j); }
         }

         // Av = A * w
         A_op.Mult(w, Av);

         // Modified Gram-Schmidt
         for (int i = 0; i <= j; i++)
         {
            Vector vi(V.GetData() + i * N, N);
            H(i, j) = Av * vi;
            Av.Add(-H(i, j), vi);
         }
         H(j + 1, j) = sqrt(Av * Av);
         if (H(j + 1, j) < 1e-30) { H(j + 1, j) = 0.0; }

         // Store next Arnoldi vector
         if (H(j + 1, j) > 0)
         {
            for (int i = 0; i < N; i++) { V(i, j + 1) = Av[i] / H(j + 1, j); }
         }

         // Apply previous Givens rotations
         for (int i = 0; i < j; i++)
         {
            double h1  =  cs[i] * H(i, j) + sn[i] * H(i + 1, j);
            double h2  = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
            H(i, j)     = h1;
            H(i + 1, j) = h2;
         }

         // Compute new Givens rotation
         double hjj = H(j, j);
         double hjp1j = H(j + 1, j);
         double rho = sqrt(hjj * hjj + hjp1j * hjp1j);
         if (rho < 1e-30) { res.breakdown = true; break; }
         cs[j] = hjj / rho;
         sn[j] = hjp1j / rho;
         H(j, j) = rho;
         H(j + 1, j) = 0.0;

         // Apply to RHS
         double g1 =  cs[j] * g[j] + sn[j] * g[j + 1];
         double g2 = -sn[j] * g[j] + cs[j] * g[j + 1];
         g[j] = g1;
         g[j + 1] = g2;

         // Compute residual
         rel_res = abs(g[j + 1]) / bnorm;
         res.res_history.push_back(rel_res);

         if (total_iter == 9)  { res.rel_10  = rel_res; }
         if (total_iter == 49) { res.rel_50  = rel_res; }
         if (total_iter == 99) { res.rel_100 = rel_res; }

         // Check convergence
         if (rel_res <= tol) { j++; break; }
      }

      // Back substitution: solve H(0:j-1,0:j-1) * y = g(0:j-1)
      for (int i = j - 1; i >= 0; i--)
      {
         y_[i] = g[i];
         for (int k = i + 1; k < j; k++) { y_[i] -= H(i, k) * y_[k]; }
         y_[i] /= H(i, i);
      }

      // x += B^{-1} * (V * y)
      for (int k = 0; k < j; k++)
      {
         Vector vk(V.GetData() + k * N, N);
         if (prec)
         {
            auto tp0 = steady_clock::now();
            prec->Mult(vk, w);
            prec_total_ms += duration<double>(steady_clock::now() - tp0).count() * 1000.0;
         }
         else
         {
            w = vk;
         }
         X.Add(y_[k], w);
      }

      // Compute residual explicitly: R = B - A * X
      A_op.Mult(X, Av);
      for (int i = 0; i < N; i++) { R[i] = B[i] - Av[i]; }
      beta = sqrt(R * R);
      rel_res = beta / bnorm;

      // Check for NaN
      if (!isfinite(rel_res)) { res.nan_detected = true; break; }
   }

   auto t_end = steady_clock::now();
   res.total_ms = duration<double>(t_end - t_start).count() * 1000.0;
   res.iters = total_iter;
   res.rel_final = rel_res;
   res.converged = (rel_res <= tol);
   res.avg_prec_ms = (prec && total_iter > 0) ? prec_total_ms / total_iter : 0.0;

   return res;
}

// ===========================================================================
// Report formatting
// ===========================================================================
void report_gmres(int myid, const string &label, const GMRESResult &r,
                  int r_val, int o_val, double f_val, int N, ostream &out)
{
   if (myid != 0) return;

   static bool header_printed = false;
   if (!header_printed)
   {
      out << "\n┌────────────────────────────────────────────────────────────────────"
          << "─────────────────────────────────────────────────────┐\n";
      out << "│ PML Pi_lumped: r=" << setw(1) << r_val
          << " o=" << setw(1) << o_val
          << " f=" << setw(4) << (int)f_val
          << "  N=" << setw(5) << N
          << " (complex, actual unknowns = 2N)                   │\n";
      out << "├─────────────────────────────────────┬──────┬──────┬──────┬──────"
          << "┬──────┬──────┬──────┬──────┬──────┬──────┤\n";
      out << "│ Preconditioner                      │ Iters│ r(10)│ r(50)│r(100)│"
          << "r(final)│Brk/Nan│Setup │Apply │nnz(Pi)│dim(Pi)│\n";
      out << "├─────────────────────────────────────┼──────┼──────┼──────┼──────"
          << "┼──────┼──────┼──────┼──────┼──────┼──────┤\n";
      header_printed = true;
   }

   out << "│ " << left << setw(35) << label << " │ "
       << right << setw(4) << r.iters << " │ ";

   auto fmt_rel = [&](double v)
   {
      if (v < 1.0) { out << scientific << setprecision(2) << setw(8) << v; }
      else { out << "    —    "; }
   };

   fmt_rel(r.rel_10); out << " │ ";
   fmt_rel(r.rel_50); out << " │ ";
   fmt_rel(r.rel_100); out << " │ ";
   out << scientific << setprecision(2) << setw(8) << r.rel_final;
   out << " │ ";
   if (r.nan_detected)       { out << " NaN  │ "; }
   else if (r.breakdown)     { out << " BRK  │ "; }
   else if (r.stagnation)    { out << " STG  │ "; }
   else                      { out << "      │ "; }
   out << fixed << setprecision(0) << setw(4) << r.setup_ms  << " │ ";
   out << fixed << setprecision(0) << setw(4) << r.total_ms   << " │ ";
   out << setw(6) << r.nnz_pi << " │";
   out << setw(5) << r.dim_pi << " │\n";
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse options
   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int    ref       = 2;
   int    order     = 2;
   double freq      = 5.0;
   int    aux_order = 1;
   int    restart   = 200;
   int    max_iter  = 5000;
   double tol       = 1e-8;
   bool   test_ams  = false;
   bool   use_shift = false;
   double shift_eta = 0.3;
   double shift_gamma = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Spline order (fine).");
   args.AddOption(&freq, "-f", "--frequency", "Frequency.");
   args.AddOption(&aux_order, "-ao", "--aux-order",
                  "Auxiliary NURBS order (default 1).");
   args.AddOption(&restart, "-kdim", "--kdim", "GMRES restart dimension.");
   args.AddOption(&max_iter, "-maxit", "--max-iter", "Max GMRES iterations.");
   args.AddOption(&tol, "-tol", "--tolerance", "GMRES relative tolerance.");
   args.AddOption(&test_ams, "-ams", "--test-ams", "-no-ams", "--no-test-ams",
                  "Test Hypre AMS on abs preconditioner.");
   args.AddOption(&use_shift, "-shift", "--shifted-ac", "-no-shift",
                  "--no-shifted-ac",
                  "Use shifted coarse operator Ac = Pi^T (K + eta*M) Pi.");
   args.AddOption(&shift_eta, "-eta", "--shift-eta",
                  "Shift parameter eta (additive mass).");
   args.AddOption(&shift_gamma, "-gamma", "--shift-gamma",
                  "Complex shift gamma (imaginary part, 0=real only).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      MPI_Finalize();
      return 1;
   }

   Device device("cpu");

   // ==========================================================================
   // Mesh, PML, and fine space
   // ==========================================================================
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   int dim = serial_mesh->Dimension();

   Array2D<real_t> pml_length(dim, 2);
   pml_length = 0.25;
   PML *pml = new PML(serial_mesh, pml_length);

   real_t omega_val = real_t(2.0 * M_PI) * freq;
   real_t mu = 1.0, epsilon = 1.0;

   // Set file-scope globals for PML stretch functions
   g_omega = omega_val; g_mu = mu; g_epsilon = epsilon;

   Array2D<real_t> comp_domain_bdr = pml->GetCompDomainBdr();
   Array2D<real_t> domain_bdr      = pml->GetDomainBdr();

   for (int l = 0; l < ref; l++) { serial_mesh->UniformRefinement(); }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   pml->SetAttributes(pmesh);

   // Fine H(curl) space
   auto *fec      = new NURBS_HCurlFECollection(order, dim);
   auto *nurbsExt = new NURBSExtension(pmesh->NURBSext, order);
   auto *fespace  = new ParFiniteElementSpace(pmesh, nurbsExt, fec);
   int   ncurl    = fespace->GetTrueVSize();

   // Auxiliary H(curl) space (order = aux_order)
   auto *aux_fec  = new NURBS_HCurlFECollection(aux_order, dim);
   auto *aux_ext  = new NURBSExtension(pmesh->NURBSext, aux_order);
   auto *aux_fes  = new ParFiniteElementSpace(pmesh, aux_ext, aux_fec);
   int   no1      = aux_fes->GetTrueVSize();

   if (myid == 0)
   {
      cout << "\n[ComplexPi] fine order=" << order << " (N=" << ncurl << ")"
           << "  aux order=" << aux_order << " (N_aux=" << no1 << ")"
           << "  freq=" << freq << "  ref=" << ref << endl;
   }

   // BCs
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   // ==========================================================================
   // Complex A_pml operator (following nurbs-25p exactly)
   // ==========================================================================
   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   // RHS: point source
   VectorFunctionCoefficient f_src(dim, point_source);
   ParComplexLinearForm b_form(fespace, conv);
   b_form.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f_src));
   b_form.Vector::operator=(0.0);
   b_form.Assemble();

   ParComplexGridFunction x_gf(fespace);
   x_gf = 0.0;

   // Bilinear form assembly
   Array<int> attr, attrPML;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());
      attrPML.SetSize(pmesh->attributes.Max());
      attr = 0;
      attr[0] = 1;
      attrPML = 0;
      if (pmesh->attributes.Max() > 1) { attrPML[1] = 1; }
   }

   ConstantCoefficient muinv_c(1.0 / mu);
   ConstantCoefficient omeg_c(-pow2(omega_val) * epsilon);
   RestrictedCoefficient restr_muinv(muinv_c, attr);
   RestrictedCoefficient restr_omeg(omeg_c, attr);

   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv), NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg), NULL);

   // PML integrators
   const int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim, detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim, detJ_inv_JT_J_Im, pml);
   ScalarVectorProductCoefficient c1_Re(muinv_c, pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv_c, pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re, attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im, attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml);
   ScalarVectorProductCoefficient c2_Re(omeg_c, pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg_c, pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re, attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im, attrPML);

   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   auto t_assemble_start = steady_clock::now();
   a.Assemble(0);

   OperatorPtr A_op_all;
   Vector B_all, X_all;
   a.FormLinearSystem(ess_tdofs, x_gf, b_form, A_op_all, X_all, B_all);
   double t_assemble = duration<double>(steady_clock::now()
                                        - t_assemble_start).count();

   // Extract real/imag blocks from the complex operator
   // The complex operator can be a BlockOperator or a ComplexHypreParMatrix
   HypreParMatrix *A_r_ptr = nullptr;
   HypreParMatrix *A_i_ptr = nullptr;

   auto *A_blk = dynamic_cast<BlockOperator*>(A_op_all.Ptr());
   if (A_blk)
   {
      A_r_ptr = dynamic_cast<HypreParMatrix*>(&A_blk->GetBlock(0, 0));
      A_i_ptr = dynamic_cast<HypreParMatrix*>(&A_blk->GetBlock(1, 0));
   }
   else
   {
      // Try ComplexHypreParMatrix
      auto *A_cpx = dynamic_cast<ComplexHypreParMatrix*>(A_op_all.Ptr());
      if (A_cpx)
      {
         A_r_ptr = &A_cpx->real();
         A_i_ptr = &A_cpx->imag();
      }
   }
   MFEM_VERIFY(A_r_ptr && A_i_ptr,
               "Cannot extract real/imag blocks from complex operator");

   if (myid == 0)
   {
      cout << "\nComplex system: 2N = " << A_op_all->Height() << " unknowns"
           << "  assemble=" << fixed << setprecision(2) << t_assemble << "s"
           << endl;
   }

   // ==========================================================================
   // Build Pi from real mass matrices
   // ==========================================================================
   ConstantCoefficient one(1.0);

   // Fine mass M_curl (real)
   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0);
   mform.Finalize(0);
   OperatorPtr M_op;
   mform.FormSystemMatrix(ess_tdofs, M_op);
   auto *M_curl = M_op.As<HypreParMatrix>();

   // Cross mass M_cross (aux × fine, real)
   ParMixedBilinearForm mmix(aux_fes, fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   mmix.Finalize(0);
   Array<int> empty_tdofs;
   OperatorHandle Mmix_h;
   mmix.FormRectangularSystemMatrix(empty_tdofs, empty_tdofs, Mmix_h);
   auto *M_cross = Mmix_h.As<HypreParMatrix>();

   // D_inv_M = diag(M_curl)^{-1}
   Vector D_inv_M(ncurl);
   M_curl->GetDiag(D_inv_M);
   for (int i = 0; i < ncurl; i++)
   {
      D_inv_M[i] = 1.0 / max(1e-14, abs(D_inv_M[i]));
   }

   // Complex diagonal of A_pml for Jacobi smoother
   Vector D_r(ncurl), D_i(ncurl);
   A_r_ptr->GetDiag(D_r);
   A_i_ptr->GetDiag(D_i);

   // ==========================================================================
   // Form complex Ac = Pi^T * A_pml * Pi (or shifted)
   // ==========================================================================
   auto t_ac_start = steady_clock::now();
   DenseMatrix Ac_r, Ac_i;
   if (use_shift)
   {
      // Shifted: A_shift = A + eta * M_curl (+ i*gamma*M_curl)
      // Ac = Pi^T * A_shift * Pi
      if (myid == 0)
      {
         cout << "[ComplexPi] Using shifted Ac: eta=" << shift_eta
              << " gamma=" << shift_gamma << endl;
      }
      FormComplexCoarseOperator(*A_r_ptr, *A_i_ptr, *M_cross, D_inv_M,
                                ncurl, no1, Ac_r, Ac_i,
                                shift_eta, shift_gamma, M_curl);
   }
   else
   {
      FormComplexCoarseOperator(*A_r_ptr, *A_i_ptr, *M_cross, D_inv_M,
                                ncurl, no1, Ac_r, Ac_i);
   }
   double t_ac = duration<double>(steady_clock::now() - t_ac_start).count();

   // Build preconditioners
   auto t_prec_start = steady_clock::now();
   ComplexPiPreconditioner prec_pi05(*M_cross, D_inv_M, Ac_r, Ac_i, D_r, D_i, 0.5);
   ComplexPiPreconditioner prec_pi07(*M_cross, D_inv_M, Ac_r, Ac_i, D_r, D_i, 0.7);
   ComplexPiPreconditioner prec_pi10(*M_cross, D_inv_M, Ac_r, Ac_i, D_r, D_i, 1.0);
   double t_prec = duration<double>(steady_clock::now() - t_prec_start).count();

   if (myid == 0)
   {
      cout << "[ComplexPi] assemble=" << fixed << setprecision(2) << t_assemble << "s"
           << "  form_Ac=" << setprecision(2) << t_ac << "s"
           << "  prec_setup=" << setprecision(4) << t_prec << "s"
           << "  nnz(M_cross)=" << M_cross->NNZ()
           << "  coarse_sys=" << (2*no1) << "x" << (2*no1) << endl;
   }

   // ==========================================================================
   // Run benchmarks: GMRES with different preconditioners
   // ==========================================================================
   const int N2 = A_op_all->Height();

   // --- none ---
   {
      auto r = run_gmres_complex(*A_op_all, B_all, nullptr,
                                 restart, max_iter, tol);
      r.setup_ms = 0;
      report_gmres(myid, "none", r, ref, order, freq, ncurl, cout);
   }

   // --- complex Jacobi ---
   // Build complex Jacobi preconditioner (simple diagonal scaling)
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
            double pr = Dr_[i] * inv * w_;
            double pi = Di_[i] * inv * w_;
            z[i]       = pr * r[i]       + pi * r[i + n];
            z[i + n]   = pr * r[i + n]   - pi * r[i];
         }
      }
      void SetOperator(const Operator &op) override
      {
         height = op.Height(); width = op.Width();
      }
   private:
      const Vector &Dr_, &Di_;
      double w_;
   };

   for (double w : {0.5, 0.7, 1.0})
   {
      ComplexJacobi jac(D_r, D_i, w);
      jac.SetOperator(*A_op_all);
      auto r = run_gmres_complex(*A_op_all, B_all, &jac,
                                 restart, max_iter, tol);
      r.setup_ms = 0;
      ostringstream os;
      os << "Jacobi(cpx) ω=" << fixed << setprecision(1) << w;
      report_gmres(myid, os.str(), r, ref, order, freq, ncurl, cout);
   }

   // --- Pi_lumped + complex Jacobi ---
   double setup_sum = (t_ac + t_prec) * 1000.0;

   {
      prec_pi05.SetOperator(*A_op_all);
      auto r = run_gmres_complex(*A_op_all, B_all, &prec_pi05,
                                 restart, max_iter, tol);
      r.setup_ms = setup_sum;
      report_gmres(myid, "Pi_lumped+jac ω=0.5", r, ref, order, freq, ncurl, cout);
   }
   {
      prec_pi07.SetOperator(*A_op_all);
      auto r = run_gmres_complex(*A_op_all, B_all, &prec_pi07,
                                 restart, max_iter, tol);
      r.setup_ms = setup_sum;
      report_gmres(myid, "Pi_lumped+jac ω=0.7", r, ref, order, freq, ncurl, cout);
   }
   {
      prec_pi10.SetOperator(*A_op_all);
      auto r = run_gmres_complex(*A_op_all, B_all, &prec_pi10,
                                 restart, max_iter, tol);
      r.setup_ms = setup_sum;
      report_gmres(myid, "Pi_lumped+jac ω=1.0", r, ref, order, freq, ncurl, cout);
   }

   // --- Hypre AMS on abs preconditioner (optional) ---
   if (test_ams)
   {
      // Build abs preconditioner matrix (same as nurbs-25p)
      ConstantCoefficient absomeg(pow2(omega_val) * epsilon);
      RestrictedCoefficient restr_absomeg(absomeg, attr);

      ParBilinearForm prec_form(fespace);
      prec_form.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
      prec_form.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

      PMLDiagMatrixCoefficient pml_c1_abs(cdim, detJ_inv_JT_J_abs, pml);
      ScalarVectorProductCoefficient c1_abs(muinv_c, pml_c1_abs);
      VectorRestrictedCoefficient restr_c1_abs(c1_abs, attrPML);

      PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs, pml);
      ScalarVectorProductCoefficient c2_abs(absomeg, pml_c2_abs);
      VectorRestrictedCoefficient restr_c2_abs(c2_abs, attrPML);

      prec_form.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
      prec_form.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));
      prec_form.Assemble();

      OperatorPtr PCOpAh;
      prec_form.FormSystemMatrix(ess_tdofs, PCOpAh);
      auto *A_abs = PCOpAh.As<HypreParMatrix>();

      const int s = (conv == ComplexOperator::HERMITIAN) ? -1 : 1;

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = ncurl;
      offsets[2] = ncurl;
      offsets.PartialSum();

      unique_ptr<Operator> pc_r, pc_i;
      try
      {
         pc_r.reset(new HypreAMS(*A_abs, fespace));
         dynamic_cast<HypreAMS*>(pc_r.get())->SetPrintLevel(0);
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }
      catch (...)
      {
         if (myid == 0)
         {
            cout << "│ " << left << setw(35) << "Hypre AMS" << " │ "
                 << "FAILED to construct (singular?)" << " │\n";
         }
      }

      if (pc_r && pc_i)
      {
         BlockDiagonalPreconditioner BlockDP(offsets);
         BlockDP.SetDiagonalBlock(0, pc_r.get());
         BlockDP.SetDiagonalBlock(1, pc_i.get());

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetPrintLevel(0);
         gmres.SetKDim(restart);
         gmres.SetMaxIter(max_iter);
         gmres.SetRelTol(tol);
         gmres.SetAbsTol(0.0);
         gmres.SetOperator(*A_op_all);
         gmres.SetPreconditioner(BlockDP);

         auto t_ams = steady_clock::now();
         gmres.Mult(B_all, X_all);
         double t_ams_s = duration<double>(steady_clock::now() - t_ams).count();

         if (myid == 0)
         {
            cout << "│ " << left << setw(35) << "Hypre AMS (abs prec)" << " │ "
                 << right << setw(4) << gmres.GetNumIterations()
                 << " │     —     │     —     │     —    │ "
                 << (gmres.GetConverged() ? "  OK  " : "  NCV ")
                 << " │    — │ " << fixed << setprecision(1) << setw(5)
                 << t_ams_s * 1000
                 << "│    — │    — │\n";
         }
      }
   }

   if (myid == 0)
   {
      cout << "└─────────────────────────────────────┴──────┴──────┴──────┴──────"
           << "┴──────┴──────┴──────┴──────┴──────┴──────┘\n";
      cout.flush();
   }

   // Cleanup
   delete aux_fes;
   delete aux_ext;
   delete aux_fec;
   delete fespace;
   delete nurbsExt;
   delete fec;
   delete pml;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

// ===========================================================================
// PML implementation
// ===========================================================================
PML::PML(Mesh *mesh_, Array2D<real_t> length_) : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void PML::SetAttributes(ParMesh *mesh_)
{
   for (int i = 0; i < mesh_->GetNBE(); ++i)
   {
      mesh_->GetBdrElement(i)->SetAttribute(i + 1);
   }
   int nrelem = mesh_->GetNE();
   elems.SetSize(nrelem);

   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = mesh_->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x, vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);
   real_t n_pml = 2.0;
   real_t c_pml = 5.0;
   real_t k = g_omega * sqrt(g_epsilon * g_mu);

   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_dom_bdr(i, 1))
      {
         real_t coeff = n_pml * c_pml / k / pow(length(i, 1), n_pml);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_dom_bdr(i, 1), n_pml - 1_r));
      }
      if (x(i) <= comp_dom_bdr(i, 0))
      {
         real_t coeff = n_pml * c_pml / k / pow(length(i, 0), n_pml);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_dom_bdr(i, 0), n_pml - 1_r));
      }
   }
}

// ===========================================================================
// PML coefficient functions
// ===========================================================================
void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   for (int i = 0; i < D.Size(); ++i) { D(i) = (det / pow2(dxs[i])).real(); }
}

void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   for (int i = 0; i < D.Size(); ++i) { D(i) = (det / pow2(dxs[i])).imag(); }
}

void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   for (int i = 0; i < D.Size(); ++i) { D(i) = abs(det / pow2(dxs[i])); }
}

void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   int sz = D.Size();
   if (sz == 2) { D = (real_t(1.0) / det).real(); }
   else { for (int i = 0; i < sz; ++i) { D(i) = (pow2(dxs[i]) / det).real(); } }
}

void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   int sz = D.Size();
   if (sz == 2) { D = (real_t(1.0) / det).imag(); }
   else { for (int i = 0; i < sz; ++i) { D(i) = (pow2(dxs[i]) / det).imag(); } }
}

void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(D.Size());
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);
   for (int i = 0; i < D.Size(); ++i) { det *= dxs[i]; }
   int sz = D.Size();
   if (sz == 2) { D = abs(real_t(1.0) / det); }
   else { for (int i = 0; i < sz; ++i) { D(i) = abs(pow2(dxs[i]) / det); } }
}

// ===========================================================================
// Point source (from nurbs-25p)
// ===========================================================================
void point_source(const Vector &x, Vector &f)
{
   int dim = x.Size();
   Vector center(dim);
   real_t r2 = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = 0.5;  // domain center on [0,1]^3
      r2 += pow(x[i] - center[i], 2.);
   }
   real_t n = 5.0 * g_omega * sqrt(g_epsilon * g_mu) / M_PI;
   real_t coeff = pow(n, 2) / M_PI;
   real_t alpha = -pow(n, 2) * r2;
   f = 0.0;
   f[0] = 1000.0 * coeff * exp(alpha);
}
