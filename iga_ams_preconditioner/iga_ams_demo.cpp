// iga_ams_demo.cpp
//
// Test three AMS-improvement strategies for IGA H(curl) PML Maxwell:
//
//   1. AMS on o=1 IGA Galerkin coarse operator
//      B_p^{-1} r = P * AMS(A_1)^{-1} * P^T r + S_p r
//
//   2. IGA-exact gradient correction
//      B_G^{-1} r = G * A_phi^{-1} * G^T r
//
//   3. Shifted AMS
//      Apply AMS to A_shift = A + eta*M, use as preconditioner for A
//
// Baselines: none, hypre AMS (default on high-order), block Jacobi.
//
// Usage:
//   mpirun -np 1 ./iga_ams_demo -m cube-nurbs.mesh -r 2 -o 2 -f 5.0

#include "mfem.hpp"
#include "iga_ams_prec.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <complex>
#include <chrono>

using namespace std;
using namespace mfem;

// ─── PML class (copied from pml_split_demo.cpp) ──────────────────────────────

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
   Array2D<real_t> GetDomainBdr()     { return dom_bdr; }
   Array<int> *GetMarkedPMLElements() { return &elems; }
   void SetAttributes(ParMesh *mesh_);
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs);
};

class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML *pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int vd, void(*F)(const Vector &, PML *, Vector &), PML *p)
      : VectorCoefficient(vd), pml(p), Function(F) {}
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &K, ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t x[3]; Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K);
   }
};

template <typename T> T pow2(const T &x) { return x * x; }

// Global problem parameters
real_t mu      = 1.0;
real_t epsilon = 1.0;
real_t omega;
int    dim;
Array2D<real_t> comp_domain_bdr;
Array2D<real_t> domain_bdr;

// Forward declarations
void source(const Vector &x, Vector &f);
void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);
void maxwell_solution(const Vector &x, vector<complex<real_t>> &E);
void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D);

// Simple MPI-global L2 norm
static double GlobalNorml2(const mfem::Vector &v, MPI_Comm comm = MPI_COMM_WORLD)
{
   double local_sq = v.Norml2(); local_sq *= local_sq;
   double global_sq = 0.0;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
   return std::sqrt(global_sq);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int    ref_levels  = 2;
   int    order       = 2;
   real_t freq        = 5.0;
   int    gmres_max   = 400;
   int    gmres_kdim  = 200;
   real_t gmres_tol   = 1e-6;
   int    gmres_print = 0;
   bool   skip_grad    = true;
   bool   skip_shifted = false;
   bool   skip_o1      = true;  // force skip for testing

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Spline order.");
   args.AddOption(&freq, "-f", "--frequency", "Frequency.");
   args.AddOption(&gmres_max, "-gmi", "--gmres-max-iter", "Max GMRES iterations.");
   args.AddOption(&gmres_kdim, "-gkd", "--gmres-kdim", "GMRES Krylov subspace dimension.");
   args.AddOption(&gmres_tol, "-grt", "--gmres-rel-tol", "GMRES relative tolerance.");
   args.AddOption(&gmres_print, "-gpl", "--gmres-print-level", "GMRES print level.");
   args.AddOption(&skip_grad, "-no-grad", "--no-gradient", "-grad", "--gradient",
                  "Disable gradient correction experiments.");
   args.AddOption(&skip_shifted, "-no-shift", "--no-shifted", "-shift", "--shifted",
                  "Disable shifted AMS experiments.");
                  args.AddOption(&skip_o1, "-no-o1", "--no-o1-ams", "-o1", "--o1-ams",
                                 "Disable o=1 Galerkin AMS experiments.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device("cpu"); device.Print();

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   Array2D<real_t> length(dim, 2); length = 0.25;
   PML *pml    = new PML(mesh, length);
   omega       = real_t(2.0 * M_PI) * freq;
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr      = pml->GetDomainBdr();

   for (int l = 0; l < ref_levels; l++) { mesh->UniformRefinement(); }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->NURBSext = nullptr; // StealNURBSext transfers ownership to ParMesh
   delete mesh;
   pml->SetAttributes(pmesh);

   FiniteElementCollection *fec = new NURBS_HCurlFECollection(order, dim);
   NURBSExtension *NURBSext = new NURBSExtension(pmesh->NURBSext, order);
   ParFiniteElementSpace *fespace =
      new ParFiniteElementSpace(pmesh, NURBSext, fec);
   if (myid == 0)
   { cout << "true_vsize=" << fespace->GetTrueVSize() << endl; }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   // ── RHS ──
   VectorFunctionCoefficient f(dim, source);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b.Vector::operator=(0.0);
   b.Assemble();

   ParComplexGridFunction x(fespace); x = 0.0;

   // ── PML attribute arrays ──
   Array<int> attr, attrPML;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());    attr    = 0; attr[0]    = 1;
      attrPML.SetSize(pmesh->attributes.Max()); attrPML = 0;
      if (pmesh->attributes.Max() > 1) { attrPML[1] = 1; }
   }

   // ── Material coefficients ──
   ConstantCoefficient muinv(1_r / mu);
   ConstantCoefficient omeg(-pow2(omega) * epsilon);
   RestrictedCoefficient restr_muinv(muinv, attr);
   RestrictedCoefficient restr_omeg(omeg, attr);

   const int cdim = (dim == 2) ? 1 : dim;

   // PML complex curl coefficients
   PMLDiagMatrixCoefficient pml_c1_Re(cdim, detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim, detJ_inv_JT_J_Im, pml);
   ScalarVectorProductCoefficient c1_Re(muinv, pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv, pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re, attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im, attrPML);

   // PML complex mass coefficients
   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml);
   ScalarVectorProductCoefficient c2_Re(omeg, pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg, pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re, attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im, attrPML);

   // Full sesquilinear form (complex PML Maxwell)
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv), NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg), NULL);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));
   a.Assemble(0);

   // Form system matrix
   OperatorPtr A;
   Vector      B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   const double r0norm = GlobalNorml2(B, MPI_COMM_WORLD);

   // ── Geometry evaluator ──
   auto geom = std::make_unique<fdfd_iga_init::SinglePatchNURBSEvaluator>(
      *pmesh, *pmesh->NURBSext, 0);

   // ── Helper: run GMRES and return diagnostic info ──────────────────────────
   struct DiagInfo { int iters; double true_rel; double init_rel; int conv; };

   auto run_gmres = [&](const std::string &label,
                        mfem::Solver *prec_ptr,
                        const mfem::Vector *x0 = nullptr) -> DiagInfo
   {
      Vector Xrun(B.Size());
      if (x0) { Xrun = *x0; } else { Xrun = 0.0; }

      // Compute initial residual ratio
      double init_rel = 1.0;
      if (x0)
      {
         Vector r0(B.Size());
         A->Mult(Xrun, r0); r0 *= -1.0; r0 += B;
         init_rel = GlobalNorml2(r0, MPI_COMM_WORLD) / r0norm;
      }

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetOperator(*A);
      if (prec_ptr) { gmres.SetPreconditioner(*prec_ptr); }
      gmres.SetRelTol(gmres_tol);
      gmres.SetAbsTol(0.0);
      gmres.SetMaxIter(gmres_max);
      gmres.SetKDim(gmres_kdim);
      gmres.SetPrintLevel(gmres_print);
      gmres.Mult(B, Xrun);

      DiagInfo diag;
      diag.iters    = gmres.GetNumIterations();
      diag.conv     = gmres.GetConverged();
      diag.init_rel = init_rel;

      // Compute true residual
      Vector resid(B.Size());
      A->Mult(Xrun, resid);
      resid *= -1.0; resid += B;
      diag.true_rel = GlobalNorml2(resid, MPI_COMM_WORLD) / r0norm;

      if (myid == 0)
      {
         cout << "  " << left << setw(28) << label;
         if (x0) { cout << " x0rel=" << scientific << setprecision(1) << init_rel; }
         else     { cout << "         "; }
         cout << "  iters=" << setw(4) << diag.iters
              << "  rel=" << scientific << setprecision(2) << diag.true_rel
              << "  conv=" << diag.conv
              << endl;
      }
      return diag;
   };

   // ═══════════════════════════════════════════════════════════════════════════
   // Main benchmark
   // ═══════════════════════════════════════════════════════════════════════════
   if (myid == 0)
   {
      cout << "\n" << string(72, '=') << "\n";
      cout << "  IGA AMS Preconditioner Benchmark\n";
      cout << "  mesh=" << mesh_file << "  r=" << ref_levels
           << "  o=" << order << "  f=" << freq << "\n";
      cout << string(72, '=') << "\n";
   }

   // ── Section 0: Baselines ──────────────────────────────────────────────────
   if (myid == 0) cout << "\n[0] Baselines\n" << string(72, '-') << "\n";

   // 0a. No preconditioner
   run_gmres("none", nullptr);

   // 0b. Default Hypre AMS (high-order IGA)
   {
      // Build a positive-definite approximation for AMS
      ConstantCoefficient abs_omeg(pow2(omega)*epsilon);
      RestrictedCoefficient restr_abs_omeg(abs_omeg, attr);
      ParBilinearForm prec_ams(fespace);
      prec_ams.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
      prec_ams.AddDomainIntegrator(new VectorFEMassIntegrator(restr_abs_omeg));
      prec_ams.Assemble();
      prec_ams.Finalize();
      OperatorPtr Ah_op;
      prec_ams.FormSystemMatrix(ess_tdof_list, Ah_op);
      auto *A_hypre = Ah_op.As<HypreParMatrix>();
      MFEM_VERIFY(A_hypre, "AMS requires HypreParMatrix.");
      HypreAMS ams(*A_hypre, fespace);
      ams.SetPrintLevel(0);

      // Wrap as real-block: apply AMS to each half
      class RealBlockAMS : public Solver
      {
      public:
         RealBlockAMS(HypreAMS &ams, int n) : Solver(2*n, 2*n), ams_(&ams), n_(n) {}
         void SetOperator(const Operator &) override {}
         void Mult(const Vector &r, Vector &z) const override
         {
            z.SetSize(2*n_); z = 0.0;
            Vector r_re(const_cast<Vector&>(r), 0, n_);
            Vector r_im(const_cast<Vector&>(r), n_, n_);
            Vector z_re(z, 0, n_), z_im(z, n_, n_);
            ams_->Mult(r_re, z_re);
            ams_->Mult(r_im, z_im);
         }
      private:
         HypreAMS *ams_; int n_;
      };
      RealBlockAMS ams_block(ams, fespace->GetTrueVSize());

      run_gmres("hypre_ams (default)", &ams_block);
   }

   // 0c. Block Jacobi (2x2 complex block diagonal)
   {
      class BlockJacobi : public Solver
      {
      public:
         BlockJacobi(const Operator &op, int n) : Solver(2*n, 2*n)
         {
            bj00_.SetSize(n); bj01_.SetSize(n); bj10_.SetSize(n); bj11_.SetSize(n);
            Vector e(2*n), Ae(2*n);
            for (int i = 0; i < n; i++)
            {
               e = 0.0; e[i] = 1.0; op.Mult(e, Ae);
               double a00 = Ae[i], a10 = Ae[i+n];
               e = 0.0; e[i+n] = 1.0; op.Mult(e, Ae);
               double a01 = Ae[i], a11 = Ae[i+n];
               double det = a00*a11 - a01*a10;
               if (std::abs(det) < 1e-14) det = 1.0;
               bj00_[i] =  a11/det; bj01_[i] = -a01/det;
               bj10_[i] = -a10/det; bj11_[i] =  a00/det;
            }
         }
         void SetOperator(const Operator &) override {}
         void Mult(const Vector &r, Vector &z) const override
         {
            const int n = bj00_.Size();
            z.SetSize(2*n);
            for (int i = 0; i < n; i++)
            {
               z[i]     = bj00_[i]*r[i]      + bj01_[i]*r[i+n];
               z[i+n]   = bj10_[i]*r[i]      + bj11_[i]*r[i+n];
            }
         }
      private:
         Vector bj00_, bj01_, bj10_, bj11_;
      };
      BlockJacobi bj(*A.Ptr(), fespace->GetTrueVSize());
      run_gmres("jacobi (2x2 block)", &bj);
   }

   // ── Section 1: AMS on o=1 Galerkin coarse ─────────────────────────────────
   if (myid == 0) cout << "\n[1] AMS on o=1 IGA Galerkin coarse operator\n"
                        << string(72, '-') << "\n";

   using namespace iga_ams;
   {
      auto prec1 = std::make_unique<CoarseAMS_GalerkinPreconditioner>(*fespace, 1);
      prec1->SetOperator(*A);
      prec1->SetCombineMode(CoarseAMS_GalerkinPreconditioner::CombineMode::coarse_only);
      run_gmres("ams_o1_gal coarse_only", prec1.get());

      prec1->SetCombineMode(CoarseAMS_GalerkinPreconditioner::CombineMode::additive);
      run_gmres("ams_o1_gal+jac add", prec1.get());

      prec1->SetCombineMode(CoarseAMS_GalerkinPreconditioner::CombineMode::multiplicative);
      run_gmres("ams_o1_gal+jac mult", prec1.get());
   }

   // ── Section 2: IGA-exact gradient correction ──────────────────────────────
   if (!skip_grad)
   {
      if (myid == 0) cout << "\n[2] IGA-exact gradient correction\n"
                        << string(72, '-') << "\n";
      auto prec2 = std::make_unique<IGA_GradientCorrectionPreconditioner>(*fespace);
      prec2->SetOperator(*A);
      prec2->SetPrintLevel(0);
      prec2->SetVerbose(myid == 0);
      run_gmres("iga_gradient_correction", prec2.get());
   }
   // ── Section 3: Shifted AMS ────────────────────────────────────────────────
   if (!skip_shifted)
   {
   if (myid == 0) cout << "\n[3] Shifted AMS\n" << string(72, '-') << "\n";

   for (double eta : {0.1, 0.3, 0.5, 1.0})
   {
      auto prec3 = std::make_unique<ShiftedAMS_Preconditioner>(*fespace, freq, epsilon);
      prec3->SetOperator(*A);
      prec3->SetEta(eta);
      prec3->SetPrintLevel(0);

      ostringstream label;
      label << "shifted_ams eta=" << fixed << setprecision(1) << eta;
      run_gmres(label.str(), prec3.get());
   }

   }
   if (myid == 0) cout << string(72, '=') << "\n";

   // ── Cleanup ───────────────────────────────────────────────────────────────
   delete fespace;
   delete fec;
   delete pmesh;
   delete pml;

   return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PML implementation
// ═══════════════════════════════════════════════════════════════════════════════

PML::PML(Mesh *mesh_, Array2D<real_t> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
   elems.SetSize(mesh->GetNE()); elems = 1;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *el = mesh->GetElement(i);
      Array<int> vertices;
      bool in_pml = false;
      el->SetAttribute(1);
      el->GetVertices(vertices);
      for (int iv = 0; iv < vertices.Size(); ++iv)
      {
         real_t *coords = mesh_->GetVertex(vertices[iv]);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            { in_pml = true; break; }
         }
         if (in_pml) { break; }
      }
      if (in_pml) { elems[i] = 0; el->SetAttribute(2); }
   }
   mesh_->SetAttributes();
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
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x, vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);
   real_t n = 2.0, c = 5.0;
   real_t k = omega * sqrt(epsilon * mu);
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         real_t coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1_r + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1_r));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         real_t coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1_r + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1_r));
      }
   }
}

// ─── Source ──────────────────────────────────────────────────────────────────

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   real_t r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = 0.5 * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
      r += pow(x[i] - center[i], 2.);
   }
   real_t n     = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   real_t coeff = pow(n, 2) / M_PI;
   real_t alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = 1000 * coeff * exp(alpha);
}

void maxwell_solution(const Vector &x, vector<complex<real_t>> &E)
{
   for (int i = 0; i < dim; ++i) { E[i] = 0.0; }
   complex<real_t> zi(0., 1.);
   real_t k = omega * sqrt(epsilon * mu);
   if (dim == 3)
   {
      real_t k10 = sqrt(k * k - M_PI * M_PI);
      E[1] = -zi * k / real_t(M_PI) * sin(real_t(M_PI) * x(2)) * exp(zi * k10 * x(0));
   }
   else
   {
      E[1] = -zi * k / real_t(M_PI) * exp(zi * k * x(0));
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      if (x(i) < comp_domain_bdr(i, 0) || x(i) > comp_domain_bdr(i, 1))
      { return; }
   }
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i) { E[i] = Eval[i].real(); }
}

void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      if (x(i) < comp_domain_bdr(i, 0) || x(i) > comp_domain_bdr(i, 1))
      { return; }
   }
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i) { E[i] = Eval[i].imag(); }
}

// ─── PML coefficient functions ───────────────────────────────────────────────

void detJ_JT_J_inv_Re(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   for (int i = 0; i < dim; ++i) { D(i) = (det / pow2(dxs[i])).real(); }
}

void detJ_JT_J_inv_Im(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   for (int i = 0; i < dim; ++i) { D(i) = (det / pow2(dxs[i])).imag(); }
}

void detJ_JT_J_inv_abs(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   for (int i = 0; i < dim; ++i) { D(i) = abs(det / pow2(dxs[i])); }
}

void detJ_inv_JT_J_Re(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   if (dim == 2) { D = (real_t(1.) / det).real(); }
   else { for (int i = 0; i < dim; ++i) { D(i) = (pow2(dxs[i]) / det).real(); } }
}

void detJ_inv_JT_J_Im(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   if (dim == 2) { D = (real_t(1.) / det).imag(); }
   else { for (int i = 0; i < dim; ++i) { D(i) = (pow2(dxs[i]) / det).imag(); } }
}

void detJ_inv_JT_J_abs(const Vector &x, PML *pml, Vector &D)
{
   vector<complex<real_t>> dxs(dim);
   pml->StretchFunction(x, dxs);
   complex<real_t> det(1., 0.);
   for (int i = 0; i < dim; ++i) { det *= dxs[i]; }
   if (dim == 2) { D = abs(real_t(1.) / det); }
   else { for (int i = 0; i < dim; ++i) { D(i) = abs(pow2(dxs[i]) / det); } }
}
