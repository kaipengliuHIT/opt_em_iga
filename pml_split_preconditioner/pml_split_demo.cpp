// pml_split_demo.cpp
//
// Benchmark comparing preconditioner strategies for the IGA H(curl)
// PML Maxwell problem on a NURBS cube mesh.
//
// Preconditioner modes tested:
//   none        : unpreconditioned GMRES (baseline)
//   jacobi      : 2x2 complex block-Jacobi (controls PML indefiniteness)
//   coarse      : Yee curl-curl coarse space only (controls gradient null space)
//   additive    : Yee curl-curl + block-Jacobi (additive combination)
//   mult        : Yee curl-curl coarse first, then block-Jacobi on updated residual
//
// Motivation: for PML Maxwell, A = A_curl + A_pml_mass.
//   A_curl = curl curl  -- real, PSD, has gradient null space (elliptic modes)
//   A_pml  = -k^2 S(eps) -- complex, indefinite (PML modes)
// The multiplicative split targets each part with a specialized sub-preconditioner.
//
// Usage:
//   mpirun -np 1 ./pml_split_demo -m /path/to/cube-nurbs.mesh -r 2 -o 2 -f 5.0

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include "split_pml_prec.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <complex>

using namespace std;
using namespace mfem;

// ─── PML class (copied from pml_point_source_demo.cpp) ───────────────────────

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

// ─── main ─────────────────────────────────────────────────────────────────────

// Simple MPI-global L2 norm for plain distributed Vector
static double GlobalNorml2(const mfem::Vector &v,
                            MPI_Comm comm = MPI_COMM_WORLD)
{
   double local_sq = v.Norml2(); local_sq *= local_sq;
   double global_sq = 0.0;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
   return std::sqrt(global_sq);
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int    ref_levels  = 2;
   int    order       = 2;
   real_t freq        = 5.0;
   int    aux_n       = 9;
   int    gmres_max   = 400;
   int    gmres_kdim  = 200;
   real_t gmres_tol   = 1e-6;
   int    gmres_print = 0;
   bool   run_legacy  = true;
   bool   run_pmg     = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Spline order.");
   args.AddOption(&freq, "-f", "--frequency", "Frequency.");
   args.AddOption(&aux_n, "-an", "--aux-n", "Yee grid points per direction.");
   args.AddOption(&gmres_max, "-gmi", "--gmres-max-iter", "Max GMRES iterations.");
   args.AddOption(&gmres_kdim, "-gkd", "--gmres-kdim", "GMRES Krylov subspace dimension (restart).");
   args.AddOption(&gmres_tol, "-grt", "--gmres-rel-tol", "GMRES relative tolerance.");
   args.AddOption(&gmres_print, "-gpl", "--gmres-print-level", "GMRES print level.");
   args.AddOption(&run_legacy, "-legacy", "--run-legacy",
                  "-no-legacy", "--no-legacy",
                  "Run legacy Yee/Jacobi split preconditioner experiments.");
   args.AddOption(&run_pmg, "-pmg", "--run-pmg",
                  "-no-pmg", "--no-pmg",
                  "Run p-level Galerkin bridge experiments.");
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

   // ── Material coefficients (matching pml_point_source_demo.cpp exactly) ──
   ConstantCoefficient muinv(1_r / mu);
   ConstantCoefficient omeg(-pow2(omega) * epsilon);   // negative: curl-curl - k^2 eps
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

   // Abs-PML coefficients (for positive-definite curl-only form)
   PMLDiagMatrixCoefficient pml_c1_abs(cdim, detJ_inv_JT_J_abs, pml);
   ScalarVectorProductCoefficient c1_abs(muinv, pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs, attrPML);

   ConstantCoefficient absomeg(pow2(omega) * epsilon);
   RestrictedCoefficient restr_absomeg(absomeg, attr);

   // ── Full sesquilinear form (complex PML Maxwell) ──
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv), NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg), NULL);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));
   a.Assemble(0);

   // ── Curl-only bilinear form (REAL, for Galerkin coarse mode: P^T A_curl P) ──
   ParBilinearForm a_curl(fespace);
   a_curl.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   a_curl.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
   a_curl.Assemble(0);

   // Form system matrices
   OperatorPtr A;
   Vector      B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   OperatorPtr A_curl_mat;
   a_curl.FormSystemMatrix(ess_tdof_list, A_curl_mat);

   const double r0norm = GlobalNorml2(B, MPI_COMM_WORLD);

   // ── Geometry evaluator for auxiliary space ──
   auto geom = std::make_unique<fdfd_iga_init::SinglePatchNURBSEvaluator>(
      *pmesh, *pmesh->NURBSext, 0);
   auto eps_fn = [](const mfem::Vector &) { return 1.0; };

   fdfd_iga_init::ReferenceGrid yee_grid;
   yee_grid.nx = yee_grid.ny = yee_grid.nz = aux_n;

   // ── Helper: run GMRES from given initial guess, return iteration count ──
   auto run_gmres = [&](const std::string &label,
                        mfem::Solver *prec_ptr,
                        const mfem::Vector *x0 = nullptr) -> int
   {
      Vector Xrun(B.Size());
      if (x0) { Xrun = *x0; } else { Xrun = 0.0; }

      // Compute initial residual ratio (warm-start quality)
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

      const int iters = gmres.GetNumIterations();
      const int conv  = gmres.GetConverged();

      // Compute true residual
      Vector resid(B.Size());
      A->Mult(Xrun, resid);
      resid *= -1.0; resid += B;
      const double true_rel = GlobalNorml2(resid, MPI_COMM_WORLD) / r0norm;

      if (myid == 0)
      {
         cout << "  " << left << setw(22) << label;
         if (x0) { cout << " x0rel=" << scientific << setprecision(1) << init_rel; }
         else     { cout << "         "; }
         cout << "  iters=" << setw(4) << iters
              << "  rel=" << scientific << setprecision(2) << true_rel
              << "  conv=" << conv
              << endl;
      }
      return iters;
   };

   // ── Helper: compute coarse initial guess via Pi A_Y^{-1} Pi^T B ──────────
   // Uses existing split_prec infrastructure (must be built first).
   auto make_coarse_init = [&](pml_split::SplitPMLPreconditioner &prec,
                                mfem::Vector &x0)
   {
      // Apply the coarse preconditioner once to B to get the initial guess
      prec.ApplyCoarsePublic(B, x0);
   };

   // ─── Benchmark ────────────────────────────────────────────────────────────
   if (myid == 0)
   {
      cout << "\n" << string(70, '=') << "\n";
      cout << "  PML Split Preconditioner Benchmark\n";
      cout << "  mesh=" << mesh_file << "  r=" << ref_levels
           << "  o=" << order << "  f=" << freq
           << "  aux_n=" << aux_n << "\n";
      cout << string(70, '=') << "\n";
   }

   std::unique_ptr<pml_split::SplitPMLPreconditioner> split_prec;
   if (run_legacy)
   {
      // 1. Baseline: no preconditioner
      run_gmres("none", nullptr);

      // Build a shared SplitPMLPreconditioner instance
      split_prec = std::make_unique<pml_split::SplitPMLPreconditioner>(
         *fespace, *geom, eps_fn);
      split_prec->SetGrid(yee_grid);
      split_prec->SetWaveNumber(omega);
      split_prec->SetOperator(*A);

      // 2. Pure block-Jacobi (known working baseline)
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::none);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::jacobi_only);
         run_gmres("jacobi_only", split_prec.get());
      }

      // 3. Pure Yee curl-curl coarse (independent, no IGA matrix)
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_curl_independent);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::coarse_only);
         run_gmres("yee_curl_indep", split_prec.get());
      }

      // 4. Yee curl-curl Galerkin coarse (P^T A_curl P)
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_curl_galerkin);
         split_prec->SetCurlOperator(A_curl_mat.Ptr());
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::coarse_only);
         run_gmres("yee_curl_galerkin", split_prec.get());
      }

      // 5. Additive: Yee curl-curl Galerkin + block-Jacobi
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_curl_galerkin);
         split_prec->SetCurlOperator(A_curl_mat.Ptr());
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::additive);
         run_gmres("gal+jacobi additive", split_prec.get());
      }

      // 6. Multiplicative: Yee curl-curl Galerkin first, then block-Jacobi
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_curl_galerkin);
         split_prec->SetCurlOperator(A_curl_mat.Ptr());
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::multiplicative);
         run_gmres("gal+jacobi mult", split_prec.get());
      }

      // 7. Multiplicative: independent Yee + block-Jacobi
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_curl_independent);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::multiplicative);
         run_gmres("indep+jacobi mult", split_prec.get());
      }

      // ── Full complex Galerkin Pi^T A_PML Pi (key experiment) ───────────────
      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_full_galerkin);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::coarse_only);
         run_gmres("full_gal coarse", split_prec.get());
      }

      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_full_galerkin);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::additive);
         run_gmres("full_gal+jac add", split_prec.get());
      }

      {
         split_prec->SetCoarseMode(pml_split::SplitPMLPreconditioner::CoarseMode::yee_full_galerkin);
         split_prec->SetCombineMode(pml_split::SplitPMLPreconditioner::CombineMode::multiplicative);
         run_gmres("full_gal+jac mult", split_prec.get());
      }
   }

   // ═══════════════════════════════════════════════════════════════════════════
   // 3-level p-auxiliary preconditioner:
   //   Yee/FDFD --Pi_{Y->1}--> o=1 IGA --I_1^p--> o=p IGA + smoother
   //
   // Exps:
   //   1. Baseline: zero init, unpreconditioned
   //   2. Yee -> o=1 -> o=p initial guess (warm-start)
   //   3. p-multigrid: dense LU exact solve on A_1 = P^T A_p P
   //   4. p-multigrid: Yee-assisted coarse (A_Y^{-1} via Pi chain)
   // ═══════════════════════════════════════════════════════════════════════════
   if (run_pmg && order > 1)
   {
      if (myid == 0)
      {
         cout << "\n" << string(70, '=') << "\n";
         cout << "  3-level p-auxiliary: Yee -> o=1 -> o=" << order << "\n";
         cout << string(70, '=') << "\n";
      }

      // ── 1. Baseline ──────────────────────────────────────────────────
      if (myid == 0) cout << "\n  [1] Baseline (zero init, no prec)\n";
      run_gmres("none", nullptr);

      // ── 2. Yee->o=1->o=p initial guess ───────────────────────────────
      if (myid == 0) cout << "\n  [2] Yee/FDFD -> o=1 -> o=p initial guess\n";
      {
         auto init_prec = std::make_unique<pml_split::PLevelGalerkinPreconditioner>(
            *fespace, 1);
         init_prec->SetOperator(*A);
         init_prec->SetCoarseSolveMode(
            pml_split::PLevelGalerkinPreconditioner::CoarseSolveMode::edge_yee);
         init_prec->SetCoarseGeometry(*geom);
         init_prec->SetCoarseYeeGrid(yee_grid);
         init_prec->SetWaveNumber(omega);
         init_prec->SetEpsFunc(eps_fn);
         init_prec->SetCombineMode(
            pml_split::PLevelGalerkinPreconditioner::CombineMode::coarse_only);

         // Force build; then use Mult(B, x0) to get one-shot coarse correction
         mfem::Vector dummy_r(2*fespace->GetTrueVSize());
         dummy_r = 0.0;
         init_prec->Mult(dummy_r, dummy_r);

         mfem::Vector x0(B.Size());
         x0 = 0.0;
         init_prec->Mult(B, x0); // x0 = P * Pi1 * A_Y^{-1} * Pi1^T * P^T * B

         run_gmres("yee_init+none", nullptr, &x0);
      }

      // ── 3. p-multigrid: exact coarse (dense LU on A_1) ───────────────
      if (myid == 0) cout << "\n  [3] p-multigrid: exact A_1 coarse (dense LU)\n";
      {
         auto p_prec = std::make_unique<pml_split::PLevelGalerkinPreconditioner>(
            *fespace, 1);
         p_prec->SetOperator(*A);
         p_prec->SetCoarseSolveMode(
            pml_split::PLevelGalerkinPreconditioner::CoarseSolveMode::dense_lu);
         p_prec->SetCombineMode(
            pml_split::PLevelGalerkinPreconditioner::CombineMode::multiplicative);
         run_gmres("p1_denseLU+jac mult", p_prec.get());
      }

      // ── 4. p-multigrid: Yee-assisted coarse ──────────────────────────
      if (myid == 0) cout << "\n  [4] p-multigrid: Yee-assisted coarse (A_Y^{-1})\n";
      {
         auto p_prec = std::make_unique<pml_split::PLevelGalerkinPreconditioner>(
            *fespace, 1);
         p_prec->SetOperator(*A);
         p_prec->SetCoarseSolveMode(
            pml_split::PLevelGalerkinPreconditioner::CoarseSolveMode::edge_yee);
         p_prec->SetCoarseGeometry(*geom);
         p_prec->SetCoarseYeeGrid(yee_grid);
         p_prec->SetWaveNumber(omega);
         p_prec->SetEpsFunc(eps_fn);

         // coarse only (one-shot Yee correction)
         p_prec->SetCombineMode(
            pml_split::PLevelGalerkinPreconditioner::CombineMode::coarse_only);
         run_gmres("p1_yee coarse", p_prec.get());

         // multiplicative: Yee correction + block Jacobi
         p_prec->SetCombineMode(
            pml_split::PLevelGalerkinPreconditioner::CombineMode::multiplicative);
         run_gmres("p1_yee+jac mult", p_prec.get());
      }

      if (myid == 0) cout << string(70, '=') << "\n";
   }

   // ── Cleanup ──────────────────────────────────────────────────────────
   if (myid == 0) { cout << string(70, '=') << "\n"; }

   delete fespace;
   // ParFiniteElementSpace takes ownership of a non-null NURBSExtension.
   delete fec;
   delete pmesh;
   delete pml;
   return 0;
}

// ─── PML class implementation ─────────────────────────────────────────────────

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
      dom_bdr(i, 0)      = pmin(i);
      dom_bdr(i, 1)      = pmax(i);
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

// ─── Source and solution functions ────────────────────────────────────────────

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

// ─── PML coefficient functions ────────────────────────────────────────────────

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
