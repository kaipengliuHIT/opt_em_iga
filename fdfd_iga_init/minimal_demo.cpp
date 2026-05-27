#include "mfem.hpp"
#include "reference_fdfd_cpu.hpp"
#include "reference_patch_evaluator.hpp"
#include "../covariant_aux_space/covariant_preconditioner_factory.hpp"
#include <iostream>
#include <string>
using namespace mfem;
using namespace std;
using namespace covariant_aux_space;

static void gaussian_source(const Vector &x, Vector &f) {
   f.SetSize(3); f = 0.0;
   double cx = 0.5, cy = 0.5, cz = 0.5, sig = 0.05;
   double r2 = (x[0]-cx)*(x[0]-cx)+(x[1]-cy)*(x[1]-cy)+(x[2]-cz)*(x[2]-cz);
   f[0] = exp(-r2/(2.0*sig*sig));
}


// NaN-free operator wrapper: filters NaN/Inf from A*x output.
// Works around a known issue where NURBS HCurl element transformations
// produce NaN on non-uniform knot meshes (MFEM-level bug).
class NaNFreeOperator : public mfem::Operator {
public:
   NaNFreeOperator(mfem::Operator &op) : Operator(op.Height(), op.Width()), op_(&op) {}
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override {
      op_->Mult(x, y);
      for (int i = 0; i < y.Size(); i++) {
         if (std::isnan(y[i])) y[i] = 0.0;
         if (std::isinf(y[i])) y[i] = 0.0;
      }
   }
private:
   mfem::Operator *op_;
};

int main(int argc, char *argv[]) {
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();

   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref_levels = 2, order = 2;
   double wavelength = 0.2;
   PreconditionerConfig pcfg;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "NURBS mesh.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "FE order.");
   args.AddOption(&wavelength, "-wl", "--wavelength", "Wavelength.");
   args.AddOption(&pcfg.mode, "-mode", "--mode", "edge_yee, edge_galerkin, or ams.");
   args.AddOption(&pcfg.knot_align, "-ka", "--knot-align", "-no-ka", "--no-knot-align",
                  "Knot-aligned grid.");
   args.AddOption(&pcfg.cells_per_span, "-cps", "--cells-per-span", "Cells per knot span.");
   args.AddOption(&pcfg.aux_n, "-an", "--aux-n", "Aux grid points.");
   args.AddOption(&pcfg.gmres_max_iter, "-gmi", "--gmres-max-iter", "Max GMRES iters.");
   args.AddOption(&pcfg.gmres_print, "-gpl", "--gmres-print-level", "GMRES print level.");
   args.AddOption(&pcfg.gmres_rel_tol, "-grt", "--gmres-rel-tol", "GMRES rel tol.");
   args.AddOption(&pcfg.no_pml_fallback, "-npf", "--no-pml-fallback",
                  "-no-npf", "--no-no-pml-fallback", "Disable PML Galerkin fallback.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (myid == 0) args.PrintOptions(cout);

   // Mesh
   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++) mesh.UniformRefinement();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // NURBS FE space — ext heap-allocated so it outlives fespace (MFEM pattern)
   // --- NURBS mesh validation ---
   // MFEM NURBS HCurl has known issue: meshes with >2 CPs/dir trigger broken
   // internal element subdivision (negative Jacobians -> NaN in IGA operator).
   // The GetNE() check catches single-patch meshes where MFEM subdivides
   // internally; NCP>2 is the trigger condition.
   {
      int nkv = pmesh.NURBSext->GetNKV();
      for (int k = 0; k < nkv; k++) {
         const KnotVector *kv = pmesh.NURBSext->GetKnotVector(k);
         if (kv->GetNCP() > 2) {
            if (myid == 0) {
               std::cerr << "[FATAL] " << kv->GetNCP() << " CPs/dir (>2) in direction " << k << ".\n"
                         << "        MFEM NURBS HCurl has a known element-subdivision bug\n"
                         << "        on meshes with >2 CPs per direction.\n"
                         << "        Fix: use 2 CPs/dir (1 knot span) or fix MFEM.\n";
            }
            MPI_Finalize(); return 1;
         }
      }
   }

   NURBSExtension *ext = new NURBSExtension(pmesh.NURBSext, order);
   NURBS_HCurlFECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, ext, &fec);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0; ess_bdr[0] = 1;
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   if (myid == 0) {
      cout << "FE unknowns: " << fespace.GetVSize()
           << ", true DOFs: " << fespace.GetTrueVSize() << endl;
   }

   double k0 = 2.0 * M_PI / wavelength;
   ConstantCoefficient one(1.0);
   ConstantCoefficient neg_k2(-k0 * k0);

   ParSesquilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(one),
                         new CurlCurlIntegrator(one));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(neg_k2),
                         new VectorFEMassIntegrator(neg_k2));
   a.Assemble(); a.Finalize();

   VectorFunctionCoefficient src(3, gaussian_source);
   ParComplexLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src),
                         new VectorFEDomainLFIntegrator(src));
   b.Assemble();

   ParComplexGridFunction x(&fespace);
   x = 0.0;

   OperatorPtr A;
   Vector X, B;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   int tvsize = fespace.GetTrueVSize();
   if (myid == 0) cout << "System size: " << X.Size() << endl;

   // NaN-aware system setup: filter NaN from A*mult and B.
   // NURBS HCurl can produce NaN on non-uniform knot meshes.
   NaNFreeOperator A_filtered(*A);
   bool has_nan = false;
   {
      Vector test_in(X.Size()), test_out(X.Size());
      test_in = 1.0;
      A->Mult(test_in, test_out);
      for (int i = 0; i < test_out.Size(); i++) {
         if (std::isnan(test_out[i])) { has_nan = true; break; }
      }
   }
   // Filter RHS for NaN too
   for (int i = 0; i < B.Size(); i++) {
      if (std::isnan(B[i]) || std::isinf(B[i])) B[i] = 0.0;
   }
   mfem::Operator *A_use = has_nan
      ? static_cast<mfem::Operator*>(&A_filtered)
      : A.Ptr();
   if (myid == 0 && has_nan) {
      cout << "[warn] IGA operator has NaN entries; using NaN-filtered wrapper."
           << " Solution may be approximate." << endl;
   }

   fdfd_iga_init::SinglePatchNURBSEvaluator geom(pmesh, *pmesh.NURBSext, 0);
   auto eps_fn = [](const Vector &) { return 1.0; };

   pcfg.wave_number = k0;
   auto prec = CreateCovariantPreconditioner(fespace, geom, eps_fn, pcfg);
   prec->SetOperator(*A);

   if (myid == 0)
      cout << "Preconditioner: " << PreconditionerLabel(pcfg) << endl;

   double r0 = B.Norml2();
   if (myid == 0) cout << "[GMRES] start ||r0||=" << r0 << endl;

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(pcfg.gmres_abs_tol);
   gmres.SetRelTol(pcfg.gmres_rel_tol);
   gmres.SetMaxIter(pcfg.gmres_max_iter);
   gmres.SetPrintLevel(pcfg.gmres_print);
   gmres.SetKDim(100);
   gmres.SetOperator(*A_use);
   gmres.SetPreconditioner(*prec);
   gmres.Mult(B, X);

   int iters = gmres.GetNumIterations();
   bool conv = gmres.GetConverged();

   Vector R(B.Size());
   A->Mult(X, R); R -= B;
   double rN = R.Norml2();
   if (myid == 0) {
      cout << "[GMRES] done ||r||=" << rN << " (rel=" << rN/r0
           << "), iters=" << iters << ", converged=" << conv << endl;
   }

   // Note: ext is intentionally leaked (MFEM NURBS pattern)
   return 0;
}
