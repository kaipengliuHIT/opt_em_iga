/**
 * gpu_bicgstab_demo.cpp
 *
 * Demo driver: IGA H(curl) Maxwell-PML  solved with GPU-native BiCGSTAB
 * (cuSPARSE SpMV + cuBLAS vector ops) and iga_ras preconditioner.
 *
 * Compile:
 *   Without CUDA (CPU BiCGSTAB only):
 *     make gpu_bicgstab_demo_cpu
 *
 *   With CUDA 12 (full GPU path):
 *     make gpu_bicgstab_demo_gpu   # adds -DHAVE_CUDA -lcusparse -lcublas
 *
 * Sample runs:
 *   # CPU BiCGSTAB with iga_ras  (same physics as pml_point_source_demo)
 *   ./gpu_bicgstab_demo_cpu -m /path/cube-nurbs.mesh -r 2 -o 3 -f 4.0 \
 *       -rasov 0 -rasw 0.8 -mode cpu
 *
 *   # GPU BiCGSTAB with iga_ras
 *   ./gpu_bicgstab_demo_gpu -m /path/cube-nurbs.mesh -r 2 -o 3 -f 4.0 \
 *       -rasov 0 -rasw 0.8 -mode gpu
 */

#include "mfem.hpp"
#include "gpu_bicgstab.hpp"
#include "../covariant_aux_space/iga_patch_ras_preconditioner.hpp"

#include <iostream>
#include <chrono>
#include <string>

using namespace mfem;

// ── Reuse PML setup from ex25p / pml_point_source_demo ──────────────────────
// (Minimal inline version for the demo – use the full PML headers for
//  production runs.)

static double omega_global;

void maxwell_source(const Vector &x, Vector &f)
{
   f = 0.0;
   f(2) = std::sin(omega_global * x(0));
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int rank = Mpi::WorldRank();

   const char *mesh_file  = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int   ref_levels        = 2;
   int   order             = 3;
   double freq             = 4.0;
   int   ras_overlap       = 0;
   double ras_damping      = 0.8;
   int   ras_iter          = 1;
   int   max_iters         = 500;
   double rel_tol          = 1e-7;
   int   print_every       = 20;
   const char *mode        = "cpu";  // "cpu" or "gpu"

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file,   "-m",  "--mesh",  "Mesh.");
   args.AddOption(&ref_levels,  "-r",  "--refine","Refinements.");
   args.AddOption(&order,       "-o",  "--order", "NURBS order.");
   args.AddOption(&freq,        "-f",  "--freq",  "Frequency (GHz).");
   args.AddOption(&ras_overlap, "-rasov","--ras-overlap","RAS overlap.");
   args.AddOption(&ras_damping, "-rasw", "--ras-damping","RAS damping.");
   args.AddOption(&ras_iter,    "-rasit","--ras-iters",  "RAS sweeps.");
   args.AddOption(&max_iters,   "-gmi", "--max-iters",   "BiCGSTAB max iters.");
   args.AddOption(&rel_tol,     "-grt", "--rel-tol",     "Relative tolerance.");
   args.AddOption(&print_every, "-gpl", "--print-level", "Print every N iters.");
   args.AddOption(&mode,        "-mode","--mode",
                  "Solver mode: cpu or gpu.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
   if (rank == 0)    { args.PrintOptions(std::cout); }

   const bool use_gpu = (std::string(mode) == "gpu");
#ifndef HAVE_CUDA
   if (use_gpu)
   {
      std::cerr << "[demo] Built without HAVE_CUDA; falling back to cpu.\n";
   }
#endif

   // ── Mesh ──────────────────────────────────────────────────────────────────
   Mesh serial_mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++) { serial_mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   const int dim = pmesh.Dimension();

   // ── NURBS H(curl) space ───────────────────────────────────────────────────
   FiniteElementCollection *fec = new NURBS_HCurlFECollection(order, dim);
   NURBSExtension *nurbs_ext    = new NURBSExtension(pmesh.NURBSext, order);
   ParFiniteElementSpace fespace(&pmesh, nurbs_ext, fec);
   if (rank == 0)
   {
      std::cout << "[demo] GlobalDOF=" << fespace.GlobalTrueVSize()
                << "  rank_DOF=" << fespace.GetTrueVSize()
                << "  np=" << Mpi::WorldSize() << "\n";
   }

   // ── Essential BCs ─────────────────────────────────────────────────────────
   Array<int> ess_tdof_list;
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // ── Bilinear form (simplified constant-coeff Maxwell, no PML) ─────────────
   // For full PML: copy coefficient setup from pml_point_source_demo.cpp.
   const double omega_rad = 2.0 * M_PI * freq * 1e9;
   omega_global = omega_rad;
   const double mu_inv  = 1.0 / (4.0 * M_PI * 1e-7);
   const double epsilon = 8.854187817e-12;

   ConstantCoefficient muinv_coef(mu_inv);
   ConstantCoefficient neg_eps_omega2(-(omega_rad * omega_rad) * epsilon);

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   ParSesquilinearForm a(&fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_coef), nullptr);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(neg_eps_omega2), nullptr);
   a.Assemble();

   // ── RHS ───────────────────────────────────────────────────────────────────
   VectorFunctionCoefficient src(dim, maxwell_source);
   ParComplexLinearForm b(&fespace, conv);
   b.AddDomainIntegrator(nullptr, new VectorFEDomainLFIntegrator(src));
   b.Vector::operator=(0.0);
   b.Assemble();

   // ── Form system ───────────────────────────────────────────────────────────
   OperatorPtr A;
   Vector B, X;
   ParComplexGridFunction x(&fespace);
   x = 0.0;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   if (rank == 0)
   {
      std::cout << "[demo] system_size=" << A->Height() << "\n";
   }

   // ── iga_ras preconditioner ─────────────────────────────────────────────────
   auto t0 = std::chrono::high_resolution_clock::now();

   covariant_aux_space::IGAPatchRASPreconditioner::Options ras_opts;
   ras_opts.overlap_layers = ras_overlap;
   ras_opts.damping        = ras_damping;
   ras_opts.iterations     = ras_iter;
   ras_opts.verbose        = (rank == 0);
   covariant_aux_space::IGAPatchRASPreconditioner ras_prec(*A, fespace, ras_opts);
   ras_prec.NumPatches();  // trigger build

   double setup_s = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - t0).count();
   if (rank == 0)
   {
      std::cout << "[demo] iga_ras setup: " << setup_s << " s"
                << "  patches=" << ras_prec.NumPatches()
                << "  max_patch=" << ras_prec.MaxPatchSize() << "\n";
   }

   // ── Solver options ────────────────────────────────────────────────────────
   gpu_solver::BiCGSTABOptions solver_opts;
   solver_opts.max_iters   = max_iters;
   solver_opts.rel_tol     = rel_tol;
   solver_opts.print_every = print_every;
   solver_opts.verbose     = (rank == 0);

   // ── Solve ─────────────────────────────────────────────────────────────────
   gpu_solver::BiCGSTABResult res;

#ifdef HAVE_CUDA
   if (use_gpu)
   {
      gpu_solver::GpuBiCGSTABSolver solver(*A, ras_prec, solver_opts);
      res = solver.Solve(B, X);
      if (rank == 0)
      {
         std::cout << "[demo] patch_inv gpu_MB=" << solver.GpuAllocBytes() / (1024*1024)
                   << "  (matrix stays on CPU, MPI-correct for all np)\n";
      }
   }
   else
#endif
   {
      gpu_solver::CpuBiCGSTABSolver solver(*A, &ras_prec, solver_opts);
      res = solver.Solve(B, X);
   }

   // ── True residual ─────────────────────────────────────────────────────────
   Vector Ax(X.Size()), resid(X.Size());
   A->Mult(X, Ax);
   subtract(B, Ax, resid);
   double local_sq  = resid.Norml2() * resid.Norml2();
   double global_sq = 0.0;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   double bnorm_sq_local = B.Norml2() * B.Norml2(), bnorm_sq = 0.0;
   MPI_Allreduce(&bnorm_sq_local, &bnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double true_rel = std::sqrt(global_sq) / std::sqrt(bnorm_sq);

   if (rank == 0)
   {
      std::cout << "\n[demo] ────── Result ──────\n"
                << "  mode:      " << mode << "\n"
                << "  iters:     " << res.iters << "\n"
                << "  converged: " << (res.converged ? "yes" : "no") << "\n"
                << "  final_rel: " << res.final_rel << "\n"
                << "  true_rel:  " << true_rel << "\n"
                << "  solve_s:   " << res.solve_seconds << "\n"
                << "──────────────────────────\n";
   }

   delete fec;
   return 0;
}
