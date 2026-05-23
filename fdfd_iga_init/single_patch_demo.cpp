#include "mfem.hpp"
#include "reference_fdfd_cpu.hpp"
#include "reference_initial_guess.hpp"
#include "../covariant_aux_space/covariant_reference_preconditioner.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace fdfd_iga_init;

namespace
{

class PhysicalGaussianSource
{
public:
   PhysicalGaussianSource(double cx, double cy, double cz, double sigma)
      : cx_(cx), cy_(cy), cz_(cz), sigma_(sigma) {}

   void operator()(const Vector &x, Vector &f) const
   {
      f.SetSize(3);
      const double dx = x[0] - cx_;
      const double dy = x[1] - cy_;
      const double dz = x[2] - cz_;
      const double amp = std::exp(-(dx * dx + dy * dy + dz * dz) / (sigma_ * sigma_));
      f = 0.0;
      f[1] = amp;
   }

private:
   double cx_, cy_, cz_, sigma_;
};

enum class SourceMode
{
   gaussian_y,
   gaussian_x,
   dipole_y,
   trig_mix
};

enum class EpsilonMode
{
   constant,
   layered_x,
   gaussian_core
};

SourceMode ParseSourceMode(const std::string &mode)
{
   if (mode == "gaussian_y") { return SourceMode::gaussian_y; }
   if (mode == "gaussian_x") { return SourceMode::gaussian_x; }
   if (mode == "dipole_y") { return SourceMode::dipole_y; }
   if (mode == "trig_mix") { return SourceMode::trig_mix; }
   MFEM_ABORT("Unknown source mode. Use gaussian_y, gaussian_x, dipole_y, or trig_mix.");
}

EpsilonMode ParseEpsilonMode(const std::string &mode)
{
   if (mode == "constant") { return EpsilonMode::constant; }
   if (mode == "layered_x") { return EpsilonMode::layered_x; }
   if (mode == "gaussian_core") { return EpsilonMode::gaussian_core; }
   MFEM_ABORT("Unknown epsilon mode. Use constant, layered_x, or gaussian_core.");
}

double EvaluateEpsilon(EpsilonMode mode, const Vector &x, double base_eps)
{
   switch (mode)
   {
      case EpsilonMode::constant:
         return base_eps;
      case EpsilonMode::layered_x:
         return (x[0] < 0.5) ? base_eps : (2.0 * base_eps);
      case EpsilonMode::gaussian_core:
      {
         const double dx = x[0] - 0.5;
         const double dy = x[1] - 0.5;
         const double dz = x[2] - 0.5;
         const double r2 = dx * dx + dy * dy + dz * dz;
         return base_eps * (1.0 + 1.5 * std::exp(-r2 / 0.04));
      }
   }
   return base_eps;
}

void EvaluatePhysicalSource(SourceMode mode, const Vector &x, Vector &f)
{
   f.SetSize(3);
   f = 0.0;

   const double cx = 0.5;
   const double cy = 0.5;
   const double cz = 0.5;
   const double sigma = 0.12;
   const double dx = x[0] - cx;
   const double dy = x[1] - cy;
   const double dz = x[2] - cz;
   const double r2 = dx * dx + dy * dy + dz * dz;
   const double g = std::exp(-r2 / (sigma * sigma));

   switch (mode)
   {
      case SourceMode::gaussian_y:
         f[1] = g;
         break;
      case SourceMode::gaussian_x:
         f[0] = g;
         break;
      case SourceMode::dipole_y:
         f[1] = dy * g / (sigma * sigma);
         break;
      case SourceMode::trig_mix:
         f[0] = std::sin(2.0 * M_PI * x[1]) * std::sin(M_PI * x[2]);
         f[1] = std::sin(2.0 * M_PI * x[0]) * std::sin(M_PI * x[2]);
         f[2] = 0.5 * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
         break;
   }
}

struct SolveStats
{
   int iters = 0;
   bool converged = false;
};

struct CaseConfig
{
   int order = 2;
   int ref_levels = 2;
   int fd_n = 41;
   int aux_n = 7;
   double wavelength = 0.2;
   double eps_r = 8.0;
   bool save_output = false;
   int gmres_print = 1;
   std::string proto_mode = "nodal_proto";
   std::string source_mode = "gaussian_y";
   std::string epsilon_mode = "constant";
   bool diagnose_coarse = false;
   bool calibrate_yee_diag = false;
};

covariant_aux_space::CovariantReferencePreconditioner::PrototypeMode
ParsePrototypeMode(const std::string &mode)
{
   using Mode = covariant_aux_space::CovariantReferencePreconditioner::PrototypeMode;
   if (mode == "nodal_proto") { return Mode::nodal_proto; }
   if (mode == "edge_galerkin_proto") { return Mode::edge_galerkin_proto; }
   if (mode == "edge_yee_proto") { return Mode::edge_yee_proto; }
   MFEM_ABORT("Unknown prototype mode. Use nodal_proto, edge_galerkin_proto, or edge_yee_proto.");
}

std::vector<int> ParseIntList(const std::string &csv)
{
   std::vector<int> vals;
   std::stringstream ss(csv);
   std::string item;
   while (std::getline(ss, item, ','))
   {
      if (!item.empty()) { vals.push_back(std::stoi(item)); }
   }
   return vals;
}

SolveStats RunGmresSolve(ParSesquilinearForm &a,
                         const Array<int> &ess_tdofs,
                         ParComplexLinearForm &b,
                         ParComplexGridFunction &x,
                         int print_level,
                         const char *label = nullptr,
                         bool scale_initial_guess = false,
                         mfem::Solver *precond = nullptr)
{
   OperatorPtr A;
   Vector X, B;
   a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

   if (scale_initial_guess)
   {
      Vector AX(X.Size());
      A->Mult(X, AX);
      const double denom = mfem::InnerProduct(MPI_COMM_WORLD, AX, AX);
      if (denom > 0.0)
      {
         const double numer = mfem::InnerProduct(MPI_COMM_WORLD, AX, B);
         const double alpha = numer / denom;
         X *= alpha;
      }
   }

   Vector AX0(X.Size());
   A->Mult(X, AX0);
   AX0 *= -1.0;
   AX0 += B;
   const double r0 = AX0.Norml2();
   const double bnorm = B.Norml2();

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetOperator(*A);
   if (precond)
   {
      precond->SetOperator(*A);
      gmres.SetPreconditioner(*precond);
   }
   gmres.SetPrintLevel(0);
   gmres.SetKDim(100);
   gmres.SetMaxIter(800);
   gmres.SetRelTol(1e-8);
   gmres.SetAbsTol(0.0);
   gmres.Mult(B, X);

   Vector AX1(X.Size());
   A->Mult(X, AX1);
   AX1 *= -1.0;
   AX1 += B;
   const double r1 = AX1.Norml2();

   if (print_level > 0 && Mpi::WorldRank() == 0)
   {
      std::cout << "[GMRES] " << (label ? label : "solve")
                << " start ||r0||=" << r0;
      if (bnorm > 0.0)
      {
         std::cout << " (rel=" << (r0 / bnorm) << ")";
      }
      std::cout << '\n'
                << "[GMRES] " << (label ? label : "solve")
                << " done  ||rN||=" << r1;
      if (bnorm > 0.0)
      {
         std::cout << " (rel=" << (r1 / bnorm) << ")";
      }
      std::cout << ", iters=" << gmres.GetNumIterations()
                << ", converged=" << gmres.GetConverged()
                << std::endl;
   }

   a.RecoverFEMSolution(X, b, x);

   SolveStats stats;
   stats.iters = gmres.GetNumIterations();
   stats.converged = gmres.GetConverged();
   return stats;
}

struct CaseResult
{
   int ndofs = 0;
   int true_vsize = 0;
   int aux_dofs = 0;
   SolveStats zero;
   SolveStats fdfd;
   SolveStats precond;
};

int AuxiliaryDofsForGrid(int aux_n)
{
   return 3 * std::max(0, aux_n - 2) * std::max(0, aux_n - 2) * std::max(0, aux_n - 2);
}

int AuxiliaryDofsForGrid(int aux_n, const std::string &proto_mode)
{
   if (proto_mode == "nodal_proto")
   {
      return 3 * std::max(0, aux_n - 2) * std::max(0, aux_n - 2) * std::max(0, aux_n - 2);
   }
   const int interior = std::max(0, aux_n - 2);
   const int edges = std::max(0, aux_n - 1);
   return edges * interior * interior * 3;
}

CaseResult RunCase(const char *mesh_file, const CaseConfig &cfg)
{
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < cfg.ref_levels; l++) { mesh->UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   NURBS_HCurlFECollection fec(cfg.order, 3);
   NURBSExtension *ext = new NURBSExtension(pmesh.NURBSext, cfg.order);
   ParFiniteElementSpace fespace(&pmesh, ext, &fec);

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdofs;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

   const double k0 = 2.0 * M_PI / cfg.wavelength;
   ConstantCoefficient muinv(1.0);
   const EpsilonMode eps_mode = ParseEpsilonMode(cfg.epsilon_mode);
   auto eps_eval = [eps_mode, eps = cfg.eps_r](const Vector &x)
   {
      return EvaluateEpsilon(eps_mode, x, eps);
   };
   FunctionCoefficient neg_eps([k0, &eps_eval](const Vector &x)
   {
      return -k0 * k0 * eps_eval(x);
   });

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   ParSesquilinearForm a(&fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(neg_eps), nullptr);
   a.Assemble();

   SinglePatchNURBSEvaluator geom(pmesh, *pmesh.NURBSext, 0);
   PhysicalGaussianSource phys_src(0.5, 0.5, 0.5, 0.12);
   const SourceMode src_mode = ParseSourceMode(cfg.source_mode);
   auto src_re_fn = [src_mode, &phys_src](const Vector &x, Vector &f)
   {
      if (src_mode == SourceMode::gaussian_y)
      {
         phys_src(x, f);
         return;
      }
      EvaluatePhysicalSource(src_mode, x, f);
   };
   VectorFunctionCoefficient src_re(3, src_re_fn);
   Vector zero_vec(3);
   zero_vec = 0.0;
   VectorConstantCoefficient zero_im(zero_vec);

   ParComplexLinearForm b(&fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_re),
                         new VectorFEDomainLFIntegrator(zero_im));
   b.Assemble();

   ReferenceFDFDCPU init_solver(geom);
   init_solver.SetGrid({cfg.fd_n, cfg.fd_n, cfg.fd_n});
   init_solver.SetWaveNumber(k0);
   init_solver.SetMaxIterations(300);
   init_solver.SetDamping(0.7);
   init_solver.SetMassShift(0.1);

   auto eps_fn = [eps_eval](const Vector &x) { return eps_eval(x); };
   auto src_re_ref_fn = [&geom, &phys_src, src_mode](const Vector &xi, Vector &f)
   {
      Vector x_phys, f_phys(3);
      DenseMatrix J, invJ;
      geom.EvalGeometry(xi, x_phys, J);
      if (src_mode == SourceMode::gaussian_y)
      {
         phys_src(x_phys, f_phys);
      }
      else
      {
         EvaluatePhysicalSource(src_mode, x_phys, f_phys);
      }
      invJ = J;
      invJ.Invert();
      f.SetSize(3);
      invJ.Mult(f_phys, f);
      f *= J.Det();
   };
   auto src_im_fn = [](const Vector &, Vector &f)
   {
      f.SetSize(3);
      f = 0.0;
   };

   SampledReferenceField ref_field =
      init_solver.Solve(eps_fn, src_re_ref_fn, src_im_fn);

   ParComplexGridFunction x_zero(&fespace);
   x_zero = 0.0;
   ParComplexGridFunction x_fdfd(&fespace);
   x_fdfd = 0.0;
   ParComplexGridFunction x_prec(&fespace);
   x_prec = 0.0;

   ReferenceFieldProjector projector(fespace, ref_field);
   projector.Project(x_fdfd);

   covariant_aux_space::CovariantReferencePreconditioner aux_prec(fespace, geom, eps_fn);
   aux_prec.SetPrototypeMode(ParsePrototypeMode(cfg.proto_mode));
   aux_prec.SetYeeDiagonalCalibration(cfg.calibrate_yee_diag);
   aux_prec.SetGrid({cfg.aux_n, cfg.aux_n, cfg.aux_n});
   aux_prec.SetWaveNumber(k0);
   aux_prec.SetMaxIterations(300);
   aux_prec.SetDamping(0.7);
   aux_prec.SetMassShift(0.1);
   if (cfg.diagnose_coarse && cfg.proto_mode != "nodal_proto")
   {
      OperatorPtr Adiag;
      Vector Xdiag, Bdiag;
      a.FormLinearSystem(ess_tdofs, x_prec, b, Adiag, Xdiag, Bdiag);
      aux_prec.SetOperator(*Adiag);
      if (Mpi::WorldRank() == 0)
      {
         aux_prec.PrintCoarseOperatorDiagnostics(std::cout);
      }
   }

   CaseResult result;
   result.ndofs = fespace.GetNDofs();
   result.true_vsize = fespace.GetTrueVSize();
   result.aux_dofs = AuxiliaryDofsForGrid(cfg.aux_n, cfg.proto_mode);
   result.zero = RunGmresSolve(a, ess_tdofs, b, x_zero, cfg.gmres_print,
                               "zero_init");
   result.fdfd = RunGmresSolve(a, ess_tdofs, b, x_fdfd, cfg.gmres_print,
                               "fdfd_init", true);
   result.precond = RunGmresSolve(a, ess_tdofs, b, x_prec, cfg.gmres_print,
                                  "aux_prec", false, &aux_prec);

   if (cfg.save_output)
   {
      ParaViewDataCollection dc_zero("ParaView/fdfd_iga_init_zero", &pmesh);
      dc_zero.SetPrefixPath(".");
      dc_zero.RegisterField("E_re", &x_zero.real());
      dc_zero.RegisterField("E_im", &x_zero.imag());
      dc_zero.SetHighOrderOutput(true);
      dc_zero.Save();

      ParaViewDataCollection dc_fdfd("ParaView/fdfd_iga_init_fdfd", &pmesh);
      dc_fdfd.SetPrefixPath(".");
      dc_fdfd.RegisterField("E_re", &x_fdfd.real());
      dc_fdfd.RegisterField("E_im", &x_fdfd.imag());
      dc_fdfd.SetHighOrderOutput(true);
      dc_fdfd.Save();
   }

   return result;
}

void PrintCaseHeader(const CaseConfig &cfg, const CaseResult &result)
{
   if (Mpi::WorldRank() != 0) { return; }
   std::cout << "[single_patch_demo] ndofs=" << result.ndofs
             << ", true_vsize=" << result.true_vsize << '\n'
             << "[single_patch_demo] order=" << cfg.order
             << ", ref_levels=" << cfg.ref_levels
             << ", fd_n=" << cfg.fd_n
             << ", aux_n=" << cfg.aux_n
             << ", proto_mode=" << cfg.proto_mode
             << ", calibrate_yee_diag=" << cfg.calibrate_yee_diag
             << ", source_mode=" << cfg.source_mode
             << ", epsilon_mode=" << cfg.epsilon_mode
             << ", aux_dofs=" << result.aux_dofs
             << ", aux_ratio=" << double(result.aux_dofs) / double(result.true_vsize)
             << ", wavelength=" << cfg.wavelength
             << ", eps_r=" << cfg.eps_r << std::endl;
}

void PrintCaseSummary(const CaseResult &result)
{
   if (Mpi::WorldRank() != 0) { return; }
   std::cout << "Zero-initialized GMRES iterations: "
             << result.zero.iters
             << ", converged=" << result.zero.converged << '\n'
             << "FDFD-initialized GMRES iterations: "
             << result.fdfd.iters
             << ", converged=" << result.fdfd.converged << '\n'
             << "Covariant-aux-preconditioned GMRES iterations: "
             << result.precond.iters
             << ", converged=" << result.precond.converged << std::endl;
}

void RunScan(const char *mesh_file,
             const CaseConfig &base_cfg,
             const std::vector<int> &orders,
             const std::vector<int> &refines,
             const std::vector<int> &fd_ns,
             const std::vector<int> &aux_ns,
             double max_aux_ratio)
{
   if (Mpi::WorldRank() == 0)
   {
      std::cout << "mesh,proto_mode,source_mode,epsilon_mode,order,ref_levels,fd_n,aux_n,true_vsize,aux_dofs,aux_ratio,"
                   "zero_iters,fdfd_iters,auxprec_iters,"
                   "zero_conv,fdfd_conv,auxprec_conv,aux_gain,status"
                << std::endl;
   }

   for (int order : orders)
   {
      for (int ref_levels : refines)
      {
         for (int fd_n : fd_ns)
         {
            for (int aux_n : aux_ns)
            {
               CaseConfig cfg = base_cfg;
               cfg.order = order;
               cfg.ref_levels = ref_levels;
               cfg.fd_n = fd_n;
               cfg.aux_n = aux_n;
               cfg.save_output = false;
               cfg.gmres_print = 0;

               const CaseResult result = RunCase(mesh_file, cfg);
               const double aux_ratio =
                  double(result.aux_dofs) / double(result.true_vsize);
               const double aux_gain =
                  (result.precond.iters > 0) ?
                  double(result.zero.iters) / double(result.precond.iters) : 0.0;
               const char *status =
                  (max_aux_ratio > 0.0 && aux_ratio > max_aux_ratio) ? "over_ratio" : "ok";
               if (max_aux_ratio > 0.0 && aux_ratio > max_aux_ratio)
               {
                  if (Mpi::WorldRank() == 0)
                  {
                     std::cout << mesh_file << ','
                               << cfg.proto_mode << ','
                               << cfg.source_mode << ','
                               << cfg.epsilon_mode << ','
                               << order << ','
                               << ref_levels << ','
                               << fd_n << ','
                               << aux_n << ','
                               << result.true_vsize << ','
                               << result.aux_dofs << ','
                               << aux_ratio << ','
                               << result.zero.iters << ','
                               << result.fdfd.iters << ','
                               << result.precond.iters << ','
                               << result.zero.converged << ','
                               << result.fdfd.converged << ','
                               << result.precond.converged << ','
                               << aux_gain << ','
                               << status << std::endl;
                     }
                  continue;
               }
               if (Mpi::WorldRank() == 0)
               {
                  std::cout << mesh_file << ','
                            << cfg.proto_mode << ','
                            << cfg.source_mode << ','
                            << cfg.epsilon_mode << ','
                            << order << ','
                            << ref_levels << ','
                            << fd_n << ','
                            << aux_n << ','
                            << result.true_vsize << ','
                            << result.aux_dofs << ','
                            << aux_ratio << ','
                            << result.zero.iters << ','
                            << result.fdfd.iters << ','
                            << result.precond.iters << ','
                            << result.zero.converged << ','
                            << result.fdfd.converged << ','
                            << result.precond.converged << ','
                            << aux_gain << ','
                            << status << std::endl;
               }
            }
         }
      }
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   Device device("cpu");
   if (Mpi::WorldRank() == 0)
   {
      device.Print();
   }

   const char *mesh_file = "/mnt/d/code/code_opt_em/mfem/data/cube-nurbs.mesh";
   CaseConfig cfg;
   bool scan = false;
   std::string scan_orders = "2,3";
   std::string scan_refines = "1,2";
   std::string scan_fd = "31,41";
   std::string scan_aux = "5,7";
   double scan_max_aux_ratio = 0.8;
   std::string scan_sources = "gaussian_y";
   std::string scan_eps = "constant";
   std::string scan_meshes = mesh_file;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Single-patch NURBS mesh.");
   args.AddOption(&cfg.order, "-o", "--order", "NURBS H(curl) order.");
   args.AddOption(&cfg.ref_levels, "-r", "--refine", "Uniform refinement levels.");
   args.AddOption(&cfg.fd_n, "-n", "--fd-n", "Reference-grid points per direction.");
   args.AddOption(&cfg.aux_n, "-an", "--aux-n",
                  "Auxiliary-space grid points per direction.");
   args.AddOption(&cfg.save_output, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable ParaView output.");
   args.AddOption(&cfg.wavelength, "-wl", "--wavelength", "Wavelength.");
   args.AddOption(&cfg.eps_r, "-eps", "--epsilon", "Relative permittivity.");
   args.AddOption(&cfg.proto_mode, "-pm", "--proto-mode",
                  "Auxiliary prototype mode: nodal_proto, edge_galerkin_proto, or edge_yee_proto.");
   args.AddOption(&cfg.source_mode, "-sm", "--source-mode",
                  "Source mode: gaussian_y, gaussian_x, dipole_y, or trig_mix.");
   args.AddOption(&cfg.epsilon_mode, "-em", "--epsilon-mode",
                  "Permittivity mode: constant, layered_x, or gaussian_core.");
   args.AddOption(&cfg.diagnose_coarse, "-diag", "--diagnose-coarse",
                  "-no-diag", "--no-diagnose-coarse",
                  "Print coarse-operator diagnostics for edge-based prototypes.");
   args.AddOption(&cfg.calibrate_yee_diag, "-cyd", "--calibrate-yee-diag",
                  "-no-cyd", "--no-calibrate-yee-diag",
                  "Scale edge_yee_proto coarse operator by matching diagonal mean to the Galerkin coarse operator.");
   args.AddOption(&scan, "-scan", "--scan", "-no-scan", "--no-scan",
                  "Run a parameter sweep instead of a single case.");
   args.AddOption(&scan_orders, "-scan-orders", "--scan-orders",
                  "Comma-separated H(curl) orders for scan mode.");
   args.AddOption(&scan_refines, "-scan-refines", "--scan-refines",
                  "Comma-separated refinement levels for scan mode.");
   args.AddOption(&scan_fd, "-scan-fd", "--scan-fd",
                  "Comma-separated reference FDFD grid sizes for scan mode.");
   args.AddOption(&scan_aux, "-scan-aux", "--scan-aux",
                  "Comma-separated auxiliary grid sizes for scan mode.");
   args.AddOption(&scan_sources, "-scan-sources", "--scan-sources",
                  "Comma-separated source modes for scan mode.");
   args.AddOption(&scan_eps, "-scan-eps", "--scan-eps",
                  "Comma-separated epsilon modes for scan mode.");
   args.AddOption(&scan_meshes, "-scan-meshes", "--scan-meshes",
                  "Comma-separated mesh paths for scan mode.");
   args.AddOption(&scan_max_aux_ratio, "-scan-max-ratio", "--scan-max-ratio",
                  "Skip scan cases whose auxiliary-space dof ratio exceeds this threshold. "
                  "Use a non-positive value to disable skipping.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::WorldRank() == 0) { args.PrintUsage(std::cout); }
      return 1;
   }

   if (scan)
   {
      const auto orders = ParseIntList(scan_orders);
      const auto refines = ParseIntList(scan_refines);
      const auto fd_ns = ParseIntList(scan_fd);
      const auto aux_ns = ParseIntList(scan_aux);
      std::stringstream ss_mesh(scan_meshes);
      std::string mesh_name;
      while (std::getline(ss_mesh, mesh_name, ','))
      {
         if (mesh_name.empty()) { continue; }
         std::stringstream ss_eps(scan_eps);
         std::string eps_name;
         while (std::getline(ss_eps, eps_name, ','))
         {
            if (eps_name.empty()) { continue; }
            cfg.epsilon_mode = eps_name;
            std::stringstream ss(scan_sources);
            std::string source_name;
            while (std::getline(ss, source_name, ','))
            {
               if (source_name.empty()) { continue; }
               cfg.source_mode = source_name;
               RunScan(mesh_name.c_str(), cfg, orders, refines, fd_ns, aux_ns,
                       scan_max_aux_ratio);
            }
         }
      }
      return 0;
   }

   const CaseResult result = RunCase(mesh_file, cfg);
   PrintCaseHeader(cfg, result);
   PrintCaseSummary(result);
   return 0;
}
