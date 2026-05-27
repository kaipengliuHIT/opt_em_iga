//                                MFEM Example nurbs_ex25p -- modified for NURBS FE
//
// Compile with: make nurbs_ex25p
//
// Sample runs:  mpirun -np 4 nurbs_ex25p -m ../../data/square-nurbs.mesh
//               mpirun -np 4 nurbs_ex25p -m ../../data/square-nurbs.mesh -o 2
//               mpirun -np 4 nurbs_ex25p -m ../../data/cube-nurbs.mesh

// Description:  This example code solves a simple electromagnetic
//               wave parallel IGA(NURBS) propagation problem
//               corresponding to the second order indefinite
//               Maxwell equation
//
//                  (1/mu) * curl curl E - \omega^2 * epsilon E = f
//
//               with a Perfectly Matched Layer (PML).
//
//               The example demonstrates discretization with Nedelec finite
//               elements in 2D or 3D, as well as the use of complex-valued
//               bilinear and linear forms.
//
//               We recommend viewing Example 25p before viewing this example.

#include "mfem.hpp"
#include "fdfd_iga_init/reference_patch_evaluator.hpp"
#include "covariant_aux_space/covariant_reference_preconditioner.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <limits>

using namespace std;
using namespace mfem;


// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region in each direction
   Array2D<real_t> length;

   // Computational Domain Boundary
   Array2D<real_t> comp_dom_bdr;

   // Domain Boundary
   Array2D<real_t> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   PML(Mesh *mesh_,Array2D<real_t> length_);

   // Return Computational Domain Boundary
   Array2D<real_t> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<real_t> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *mesh_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, PML *,
                                              Vector &),
                            PML * pml_)
      : VectorCoefficient(dim), pml(pml_), Function(F)
   {}

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

class TrueResidualController : public IterativeSolverController
{
private:
   const Operator &A_;
   const Vector &B_;
   const real_t bnorm_;
   const real_t rel_tol_;
   const real_t abs_tol_;
   const int print_every_;
   const int rank_;
   mutable Vector residual_;
   int last_iter_ = -1;
   real_t last_norm_ = numeric_limits<real_t>::infinity();
   real_t last_rel_ = numeric_limits<real_t>::infinity();

public:
   TrueResidualController(const Operator &A, const Vector &B,
                          real_t bnorm, real_t rel_tol, real_t abs_tol,
                          int print_every, int rank)
      : A_(A), B_(B), bnorm_(bnorm), rel_tol_(rel_tol), abs_tol_(abs_tol),
        print_every_(print_every), rank_(rank)
   {
      residual_.SetSize(B.Size());
   }

   bool RequiresUpdatedSolution() const override { return true; }

   void Reset() override
   {
      IterativeSolverController::Reset();
      last_iter_ = -1;
      last_norm_ = numeric_limits<real_t>::infinity();
      last_rel_ = numeric_limits<real_t>::infinity();
   }

   void MonitorSolution(int it, real_t, const Vector &x, bool final) override
   {
      A_.Mult(x, residual_);
      residual_ *= -1.0;
      residual_ += B_;
      last_iter_ = it;
      last_norm_ = residual_.Norml2();
      last_rel_ = (bnorm_ > 0.0) ? last_norm_ / bnorm_ : last_norm_;

      const bool true_stop =
         (bnorm_ > 0.0) ? (last_rel_ <= rel_tol_) : (last_norm_ <= abs_tol_);
      if (true_stop)
      {
         converged = true;
      }

      if (rank_ == 0 && print_every_ > 0 &&
          (final || true_stop || it == 0 || it % print_every_ == 0))
      {
         cout << "[true-residual monitor] iter=" << it
              << " ||r||=" << last_norm_;
         if (bnorm_ > 0.0) { cout << " (rel=" << last_rel_ << ")"; }
         cout << ", true_converged=" << true_stop << '\n';
      }
   }

   int LastIter() const { return last_iter_; }
   real_t LastNorm() const { return last_norm_; }
   real_t LastRel() const { return last_rel_; }
};

template <typename T> T pow2(const T &x) { return x*x; }
void maxwell_solution(const Vector &x, vector<complex<real_t>> &Eval);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D);

Array2D<real_t> comp_domain_bdr;
Array2D<real_t> domain_bdr;

real_t mu = 1.0;
real_t epsilon = 1.0;
real_t omega;
int dim;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = 8;
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 0;
   real_t freq = 5.0;
   std::string prec_mode = "ams";
   int aux_n = 7;
   int gmres_max_iter = 10000;
   int gmres_print = 0;
   real_t gmres_rel_tol = 1e-5;
   bool diagnose_yee = false;
   bool diagnose_only = false;
   bool yee_pml = true;
   int cells_per_span = 0;
   bool knot_align = false;
   bool no_pml_fallback = false;
   bool yee_calib = true;
   real_t coarse_correction_weight = 1.0;
   real_t identity_smoother_weight = 0.0;
   real_t jacobi_smoother_weight = 0.0;
   int jacobi_smoother_iterations = 1;
   real_t block_jacobi_smoother_weight = 0.0;
   int block_jacobi_smoother_iterations = 1;
   bool true_residual_control = false;
   int true_residual_print_every = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&prec_mode, "-prec", "--preconditioner",
                  "Preconditioner: none, ams, amg, edge_galerkin, or edge_yee.");
   args.AddOption(&aux_n, "-an", "--aux-n",
                  "Auxiliary Yee grid points per direction for edge_yee.");
   args.AddOption(&gmres_max_iter, "-gmi", "--gmres-max-iter",
                  "Maximum GMRES iterations.");
   args.AddOption(&gmres_print, "-gpl", "--gmres-print-level",
                  "MFEM GMRES print level. Use 0 for summary-only output.");
   args.AddOption(&gmres_rel_tol, "-grt", "--gmres-rel-tol",
                  "GMRES relative tolerance.");
   args.AddOption(&diagnose_yee, "-dy", "--diagnose-yee",
                  "-no-dy", "--no-diagnose-yee",
                  "Print Yee coarse operator comparison against P^T A P.");
   args.AddOption(&diagnose_only, "-do", "--diagnose-only",
                  "-no-do", "--no-diagnose-only",
                  "Exit after diagnostics without running GMRES.");
   args.AddOption(&yee_pml, "-ypml", "--yee-pml",
                  "-no-ypml", "--no-yee-pml",
                  "Enable PML stretching weights inside the Yee coarse "
                  "operator. Auto-disabled if the mesh has no PML region.");
   args.AddOption(&cells_per_span, "-cps", "--cells-per-span",
                  "Cells per knot span for knot-aligned Yee grid (0=use -an).");
   args.AddOption(&knot_align, "-ka", "--knot-align",
                  "-no-ka", "--no-knot-align",
                  "Align Yee grid lines to NURBS knot positions.");
   args.AddOption(&no_pml_fallback, "-npf", "--no-pml-fallback",
                  "-no-npf", "--no-no-pml-fallback",
                  "Disable PML Galerkin fallback; force pure Yee FDFD.");
   args.AddOption(&yee_calib, "-ycal", "--yee-calib",
                  "-no-ycal", "--no-yee-calib",
                  "Calibrate Yee coarse operator diagonal mean to match the "
                  "IGA Galerkin restriction P^T A_h P (mesh-dependent fix).");
   args.AddOption(&coarse_correction_weight, "-cw", "--coarse-weight",
                  "Weight multiplying the auxiliary coarse correction "
                  "Pi Aaux^{-1} Pi^T r. Use 0 for smoother-only ablations.");
   args.AddOption(&identity_smoother_weight, "-sid", "--identity-smoother",
                  "Add beta*I to the auxiliary preconditioner output: "
                  "z = beta*r + Pi Aaux^{-1} Pi^T r.");
   args.AddOption(&jacobi_smoother_weight, "-sjac", "--jacobi-smoother",
                  "Add omega*diag(A)^{-1}r to the auxiliary preconditioner output.");
   args.AddOption(&jacobi_smoother_iterations, "-sjit", "--jacobi-smoother-iters",
                  "Number of weighted Jacobi smoothing iterations.");
   args.AddOption(&block_jacobi_smoother_weight, "-sbjac",
                  "--block-jacobi-smoother",
                  "Add omega times the inverse 2x2 real/imag diagonal block "
                  "of A to the auxiliary preconditioner output.");
   args.AddOption(&block_jacobi_smoother_iterations, "-sbjit",
                  "--block-jacobi-smoother-iters",
                  "Number of weighted 2x2 block-Jacobi smoothing iterations.");
   args.AddOption(&true_residual_control, "-trc", "--true-residual-control",
                  "-no-trc", "--no-true-residual-control",
                  "Stop GMRES using the unpreconditioned true residual norm.");
   args.AddOption(&true_residual_print_every, "-trp",
                  "--true-residual-print-every",
                  "Print true residual every N iterations when -trc is enabled "
                  "(0=only final summary).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0 )
   {
      args.PrintOptions(cout);
   }

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 3. Setup PML length
   Array2D<real_t> length(dim, 2); length = 0.25;
   PML * pml = new PML(mesh,length);
   // Angular frequency
   omega = real_t(2.0 * M_PI) * freq;

   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 4. Refine the mesh to increase the resolution.
   if (ref_levels == -1)
   {
      ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pml->SetAttributes(pmesh);

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext,
                                                              fec);
   cout <<"Number of finite element unknowns in rank "<<myid<<": "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "Number of knowns in essential BCs: "
        << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   VectorFunctionCoefficient f(dim, source);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b.Vector::operator=(0.0);
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);

   // 9. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Array<int> attr;
   Array<int> attrPML;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());
      attrPML.SetSize(pmesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (pmesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   ConstantCoefficient muinv(1_r / mu);
   ConstantCoefficient omeg(-pow2(omega) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);

   const int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
   ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   ConstantCoefficient absomeg(pow2(omega) * epsilon);
   RestrictedCoefficient restr_absomeg(absomeg,attr);

   // Define bilinear form for the preconditioner
   // This creates a Hermitian, positive definite approximation to the original
   // sesquilinear form for use in the preconditioner
   ParBilinearForm prec(fespace);
   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

   PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml);
   ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);

   PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs,pml);
   ScalarVectorProductCoefficient c2_abs(absomeg,pml_c2_abs);
   VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));

   prec.Assemble();

   // 11. Define and apply a GMRES solver for AU=B. The AMS/AMG paths follow
   //     the positive-definite PML preconditioner used by MFEM ex25p, while
   //     edge_yee uses the project auxiliary-space preconditioner.
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fespace->GetTrueVSize();
   offsets[2] = fespace->GetTrueVSize();
   offsets.PartialSum();

   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;
   const int s = (conv == ComplexOperator::HERMITIAN) ? -1 : 1;

   OperatorPtr PCOpAh;
   prec.FormSystemMatrix(ess_tdof_list, PCOpAh);
   OperatorPtr PCOpCurlAh;
   OperatorPtr PCOpMassAh;
   std::unique_ptr<ParBilinearForm> prec_curl_form;
   std::unique_ptr<ParBilinearForm> prec_mass_form;

   if (diagnose_yee)
   {
      prec_curl_form = std::make_unique<ParBilinearForm>(fespace);
      prec_curl_form->AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
      prec_curl_form->AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
      prec_curl_form->Assemble();
      prec_curl_form->FormSystemMatrix(ess_tdof_list, PCOpCurlAh);

      prec_mass_form = std::make_unique<ParBilinearForm>(fespace);
      prec_mass_form->AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));
      prec_mass_form->AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));
      prec_mass_form->Assemble();
      prec_mass_form->FormSystemMatrix(ess_tdof_list, PCOpMassAh);
   }

   BlockDiagonalPreconditioner BlockDP(offsets);
   std::unique_ptr<fdfd_iga_init::SinglePatchNURBSEvaluator> geom;
   std::unique_ptr<covariant_aux_space::CovariantReferencePreconditioner> edge_prec;

   if (prec_mode == "ams")
   {
      // Use Hypre AMS preconditioner for NURBS HCurl space
      if (myid == 0)
      {
         cout << "\n" << string(80, '-') << endl;
         cout << "Using AMS preconditioner for NURBS HCurl space..." << endl;
      }
      pc_r.reset(new HypreAMS(*PCOpAh.As<HypreParMatrix>(), fespace));
      pc_i.reset(new ScaledOperator(pc_r.get(), s));
      BlockDP.SetDiagonalBlock(0, pc_r.get());
      BlockDP.SetDiagonalBlock(1, pc_i.get());
   }
   else if (prec_mode == "amg")
   {
      // Use Hypre BoomerAMG preconditioner
      if (myid == 0)
      {
         cout << "\n" << string(80, '-') << endl;
         cout << "Using BoomerAMG preconditioner..." << endl;
      }
      HypreBoomerAMG *amg = new HypreBoomerAMG(*PCOpAh.As<HypreParMatrix>());
      amg->SetPrintLevel(0);
      pc_r.reset(amg);
      pc_i.reset(new ScaledOperator(pc_r.get(), s));
      BlockDP.SetDiagonalBlock(0, pc_r.get());
      BlockDP.SetDiagonalBlock(1, pc_i.get());
   }
   else if (prec_mode == "edge_yee" || prec_mode == "edge_galerkin")
   {
      if (myid == 0)
      {
         cout << "\n" << string(80, '-') << endl;
         cout << "Using " << prec_mode
              << " auxiliary-space preconditioner..." << endl;
         if (prec_mode == "edge_yee")
         {
            cout << "Using abs tensor-stretched PML Yee coarse operator."
                 << endl;
         }
      }
      geom = std::make_unique<fdfd_iga_init::SinglePatchNURBSEvaluator>(
         *pmesh, *pmesh->NURBSext, 0);
      std::function<double(const Vector &)> eps_fn =
         [](const Vector &) { return epsilon; };
      edge_prec =
         std::make_unique<covariant_aux_space::CovariantReferencePreconditioner>(
            *fespace, *geom, eps_fn);
      const auto mode =
         (prec_mode == "edge_galerkin") ?
         covariant_aux_space::CovariantReferencePreconditioner::PrototypeMode::edge_galerkin_proto :
         covariant_aux_space::CovariantReferencePreconditioner::PrototypeMode::edge_yee_proto;
      edge_prec->SetPrototypeMode(mode);
      if (knot_align && cells_per_span > 0)
      {
         edge_prec->SetKnotAlignGrid(true, cells_per_span);
      }
      edge_prec->SetGrid({aux_n, aux_n, aux_n});
      edge_prec->SetWaveNumber(omega * sqrt(epsilon * mu));
      edge_prec->SetCoarseCorrectionWeight(coarse_correction_weight);
      edge_prec->SetIdentitySmootherWeight(identity_smoother_weight);
      edge_prec->SetOperatorJacobiSmootherWeight(jacobi_smoother_weight);
      edge_prec->SetOperatorJacobiSmootherIterations(jacobi_smoother_iterations);
      edge_prec->SetOperatorBlockJacobiSmootherWeight(block_jacobi_smoother_weight);
      edge_prec->SetOperatorBlockJacobiSmootherIterations(block_jacobi_smoother_iterations);
      if (prec_mode == "edge_yee" && yee_calib)
      {
         edge_prec->SetYeeDiagonalCalibration(true);
      }
      const bool mesh_has_pml = (pmesh->attributes.Max() > 1);
      const bool use_yee_pml =
         (prec_mode == "edge_yee") && yee_pml && mesh_has_pml;
      if (prec_mode == "edge_yee")
      {
         if (use_yee_pml)
         {
            edge_prec->SetYeeReferencePML(true, 0.25, 5.0, 2.0);
            edge_prec->SetYeePMLGalerkinFallback(!no_pml_fallback);
            if (myid == 0)
            {
               cout << "[edge_yee] Yee PML stretching enabled; "
                    << (no_pml_fallback ? "Galerkin fallback disabled"
                                         : "Galerkin fallback enabled")
                    << "." << endl;
            }
         }
         else if (myid == 0)
         {
            cout << "[edge_yee] PML stretching disabled ("
                 << (mesh_has_pml ? "user override" : "mesh has no PML region")
                 << ")." << endl;
         }
      }
      edge_prec->SetOperator(*A);
      if (diagnose_yee && prec_mode == "edge_yee" && myid == 0)
      {
         mfem::DenseMatrix AyeeCurl, AyeeMass;
         covariant_aux_space::YeeOperatorBuilder yee_diag_builder(*geom);
         yee_diag_builder.SetGrid(edge_prec->GetYeeGrid());
         if (use_yee_pml)
         {
            yee_diag_builder.SetReferencePML(true, 0.25, 5.0, 2.0);
         }
         yee_diag_builder.AssembleYeeCurlOperator(AyeeCurl, omega * sqrt(epsilon * mu));
         yee_diag_builder.AssembleYeeMassOperator(
            eps_fn, AyeeMass, omega * sqrt(epsilon * mu));

         edge_prec->PrintYeeGalerkinComparison(cout);
         edge_prec->PrintYeeGalerkinComparison(*PCOpAh, "pml_prec_abs_matrix", cout);
         edge_prec->PrintYeeCandidateComparison(*PCOpCurlAh, AyeeCurl,
                                               "pml_prec_curl_abs_matrix",
                                               "CtMC", cout);
         edge_prec->PrintYeeCandidateComparison(*PCOpMassAh, AyeeMass,
                                               "pml_prec_mass_abs_matrix",
                                               "k0^2Meps", cout);
         edge_prec->PrintCoarseOperatorDiagnostics(cout);
      }
      if (diagnose_only)
      {
         return 0;
      }
   }
   else if (prec_mode != "none")
   {
      MFEM_ABORT("Unknown preconditioner. Use none, ams, amg, edge_galerkin, or edge_yee.");
   }

   GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(gmres_print);
   gmres.SetKDim(200);
   gmres.SetMaxIter(gmres_max_iter);
   gmres.SetRelTol(gmres_rel_tol);
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*A);
   if (prec_mode == "ams" || prec_mode == "amg")
   {
      gmres.SetPreconditioner(BlockDP);
   }
   else if (prec_mode == "edge_yee" || prec_mode == "edge_galerkin")
   {
      gmres.SetPreconditioner(*edge_prec);
   }

   Vector R0(B.Size());
   A->Mult(X, R0);
   R0 *= -1.0;
   R0 += B;
   const real_t r0 = R0.Norml2();
   const real_t bnorm = B.Norml2();
   if (myid == 0)
   {
      cout << "[PML GMRES] " << prec_mode
           << " start ||r0||=" << r0;
      if (bnorm > 0.0) { cout << " (rel=" << r0 / bnorm << ")"; }
      cout << endl;
      if (true_residual_control)
      {
         cout << "[PML GMRES] true residual controller enabled"
              << " (rel_tol=" << gmres_rel_tol << ")\n";
      }
   }

   unique_ptr<TrueResidualController> true_controller;
   if (true_residual_control)
   {
      true_controller.reset(new TrueResidualController(*A, B, bnorm,
                                                       gmres_rel_tol, 0.0,
                                                       true_residual_print_every,
                                                       myid));
      gmres.SetController(*true_controller);
   }

   gmres.Mult(B, X);

   Vector R1(B.Size());
   A->Mult(X, R1);
   R1 *= -1.0;
   R1 += B;
   const real_t r1 = R1.Norml2();
   const bool true_converged =
      (bnorm > 0.0) ? (r1 / bnorm <= gmres_rel_tol) : (r1 <= gmres_rel_tol);

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   if (myid == 0)
   {
      cout << "[PML GMRES] " << prec_mode
           << " done  ||rN||=" << r1;
      if (bnorm > 0.0) { cout << " (rel=" << r1 / bnorm << ")"; }
      cout << ", iters=" << gmres.GetNumIterations()
           << ", converged=" << gmres.GetConverged()
           << ", true_converged=" << true_converged << "\n";
      cout << " Preconditioner: " << prec_mode << "\n";
      cout << string(80, '-') << endl;
   }

   // 13. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m refined.mesh -g sol_r.gf -g sol_i.gf".
   {
      // Add rank number to filenames to prevent conflicts in parallel
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "refined.mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "sol_r.gf." << setfill('0') << setw(6) << myid;
      sol_i_name << "sol_i.gf." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      x.real().Save(sol_r_ofs);
      x.imag().Save(sol_i_ofs);
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : "keys amrRljcUUuu\n";

      char vishost[] = "localhost";
      int visport = 19916;

      // Send real part
      {
         socketstream sol_sock_re(vishost, visport);
         sol_sock_re.precision(8);
         sol_sock_re << "parallel " << Mpi::WorldSize() << " " << myid << "\n"
                     << "solution\n" << *pmesh << x.real() << keys
                     << "window_title 'Solution real part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      MPI_Barrier(MPI_COMM_WORLD); // Additional barrier between real and imag
      // parts

      // Send imaginary part
      {
         socketstream sol_sock_im(vishost, visport);
         sol_sock_im.precision(8);
         sol_sock_im << "parallel " << Mpi::WorldSize() << " " << myid << "\n"
                     << "solution\n" << *pmesh << x.imag() << keys
                     << "window_title 'Solution imag part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      // Send time-harmonic solution as animation
      {
         ParGridFunction x_t(fespace);
         x_t = x.real();

         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << Mpi::WorldSize() << " " << myid << "\n"
                  << "solution\n" << *pmesh << x_t << keys << "autoscale off\n"
                  << "window_title 'Harmonic Solution (t = 0.0 T)'"
                  << "pause\n" << flush;

         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }

         int num_frames = 32;
         int i = 0;
         while (sol_sock)
         {
            real_t t = (real_t)(i % num_frames) / num_frames;
            ostringstream oss;
            oss << "Harmonic Solution (t = " << t << " T)";

            add(cos(real_t(2.0*M_PI)*t), x.real(),
                sin(real_t(2.0*M_PI)*t), x.imag(), x_t);
            sol_sock << "parallel " << Mpi::WorldSize() << " " << myid << "\n";
            sol_sock << "solution\n" << *pmesh << x_t
                     << "window_title '" << oss.str() << "'" << flush;
            i++;
         }
      }
   }

   // 14. Save data in ParaView format.
   ParaViewDataCollection *pd = NULL;
   if (myid == 0)
   {
      cout << " Saving ParaView output to: ParaView" << endl;
   }
   pd = new ParaViewDataCollection("nurbs_ex25p", pmesh);
   pd->SetPrefixPath("./ParaView");
   pd->RegisterField("solution_real", &(x.real()));
   pd->RegisterField("solution_imag", &(x.imag()));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;

   if (myid == 0)
   {
      cout << "\n" << string(80, '-') << endl;
   }

   // 15. Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete pmesh;

   return 0;
}

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   real_t r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = 0.5 * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
      r += pow(x[i] - center[i], 2.);
   }
   real_t n = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   real_t coeff = pow(n, 2) / M_PI;
   real_t alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = 1000* coeff * exp(alpha);
}

void maxwell_solution(const Vector &x, vector<complex<real_t>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }

   complex<real_t> zi = complex<real_t>(0., 1.);
   real_t k = omega * sqrt(epsilon * mu);
   // T_10 mode
   if (dim == 3)
   {
      real_t k10 = sqrt(k * k - M_PI * M_PI);
      E[1] = -zi * k / real_t(M_PI) * sin(real_t(M_PI)*x(2))*exp(zi * k10 * x(0));
   }
   else if (dim == 2)
   {
      E[1] = -zi * k / real_t(M_PI) * exp(zi * k * x(0));
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   vector<complex<real_t>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;


   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<real_t>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<real_t>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow2(dxs[i])).real();
   }

}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow2(dxs[i])).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow2(dxs[i]));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      D = (real_t(1.0) / det).real();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow2(dxs[i]) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (real_t(1.0) / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow2(dxs[i]) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<real_t>> dxs(dim);
   complex<real_t> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(real_t(1.0) / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow2(dxs[i]) / det);
      }
   }
}

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
   // Initialize bdr attributes
   for (int i = 0; i < mesh_->GetNBE(); ++i)
   {
      mesh_->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = mesh_->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
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

void PML::StretchFunction(const Vector &x,
                          vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);

   real_t n = 2.0;
   real_t c = 5.0;
   real_t coeff;
   real_t k = omega * sqrt(epsilon * mu);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1_r));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1_r));
      }
   }
}
