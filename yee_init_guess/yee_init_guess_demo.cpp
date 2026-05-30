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
#include "yee_to_iga_transfer.hpp"
#include "../covariant_aux_space/yee_operator.hpp"
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
   int    yee_n       = 9;
   int    gmres_max   = 400;
   int    gmres_kdim  = 200;
   real_t gmres_tol   = 1e-6;
   int    gmres_print = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
   args.AddOption(&order, "-o", "--order", "Spline order.");
   args.AddOption(&freq, "-f", "--frequency", "Frequency.");
   args.AddOption(&yee_n, "-yn", "--yee-n", "Yee grid points per direction.");
   args.AddOption(&gmres_max, "-gmi", "--gmres-max-iter", "Max GMRES iterations.");
   args.AddOption(&gmres_kdim, "-gkd", "--gmres-kdim", "GMRES Krylov subspace dimension.");
   args.AddOption(&gmres_tol, "-grt", "--gmres-rel-tol", "GMRES relative tolerance.");
   args.AddOption(&gmres_print, "-gpl", "--gmres-print-level", "GMRES print level.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   if (myid == 0) { args.PrintOptions(cout); }

   Device device("cpu"); device.Print();
   if (myid==0) cout << "DEBUG: device done" << endl;

   // ── Mesh and FE space ──────────────────────────────────────────────
   if (myid==0) cout << "DEBUG: loading mesh" << endl;
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   for (int lev = 0; lev < ref_levels; lev++) mesh->UniformRefinement();

   if (myid==0) cout << "DEBUG: creating ParMesh" << endl;
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   omega = freq;

   if (myid==0) cout << "DEBUG: creating geometry" << endl;
   // ── Geometry evaluator ─────────────────────────────────────────────
   // NOTE: pmesh->NURBSext may be null after ParMesh construction from certain MFEM builds.
   // Use the original mesh's NURBSext (accessed via the mesh that we just deleted).
   // Instead, access the NURBSext from the ParMesh's internal storage.
   NURBSExtension *nurbs_ext_geom = pmesh->NURBSext;
   MFEM_VERIFY(nurbs_ext_geom != nullptr, "pmesh->NURBSext is null!");
   fdfd_iga_init::SinglePatchNURBSEvaluator geom(*pmesh, *nurbs_ext_geom, 0);

   // ── PML ────────────────────────────────────────────────────────────
   Array2D<real_t> length(dim, 2);
   length = 0.0; length(0,1) = 1.0; length(1,1) = 1.0;
   if (dim > 2) length(2,1) = 1.0;
   PML *pml = new PML(pmesh, length);
   pml->SetAttributes(pmesh);

   // ── NURBS FE space ─────────────────────────────────────────────────
   NURBSExtension *nurbs_ext = new NURBSExtension(pmesh->NURBSext, order);
   FiniteElementCollection *fec = new NURBS_HCurlFECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, nurbs_ext, fec);
   fespace->StealNURBSext();

   int tvsize = fespace->GetTrueVSize();
   if (myid == 0) cout << "true_vsize=" << tvsize << "\n";

   // ── 1. Assemble IGA H(curl) PML Maxwell ────────────────────────────
   if (myid == 0) cout << "\n[1] Assembling IGA H(curl) PML Maxwell...\n";

   ConstantCoefficient one(1.0), neg_one(-1.0);
   ConstantCoefficient muinv(1.0/mu), omeg(-omega*omega*epsilon);

   PMLDiagMatrixCoefficient pml_c1_Re(dim, detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(dim, detJ_inv_JT_J_Im, pml);
   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml);

   Array<int> attrPML;
   VectorRestrictedCoefficient c1_Re(pml_c1_Re, attrPML);
   VectorRestrictedCoefficient c1_Im(pml_c1_Im, attrPML);
   VectorRestrictedCoefficient c2_Re(pml_c2_Re, attrPML);
   VectorRestrictedCoefficient c2_Im(pml_c2_Im, attrPML);

   Array<int> attr;
   for (int i = 0; i < pmesh->attributes.Size(); i++) attr.Append(i);
   RestrictedCoefficient restr_muinv(muinv, attr);
   RestrictedCoefficient restr_omeg(omeg, attr);

   // Real part
   ParBilinearForm a_re(fespace);
   a_re.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   a_re.AddDomainIntegrator(new VectorFEMassIntegrator(c1_Re));
   a_re.Assemble(); a_re.Finalize();

   // Imag part
   ParBilinearForm a_im(fespace);
   a_im.AddDomainIntegrator(new VectorFEMassIntegrator(c1_Im));
   a_im.Assemble(); a_im.Finalize();

   VectorFunctionCoefficient src_coeff(dim, source);
   ParLinearForm b(fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();

   Array<int> ess_tdof_list;
   OperatorHandle Ah_re, Ah_im;
   Vector Xd, Bd;
   a_re.FormLinearSystem(ess_tdof_list, Xd, Bd, Ah_re, Xd, Bd);
   a_im.FormLinearSystem(ess_tdof_list, Xd, Bd, Ah_im, Xd, Bd);

   SparseMatrix *A_re = Ah_re.As<SparseMatrix>();
   SparseMatrix *A_im = Ah_im.As<SparseMatrix>();

   Array<int> block_offsets({0, tvsize, 2*tvsize});
   BlockOperator A(block_offsets);
   A.SetBlock(0, 0, A_re);
   A.SetBlock(0, 1, A_im, -1.0);
   A.SetBlock(1, 0, A_im);
   A.SetBlock(1, 1, A_re);

   Vector B(2*tvsize), X(2*tvsize);
   B = 0.0;
   for (int i = 0; i < tvsize; i++) B[i] = b[i];
   double bnorm = GlobalNorml2(B);
   if (myid == 0) cout << "  IGA system: ||B||=" << bnorm << "\n";

   // ── 2. Solve Yee/FDFD PML ──────────────────────────────────────────
   if (myid == 0) cout << "\n[2] Solving Yee/FDFD PML system...\n";

   fdfd_iga_init::ReferenceGrid yee_grid;
   yee_grid.nx = yee_n; yee_grid.ny = yee_n; yee_grid.nz = yee_n;

   auto eps_fn = [](const Vector &x) -> double { return 1.0; };
   covariant_aux_space::YeeOperatorBuilder yee_op(geom);
   yee_op.SetGrid(yee_grid);
   yee_op.SetReferencePML(true, 0.25, 5.0, 2.0);

   DenseMatrix AyeeRe, AyeeIm;
   yee_op.AssembleYeeMaxwellOperatorComplex(eps_fn, omega, AyeeRe, AyeeIm);

   int na = AyeeRe.Height();
   DenseMatrix AyeeBlock(2*na, 2*na); AyeeBlock = 0.0;
   for (int i=0; i<na; i++) for (int j=0; j<na; j++) {
      AyeeBlock(i,j)       =  AyeeRe(i,j);
      AyeeBlock(i,j+na)    = -AyeeIm(i,j);
      AyeeBlock(i+na,j)    =  AyeeIm(i,j);
      AyeeBlock(i+na,j+na) =  AyeeRe(i,j);
   }
   double max_d = 0.0;
   for (int i=0; i<2*na; i++) max_d = max(max_d, abs(AyeeBlock(i,i)));
   for (int i=0; i<2*na; i++) AyeeBlock(i,i) += 1e-8*max_d;
   DenseMatrixInverse AyeeInv(AyeeBlock);

   // Yee RHS: evaluate source at Yee edge midpoints
   const auto &yee_edges = yee_op.GetEdgeDofs();
   double dxyz = 1.0/(yee_n-1);
   Vector yee_rhs(2*na); yee_rhs = 0.0;

   for (int e = 0; e < na; e++) {
      const auto &ed = yee_edges[e];
      double xi_x, xi_y, xi_z;
      if (ed.axis == 0)      { xi_x = (ed.i+0.5)*dxyz; xi_y = ed.j*dxyz; xi_z = ed.k*dxyz; }
      else if (ed.axis == 1) { xi_x = ed.i*dxyz; xi_y = (ed.j+0.5)*dxyz; xi_z = ed.k*dxyz; }
      else                   { xi_x = ed.i*dxyz; xi_y = ed.j*dxyz; xi_z = (ed.k+0.5)*dxyz; }

      Vector xi(3); xi[0]=xi_x; xi[1]=xi_y; xi[2]=xi_z;
      Vector x_phys(3); DenseMatrix Jac(3,3);
      geom.EvalGeometry(xi, x_phys, Jac);

      Vector J(3); source(x_phys, J);
      double jhat = 0.0;
      for (int d=0; d<3; d++) jhat += Jac(d, ed.axis)*J[d];
      yee_rhs[e] = jhat * dxyz*dxyz*dxyz;
   }

   Vector yee_sol(2*na);
   AyeeInv.Mult(yee_rhs, yee_sol);
   Vector u_yee(yee_sol, 0, na);
   if (myid == 0) cout << "  Yee solved: " << na << " edges, ||u_yee||=" << u_yee.Norml2() << "\n";

   // ── 3. Edge-integral mapping: u_yee → u_1 ─────────────────────────
   if (myid == 0) cout << "\n[3] Edge-integral mapping Yee → o=1 IGA...\n";

   yee_init_guess::YeeToIGATransfer transfer(*fespace, 1);
   transfer.BuildYeeEdges(yee_n, yee_n, yee_n);

   int nc = transfer.GetCoarseTrueVSize();
   Vector u1_re(nc);
   transfer.MapYeeToO1(u_yee, u1_re);
   if (myid == 0) cout << "  o=1 IGA: ||u1||=" << u1_re.Norml2() << "\n";

   // ── 4. Prolongate: u_1 → u_p ──────────────────────────────────────
   if (myid == 0) cout << "\n[4] Prolongating o=1 → o=" << order << " IGA...\n";
   Vector u_p_re(tvsize);
   transfer.ProlongateO1ToOp(u1_re, u_p_re);

   Vector X0(2*tvsize); X0 = 0.0;
   for (int i=0; i<tvsize; i++) X0[i] = u_p_re[i];
   if (myid == 0) cout << "  Initial guess: ||X0||=" << GlobalNorml2(X0) << "\n";

   Vector R0(2*tvsize);
   A.Mult(X0, R0);
   for (int i=0; i<2*tvsize; i++) R0[i] = B[i] - R0[i];
   double r0norm = GlobalNorml2(R0);
   if (myid == 0) cout << "  Initial true residual: ||r0||/||B|| = "
                        << r0norm/bnorm << "\n";

   // ── 5. GMRES comparison ────────────────────────────────────────────
   if (myid == 0) {
      cout << "\n" << string(70, '=') << "\n";
      cout << "  GMRES Comparison\n" << string(70, '=') << "\n";
   }

   auto run_gmres = [&](const char *label, const Vector *x0) {
      Vector Xrun(2*tvsize);
      if (x0) Xrun = *x0; else Xrun = 0.0;
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetOperator(A);
      gmres.SetRelTol(gmres_tol);
      gmres.SetMaxIter(gmres_max);
      gmres.SetKDim(gmres_kdim);
      gmres.SetPrintLevel(gmres_print);
      gmres.Mult(B, Xrun);

      Vector R1(2*tvsize);
      A.Mult(Xrun, R1);
      for (int i=0; i<2*tvsize; i++) R1[i] = B[i] - R1[i];
      double r1norm = GlobalNorml2(R1);

      if (myid == 0) {
         cout << left << setw(28) << label << " iters=" << setw(4) << gmres.GetNumIterations()
              << " rel=" << scientific << setprecision(2) << r1norm/bnorm
              << " conv=" << gmres.GetConverged() << "\n";
      }
   };

   if (myid == 0) cout << "\n  [A] Zero initial guess\n";
   run_gmres("zero_init", nullptr);

   if (myid == 0) cout << "\n  [B] Yee->o=1->o=p init guess\n";
   run_gmres("yee_init_guess", &X0);

   if (myid == 0) cout << string(70, '=') << "\n";

   delete fespace; delete fec; delete pmesh; delete pml;
   return 0;
}

// ─── PML class implementation ───────────────────────────────────────────── ─────────────────────────────────────────────────

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
