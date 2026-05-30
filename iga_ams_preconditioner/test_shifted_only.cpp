// Quick test: Shifted AMS only (no other HypreAMS to corrupt memory)
#include "mfem.hpp"
#include "iga_ams_prec.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include <iostream>
#include <complex>
using namespace std;
using namespace mfem;

// ... PML class and all the PML functions from the main demo (copied inline for brevity)
// Actually, let me just use a minimal source function

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   int myid = Mpi::WorldRank();

   const char *mesh_file = "/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref = 2, order = 2;
   double freq = 5.0;

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   for (int l = 0; l < ref; l++) mesh->UniformRefinement();
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->NURBSext = nullptr; delete mesh;

   auto *fec = new NURBS_HCurlFECollection(order, dim);
   auto *NURBSext = new NURBSExtension(pmesh->NURBSext, order);
   auto *fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);

   if (myid == 0) cout << "true_vsize=" << fespace->GetTrueVSize() << endl;

   // Build a simple real curl-curl + mass operator
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient muinv(1.0);
   ConstantCoefficient neg_k2(-pow(2.0*M_PI*freq, 2));
   RestrictedCoefficient restr_muinv(muinv, attr);
   RestrictedCoefficient restr_neg_k2(neg_k2, attr);

   Array<int> attr_v, attrPML;
   attr_v.SetSize(pmesh->attributes.Max()); attr_v = 0; attr_v[0] = 1;
   attrPML.SetSize(pmesh->attributes.Max()); attrPML = 0;

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv), NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_neg_k2), NULL);
   a.Assemble(0);

   ParComplexGridFunction x(fespace); x = 0.0;
   VectorFunctionCoefficient f_src(dim, [](const Vector &x, Vector &f) {
      f = 0.0; f[0] = 1.0 * exp(-100.0*(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)+pow(x[2]-0.5,2)));
   });
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f_src));
   b.Vector::operator=(0.0); b.Assemble();

   OperatorPtr A; Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   double r0norm = [&](){ double s=B.Norml2(); s*=s; double g=0; MPI_Allreduce(&s,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); return sqrt(g); }();

   // Test shifted AMS for eta=0.3
   cout << "\n=== Testing Shifted AMS (eta=0.3) ===\n";
   {
      auto prec = std::make_unique<iga_ams::ShiftedAMS_Preconditioner>(*fespace, freq, 1.0);
      prec->SetOperator(*A);
      prec->SetEta(0.3);
      prec->SetPrintLevel(0);

      Vector Xrun(B.Size()); Xrun = 0.0;
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetOperator(*A);
      gmres.SetPreconditioner(*prec);
      gmres.SetRelTol(1e-6);
      gmres.SetAbsTol(0.0);
      gmres.SetMaxIter(400);
      gmres.SetKDim(200);
      gmres.SetPrintLevel(0);
      gmres.Mult(B, Xrun);

      Vector resid(B.Size());
      A->Mult(Xrun, resid); resid *= -1.0; resid += B;
      double rel = resid.Norml2() / r0norm;
      if (myid == 0) cout << "shifted_ams eta=0.3: iters=" << gmres.GetNumIterations()
                           << " rel=" << scientific << rel << endl;
   }

   cout << "=== ALL DONE ===\n" << flush;

   delete fespace; delete fec; delete pmesh;
   return 0;
}
