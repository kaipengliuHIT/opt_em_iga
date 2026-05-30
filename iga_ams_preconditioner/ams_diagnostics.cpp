// ams_diagnostics.cpp — Minimal, correct AMS internals inspection
#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
using namespace mfem;

double gl2(const Vector &v, MPI_Comm c=MPI_COMM_WORLD) {
   double s=v.Norml2(); s*=s; double g=0; MPI_Allreduce(&s,&g,1,MPI_DOUBLE,MPI_SUM,c); return sqrt(g);
}

int main(int argc, char *argv[]) {
   Mpi::Init(argc,argv); int myid=Mpi::WorldRank(); Hypre::Init();

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2,order=2; double freq=5.0;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.Parse(); if(!args.Good()){args.PrintUsage(cout);return 1;}
   if(myid==0) args.PrintOptions(cout);

   Mesh *mesh=new Mesh(mesh_file,1,1); int dim=mesh->Dimension();
   for(int l=0;l<ref;l++) mesh->UniformRefinement();
   ParMesh *pmesh=new ParMesh(MPI_COMM_WORLD,*mesh);
   mesh->NURBSext=nullptr; delete mesh;

   // ── H(curl) IGA space ───────────────────────────────────────────
   auto *fec=new NURBS_HCurlFECollection(order,dim);
   auto *nurbsExt=new NURBSExtension(pmesh->NURBSext,order);
   auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int tvsize=fespace->GetTrueVSize();
   if(myid==0) cout<<"HCurl(order="<<order<<"): true_vsize="<<tvsize<<endl;

   // ── H1 IGA scalar space ─────────────────────────────────────────
   auto *h1_fec=new NURBSFECollection(order);
   auto *h1_ext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1_space=new ParFiniteElementSpace(pmesh,h1_ext,h1_fec);
   int h1size=h1_space->GetTrueVSize();
   if(myid==0) cout<<"H1(order="<<order<<"):    true_vsize="<<h1size<<endl;

   // ── Essential BCs ───────────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);

   // ── Build real pos-def system: A = curl-curl + k^2 * M ──────────
   double omega=2.0*M_PI*freq, k2=omega*omega;
   Array<int> attr_dom(pmesh->attributes.Max()); attr_dom=0; attr_dom[0]=1;
   ConstantCoefficient muinv_coeff(1.0), k2_coeff(k2);
   RestrictedCoefficient muinv_r(muinv_coeff,attr_dom);
   RestrictedCoefficient k2_r(k2_coeff,attr_dom);

   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   a.Assemble(); a.Finalize();
   OperatorPtr A_op; a.FormSystemMatrix(ess_tdofs,A_op);
   auto *Ah=A_op.As<HypreParMatrix>();
   if(myid==0) cout<<"System A: "<<Ah->M()<<"x"<<Ah->N()<<endl;

   // RHS
   ParGridFunction x0(fespace); x0=0.0;
   ParLinearForm b(fespace);
   VectorFunctionCoefficient src(dim,[](const Vector &x,Vector &f){
      f=0.0; double r2=0; for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
      f[0]=exp(-100.0*r2);
   });
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src));
   b.Assemble();
   Vector B(tvsize),X(tvsize);
   b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;
   double bnorm=gl2(B); if(myid==0) cout<<"||b||="<<bnorm<<endl;

   // ═══════════════════════════════════════════════════════════════
   // DIAGNOSTIC 1: AMS internals
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ AMS Internals ═══\n";

   // 1a. Build G (discrete gradient H1→H(curl))
   ParDiscreteLinearOperator *grad_op=new ParDiscreteLinearOperator(h1_space,fespace);
   grad_op->AddDomainInterpolator(new GradientInterpolator);
   grad_op->Assemble(); grad_op->Finalize();
   HypreParMatrix *G=grad_op->ParallelAssemble(); delete grad_op;
   if(myid==0) cout<<"G: "<<G->M()<<"x"<<G->N()<<" nnz="<<G->NNZ()<<endl;

   // 1b. Build Pi (vector H1→H(curl) via IdentityInterpolator)
   int vdim=fec->GetRangeDim(dim);
   auto *h1v_fec=new NURBSFECollection(order);
   auto *h1v_ext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1v_space=new ParFiniteElementSpace(pmesh,h1v_ext,h1v_fec,
                                              max(pmesh->SpaceDimension(),vdim),
                                              Ordering::byVDIM);
   if(myid==0) cout<<"Vector H1(byVDIM): true_vsize="<<h1v_space->GetTrueVSize()<<endl;

   ParDiscreteLinearOperator *id_op=new ParDiscreteLinearOperator(h1v_space,fespace);
   id_op->AddDomainInterpolator(new IdentityInterpolator);
   id_op->Assemble(); id_op->Finalize();
   HypreParMatrix *Pi=id_op->ParallelAssemble(); delete id_op;

   if(myid==0) {
      cout<<"Pi: "<<Pi->M()<<"x"<<Pi->N()<<" nnz="<<Pi->NNZ()<<endl;
      cout<<"  nnz_per_row="<<(Pi->NNZ()/(double)Pi->M())<<endl;
      cout<<"  Construction: IdentityInterpolator(vector NURBS H1 → NURBS HCurl)"<<endl;
      cout<<"  BOTH spaces use order "<<order<< " — this is NOT a low→high order mapping!"<<endl;
   }

   // 1c. Check: Curl * G ≈ 0?
   {
      double max_nrm=0, max_curl_nrm=0;
      Vector gcol(G->N()), Cgcol(G->M()), ACgcol(G->M());
      for(int j=0;j<min(10,G->N());j++) {
         gcol=0.0; gcol[j]=1.0; G->Mult(gcol,Cgcol);
         double nrm_g=Cgcol.Norml2();
         Ah->Mult(Cgcol,ACgcol);
         double k2_term=k2*nrm_g; // what k^2*M adds
         max_nrm=max(max_nrm,nrm_g);
         max_curl_nrm=max(max_curl_nrm,ACgcol.Norml2());
      }
      if(myid==0) {
         cout<<"Curl*G check: ||(C^T*C + k^2*M)*G_j|| / ||G_j|| (10 cols):"<<endl;
         cout<<"  max ||G_j||  = "<<scientific<<max_nrm<<endl;
         cout<<"  max ||A*G_j|| = "<<scientific<<max_curl_nrm<<endl;
         cout<<"  If A*G_j ≈ k^2 * G_j (since Curl*G_j=0), ratio should be ≈ k^2="<<k2<<endl;
         if(max_nrm>1e-12) cout<<"  Observed ratio = "<<scientific<<max_curl_nrm/max_nrm<<endl;
      }
   }

   // ═══════════════════════════════════════════════════════════════
   // DIAGNOSTIC 2: AMS solve + component analysis
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ AMS Solve Diagnostics ═══\n";

   unique_ptr<HypreAMS> ams(new HypreAMS(*Ah,fespace));
   ams->SetPrintLevel(0);

   // Apply AMS as standalone solver (few iterations)
   Vector Xams(B.Size()); Xams=0.0;
   {
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(*Ah);
      cg.SetPreconditioner(*ams);
      cg.SetRelTol(1e-8); cg.SetMaxIter(100); cg.SetPrintLevel(0);
      cg.Mult(B,Xams);
      Vector resid(B.Size()); Ah->Mult(Xams,resid); resid-=B;
      if(myid==0) cout<<"AMS as CG prec: iters="<<cg.GetNumIterations()
                        <<" rel="<<scientific<<gl2(resid)/bnorm<<endl;
   }

   // Compare: no preconditioner
   {
      CGSolver cg(MPI_COMM_WORLD); Vector Xn(B.Size()); Xn=0.0;
      cg.SetOperator(*Ah); cg.SetRelTol(1e-8); cg.SetMaxIter(500); cg.SetPrintLevel(0);
      cg.Mult(B,Xn);
      Vector resid(B.Size()); Ah->Mult(Xn,resid); resid-=B;
      if(myid==0) cout<<"No prec CG:      iters="<<cg.GetNumIterations()
                        <<" rel="<<scientific<<gl2(resid)/bnorm<<endl;
   }

   // ═══════════════════════════════════════════════════════════════
   // DIAGNOSTIC 3: AMS on o=1 Galerkin
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ o=1 Galerkin AMS (separate) ═══\n";

   // Build o=1 H(curl) space
   auto *c_fec=new NURBS_HCurlFECollection(1,dim);
   auto *c_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *c_fes=new ParFiniteElementSpace(pmesh,c_ext,c_fec);
   int nc=c_fes->GetTrueVSize(), nf=tvsize;
   if(myid==0) cout<<"o=1 HCurl: true_vsize="<<nc<<endl;

   // Build mass prolongation P: M_f^{-1} * M_{1,f}
   ConstantCoefficient one(1.0); Array<int> empty;
   ParBilinearForm mf(fespace); mf.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mf.Assemble(0); OperatorPtr Mf; mf.FormSystemMatrix(empty,Mf);
   ParMixedBilinearForm mm(c_fes,fespace); mm.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mm.Assemble(0); OperatorHandle Mm; mm.FormRectangularSystemMatrix(empty,empty,Mm);

   DenseMatrix Mfd(nf,nf), Mmd(nf,nc);
   {Vector e(nf),col(nf); for(int j=0;j<nf;j++){e=0.0;e[j]=1.0;Mf->Mult(e,col);for(int i=0;i<nf;i++)Mfd(i,j)=col[i];}}
   {Vector e(nc),col(nf); for(int j=0;j<nc;j++){e=0.0;e[j]=1.0;Mm->Mult(e,col);for(int i=0;i<nf;i++)Mmd(i,j)=col[i];}}
   DenseMatrixInverse MfdInv(Mfd); DenseMatrix P(nf,nc);
   {Vector rhs(nf),sol(nf); for(int j=0;j<nc;j++){for(int i=0;i<nf;i++)rhs[i]=Mmd(i,j);MfdInv.Mult(rhs,sol);for(int i=0;i<nf;i++)P(i,j)=sol[i];}}

   // A_1_real = P^T * A * P
   DenseMatrix A1r(nc,nc); A1r=0.0;
   {Vector probe(nf),Ap(nf); for(int j=0;j<nc;j++){probe=0.0;for(int i=0;i<nf;i++)probe[i]=P(i,j);
      Ah->Mult(probe,Ap);for(int ii=0;ii<nc;ii++){double v=0;for(int k=0;k<nf;k++)v+=P(k,ii)*Ap[k];A1r(ii,j)=v;}}}

   // Convert to HypreParMatrix
   auto *Asp=new SparseMatrix(nc,nc);
   for(int i=0;i<nc;i++)for(int j=0;j<nc;j++)if(abs(A1r(i,j))>1e-15)Asp->Set(i,j,A1r(i,j));
   Asp->Finalize();
   HYPRE_BigInt *rs=new HYPRE_BigInt[2]; rs[0]=0; rs[1]=nc;
   HypreParMatrix A1(MPI_COMM_WORLD,nc,rs,Asp);

   // Restrict RHS: b_1 = P^T * B
   Vector B1(nc); P.MultTranspose(B,B1);

   // AMS on o=1
   if(myid==0) cout<<"Building AMS on o=1...\n";
   unique_ptr<HypreAMS> ams1(new HypreAMS(A1,c_fes));
   ams1->SetPrintLevel(0);

   // Solve on o=1 and prolongate
   Vector X1(nc); X1=0.0;
   {
      CGSolver cg(MPI_COMM_WORLD); cg.SetOperator(A1); cg.SetPreconditioner(*ams1);
      cg.SetRelTol(1e-8); cg.SetMaxIter(100); cg.SetPrintLevel(0);
      cg.Mult(B1,X1);
      if(myid==0) cout<<"o=1 AMS-CG: iters="<<cg.GetNumIterations()<<endl;
   }
   Vector Xp(nf); P.Mult(X1,Xp);

   // Compute residual: r = B - A * Xp
   Vector AXp(nf), R(nf); Ah->Mult(Xp,AXp);
   for(int i=0;i<nf;i++) R[i]=B[i]-AXp[i];

   if(myid==0) {
      double rel=gl2(R)/bnorm;
      cout<<"o=1 Galerkin AMS solve:\n"
          <<"  ||r||/||b|| = "<<scientific<<rel<<endl;

      // Also check component: r_re (curl-curl) vs r_im (mass)
      // Since this is real system, we just have one part
      // But the residual norm gives the overall quality
   }

   // Cleanup
   delete G; delete Pi; delete h1v_space;
   delete c_fes; delete h1_space;
   delete fespace; delete fec; delete pmesh;
   if(myid==0) cout<<"\n═══ Done ═══\n"<<flush;
   return 0;
}
