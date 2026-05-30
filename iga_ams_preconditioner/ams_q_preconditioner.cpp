// ams_q_preconditioner.cpp
// Full Q=[G, Pi_exact] preconditioner in PCG/GMRES
//
// Variants:
//   none, jacobi, hypre_ams, grad-only, Pi-only,
//   Q-only, Q+jac_add, Jac→Q mult, Q→Jac mult
// Sweep ω ∈ {0.3, 0.5, 0.7, 1.0}
//
// Reports: iters, rel(10), rel(20), rel(final),
//   breakdown, cond_est, setup/apply/total time

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace mfem;

// ─── Dense helpers ──────────────────────────────────────────────────
void DenseMult(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
   int m=A.Height(),k=A.Width(),n=B.Width(); C.SetSize(m,n); C=0.0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int l=0;l<k;l++)C(i,j)+=A(i,l)*B(l,j);
}
void DenseTransposeMult(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
   int k=A.Height(),m=A.Width(),n=B.Width(); C.SetSize(m,n); C=0.0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int l=0;l<k;l++)C(i,j)+=A(l,i)*B(l,j);
}
void AddDiagReg(DenseMatrix &M, double eps=1e-12) {
   int n=min(M.Height(),M.Width()); double mx=0;
   for(int i=0;i<n;i++) mx=max(mx,abs(M(i,i)));
   for(int i=0;i<n;i++) M(i,i)+=max(eps,mx*eps);
}

// ─── PCG with full statistics ──────────────────────────────────────
struct CGResult {
   int    iters=0;
   double rel_final=1.0;
   double rel_10=1.0, rel_20=1.0;
   bool   converged=false;
   bool   breakdown=false;  // pap ≤ 0 or nan
   double setup_ms=0, apply_ms=0, total_ms=0;
};

CGResult run_pcg(const DenseMatrix &A, const Vector &B,
                 const DenseMatrix *G,
                 const DenseMatrix *Pi,
                 double omega, bool mult_mode, bool jac_first,
                 double tol=1e-8, int max_iter=400) {
   CGResult res;
   int N=A.Height();
   int nq=0;
   if(G) nq+=G->Width();
   if(Pi) nq+=Pi->Width();

   auto t0=steady_clock::now();

   // ── Precompute Q^TAQ inverse ─────────────────────────────────
   DenseMatrix *QTAQ=nullptr; DenseMatrixInverse *QTAQinv=nullptr;
   DenseMatrix *Qmat=nullptr;
   vector<double> jac_inv(N);
   for(int i=0;i<N;i++) jac_inv[i]=1.0/max(1e-14,abs(A(i,i)));

   if(nq>0) {
      Qmat=new DenseMatrix(N,nq);
      int col=0;
      if(G) {for(int i=0;i<N;i++)for(int j=0;j<G->Width();j++) (*Qmat)(i,col+j)=(*G)(i,j); col+=G->Width();}
      if(Pi){for(int i=0;i<N;i++)for(int j=0;j<Pi->Width();j++) (*Qmat)(i,col+j)=(*Pi)(i,j); col+=Pi->Width();}

      // Q^T A Q
      QTAQ=new DenseMatrix(nq,nq);
      DenseMatrix QT(nq,N), QTA(nq,N);
      DenseTransposeMult(*Qmat,A,QT);
      DenseMult(QT,*Qmat,*QTAQ);

      // Regularize
      double mxQTAQ=0;
      for(int i=0;i<nq;i++) mxQTAQ=max(mxQTAQ,abs((*QTAQ)(i,i)));
      double reg=max(1e-12, mxQTAQ*1e-10);
      for(int i=0;i<nq;i++) (*QTAQ)(i,i)+=reg;

      QTAQinv=new DenseMatrixInverse(*QTAQ);
   }

   auto t1=steady_clock::now();
   res.setup_ms=duration<double>(t1-t0).count()*1000;

   // ── Apply functions ──────────────────────────────────────────
   auto apply_coarse=[&](const Vector &r, Vector &z){
      int nq2=Qmat->Width();
      Vector rq(nq2); rq=0.0;
      for(int i=0;i<nq2;i++){double s=0;for(int j=0;j<N;j++)s+=(*Qmat)(j,i)*r[j];rq[i]=s;}
      Vector zq(nq2);
      QTAQinv->Mult(rq,zq);
      z.SetSize(N); z=0.0;
      for(int i=0;i<N;i++)for(int j=0;j<nq2;j++)z[i]+=(*Qmat)(i,j)*zq[j];
   };

   auto apply_jac=[&](const Vector &r, Vector &z){
      z.SetSize(N);
      for(int i=0;i<N;i++) z[i]=omega*jac_inv[i]*r[i];
   };

   // ── PCG main loop ────────────────────────────────────────────
   Vector X(N); X=0.0;
   Vector R(B.Size()); R=B;
   double bnorm=sqrt(B*B);
   Vector Z(N), P(N);

   // Initial preconditioned residual
   if(nq==0 || omega==0){
      // Jacobi or identity
      for(int i=0;i<N;i++) Z[i]=(omega>0?omega*jac_inv[i]*R[i]:R[i]);
   } else {
      // Apply preconditioner
      if(!mult_mode){
         // Additive: Z = coarse + jacobi
         apply_coarse(R,Z);
         Vector Zj; apply_jac(R,Zj);
         for(int i=0;i<N;i++) Z[i]+=Zj[i];
      } else if(jac_first){
         // Jacobi → Q: z_J = ω D^{-1} r, r1 = r-A*z_J, z_Q, z = z_J+z_Q
         Vector Zj; apply_jac(R,Zj);
         Vector AZj(N); A.Mult(Zj,AZj);
         Vector R1(N); for(int i=0;i<N;i++) R1[i]=R[i]-AZj[i];
         Vector Zcoarse; apply_coarse(R1,Zcoarse);
         Z=Zcoarse; Z+=Zj;
      } else {
         // Q → Jacobi: z_Q, r1 = r-A*z_Q, z_J, z = z_Q+z_J
         Vector Zcoarse; apply_coarse(R,Zcoarse);
         Vector AZc(N); A.Mult(Zcoarse,AZc);
         Vector R1(N); for(int i=0;i<N;i++) R1[i]=R[i]-AZc[i];
         Vector Zj; apply_jac(R1,Zj);
         Z=Zcoarse; Z+=Zj;
      }
   }

   P=Z;
   double rz=R*Z;
   auto t2=steady_clock::now();
   res.apply_ms=0;

   for(int iter=0;iter<max_iter;iter++){
      auto ia_start=steady_clock::now();

      Vector AP(N); A.Mult(P,AP);
      double pap=P*AP;

      if(pap<=0 || isnan(pap) || isinf(pap)){
         res.breakdown=true; res.iters=iter+1;
         // Compute current residual
         Vector AX(N); A.Mult(X,AX);
         Vector Res(N); for(int i=0;i<N;i++) Res[i]=B[i]-AX[i];
         res.rel_final=sqrt(Res*Res)/bnorm;
         break;
      }

      double alpha=rz/pap;
      for(int i=0;i<N;i++) X[i]+=alpha*P[i];
      for(int i=0;i<N;i++) R[i]-=alpha*AP[i];

      double rnorm=sqrt(R*R)/bnorm;
      if(iter==9)  res.rel_10=rnorm;
      if(iter==19) res.rel_20=rnorm;

      if(rnorm<tol){
         res.converged=true; res.iters=iter+1; res.rel_final=rnorm;
         break;
      }

      // Preconditioner apply
      if(nq==0 || omega==0){
         for(int i=0;i<N;i++) Z[i]=(omega>0?omega*jac_inv[i]*R[i]:R[i]);
      } else if(!mult_mode){
         apply_coarse(R,Z);
         Vector Zj; apply_jac(R,Zj);
         for(int i=0;i<N;i++) Z[i]+=Zj[i];
      } else if(jac_first){
         Vector Zj; apply_jac(R,Zj);
         Vector AZj(N); A.Mult(Zj,AZj);
         Vector R1(N); for(int i=0;i<N;i++) R1[i]=R[i]-AZj[i];
         Vector Zcoarse; apply_coarse(R1,Zcoarse);
         Z=Zcoarse; Z+=Zj;
      } else {
         Vector Zcoarse; apply_coarse(R,Zcoarse);
         Vector AZc(N); A.Mult(Zcoarse,AZc);
         Vector R1(N); for(int i=0;i<N;i++) R1[i]=R[i]-AZc[i];
         Vector Zj; apply_jac(R1,Zj);
         Z=Zcoarse; Z+=Zj;
      }

      double rz_new=R*Z;
      if(rz_new<=0 || isnan(rz_new)){
         res.breakdown=true; res.iters=iter+1; res.rel_final=rnorm;
         break;
      }
      double beta=rz_new/rz;
      for(int i=0;i<N;i++) P[i]=Z[i]+beta*P[i];
      rz=rz_new;

      auto ia_end=steady_clock::now();
      if(iter==0) res.apply_ms=duration<double>(ia_end-ia_start).count()*1000;
   }

   if(res.iters==0){ // reached max without ever breaking
      res.iters=max_iter;
      Vector AX(N); A.Mult(X,AX);
      Vector Res(N); for(int i=0;i<N;i++) Res[i]=B[i]-AX[i];
      res.rel_final=sqrt(Res*Res)/bnorm;
   }

   auto t3=steady_clock::now();
   res.total_ms=duration<double>(t3-t0).count()*1000;

   delete Qmat; delete QTAQ; delete QTAQinv;
   return res;
}

// ─── rank estimation via column norms ────────────────────────────────
double rank_quality(const DenseMatrix &Q) {
   // Simple metric: ratio of min to max column norm
   int nc=Q.Width(), nr=Q.Height();
   double mx=0, mn=1e300;
   for(int j=0;j<nc;j++){
      double s=0; for(int i=0;i<nr;i++){double v=Q(i,j);s+=v*v;}
      s=sqrt(s);
      mx=max(mx,s); mn=min(mn,s);
   }
   return mn/mx;
}

// ─── Main ───────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
   Mpi::Init(argc,argv); Hypre::Init(); int myid=Mpi::WorldRank();
   if(myid==0) cout<<scientific<<setprecision(3);

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2,order=2; double freq=5.0;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.Parse(); if(!args.Good()){args.PrintUsage(cout);return 1;}

   Mesh *mesh=new Mesh(mesh_file,1,1); int dim=mesh->Dimension();
   for(int l=0;l<ref;l++) mesh->UniformRefinement();
   ParMesh *pmesh=new ParMesh(MPI_COMM_WORLD,*mesh);
   mesh->NURBSext=nullptr; delete mesh;

   // ── Spaces ───────────────────────────────────────────────────
   auto *fec=new NURBS_HCurlFECollection(order,dim);
   auto *nurbsExt=new NURBSExtension(pmesh->NURBSext,order);
   auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int ncurl=fespace->GetTrueVSize();

   auto *h1fec=new NURBSFECollection(order);
   auto *h1ext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1space=new ParFiniteElementSpace(pmesh,h1ext,h1fec);
   int nh1=h1space->GetTrueVSize();

   auto *o1_fec=new NURBS_HCurlFECollection(1,dim);
   auto *o1_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *o1_fes=new ParFiniteElementSpace(pmesh,o1_ext,o1_fec);
   int no1=o1_fes->GetTrueVSize();
   if(myid==0) cout<<"Hcurl="<<ncurl<<" H1="<<nh1<<" o1HCurl="<<no1<<endl;

   // ── BCs + Assembly ───────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);
   double omega_val=2.0*M_PI*freq, k2=omega_val*omega_val;
   Array<int> attr_dom(pmesh->attributes.Max()); attr_dom=0; attr_dom[0]=1;
   ConstantCoefficient muinv_c(1.0), k2_c(k2), one(1.0);
   RestrictedCoefficient muinv_r(muinv_c,attr_dom), k2_r(k2_c,attr_dom);

   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   a.Assemble(0); a.Finalize(0);
   OperatorPtr A_op; a.FormSystemMatrix(ess_tdofs,A_op);
   auto *Ah=A_op.As<HypreParMatrix>();

   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0); mform.Finalize(0);
   OperatorPtr M_op; mform.FormSystemMatrix(ess_tdofs,M_op);

   // ── RHS ──────────────────────────────────────────────────────
   ParLinearForm b(fespace);
   VectorFunctionCoefficient src_coeff(dim,[](const Vector &x,Vector &f){
      f=0.0; double r2=0; for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
      f[0]=exp(-100.0*r2);
   });
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();
   Vector B(ncurl); b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;

   // ── G ────────────────────────────────────────────────────────
   ParDiscreteLinearOperator *gdo=new ParDiscreteLinearOperator(h1space,fespace);
   gdo->AddDomainInterpolator(new GradientInterpolator);
   gdo->Assemble(); gdo->Finalize();
   HypreParMatrix *Ghyp=gdo->ParallelAssemble(); delete gdo;

   // ── M_cross (o=1 × o=p HCurl) ───────────────────────────────
   Array<int> empty_tdofs;
   ParMixedBilinearForm mmix(o1_fes,fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_op);

   // ── Convert to dense ─────────────────────────────────────────
   DenseMatrix Gd(ncurl,nh1);
   {Vector e(nh1),Ge(ncurl); for(int j=0;j<nh1;j++){e=0.0;e[j]=1.0;Ghyp->Mult(e,Ge);for(int i=0;i<ncurl;i++)Gd(i,j)=Ge[i];}}

   DenseMatrix Mcross(ncurl,no1);
   {Vector e(no1),Me(ncurl); for(int j=0;j<no1;j++){e=0.0;e[j]=1.0;Mmix_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Mcross(i,j)=Me[i];}}

   DenseMatrix Ad(ncurl,ncurl), Md(ncurl,ncurl);
   {Vector e(ncurl),Ae(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;Ah->Mult(e,Ae);for(int i=0;i<ncurl;i++)Ad(i,j)=Ae[i];}}
   {Vector e(ncurl),Me(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;M_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Md(i,j)=Me[i];}}

   // Pi_exact = M^{-1} * Mcross
   DenseMatrix PiExact(ncurl,no1); PiExact=0.0;
   {DenseMatrixInverse MInv(Md); Vector rhs(ncurl),sol(ncurl);
    for(int j=0;j<no1;j++){for(int i=0;i<ncurl;i++)rhs[i]=Mcross(i,j);MInv.Mult(rhs,sol);for(int i=0;i<ncurl;i++)PiExact(i,j)=sol[i];}}

   delete Ghyp;

   // ── Hypre AMS baseline (separate, one-time) ─────────────────
   auto *A_hypre=Ah;
   unique_ptr<HypreAMS> ams(new HypreAMS(*A_hypre,fespace));
   ams->SetPrintLevel(0);

   // ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
   // Benchmark
   // ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

   if(myid==0) {
      cout<<"\n┌──────────────────────────────────────────────────────────────────────────────┐\n";
      cout<<"│  Q=[G,Pi_exact] PRECONDITIONER — PCG BENCHMARK  (SPD, tol=1e-8)            │\n";
      string header="mesh="+string(mesh_file)+" r="+to_string(ref)+" o="+to_string(order)+" f="+to_string((int)freq);
      cout<<"│  "<<left<<setw(76)<<header<<"│\n";
      cout<<"│  N="<<setw(5)<<ncurl<<"  dim(G)="<<setw(5)<<Gd.Width()<<"  dim(Pi)="<<setw(5)<<PiExact.Width()<<"  dim(Q)="<<setw(5)<<(Gd.Width()+PiExact.Width())<<"          │\n";
      cout<<"│  rank_quality(Q)="<<setw(8)<<rank_quality([&](){DenseMatrix Q(ncurl,nh1+no1);
         for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++)Q(i,j)=Gd(i,j);for(int j=0;j<no1;j++)Q(i,nh1+j)=PiExact(i,j);}return Q;}())<<"                       │\n";
      cout<<"├──────────────────────────────────────┬──────┬─────────┬─────────┬─────────┬──────┬─────────┬──────────┤\n";
      cout<<"│ Preconditioner                       │ Iters│ rel(10) │ rel(20) │ rel(end)│Break?│ Setup(s)│Apply(ms) │\n";
      cout<<"├──────────────────────────────────────┼──────┼─────────┼─────────┼─────────┼──────┼─────────┼──────────┤\n";
   }

   auto report=[&](const string &label, const CGResult &r){
      if(myid==0) {
         cout<<"│ "<<left<<setw(36)<<label<<" │ "
             <<right<<setw(5)<<r.iters<<" │ "
             <<setw(7)<<(r.rel_10<1.0?to_string((int)(r.rel_10*1e6))+"e-6":r.rel_10>=1.0?"  —    ":"N/A")
             <<" │ "
             <<setw(7)<<(r.rel_20<1.0?to_string((int)(r.rel_20*1e6))+"e-6":r.rel_20>=1.0?"  —    ":"N/A")
             <<" │ "
             <<setw(7)<<scientific<<setprecision(1)<<r.rel_final;
         cout<<" │ "<<(r.breakdown?"BRK":"   ");
         cout<<" │ "<<fixed<<setprecision(2)<<setw(7)<<r.setup_ms/1000
             <<" │ "<<setw(8)<<r.apply_ms;
         cout<<" │\n";
      }
   };

   // ── Baselines ──────────────────────────────────────────────────
   report("none",  run_pcg(Ad,B,nullptr,nullptr,1.0,false,false));

   // Hypre AMS (wrap as PCG preconditioner)
   {
      auto t0=steady_clock::now();
      Vector X(ncurl); X=0.0; Vector R(B.Size()); R=B;
      double bnorm=sqrt(B*B);
      CGSolver pcg(MPI_COMM_WORLD);
      pcg.SetOperator(*Ah);
      pcg.SetPreconditioner(*ams);
      pcg.SetRelTol(1e-8); pcg.SetMaxIter(400); pcg.SetPrintLevel(0);
      pcg.Mult(B,X);
      auto t1=steady_clock::now();
      Vector AX(ncurl); Ah->Mult(X,AX);
      Vector Res(ncurl); for(int i=0;i<ncurl;i++) Res[i]=B[i]-AX[i];
      double rel_f=sqrt(Res*Res)/bnorm;
      if(myid==0) {
         cout<<"│ "<<left<<setw(36)<<"hypre_ams"<<" │ "
             <<right<<setw(5)<<pcg.GetNumIterations()<<" │ "
             <<"  —     │  —     │ "
             <<setw(7)<<scientific<<setprecision(1)<<rel_f
             <<" │ "<<(pcg.GetConverged()?"   ":"NCV")
             <<" │ "<<fixed<<setprecision(2)<<setw(7)<<duration<double>(t1-t0).count()
             <<" │ "<<setw(8)<<"— "
             <<" │\n";
      }
   }

   // ── grad-only (ω=1.0) ─────────────────────────────────────────
   report("Q=[G] + jac_add ω=1.0", run_pcg(Ad,B,&Gd,nullptr,1.0,false,false));

   // ── Pi-only (ω=1.0) ───────────────────────────────────────────
   report("Q=[Pi_exact] + jac_add ω=1.0", run_pcg(Ad,B,nullptr,&PiExact,1.0,false,false));

   // ── Q=[G,Pi] variants ─────────────────────────────────────────
   // Coarse-only
   report("Q=[G,Pi] coarse-only", run_pcg(Ad,B,&Gd,&PiExact,0.0,false,false));

   // Additive + ω sweep
   for(double w : {0.3,0.5,0.7,1.0}) {
      ostringstream os; os<<"Q=[G,Pi] + jac_add ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_pcg(Ad,B,&Gd,&PiExact,w,false,false));
   }

   // Multiplicative: Jac→Q
   for(double w : {0.3,0.5,0.7,1.0}) {
      ostringstream os; os<<"Q=[G,Pi] Jac→Q mult ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_pcg(Ad,B,&Gd,&PiExact,w,true,true));
   }

   // Multiplicative: Q→Jac
   for(double w : {0.3,0.5,0.7,1.0}) {
      ostringstream os; os<<"Q=[G,Pi] Q→Jac mult ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_pcg(Ad,B,&Gd,&PiExact,w,true,false));
   }

   if(myid==0) {
      cout<<"└──────────────────────────────────────┴──────┴─────────┴─────────┴─────────┴──────┴─────────┴──────────┘\n";
      cout<<"\nLegend: BRK=breakdown NCV=not converged N/A=not reached\n"<<endl;
   }

   delete o1_fes; delete h1space;
   delete fespace; delete fec; delete pmesh;
   return 0;
}
