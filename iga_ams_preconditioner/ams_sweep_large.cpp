//patched
// ams_sweep.cpp
// Parameter sweep: Pi_exact + Jacobi preconditioner for IGA H(curl) SPD Maxwell
//
// Sweeps: o=2,3; r=1,2,3,4; f=1,5,10
// Compares: none, Jacobi, Pi_lumped+Jacobi, Pi_exact+Jacobi (ω=0.7,1.0)
// Uses sparse CG + dense auxiliary space preconditioner

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace mfem;

// ─── Gaussian RHS ──────────────────────────────────────────────────
void gaussian_rhs(const Vector &x, Vector &f) {
   f=0.0;
   double r2=0;
   for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
   f[0]=exp(-100.0*r2);
}

// ─── Pi construction ────────────────────────────────────────────────
void BuildPi(const Operator &Mcurl_op, const Operator &Mcross_op,
             int ncurl, int no1, DenseMatrix &Pi, bool lumped) {
   Pi.SetSize(ncurl, no1);
   if(lumped) {
      Vector diag(ncurl);
      {Vector e(ncurl), Me(ncurl);
       for(int i=0;i<ncurl;i++){e=0.0;e[i]=1.0;Mcurl_op.Mult(e,Me);diag[i]=Me[i];}}
      Vector e(no1), Mc(ncurl);
      for(int j=0;j<no1;j++) {
         e=0.0; e[j]=1.0; Mc=0.0; Mcross_op.Mult(e,Mc);
         for(int i=0;i<ncurl;i++) Pi(i,j)=Mc[i]/max(1e-14,diag[i]);
      }
   } else {
      DenseMatrix Md(ncurl,ncurl);
      {Vector e(ncurl), Me(ncurl);
       for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;Mcurl_op.Mult(e,Me);for(int i=0;i<ncurl;i++)Md(i,j)=Me[i];}}
      DenseMatrixInverse MInv(Md);
      Vector rhs(ncurl), sol(ncurl);
      {Vector e(no1); Vector Mc(ncurl);
       for(int j=0;j<no1;j++){e=0.0;e[j]=1.0;Mcross_op.Mult(e,Mc);MInv.Mult(Mc,sol);for(int i=0;i<ncurl;i++)Pi(i,j)=sol[i];}}
   }
}

// ─── Build Q^T A Q = Pi^T * A * Pi ─────────────────────────────────
void BuildQTAQ(const Operator &A_op, const DenseMatrix &Pi, DenseMatrix &QTAQ) {
   int ncurl=Pi.Height(), no1=Pi.Width();
   QTAQ.SetSize(no1, no1);
   Vector pj(ncurl), Apj(ncurl);
   for(int j=0;j<no1;j++) {
      for(int i=0;i<ncurl;i++) pj[i]=Pi(i,j);
      A_op.Mult(pj,Apj);
      for(int k=0;k<no1;k++) {
         double s=0;
         for(int i=0;i<ncurl;i++) s+=Pi(i,k)*Apj[i];
         QTAQ(k,j)=s;
      }
   }
}

// ─── CG result struct ──────────────────────────────────────────────
struct CGResult {
   int iters=0;
   double rel_final=1.0, rel_10=1.0, rel_20=1.0;
   bool converged=false, breakdown=false;
   double setup_ms=0, apply_ms=0;
};

// ─── CG: none / Jacobi baseline ────────────────────────────────────
CGResult run_cg_baseline(const Operator &A_op, const Vector &B,
                          bool use_jacobi, double omega,
                          double tol=1e-8, int max_iter=500) {
   CGResult res;
   int N=A_op.Height();
   auto t0=steady_clock::now();

   Vector jac_inv(N);
   if(use_jacobi){
      Vector e(N),Ae(N);
      for(int i=0;i<N;i++){e=0.0;e[i]=1.0;A_op.Mult(e,Ae);jac_inv[i]=1.0/max(1e-14,abs(Ae[i]));}
   }
   res.setup_ms=duration<double>(steady_clock::now()-t0).count()*1000;

   Vector X(N); X=0.0;
   Vector R(B.Size()); R=B;
   double bnorm=sqrt(B*B); if(bnorm<1e-30) bnorm=1.0;

   Vector Z(N), P(N), AP(N);
   if(use_jacobi){for(int i=0;i<N;i++)Z[i]=omega*jac_inv[i]*R[i];}
   else Z=R;
   P=Z;
   double rz=R*Z;
   if(rz<=0){res.breakdown=true;return res;}

   double rel_res=1.0;
   int iter;
   auto ta0=steady_clock::now();
   for(iter=1;iter<=max_iter;iter++){
      A_op.Mult(P,AP);
      double pap=P*AP;
      if(pap<=0||!isfinite(pap)){res.breakdown=true;break;}
      double alpha=rz/pap;
      X.Add(alpha,P);
      R.Add(-alpha,AP);
      rel_res=sqrt(R*R)/bnorm;
      if(iter==10)res.rel_10=rel_res;
      if(iter==20)res.rel_20=rel_res;
      if(rel_res<=tol)break;
      if(use_jacobi){for(int i=0;i<N;i++)Z[i]=omega*jac_inv[i]*R[i];}
      else Z=R;
      double rz_new=R*Z;
      if(rz_new<=0||!isfinite(rz_new)){res.breakdown=true;break;}
      double beta=rz_new/rz;
      for(int i=0;i<N;i++)P[i]=Z[i]+beta*P[i];
      rz=rz_new;
   }
   res.apply_ms=duration<double>(steady_clock::now()-ta0).count()*1000;
   res.iters=min(iter,max_iter);
   res.rel_final=rel_res;
   res.converged=(rel_res<=tol);
   return res;
}

// ─── CG: Pi + Jacobi additive ──────────────────────────────────────
CGResult run_cg_pi_jacobi(const Operator &A_op, const Vector &B,
                           const DenseMatrix &Pi, const DenseMatrix &QTAQ,
                           double omega, double tol=1e-8, int max_iter=500) {
   CGResult res;
   int N=A_op.Height(), no1=Pi.Width();
   auto t0=steady_clock::now();

   // Jacobi diagonal
   Vector jac_inv(N);
   {Vector e(N),Ae(N);
    for(int i=0;i<N;i++){e=0.0;e[i]=1.0;A_op.Mult(e,Ae);jac_inv[i]=1.0/max(1e-14,abs(Ae[i]));}}

   // QTAQ inverse with regularization
   DenseMatrix QTAQ_reg(QTAQ);
   double mxQ=0;
   for(int i=0;i<no1;i++) mxQ=max(mxQ,abs(QTAQ_reg(i,i)));
   double reg=max(1e-12, mxQ*1e-10);
   for(int i=0;i<no1;i++) QTAQ_reg(i,i)+=reg;
   DenseMatrixInverse QTAQinv(QTAQ_reg);

   res.setup_ms=duration<double>(steady_clock::now()-t0).count()*1000;

   // Preconditioner apply function
   auto apply_prec=[&](const Vector &r, Vector &z){
      Vector rq(no1);
      for(int j=0;j<no1;j++){double s=0;for(int i=0;i<N;i++)s+=Pi(i,j)*r[i];rq[j]=s;}
      Vector zq(no1); QTAQinv.Mult(rq,zq);
      z.SetSize(N); z=0.0;
      for(int i=0;i<N;i++) for(int j=0;j<no1;j++) z[i]+=Pi(i,j)*zq[j];
      for(int i=0;i<N;i++) z[i]+=omega*jac_inv[i]*r[i];
   };

   // PCG
   Vector X(N); X=0.0;
   Vector R(B.Size()); R=B;
   double bnorm=sqrt(B*B); if(bnorm<1e-30) bnorm=1.0;

   Vector Z(N), P(N), AP(N);
   apply_prec(R,Z);
   for(int i=0;i<N;i++) if(!isfinite(Z[i])){res.breakdown=true;return res;}
   P=Z;
   double rz=R*Z;
   if(rz<=0){res.breakdown=true;return res;}

   double rel_res=1.0;
   int iter;
   auto ta0=steady_clock::now();
   for(iter=1;iter<=max_iter;iter++){
      A_op.Mult(P,AP);
      double pap=P*AP;
      if(pap<=0||!isfinite(pap)){res.breakdown=true;break;}
      double alpha=rz/pap;
      X.Add(alpha,P);
      R.Add(-alpha,AP);
      rel_res=sqrt(R*R)/bnorm;
      if(iter==10)res.rel_10=rel_res;
      if(iter==20)res.rel_20=rel_res;
      if(rel_res<=tol)break;
      apply_prec(R,Z);
      double rz_new=R*Z;
      if(rz_new<=0||!isfinite(rz_new)){res.breakdown=true;break;}
      double beta=rz_new/rz;
      for(int i=0;i<N;i++)P[i]=Z[i]+beta*P[i];
      rz=rz_new;
   }
   res.apply_ms=duration<double>(steady_clock::now()-ta0).count()*1000;
   res.iters=min(iter,max_iter);
   res.rel_final=rel_res;
   res.converged=(rel_res<=tol);
   return res;
}

// ─── Main ──────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
   std::cerr << "SWEEP_START" << std::endl;
   MPI_Init(&argc,&argv);
   int myid; MPI_Comm_rank(MPI_COMM_WORLD,&myid); std::cerr << "RANK=" << myid << std::endl;

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2, order=2; double freq=5.0;
   int test_hypre=0;

   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.AddOption(&test_hypre,"-hypre","--hypre","Test hypre AMS.");
   args.Parse();
   if(!args.Good()){args.PrintUsage(cout);MPI_Finalize();return 1;}

   int dim=3;
   Mesh *mesh=new Mesh(mesh_file,1,1);
   for(int l=0;l<ref;l++) mesh->UniformRefinement();
   ParMesh *pmesh=new ParMesh(MPI_COMM_WORLD,*mesh);
   mesh->NURBSext=nullptr; delete mesh;

   // ── Spaces ──────────────────────────────────────────────────
   auto *fec=new NURBS_HCurlFECollection(order,dim);
   auto *nurbsExt=new NURBSExtension(pmesh->NURBSext,order);
   auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int ncurl=fespace->GetTrueVSize();

   auto *o1_fec=new NURBS_HCurlFECollection(1,dim);
   auto *o1_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *o1_fes=new ParFiniteElementSpace(pmesh,o1_ext,o1_fec);
   int no1=o1_fes->GetTrueVSize();

   // ── BCs ─────────────────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);
   double omega_val=2.0*M_PI*freq, k2=omega_val*omega_val;
   Array<int> attr_dom(pmesh->attributes.Max()); attr_dom=0; attr_dom[0]=1;
   ConstantCoefficient muinv_c(1.0), k2_c(k2), one(1.0);
   RestrictedCoefficient muinv_r(muinv_c,attr_dom), k2_r(k2_c,attr_dom);

   // ── Assemble A = C^T C + k^2 M ─────────────────────────────
   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   a.Assemble(0); a.Finalize(0);
   OperatorPtr A_op; a.FormSystemMatrix(ess_tdofs,A_op);
   auto *Ah=A_op.As<HypreParMatrix>();

   // ── RHS ─────────────────────────────────────────────────────
   ParLinearForm b(fespace);
   VectorFunctionCoefficient src_coeff(dim, gaussian_rhs);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();
   Vector B(ncurl); b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;

   // ── Mass matrix ─────────────────────────────────────────────
   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mform.Assemble(0); mform.Finalize(0);
   OperatorPtr M_op; mform.FormSystemMatrix(ess_tdofs,M_op);

   // ── M_cross: o=1 × order=p ──────────────────────────────────
   Array<int> empty_tdofs;
   ParMixedBilinearForm mmix(o1_fes,fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_op);

   // ── Build Pi ────────────────────────────────────────────────
   DenseMatrix PiExact, PiLumped;
   BuildPi(*M_op, *Mmix_op, ncurl, no1, PiExact, false);
   BuildPi(*M_op, *Mmix_op, ncurl, no1, PiLumped, true);

   // ── Build Q^T A Q ──────────────────────────────────────────
   DenseMatrix QTAQ_exact, QTAQ_lumped;
   BuildQTAQ(*Ah, PiExact, QTAQ_exact);
   BuildQTAQ(*Ah, PiLumped, QTAQ_lumped);

   // ── Hypre AMS (optional, only one instantiation) ────────────
   HypreAMS *ams=nullptr;
   if(test_hypre!=0) { ams=new HypreAMS(*Ah,fespace); ams->SetPrintLevel(0); }

   // ── Header ──────────────────────────────────────────────────
   if(myid==0) {
      cout<<"\n┌─────────────────────────────────────────────────────────────────────────────────────┐\n";
      cout<<"│  Sweep: r="<<setw(1)<<ref<<" o="<<setw(1)<<order<<" f="<<setw(4)<<(int)freq
           <<"  N="<<setw(5)<<ncurl<<"  dim(Pi)="<<setw(4)<<no1<<"                           │\n";
      cout<<"├────────────────────────────────────────┬──────┬──────────┬──────────┬──────────┬──────┤\n";
      cout<<"│ Preconditioner                         │ Iters│ rel(10)  │ rel(20)  │ rel(end) │ Break│\n";
      cout<<"├────────────────────────────────────────┼──────┼──────────┼──────────┼──────────┼──────┤\n";
   }

   auto report=[&](const string &label, const CGResult &r){
      if(myid==0){
         cout<<"│ "<<left<<setw(38)<<label<<" │ "<<right<<setw(4)<<r.iters<<" │ ";
         if(r.rel_10<1.0) cout<<scientific<<setprecision(2)<<setw(8)<<r.rel_10;
         else cout<<"    —    ";
         cout<<" │ ";
         if(r.rel_20<1.0) cout<<scientific<<setprecision(2)<<setw(8)<<r.rel_20;
         else cout<<"    —    ";
         cout<<" │ "<<scientific<<setprecision(2)<<setw(8)<<r.rel_final;
         cout<<" │ "<<(r.breakdown?" BRK":"    ")<<" │\n";
      }
   };

   // ── Run all methods ─────────────────────────────────────────
   report("none", run_cg_baseline(*Ah,B,false,1.0));
   report("Jacobi ω=1.0", run_cg_baseline(*Ah,B,true,1.0));

   for(double w : {0.7,1.0}) {
      ostringstream os; os<<"Pi_lumped+jac ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_cg_pi_jacobi(*Ah,B,PiLumped,QTAQ_lumped,w));
   }
   for(double w : {0.7,1.0}) {
      ostringstream os; os<<"Pi_exact+jac ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_cg_pi_jacobi(*Ah,B,PiExact,QTAQ_exact,w));
   }

   // Hypre AMS
   if(test_hypre!=0 && ams) {
      auto t0=steady_clock::now();
      Vector X(ncurl); X=0.0;
      Vector R(B.Size()); R=B; double bnorm=sqrt(B*B);
      CGSolver pcg(MPI_COMM_WORLD);
      pcg.SetOperator(*Ah); pcg.SetPreconditioner(*ams);
      pcg.SetRelTol(1e-8); pcg.SetMaxIter(500); pcg.SetPrintLevel(0);
      pcg.Mult(B,X);
      auto t1=steady_clock::now();
      Vector AX(ncurl); Ah->Mult(X,AX);
      Vector Res(ncurl); for(int i=0;i<ncurl;i++)Res[i]=B[i]-AX[i];
      double rel_f=sqrt(Res*Res)/bnorm;
      if(myid==0){
         cout<<"│ "<<left<<setw(38)<<"hypre_ams"<<" │ "<<right<<setw(4)<<pcg.GetNumIterations()<<" │ "
             <<"    —     │     —    │ "<<scientific<<setprecision(2)<<setw(8)<<rel_f
             <<" │ "<<(pcg.GetConverged()?"    ":" NCV")<<" │\n";
      }
   }

   if(myid==0) {
      cout<<"└────────────────────────────────────────┴──────┴──────────┴──────────┴──────────┴──────┘\n";
      cout.flush();
   }

   if(ams) delete ams;
   delete o1_fes; delete o1_ext; delete o1_fec;
   delete fespace; delete nurbsExt; delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}
