// ams_sweep_v2.cpp — sparse Pi construction, scalable to larger systems
// Pi_lumped:  diagonal M^{-1} * M_cross (cheap, always works)
// Pi_exact:   CG-solved M^{-1} * M_cross (iterative, no dense M)
// Both use sparse mat-vec to build QTAQ = Pi^T * A * Pi (dense, no1×no1 small)

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace mfem;

void gaussian_rhs(const Vector &x, Vector &f) {
   f=0.0; double r2=0;
   for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
   f[0]=exp(-100.0*r2);
}

// ─── Dense helpers ──────────────────────────────────────────────────
void DenseMultAtB(const DenseMatrix &A, const Vector &x, Vector &y){
   int m=A.Height(), n=A.Width(); y.SetSize(n); y=0.0;
   for(int j=0;j<n;j++) for(int i=0;i<m;i++) y[j]+=A(i,j)*x[i];
}
void DenseMultAB(const DenseMatrix &A, const Vector &x, Vector &y){
   int m=A.Height(), n=A.Width(); y.SetSize(m); y=0.0;
   for(int i=0;i<m;i++) for(int j=0;j<n;j++) y[i]+=A(i,j)*x[j];
}

// ─── Pi_lumped construction ─────────────────────────────────────────
void BuildPiLumped(const Operator &M_op, const Operator &Mcross_op,
                   int ncurl, int no1, DenseMatrix &Pi) {
   Pi.SetSize(ncurl, no1);
   // Extract diagonal of M — use unit-vector sparse matvec
   Vector diag(ncurl);
   {Vector e(ncurl), Me(ncurl);
    for(int i=0;i<ncurl;i++){e=0.0;e[i]=1.0;M_op.Mult(e,Me);diag[i]=Me[i];}}
   // Pi_lumped(i,j) = Mcross(i,j) / diag(i)
   Vector e(no1), Mc(ncurl);
   for(int j=0;j<no1;j++) {
      e=0.0; e[j]=1.0; Mc=0.0; Mcross_op.Mult(e,Mc);
      for(int i=0;i<ncurl;i++) Pi(i,j)=Mc[i]/max(1e-14,diag[i]);
   }
}

// ─── Pi_exact via CG ────────────────────────────────────────────────
void BuildPiExactCG(const Operator &M_op, const Operator &Mcross_op,
                    int ncurl, int no1, DenseMatrix &Pi, int cg_max=200) {
   Pi.SetSize(ncurl, no1);
   Vector e(no1), rhs(ncurl), sol(ncurl);

   // Jacobi preconditioner for M
   Vector diag(ncurl);
   {Vector e2(ncurl), Me(ncurl);
    for(int i=0;i<ncurl;i++){e2=0.0;e2[i]=1.0;M_op.Mult(e2,Me);diag[i]=Me[i];}}

   for(int j=0;j<no1;j++) {
      e=0.0; e[j]=1.0; rhs=0.0; Mcross_op.Mult(e,rhs);
      sol=0.0;

      // CG for M * sol = rhs
      Vector R(rhs.Size()); R=rhs;
      Vector Z(ncurl), P(ncurl), MP(ncurl);
      for(int i=0;i<ncurl;i++) Z[i]=R[i]/max(1e-14,diag[i]);
      P=Z;
      double rz=R*Z;
      if(rz<=0) { for(int i=0;i<ncurl;i++) Pi(i,j)=0.0; continue; }

      for(int iter=1; iter<=cg_max; iter++) {
         M_op.Mult(P,MP);
         double pap=P*MP;
         if(pap<=0 || !isfinite(pap)) break;
         double alpha=rz/pap;
         sol.Add(alpha,P);
         R.Add(-alpha,MP);
         if(sqrt(R*R)/sqrt(rhs*rhs) < 1e-6) break;
         for(int i=0;i<ncurl;i++) Z[i]=R[i]/max(1e-14,diag[i]);
         double rz_new=R*Z;
         if(rz_new<=0 || !isfinite(rz_new)) break;
         double beta=rz_new/rz;
         for(int i=0;i<ncurl;i++) P[i]=Z[i]+beta*P[i];
         rz=rz_new;
      }
      for(int i=0;i<ncurl;i++) Pi(i,j)=sol[i];
   }
}

// ─── Build QTAQ = Pi^T * A * Pi (sparse matvec, dense result) ──────
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

// ─── CG result ──────────────────────────────────────────────────────
struct CGResult {
   int iters=0; double rel_final=1.0, rel_10=1.0, rel_20=1.0;
   bool converged=false, breakdown=false;
   double setup_ms=0, apply_ms=0;
};

// ─── CG: none / Jacobi ──────────────────────────────────────────────
CGResult run_cg_baseline(const Operator &A_op, const Vector &B,
                          bool use_jac, double omega, double tol=1e-8, int max_iter=500) {
   CGResult res; int N=A_op.Height();
   auto t0=steady_clock::now();
   Vector jac_inv(N);
   if(use_jac){
      Vector e(N),Ae(N);
      for(int i=0;i<N;i++){e=0.0;e[i]=1.0;A_op.Mult(e,Ae);jac_inv[i]=1.0/max(1e-14,abs(Ae[i]));}
   }
   res.setup_ms=duration<double>(steady_clock::now()-t0).count()*1000;
   Vector X(N); X=0.0; Vector R(B.Size()); R=B;
   double bnorm=sqrt(B*B); if(bnorm<1e-30) bnorm=1.0;
   Vector Z(N), P(N), AP(N);
   if(use_jac){for(int i=0;i<N;i++)Z[i]=omega*jac_inv[i]*R[i];} else Z=R;
   P=Z; double rz=R*Z; if(rz<=0){res.breakdown=true;return res;}
   double rel_res=1.0; int iter; auto ta0=steady_clock::now();
   for(iter=1;iter<=max_iter;iter++){
      A_op.Mult(P,AP); double pap=P*AP;
      if(pap<=0||!isfinite(pap)){res.breakdown=true;break;}
      double alpha=rz/pap; X.Add(alpha,P); R.Add(-alpha,AP);
      rel_res=sqrt(R*R)/bnorm;
      if(iter==10)res.rel_10=rel_res; if(iter==20)res.rel_20=rel_res;
      if(rel_res<=tol)break;
      if(use_jac){for(int i=0;i<N;i++)Z[i]=omega*jac_inv[i]*R[i];} else Z=R;
      double rz_new=R*Z; if(rz_new<=0||!isfinite(rz_new)){res.breakdown=true;break;}
      double beta=rz_new/rz;
      for(int i=0;i<N;i++)P[i]=Z[i]+beta*P[i]; rz=rz_new;
   }
   res.apply_ms=duration<double>(steady_clock::now()-ta0).count()*1000;
   res.iters=min(iter,max_iter); res.rel_final=rel_res; res.converged=(rel_res<=tol);
   return res;
}

// ─── CG: Pi + Jacobi ────────────────────────────────────────────────
CGResult run_cg_pi_jacobi(const Operator &A_op, const Vector &B,
                           const DenseMatrix &Pi, const DenseMatrix &QTAQ,
                           double omega, double tol=1e-8, int max_iter=500) {
   CGResult res; int N=A_op.Height(), no1=Pi.Width();
   auto t0=steady_clock::now();
   Vector jac_inv(N);
   {Vector e(N),Ae(N);
    for(int i=0;i<N;i++){e=0.0;e[i]=1.0;A_op.Mult(e,Ae);jac_inv[i]=1.0/max(1e-14,abs(Ae[i]));}}
   DenseMatrix QTAQ_reg(QTAQ); double mxQ=0;
   for(int i=0;i<no1;i++) mxQ=max(mxQ,abs(QTAQ_reg(i,i)));
   double reg=max(1e-12, mxQ*1e-10);
   for(int i=0;i<no1;i++) QTAQ_reg(i,i)+=reg;
   DenseMatrixInverse QTAQinv(QTAQ_reg);
   res.setup_ms=duration<double>(steady_clock::now()-t0).count()*1000;

   auto apply_prec=[&](const Vector &r, Vector &z){
      Vector rq(no1);
      for(int j=0;j<no1;j++){double s=0;for(int i=0;i<N;i++)s+=Pi(i,j)*r[i];rq[j]=s;}
      Vector zq(no1); QTAQinv.Mult(rq,zq);
      z.SetSize(N); z=0.0;
      for(int i=0;i<N;i++) for(int j=0;j<no1;j++) z[i]+=Pi(i,j)*zq[j];
      for(int i=0;i<N;i++) z[i]+=omega*jac_inv[i]*r[i];
   };

   Vector X(N); X=0.0; Vector R(B.Size()); R=B;
   double bnorm=sqrt(B*B); if(bnorm<1e-30) bnorm=1.0;
   Vector Z(N), P(N), AP(N); apply_prec(R,Z);
   for(int i=0;i<N;i++) if(!isfinite(Z[i])){res.breakdown=true;return res;}
   P=Z; double rz=R*Z; if(rz<=0){res.breakdown=true;return res;}
   double rel_res=1.0; int iter; auto ta0=steady_clock::now();
   for(iter=1;iter<=max_iter;iter++){
      A_op.Mult(P,AP); double pap=P*AP;
      if(pap<=0||!isfinite(pap)){res.breakdown=true;break;}
      double alpha=rz/pap; X.Add(alpha,P); R.Add(-alpha,AP);
      rel_res=sqrt(R*R)/bnorm;
      if(iter==10)res.rel_10=rel_res; if(iter==20)res.rel_20=rel_res;
      if(rel_res<=tol)break;
      apply_prec(R,Z); double rz_new=R*Z;
      if(rz_new<=0||!isfinite(rz_new)){res.breakdown=true;break;}
      double beta=rz_new/rz;
      for(int i=0;i<N;i++)P[i]=Z[i]+beta*P[i]; rz=rz_new;
   }
   res.apply_ms=duration<double>(steady_clock::now()-ta0).count()*1000;
   res.iters=min(iter,max_iter); res.rel_final=rel_res; res.converged=(rel_res<=tol);
   return res;
}

// ─── Main ──────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
   MPI_Init(&argc,&argv);
   int myid; MPI_Comm_rank(MPI_COMM_WORLD,&myid);

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2, order=2; double freq=5.0;
   int use_exact_cg=0;  // default: Pi_lumped only (fast)
   int test_hypre=0;

   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.AddOption(&use_exact_cg,"-exact","--exact-cg","Use CG-based Pi_exact.");
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
   cerr<<"[step1 spaces]" <<endl; auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int ncurl=fespace->GetTrueVSize();

   auto *o1_fec=new NURBS_HCurlFECollection(1,dim);
   auto *o1_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *o1_fes=new ParFiniteElementSpace(pmesh,o1_ext,o1_fec);
   int no1=o1_fes->GetTrueVSize();

   // ── BCs & Assembly ──────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   cerr<<"[step2 bcs]" <<endl; fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);
   double omega_val=2.0*M_PI*freq, k2=omega_val*omega_val;
   Array<int> attr_dom(pmesh->attributes.Max()); attr_dom=0; attr_dom[0]=1;
   ConstantCoefficient muinv_c(1.0), k2_c(k2), one(1.0);
   RestrictedCoefficient muinv_r(muinv_c,attr_dom), k2_r(k2_c,attr_dom);

   ParBilinearForm a(fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv_r));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(k2_r));
   cerr<<"[step3 assemble A]" <<endl; a.Assemble(0); a.Finalize(0);
   OperatorPtr A_op; a.FormSystemMatrix(ess_tdofs,A_op);
   cerr<<"[step4 form A]" <<endl; auto *Ah=A_op.As<HypreParMatrix>();

   ParLinearForm b(fespace);
   VectorFunctionCoefficient src_coeff(dim, gaussian_rhs);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();
   Vector B(ncurl); cerr<<"[step5 rhs]" <<endl; b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;

   ParBilinearForm mform(fespace);
   mform.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   cerr<<"[step6 assemble M]" <<endl; mform.Assemble(0); mform.Finalize(0);
   cerr<<"[step7 form M]" <<endl; OperatorPtr M_op; mform.FormSystemMatrix(ess_tdofs,M_op);

   Array<int> empty_tdofs;
   ParMixedBilinearForm mmix(o1_fes,fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   cerr<<"[step8 assemble Mcross]" <<endl; mmix.Assemble(0);
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_op);

   // ── Build Pi ────────────────────────────────────────────────
   DenseMatrix PiLumped, PiExact;
   cerr<<"[step9 BuildPiLumped]" <<endl; BuildPiLumped(*M_op, *Mmix_op, ncurl, no1, PiLumped);
   if(use_exact_cg!=0) {
      BuildPiExactCG(*M_op, *Mmix_op, ncurl, no1, PiExact, 200);
   }

   DenseMatrix QTAQ_lumped, QTAQ_exact;
   cerr<<"[step10 BuildQTAQ]" <<endl; BuildQTAQ(*Ah, PiLumped, QTAQ_lumped);
   if(use_exact_cg!=0) BuildQTAQ(*Ah, PiExact, QTAQ_exact);

   // ── Hypre AMS (optional) ────────────────────────────────────
   HypreAMS *ams=nullptr;
   if(test_hypre!=0) { ams=new HypreAMS(*Ah,fespace); ams->SetPrintLevel(0); }

   // ── Print Header ────────────────────────────────────────────
   if(myid==0) {
      cout<<"\n┌─────────────────────────────────────────────────────────────────────────────────────┐\n";
      cout<<"│  Sweep v2: r="<<setw(1)<<ref<<" o="<<setw(1)<<order<<" f="<<setw(4)<<(int)freq
           <<"  N="<<setw(5)<<ncurl<<"  dim(Pi)="<<setw(4)<<no1<<"  method="<<(use_exact_cg!=0?"lump+exact":"lumped")<<"  │\n";
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

   report("none", run_cg_baseline(*Ah,B,false,1.0));
   report("Jacobi ω=1.0", run_cg_baseline(*Ah,B,true,1.0));

   for(double w : {0.7,1.0}) {
      ostringstream os; os<<"Pi_lumped+jac ω="<<fixed<<setprecision(1)<<w;
      report(os.str(), run_cg_pi_jacobi(*Ah,B,PiLumped,QTAQ_lumped,w));
   }

   if(use_exact_cg!=0) {
      for(double w : {0.7,1.0}) {
         ostringstream os; os<<"Pi_exact_cg+jac ω="<<fixed<<setprecision(1)<<w;
         report(os.str(), run_cg_pi_jacobi(*Ah,B,PiExact,QTAQ_exact,w));
      }
   }

   if(test_hypre!=0 && ams) {
      auto t0=steady_clock::now();
      Vector X(ncurl); X=0.0; Vector R(B.Size()); R=B; double bnorm=sqrt(B*B);
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

   if(myid==0) cout<<"└────────────────────────────────────────┴──────┴──────────┴──────────┴──────────┴──────┘\n";

   if(ams) delete ams;
   delete o1_fes; delete o1_ext; delete o1_fec;
   delete fespace; delete nurbsExt; delete fec;
   delete pmesh;
   MPI_Finalize();
   return 0;
}
