// ams_iga_coupled.cpp
// Stage 2: IGA-compatible Pi + multiplicative/coupled AMS-like corrections
//
// Implements:
//   1. Pi variants: Pi_default, Pi_mass_lumped, Pi_mass_exact, Pi_o1_prolong
//   2. Multiplicative: Pi-then-G, G-then-Pi
//   3. Coupled: Q = [G, Pi] + solve(Q^T A Q)
//   4. Full preconditioner: B^{-1} = Q*(Q^T A Q)^{-1}*Q^T + Jacobi + CG test
//
// Usage: mpirun -np 1 ./ams_iga_coupled -r 2 -o 2 -f 5.0

#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;
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
void AddDiagReg(DenseMatrix &A, double rel=1e-14) {
   int n=min(A.Height(),A.Width()); double mx=0;
   for(int i=0;i<n;i++) mx=max(mx,abs(A(i,i)));
   for(int i=0;i<n;i++) A(i,i)+=max(1e-15,rel*mx);
}

// ─── Preconditioner base: Q-based auxiliary correction + smoother ───
class QBasedPreconditioner : public Solver {
public:
   QBasedPreconditioner(const DenseMatrix &A, const DenseMatrix &Q)
      : Solver(A.Height(),A.Height()), A_(&A), Q_(&Q) { Build(); }
   void SetOperator(const Operator&) override {}
   void Mult(const Vector &r, Vector &z) const override {
      // z = Q * (Q^T A Q)^{-1} * Q^T * r + Jacobi_residual(r - A*Q*...)
      int n=A_->Height(), nq=Q_->Width();

      // 1. Restrict: rq = Q^T * r
      Vector rq(nq); rq=0.0;
      for(int i=0;i<nq;i++){double s=0;for(int j=0;j<n;j++)s+=Q_->operator()(j,i)*r[j];rq[i]=s;}

      // 2. Solve: zq = (Q^T A Q)^{-1} * rq
      Vector zq(nq);
      QTAQinv_->Mult(rq,zq);

      // 3. Prolongate: z_Q = Q * zq
      z.SetSize(n); z=0.0;
      for(int i=0;i<n;i++)for(int j=0;j<nq;j++)z[i]+=Q_->operator()(i,j)*zq[j];

      // 4. Residual: r1 = r - A * z_Q
      Vector Az(n); A_->Mult(z,Az);
      Vector r1(n); for(int i=0;i<n;i++) r1[i]=r[i]-Az[i];

      // 5. Jacobi smoother on r1
      for(int i=0;i<n;i++) z[i]+=jacobi_inv_[i]*r1[i];
   }

private:
   const DenseMatrix *A_, *Q_;
   unique_ptr<DenseMatrixInverse> QTAQinv_;
   Vector jacobi_inv_;

   void Build() {
      int n=A_->Height(), nq=Q_->Width();
      // Q^T A Q
      DenseMatrix QT(nq,n), QTA(nq,n), QTAQ(nq,nq);
      DenseTransposeMult(*Q_,*A_,QT);
      DenseMult(QT,*Q_,QTAQ);
      AddDiagReg(QTAQ,1e-12);
      QTAQinv_=make_unique<DenseMatrixInverse>(QTAQ);
      // Jacobi
      jacobi_inv_.SetSize(n);
      for(int i=0;i<n;i++) jacobi_inv_[i]=1.0/max(1e-14,abs(A_->operator()(i,i)));
   }
};

// ─── Main ───────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
   Mpi::Init(argc,argv); Hypre::Init(); int myid=Mpi::WorldRank();
   Device dev("cpu"); if(myid==0) dev.Print();

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2,order=2; double freq=5.0;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.Parse(); if(!args.Good()){args.PrintUsage(cout);return 1;}
   if(myid==0){args.PrintOptions(cout); cout<<scientific<<setprecision(3);}

   // ── Mesh + Spaces ──────────────────────────────────────────────
   Mesh *mesh=new Mesh(mesh_file,1,1); int dim=mesh->Dimension();
   for(int l=0;l<ref;l++) mesh->UniformRefinement();
   ParMesh *pmesh=new ParMesh(MPI_COMM_WORLD,*mesh);
   mesh->NURBSext=nullptr; delete mesh;

   auto *fec=new NURBS_HCurlFECollection(order,dim);
   auto *nurbsExt=new NURBSExtension(pmesh->NURBSext,order);
   auto *fespace=new ParFiniteElementSpace(pmesh,nurbsExt,fec);
   int ncurl=fespace->GetTrueVSize();

   auto *h1fec=new NURBSFECollection(order);
   auto *h1ext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1space=new ParFiniteElementSpace(pmesh,h1ext,h1fec);
   int nh1=h1space->GetTrueVSize();

   // o=1 HCurl for proper prolongation
   auto *o1_fec=new NURBS_HCurlFECollection(1,dim);
   auto *o1_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *o1_fes=new ParFiniteElementSpace(pmesh,o1_ext,o1_fec);
   int no1=o1_fes->GetTrueVSize();

   // Vec H1 (order=order) for default Pi
   int vdim=fec->GetRangeDim(dim);
   auto *h1vfec=new NURBSFECollection(order);
   auto *h1vext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1vspace=new ParFiniteElementSpace(pmesh,h1vext,h1vfec,
                                             max(pmesh->SpaceDimension(),vdim),
                                             Ordering::byVDIM);
   int nh1v=h1vspace->GetTrueVSize();
   if(myid==0) cout<<"Spaces: Hcurl="<<ncurl<<" H1="<<nh1
                    <<" vecH1="<<nh1v<<" o1HCurl="<<no1<<endl;

   // ── BCs + Assembly ──────────────────────────────────────────────
   Array<int> ess_tdofs, ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr=1;
   fespace->GetEssentialTrueDofs(ess_bdr,ess_tdofs);
   double omega=2.0*M_PI*freq, k2=omega*omega;

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

   // RHS
   ParLinearForm b(fespace);
   VectorFunctionCoefficient src_coeff(dim,[](const Vector &x,Vector &f){
      f=0.0; double r2=0; for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
      f[0]=exp(-100.0*r2);
   });
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
   b.Assemble();
   Vector B(ncurl); b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;
   
   // Also free-DOF only vector for consistency
   Vector B_free(ncurl); B_free=0.0;
   for(int i=0;i<ncurl;i++){bool found=false;for(int j=0;j<ess_tdofs.Size();j++)if(ess_tdofs[j]==i){found=true;break;}if(!found)B_free[i]=1.0;}

   // ── Build G ─────────────────────────────────────────────────────
   ParDiscreteLinearOperator *gdo=new ParDiscreteLinearOperator(h1space,fespace);
   gdo->AddDomainInterpolator(new GradientInterpolator);
   gdo->Assemble(); gdo->Finalize();
   HypreParMatrix *Ghyp=gdo->ParallelAssemble(); delete gdo;

   // ── Build Pi_default (IdentityInterpolator, same-order) ────────
   ParDiscreteLinearOperator *pdo=new ParDiscreteLinearOperator(h1vspace,fespace);
   pdo->AddDomainInterpolator(new IdentityInterpolator);
   pdo->Assemble(); pdo->Finalize();
   HypreParMatrix *Pihyp=pdo->ParallelAssemble(); delete pdo;

   // ── Build M_cross (o=1 HCurl × o=p HCurl) ─────────────────────
   Array<int> empty_tdofs;
   ParMixedBilinearForm mmix(o1_fes,fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_op);

   // ── Convert to Dense ────────────────────────────────────────────
   DenseMatrix Gd(ncurl,nh1);
   {Vector e(nh1),Ge(ncurl); for(int j=0;j<nh1;j++){e=0.0;e[j]=1.0;Ghyp->Mult(e,Ge);for(int i=0;i<ncurl;i++)Gd(i,j)=Ge[i];}}

   DenseMatrix Pid(ncurl,nh1v);  // Pi_default
   {Vector e(nh1v),Pe(ncurl); for(int j=0;j<nh1v;j++){e=0.0;e[j]=1.0;Pihyp->Mult(e,Pe);for(int i=0;i<ncurl;i++)Pid(i,j)=Pe[i];}}

   DenseMatrix Mcross(ncurl,no1);  // Cross-mass M_cross(o1,op)
   {Vector e(no1),Me(ncurl); for(int j=0;j<no1;j++){e=0.0;e[j]=1.0;Mmix_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Mcross(i,j)=Me[i];}}

   DenseMatrix Ad(ncurl,ncurl), Md(ncurl,ncurl);
   {Vector e(ncurl),Ae(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;Ah->Mult(e,Ae);for(int i=0;i<ncurl;i++)Ad(i,j)=Ae[i];}}
   {Vector e(ncurl),Me(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;M_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Md(i,j)=Me[i];}}

   // ── Pi variants ─────────────────────────────────────────────────
   // Pi_def: already have Pid

   // Pi_mass_lumped: diag(M)^{-1} * Mcross
   DenseMatrix PiLumped(ncurl,no1); PiLumped=0.0;
   for(int i=0;i<ncurl;i++){double inv=1.0/max(1e-14,Md(i,i));for(int j=0;j<no1;j++)PiLumped(i,j)=inv*Mcross(i,j);}

   // Pi_mass_exact: M^{-1} * Mcross (full dense solve)
   DenseMatrix PiExact(ncurl,no1); PiExact=0.0;
   {DenseMatrixInverse MInv(Md); Vector rhs(ncurl),sol(ncurl);
    for(int j=0;j<no1;j++){for(int i=0;i<ncurl;i++)rhs[i]=Mcross(i,j);MInv.Mult(rhs,sol);for(int i=0;i<ncurl;i++)PiExact(i,j)=sol[i];}}

   // ── Condition numbers ────────────────────────────────────────────
   if(myid==0) cout<<"\n── Pi sizes ──\n   Pi_default: "<<ncurl<<"×"<<nh1v<<"\n   Pi_lumped:  "<<ncurl<<"×"<<no1<<"\n   Pi_exact:   "<<ncurl<<"×"<<no1<<endl;

   // ─── Correction helpers ─────────────────────────────────────────
   auto apply_correction=[&](const DenseMatrix &T, const Vector &r, Vector &z){
      int np=T.Width();
      DenseMatrix TT(np,ncurl); DenseTransposeMult(T,Ad,TT);
      DenseMatrix TTAT(np,np); DenseMult(TT,T,TTAT);
      AddDiagReg(TTAT,1e-12);
      DenseMatrixInverse TTATInv(TTAT);
      Vector Tr(np); for(int i=0;i<np;i++){double s=0;for(int j=0;j<ncurl;j++)s+=T(j,i)*r[j];Tr[i]=s;}
      Vector w(np); TTATInv.Mult(Tr,w);
      z.SetSize(ncurl);z=0.0;for(int i=0;i<ncurl;i++)for(int j=0;j<np;j++)z[i]+=T(i,j)*w[j];
   };

   auto res_ratio=[&](const Vector &r,const Vector &z)->double{
      Vector Az(ncurl); Ad.Mult(z,Az);
      double r2=0,res2=0; for(int i=0;i<ncurl;i++){double d=r[i]-Az[i];r2+=r[i]*r[i];res2+=d*d;}
      return sqrt(res2)/sqrt(r2);
   };

   // ═══════════════════════════════════════════════════════════════
   // TEST: Multiplicative corrections
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Multiplicative & Coupled Corrections ═══\n";

   mt19937 rng(789); normal_distribution<double> nd(0,1);

   // a) Random r
   { Vector r(ncurl); for(int i=0;i<ncurl;i++)r[i]=nd(rng);
     // Pi-then-G
     Vector zPi; apply_correction(PiExact,r,zPi);
     Vector A_zPi(ncurl); Ad.Mult(zPi,A_zPi);
     Vector r1(ncurl); for(int i=0;i<ncurl;i++) r1[i]=r[i]-A_zPi[i];
     Vector zG; apply_correction(Gd,r1,zG);
     Vector zPG(ncurl); for(int i=0;i<ncurl;i++)zPG[i]=zPi[i]+zG[i];
     double rr_pg=res_ratio(r,zPG);
     // G-then-Pi
     Vector zG2; apply_correction(Gd,r,zG2);
     Vector A_zG2(ncurl); Ad.Mult(zG2,A_zG2);
     Vector r2(ncurl); for(int i=0;i<ncurl;i++) r2[i]=r[i]-A_zG2[i];
     Vector zPi2; apply_correction(PiExact,r2,zPi2);
     Vector zGP(ncurl); for(int i=0;i<ncurl;i++)zGP[i]=zG2[i]+zPi2[i];
     double rr_gp=res_ratio(r,zGP);
     // Coupled Q=[G,Pi]
     DenseMatrix Q(ncurl,nh1+no1);
     for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++) Q(i,j)=Gd(i,j);for(int j=0;j<no1;j++) Q(i,nh1+j)=PiExact(i,j);}
     Vector zQ; apply_correction(Q,r,zQ);
     double rr_q=res_ratio(r,zQ);
     // Single
     Vector zGs; apply_correction(Gd,r,zGs); double rr_g=res_ratio(r,zGs);
     Vector zPs; apply_correction(PiExact,r,zPs); double rr_p=res_ratio(r,zPs);
     if(myid==0) cout<<"  Random r:\n"
        <<"    grad-only: "<<setw(10)<<rr_g
        <<"  Pi-only: "<<setw(10)<<rr_p
        <<"  add G+Pi: "<<setw(10)<<res_ratio(r,[&](){Vector z;z.SetSize(ncurl);
           for(int i=0;i<ncurl;i++)z[i]=zGs[i]+zPs[i];return z;}())
        <<"\n    Pi→G:     "<<setw(10)<<rr_pg
        <<"  G→Pi:     "<<setw(10)<<rr_gp
        <<"  Q=[G,Pi]: "<<setw(10)<<rr_q<<endl;
   }

   // b) CG residual
   { Vector X(ncurl);X=0.0; Vector R(B.Size());R=B;
     Vector Z(R),Pz=Z; double rz=R*Z; Vector AP(ncurl); Ad.Mult(Pz,AP);
     double alpha=rz/(Pz*AP); for(int i=0;i<ncurl;i++){X[i]+=alpha*Pz[i];R[i]-=alpha*AP[i];}
     Vector zG, zPi; apply_correction(Gd,R,zG); apply_correction(PiExact,R,zPi);
     double rr_g=res_ratio(R,zG), rr_p=res_ratio(R,zPi);
     double rr_add=res_ratio(R, [&](){Vector z(ncurl);for(int i=0;i<ncurl;i++)z[i]=zG[i]+zPi[i];return z;}());
     // Pi→G
     Vector AzPi(ncurl); Ad.Mult(zPi,AzPi);
     Vector r1(ncurl); for(int i=0;i<ncurl;i++)r1[i]=R[i]-AzPi[i];
     Vector zG2; apply_correction(Gd,r1,zG2);
     Vector zPG(ncurl); for(int i=0;i<ncurl;i++)zPG[i]=zPi[i]+zG2[i];
     double rr_pg=res_ratio(R,zPG);
     // G→Pi
     Vector AzG(ncurl); Ad.Mult(zG,AzG);
     Vector r2(ncurl); for(int i=0;i<ncurl;i++)r2[i]=R[i]-AzG[i];
     Vector zPi2; apply_correction(PiExact,r2,zPi2);
     Vector zGP(ncurl); for(int i=0;i<ncurl;i++)zGP[i]=zG[i]+zPi2[i];
     double rr_gp=res_ratio(R,zGP);
     // Coupled
     DenseMatrix Q(ncurl,nh1+no1);
     for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++) Q(i,j)=Gd(i,j);for(int j=0;j<no1;j++) Q(i,nh1+j)=PiExact(i,j);}
     Vector zQ; apply_correction(Q,R,zQ); double rr_q=res_ratio(R,zQ);

     if(myid==0) cout<<"  CG residual (1 step, ||r||/||b||≈0.47):\n"
        <<"    grad-only: "<<setw(10)<<rr_g
        <<"  Pi-only: "<<setw(10)<<rr_p
        <<"  add G+Pi: "<<setw(10)<<rr_add
        <<"\n    Pi→G:     "<<setw(10)<<rr_pg
        <<"  G→Pi:     "<<setw(10)<<rr_gp
        <<"  Q=[G,Pi]: "<<setw(10)<<rr_q<<endl;
   }

   // c) r = A*G*phi (gradient-dominated)
   { Vector phi(nh1); for(int i=0;i<nh1;i++) phi[i]=nd(rng);
     Vector u(ncurl); Gd.Mult(phi,u);
     Vector r(ncurl); Ad.Mult(u,r);
     Vector zG, zPi; apply_correction(Gd,r,zG); apply_correction(PiExact,r,zPi);
     double rr_g=res_ratio(r,zG), rr_p=res_ratio(r,zPi);
     Vector AzPi(ncurl); Ad.Mult(zPi,AzPi);
     Vector r1(ncurl); for(int i=0;i<ncurl;i++)r1[i]=r[i]-AzPi[i];
     Vector zG2; apply_correction(Gd,r1,zG2);
     Vector zPG(ncurl); for(int i=0;i<ncurl;i++)zPG[i]=zPi[i]+zG2[i];
     double rr_pg=res_ratio(r,zPG);
     Vector AzG(ncurl); Ad.Mult(zG,AzG);
     Vector r2(ncurl); for(int i=0;i<ncurl;i++)r2[i]=r[i]-AzG[i];
     Vector zPi2; apply_correction(PiExact,r2,zPi2);
     Vector zGP(ncurl); for(int i=0;i<ncurl;i++)zGP[i]=zG[i]+zPi2[i];
     double rr_gp=res_ratio(r,zGP);
     DenseMatrix Q(ncurl,nh1+no1);
     for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++) Q(i,j)=Gd(i,j);for(int j=0;j<no1;j++) Q(i,nh1+j)=PiExact(i,j);}
     Vector zQ; apply_correction(Q,r,zQ); double rr_q=res_ratio(r,zQ);
     if(myid==0) cout<<"  r=A*G*phi (grad-dominated):\n"
        <<"    grad-only: "<<setw(10)<<rr_g
        <<"  Pi-only:  "<<setw(10)<<rr_p
        <<"  Pi→G:     "<<setw(10)<<rr_pg
        <<"  G→Pi:     "<<setw(10)<<rr_gp
        <<"  Q=[G,Pi]: "<<setw(10)<<rr_q<<endl;
   }

   // ═══════════════════════════════════════════════════════════════
   // CG BENCHMARKS (SPD system, no PML)
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ CG Benchmark (SPD, tol=1e-8, max=400) ═══\n";

   double bnorm=sqrt(B*B);

   auto run_cg=[&](const string &label, const DenseMatrix *Qmat){
      vector<double> res_hist;
      Vector X(ncurl); X=0.0;
      Vector R(B.Size()); R=B;
      Vector Z(R), Pz=Z;
      double rz=R*Z;
      int iter=0;
      for(;iter<400;iter++){
         Vector AP(ncurl); Ad.Mult(Pz,AP);
         double pap=Pz*AP;
         if(pap<1e-30) break;
         double alpha=rz/pap;
         for(int i=0;i<ncurl;i++) X[i]+=alpha*Pz[i];
         for(int i=0;i<ncurl;i++) R[i]-=alpha*AP[i];
         double rnorm=sqrt(R*R)/bnorm;
         res_hist.push_back(rnorm);
         if(rnorm<1e-8) break;
         // Preconditioner
         Z.SetSize(ncurl);
         if(Qmat==nullptr){
            // Block Jacobi
            for(int i=0;i<ncurl;i++) Z[i]=R[i]/max(1e-14,Ad(i,i));
         } else {
            QBasedPreconditioner prec(Ad,*Qmat);
            prec.Mult(R,Z);
         }
         double rz_new=R*Z;
         double beta=rz_new/rz;
         for(int i=0;i<ncurl;i++){Pz[i]=Z[i]+beta*Pz[i];}
         rz=rz_new;
      }
      if(myid==0){
         cout<<"  "<<left<<setw(32)<<label
             <<" iters="<<setw(4)<<iter+1
             <<" rel(10)="<<(res_hist.size()>9?res_hist[9]:-1.0)
             <<" rel(final)="<<(res_hist.empty()?0.0:res_hist.back());
         if(iter>=399) cout<<" (unconv.)";
         cout<<endl;
      }
      return iter+1;
   };

   // Build Q matrices
   DenseMatrix Q_default(ncurl,nh1+nh1v);
   for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++)Q_default(i,j)=Gd(i,j);for(int j=0;j<nh1v;j++)Q_default(i,nh1+j)=Pid(i,j);}
   DenseMatrix Q_exact(ncurl,nh1+no1);
   for(int i=0;i<ncurl;i++){for(int j=0;j<nh1;j++)Q_exact(i,j)=Gd(i,j);for(int j=0;j<no1;j++)Q_exact(i,nh1+j)=PiExact(i,j);}

   // Baselines
   run_cg("none (diag prec = Jacobi)",nullptr);
   // Wait, run_cg with Qmat==nullptr uses block Jacobi. Need no-preconditioner too.
   auto run_cg_none=[&](const string &label){
      Vector X(ncurl);X=0.0; Vector R(B.Size());R=B;Vector Z(R),Pz=Z;double rz=R*Z;int iter=0;vector<double> rh;
      for(;iter<400;iter++){
         Vector AP(ncurl);Ad.Mult(Pz,AP);double pap=Pz*AP;if(pap<1e-30)break;
         double alpha=rz/pap;
         for(int i=0;i<ncurl;i++)X[i]+=alpha*Pz[i];
         for(int i=0;i<ncurl;i++)R[i]-=alpha*AP[i];
         double rn=sqrt(R*R)/bnorm; rh.push_back(rn); if(rn<1e-8)break;
         Z=R; double rz_new=R*Z;double beta=rz_new/rz;
         for(int i=0;i<ncurl;i++)Pz[i]=Z[i]+beta*Pz[i]; rz=rz_new;
      }
      if(myid==0)cout<<"  "<<left<<setw(32)<<label<<" iters="<<setw(4)<<iter+1
               <<" rel(final)="<<scientific<<setprecision(3)<<(rh.empty()?0.0:rh.back());
         if(rh.size()>9) cout<<" rel(10)="<<rh[9];
      if(iter>=399)cout<<" (unconv.)";cout<<endl;
   };
   run_cg_none("none");

   auto run_cg_jac=[&](const string &label){
      Vector X(ncurl);X=0.0; Vector R(B.Size());R=B;Vector Z(ncurl),Pz;double rz=0;int iter=0;vector<double>rh;
      // Jacobi
      for(int i=0;i<ncurl;i++)Z[i]=R[i]/max(1e-14,Ad(i,i));
      Pz=Z; rz=R*Z;
      for(;iter<400;iter++){
         Vector AP(ncurl);Ad.Mult(Pz,AP);double pap=Pz*AP;if(pap<1e-30)break;
         double alpha=rz/pap;
         for(int i=0;i<ncurl;i++)X[i]+=alpha*Pz[i];
         for(int i=0;i<ncurl;i++)R[i]-=alpha*AP[i];
         double rn=sqrt(R*R)/bnorm; rh.push_back(rn); if(rn<1e-8)break;
         for(int i=0;i<ncurl;i++)Z[i]=R[i]/max(1e-14,Ad(i,i));
         double rz_new=R*Z;double beta=rz_new/rz;
         for(int i=0;i<ncurl;i++)Pz[i]=Z[i]+beta*Pz[i];rz=rz_new;
      }
      if(myid==0)cout<<"  "<<left<<setw(32)<<label<<" iters="<<setw(4)<<iter+1
               <<" rel(final)="<<scientific<<setprecision(3)<<(rh.empty()?0.0:rh.back());
         if(rh.size()>9) cout<<" rel(10)="<<rh[9];
      if(iter>=399)cout<<" (unconv.)";cout<<endl;
   };
   run_cg_jac("jacobi (diag)");

   // Coupled Q=[G,Pi] variants
   run_cg("Q=[G,Pi_def] + Jacobi",     &Q_default);
   run_cg("Q=[G,Pi_exact] + Jacobi",   &Q_exact);

   // Grad-only + Jacobi
   run_cg("Q=[G] + Jacobi", &Gd);
   // Pi_exact-only + Jacobi
   run_cg("Q=[Pi_exact] + Jacobi", &PiExact);

   if(myid==0) cout<<"\n═══ Done ═══\n"<<flush;

   delete Ghyp; delete Pihyp;
   delete o1_fes; delete h1vspace; delete h1space;
   delete fespace; delete fec; delete pmesh;
   return 0;
}
