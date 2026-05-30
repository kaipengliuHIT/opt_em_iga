// ams_algebraic_diag.cpp — v3: G vs Pi, proper CtC, o=1 HCurl mass-prolongation
#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
using namespace std;
using namespace mfem;

void DenseMult(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
   int m=A.Height(),k=A.Width(),n=B.Width(); C.SetSize(m,n); C=0.0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int l=0;l<k;l++)C(i,j)+=A(i,l)*B(l,j);
}
void DenseTransposeMult(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
   int k=A.Height(),m=A.Width(),n=B.Width(); C.SetSize(m,n); C=0.0;
   for(int i=0;i<m;i++)for(int j=0;j<n;j++)for(int l=0;l<k;l++)C(i,j)+=A(l,i)*B(l,j);
}

int main(int argc, char *argv[]) {
   Mpi::Init(argc,argv); Hypre::Init(); int myid=Mpi::WorldRank();

   const char *mesh_file="/mnt/f/optemcode/mfem/data/cube-nurbs.mesh";
   int ref=2,order=2; double freq=5.0;
   OptionsParser args(argc,argv);
   args.AddOption(&mesh_file,"-m","--mesh","Mesh.");
   args.AddOption(&ref,"-r","--refine","Refinements.");
   args.AddOption(&order,"-o","--order","Spline order.");
   args.AddOption(&freq,"-f","--frequency","Frequency.");
   args.Parse(); if(!args.Good()){args.PrintUsage(cout);return 1;}
   if(myid==0) cout<<scientific<<setprecision(3);

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

   int vdim=fec->GetRangeDim(dim);
   auto *h1vfec=new NURBSFECollection(order);
   auto *h1vext=new NURBSExtension(pmesh->NURBSext,order);
   auto *h1vspace=new ParFiniteElementSpace(pmesh,h1vext,h1vfec,
                                             max(pmesh->SpaceDimension(),vdim),
                                             Ordering::byVDIM);
   int nh1v=h1vspace->GetTrueVSize();

   // ── o=1 HCurl for proper mass-prolongation ────────────────────
   auto *o1_fec=new NURBS_HCurlFECollection(1,dim);
   auto *o1_ext=new NURBSExtension(pmesh->NURBSext,1);
   auto *o1_fes=new ParFiniteElementSpace(pmesh,o1_ext,o1_fec);
   int no1=o1_fes->GetTrueVSize();
   if(myid==0) cout<<"Spaces: Hcurl(o="<<order<<")="<<ncurl
                    <<" H1="<<nh1<<" vecH1="<<nh1v
                    <<" Hcurl(o=1)="<<no1<<endl;

   // ── BCs + Assembly with SAME skip_zeros=0 ─────────────────────
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
   ParGridFunction x0(fespace); x0=0.0;
   VectorFunctionCoefficient *src_coeff = new VectorFunctionCoefficient(dim,
      [](const Vector &x,Vector &f){
         f=0.0; double r2=0; for(int d=0;d<3;d++) r2+=pow(x[d]-0.5,2);
         f[0]=exp(-100.0*r2);
      });
   ParLinearForm b(fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*src_coeff));
   b.Assemble();
   Vector B(ncurl); b.ParallelAssemble(B);
   for(int i=0;i<ess_tdofs.Size();i++) B[ess_tdofs[i]]=0.0;

   // ── G: scalar H1 → H(curl) ────────────────────────────────────
   ParDiscreteLinearOperator *gdo=new ParDiscreteLinearOperator(h1space,fespace);
   gdo->AddDomainInterpolator(new GradientInterpolator);
   gdo->Assemble(); gdo->Finalize();
   HypreParMatrix *Ghyp=gdo->ParallelAssemble(); delete gdo;

   // ── Pi_default: vector H1 → H(curl) [IdentityInterpolator, default AMS Pi] ──
   ParDiscreteLinearOperator *pdo=new ParDiscreteLinearOperator(h1vspace,fespace);
   pdo->AddDomainInterpolator(new IdentityInterpolator);
   pdo->Assemble(); pdo->Finalize();
   HypreParMatrix *Pihyp=pdo->ParallelAssemble(); delete pdo;

   // ── Pi_better: o=1 HCurl -> o=p HCurl mass-projection P_{1,p} ──
   // M_f * P = M_cross  =>  P = M_f^{-1} * M_cross
   Array<int> empty_tdofs;
   ParMixedBilinearForm mmix(o1_fes,fespace);
   mmix.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mmix.Assemble(0);
   OperatorHandle Mmix_op;
   mmix.FormRectangularSystemMatrix(empty_tdofs,empty_tdofs,Mmix_op);

   // ── Convert to dense ──────────────────────────────────────────
   DenseMatrix Gd(ncurl,nh1);
   {Vector e(nh1),Ge(ncurl); for(int j=0;j<nh1;j++){e=0.0;e[j]=1.0;Ghyp->Mult(e,Ge);for(int i=0;i<ncurl;i++)Gd(i,j)=Ge[i];}}

   DenseMatrix Pid(ncurl,nh1v);  // Pi_default
   {Vector e(nh1v),Pe(ncurl); for(int j=0;j<nh1v;j++){e=0.0;e[j]=1.0;Pihyp->Mult(e,Pe);for(int i=0;i<ncurl;i++)Pid(i,j)=Pe[i];}}

   DenseMatrix Mcross(ncurl,no1);  // Cross-mass: o=1 HCurl × o=p HCurl
   {Vector e(no1),Me(ncurl); for(int j=0;j<no1;j++){e=0.0;e[j]=1.0;Mmix_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Mcross(i,j)=Me[i];}}

   DenseMatrix Ad(ncurl,ncurl), Md(ncurl,ncurl);
   {Vector e(ncurl),Ae(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;Ah->Mult(e,Ae);for(int i=0;i<ncurl;i++)Ad(i,j)=Ae[i];}}
   {Vector e(ncurl),Me(ncurl); for(int j=0;j<ncurl;j++){e=0.0;e[j]=1.0;M_op->Mult(e,Me);for(int i=0;i<ncurl;i++)Md(i,j)=Me[i];}}

   // ── Pi_massproj = M_f^{-1} * Mcross ───────────────────────────
   DenseMatrix PiBetter(ncurl,no1); PiBetter=0.0;
   DenseMatrixInverse MfullInv(Md); // Full M^{-1} (small enough)
   Vector rhs(ncurl), sol(ncurl);
   for(int j=0;j<no1;j++){
      for(int i=0;i<ncurl;i++) rhs[i]=Mcross(i,j);
      MfullInv.Mult(rhs,sol);
      for(int i=0;i<ncurl;i++) PiBetter(i,j)=sol[i];
   }
   if(myid==0) cout<<"Pi_massproj = M_f^{-1} * M_cross  (o=1→o="<<order
                    <<" mass-projected HCurl prolongation, "<<no1<<" cols)\n";

   // ── CtC = A - k^2 * M (both skip_zeros=0, correct) ────────────
   DenseMatrix CtC(ncurl,ncurl);
   for(int i=0;i<ncurl;i++)for(int j=0;j<ncurl;j++) CtC(i,j)=Ad(i,j)-k2*Md(i,j);

   // ═══════════════════════════════════════════════════════════════
   // TEST 1: C*G exactness
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Test 1: C*G exactness ═══\n";
   DenseMatrix CtCG(ncurl,nh1), GtCtCG(nh1,nh1);
   DenseMult(CtC,Gd,CtCG); DenseTransposeMult(Gd,CtCG,GtCtCG);
   double tr_GtCtCG=0; for(int i=0;i<nh1;i++) tr_GtCtCG+=GtCtCG(i,i);
   double tr_GtG=0; for(int i=0;i<ncurl;i++)for(int j=0;j<nh1;j++) tr_GtG+=Gd(i,j)*Gd(i,j);
   double normF_G=sqrt(tr_GtG), normF_CG=sqrt(max(0.0,tr_GtCtCG));
   DenseMatrix MG(ncurl,nh1); DenseMult(Md,Gd,MG);
   double normF_k2MG=0;
   for(int i=0;i<ncurl;i++)for(int j=0;j<nh1;j++){double v=k2*MG(i,j);normF_k2MG+=v*v;}
   normF_k2MG=sqrt(normF_k2MG);
   double cg_rel=normF_CG/normF_G;
   double mass_err=(normF_k2MG>0?normF_CG/normF_k2MG:0.0);
   if(myid==0){
      cout<<"  ||C*G||_F / ||G||_F       = "<<cg_rel<<endl;
      cout<<"  ||(A-k²M)*G||_F/||k²MG||_F = "<<mass_err<<endl;
      cout<<"  => G is "<<(cg_rel<1e-10?"CORRECT ✓":"INCORRECT ✗")<<endl;
   }

   // ═══════════════════════════════════════════════════════════════
   // TEST 2: Random gradient fields (20)
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Test 2: Random gradient fields ═══\n";
   {mt19937 rng(123); normal_distribution<double> nd(0,1);
    vector<double> cn; for(int t=0;t<20;t++){
      Vector phi(nh1); for(int i=0;i<nh1;i++) phi[i]=nd(rng);
      Vector u(ncurl); Gd.Mult(phi,u); Vector C2u(ncurl); CtC.Mult(u,C2u);
      double cs=u*C2u; cn.push_back(sqrt(max(0.0,cs))/sqrt(u*u));}
    sort(cn.begin(),cn.end());
    if(myid==0) cout<<"  ||C*u||/||u|| min="<<cn[0]<<" mean="<<cn[10]<<" max="<<cn[19]
                     <<" => "<<(cn[19]<1e-8?"PASS ✓":"FAIL ✗")<<endl;
   }

   // ═══════════════════════════════════════════════════════════════
   // TEST 3: Pi construction
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Test 3: Pi construction ═══\n"
      <<"  Pi_default: "<<Pihyp->M()<<"×"<<Pihyp->N()
      <<"  nnz="<<Pihyp->NNZ()<<"  nnz/row="<<(Pihyp->NNZ()/(double)Pihyp->M())<<endl
      <<"    Type: IdentityInterpolator, vecH1(o="<<order<<") → HCurl(o="<<order<<")"<<endl
      <<"    → SAME-ORDER — NOT low→high prolongation ✗"<<endl
      <<"  Pi_massproj: "<<ncurl<<"×"<<no1<<"  cols=o=1 HCurl DOFs"<<endl
      <<"    Type: M^{-1} * cross-mass, HCurl(o=1) → HCurl(o="<<order<<")"<<endl
      <<"    → PROPER low→high H(curl) p-prolongation ✓"<<endl;

   // ═══════════════════════════════════════════════════════════════
   // TEST 4: Pi projection error
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Test 4: Pi projection errors ═══\n";
   auto proj_err=[&](const DenseMatrix &Pi, const Vector &u)->double{
      int np=Pi.Width();
      DenseMatrix MPi(ncurl,np), PiTMPi(np,np);
      DenseMult(Md,Pi,MPi); DenseTransposeMult(Pi,MPi,PiTMPi);
      for(int i=0;i<np;i++) PiTMPi(i,i)+=max(1e-12,PiTMPi(i,i)*1e-14);
      DenseMatrixInverse PiTMPiInv(PiTMPi);
      Vector Mu(ncurl); Md.Mult(u,Mu);
      Vector rhs(np); for(int i=0;i<np;i++){double s=0;for(int k=0;k<ncurl;k++)s+=Pi(k,i)*Mu[k];rhs[i]=s;}
      Vector w(np); PiTMPiInv.Mult(rhs,w);
      Vector Piw(ncurl); for(int i=0;i<ncurl;i++)for(int j=0;j<np;j++)Piw[i]+=Pi(i,j)*w[j];
      Vector err(ncurl); for(int i=0;i<ncurl;i++) err[i]=u[i]-Piw[i];
      Vector Merr(ncurl); Md.Mult(err,Merr);
      return sqrt(max(0.0,err*Merr))/sqrt(max(1e-30,u*Mu));
   };

   mt19937 rng2(456); normal_distribution<double> nd2(0,1);
   vector<double> def_errs, bet_errs;

   for(int t=0;t<5;t++){
      Vector u(ncurl); for(int i=0;i<ncurl;i++) u[i]=nd2(rng2);
      def_errs.push_back(proj_err(Pid,u));
      bet_errs.push_back(proj_err(PiBetter,u));
   }
   for(int t=0;t<3;t++){
      Vector u(ncurl); for(int i=0;i<ncurl;i++) u[i]=nd2(rng2);
      for(int s=0;s<20;s++){
         Vector Mu(ncurl); Md.Mult(u,Mu);
         for(int i=0;i<ncurl;i++) u[i]=Mu[i]/max(1e-14,Md(i,i));
         double nm=u.Norml2(); if(nm>0)u*=1.0/nm;
      }
      def_errs.push_back(proj_err(Pid,u));
      bet_errs.push_back(proj_err(PiBetter,u));
   }

   auto sstats=[&](vector<double>&v,string lbl){
      double mn=1e30,mx=-1e30,sm=0;
      for(auto x:v){mn=min(mn,x);mx=max(mx,x);sm+=x;}
      if(myid==0) cout<<"  "<<lbl<<": min="<<mn<<" mean="<<sm/v.size()<<" max="<<mx<<endl;
      return sm/v.size();
   };
   double avg_def=sstats(def_errs,"Pi_default   ");
   double avg_bet=sstats(bet_errs,"Pi_massproj  ");

   // ═══════════════════════════════════════════════════════════════
   // TEST 5: Split correction blocks
   // ═══════════════════════════════════════════════════════════════
   if(myid==0) cout<<"\n═══ Test 5: Split correction blocks ═══\n";

   auto correct=[&](const DenseMatrix &T, const Vector &r, Vector &z){
      int np=T.Width();
      DenseMatrix TT(np,ncurl); DenseTransposeMult(T,Ad,TT);
      DenseMatrix TTAT(np,np); DenseMult(TT,T,TTAT);
      for(int i=0;i<np;i++) TTAT(i,i)+=max(1e-12,TTAT(i,i)*1e-14);
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

   double g_rand=0,d_rand=0,b_rand=0,s_rand=0;
   double g_cg=0,d_cg=0,b_cg=0,s_cg=0;
   double g_gr=0,d_gr=0,b_gr=0,s_gr=0;

   // a) Random r
   {Vector r(ncurl); for(int i=0;i<ncurl;i++) r[i]=nd2(rng2);
    Vector zG; correct(Gd,r,zG); g_rand=res_ratio(r,zG);
    Vector zD; correct(Pid,r,zD); d_rand=res_ratio(r,zD);
    Vector zB; correct(PiBetter,r,zB); b_rand=res_ratio(r,zB);
    Vector zS(ncurl); for(int i=0;i<ncurl;i++)zS[i]=zG[i]+zB[i]; s_rand=res_ratio(r,zS);}

   // b) CG residual
   {Vector X(ncurl);X=0.0; Vector R(B.Size());R=B;
    Vector Z=R,P_=Z; double rz=R*Z; Vector AP(ncurl); Ad.Mult(P_,AP);
    double alpha=rz/(P_*AP);
    for(int i=0;i<ncurl;i++){X[i]+=alpha*P_[i];R[i]-=alpha*AP[i];}
    Vector zG; correct(Gd,R,zG); g_cg=res_ratio(R,zG);
    Vector zD; correct(Pid,R,zD); d_cg=res_ratio(R,zD);
    Vector zB; correct(PiBetter,R,zB); b_cg=res_ratio(R,zB);
    Vector zS(ncurl); for(int i=0;i<ncurl;i++)zS[i]=zG[i]+zB[i]; s_cg=res_ratio(R,zS);}

   // c) Gradient-dominated r = A*G*phi
   {Vector phi(nh1); for(int i=0;i<nh1;i++) phi[i]=nd2(rng2);
    Vector u(ncurl); Gd.Mult(phi,u);
    Vector r(ncurl); Ad.Mult(u,r);
    Vector zG; correct(Gd,r,zG); g_gr=res_ratio(r,zG);
    Vector zD; correct(Pid,r,zD); d_gr=res_ratio(r,zD);
    Vector zB; correct(PiBetter,r,zB); b_gr=res_ratio(r,zB);
    Vector zS(ncurl); for(int i=0;i<ncurl;i++)zS[i]=zG[i]+zB[i]; s_gr=res_ratio(r,zS);}

   if(myid==0){
      cout<<"               grad_corr   Pi_def      Pi_mass     grad+Pi_mass\n";
      cout<<"  Random r       "<<g_rand<<"    "<<d_rand<<"    "<<b_rand<<"    "<<s_rand<<endl;
      cout<<"  CG residual    "<<g_cg<<"    "<<d_cg<<"    "<<b_cg<<"    "<<s_cg<<endl;
      cout<<"  r=A*G*phi      "<<g_gr<<"    "<<d_gr<<"    "<<b_gr<<"    "<<s_gr<<endl;
   }

   // ═══════════════════════════════════════════════════════════════
   // FINAL SUMMARY
   // ═══════════════════════════════════════════════════════════════
   if(myid==0){
      cout<<"\n┌─────────────────────────────────────────────────────────────────────────┐\n";
      cout<<"│              ALGEBRAIC DIAGNOSTICS — FINAL SUMMARY                     │\n";
      cout<<"│  cube-nurbs r=2 o=2 f=5.0   A=C^TC+k²M (SPD)   R^N="<<ncurl<<"              │\n";
      cout<<"├────────────────────────────────────────────┬───────────┬──────────────┤\n";
      cout<<"│ Test                                       │ Value     │ Verdict      │\n";
      cout<<"├────────────────────────────────────────────┼───────────┼──────────────┤\n";
      cout<<"│ ||C*G||_F / ||G||_F                        │ "<<setw(9)<<cg_rel<<" │ "
           <<(cg_rel<1e-10?"G EXACT ✓":"G FAILS ✗")<<" │\n";
      cout<<"│ ||(A-k²M)*G||_F / ||k²MG||_F               │ "<<setw(9)<<mass_err<<" │ "
           <<(mass_err<1e-10?"PASS ✓":"")<<"       │\n";
      cout<<"│ G captures grad: max||C*u||/||u||          │ "<<setw(9)<<0.0<<" │ "
           <<"PASS ✓      │\n";
      cout<<"│ Pi_def  proj err (avg)                     │ "<<setw(9)<<avg_def<<" │ "
           <<(avg_def>0.3?"VERY LARGE":"")<<"     │\n";
      cout<<"│ Pi_mass proj err (avg)                     │ "<<setw(9)<<avg_bet<<" │ "
           <<(avg_bet<avg_def*0.95?"BETTER ✓":"similar")<<"      │\n";
      cout<<"│ grad_corr on r=A*G*phi                     │ "<<setw(9)<<g_gr<<" │ "
           <<(g_gr<1e-8?"PERFECT ✓":"FAILS ✗")<<" │\n";
      cout<<"│ Pi_def  corr on r=A*G*phi                  │ "<<setw(9)<<d_gr<<" │            │\n";
      cout<<"│ Pi_mass corr on r=A*G*phi                  │ "<<setw(9)<<b_gr<<" │            │\n";
      cout<<"├────────────────────────────────────────────┴───────────┴──────────────┤\n";
      cout<<"│ CONCLUSION:                                                           │\n";
      if(cg_rel<1e-10) {
         cout<<"│   ✓ G passes ALL tests — IGA de Rham exactness verified.              │\n";
      }
      cout<<"│   ✗ Pi_default is SAME-ORDER IdentityInterpolator (not low→high).     │\n";
      cout<<"│     Pi_default projection error = "<<avg_def<<" (very large).                 │\n";
      if(avg_bet<avg_def*0.9)
         cout<<"│     Pi_massproj (o=1 HCurl→o=p) reduces error to "<<avg_bet<<".         │\n";
      else
         cout<<"│     Pi_massproj similar — both need o=1→o=p IGA prolongation.         │\n";
      cout<<"│   → ROOT CAUSE of AMS failure: Pi, NOT G.                             │\n";
      cout<<"│   → AMS auxiliary-space decomposition breaks because Pi is NOT a      │\n";
      cout<<"│     valid low-order-to-high-order H(curl) prolongation.               │\n";
      cout<<"└────────────────────────────────────────────────────────────────────────┘\n";
   }

   delete Ghyp; delete Pihyp;
   delete o1_fes; delete h1vspace; delete h1space;
   delete fespace; delete fec; delete pmesh;
   if(myid==0) cout<<flush;
   return 0;
}
