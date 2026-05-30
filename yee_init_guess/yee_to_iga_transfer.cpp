#include "yee_to_iga_transfer.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace yee_init_guess
{

YeeToIGATransfer::YeeToIGATransfer(
   mfem::ParFiniteElementSpace &fine_fespace, int coarse_order)
   : fine_fespace_(fine_fespace), coarse_order_(coarse_order) {}

void YeeToIGATransfer::BuildYeeEdges(int nx, int ny, int nz)
{
   yee_edges_.clear();
   dx_ = 1.0/(nx-1); dy_ = 1.0/(ny-1); dz_ = 1.0/(nz-1);
   for (int k=1; k<nz-1; k++) for (int j=1; j<ny-1; j++) for (int i=0; i<nx-1; i++)
      yee_edges_.push_back({i,j,k,0});
   for (int k=1; k<nz-1; k++) for (int j=0; j<ny-1; j++) for (int i=1; i<nx-1; i++)
      yee_edges_.push_back({i,j,k,1});
   for (int k=0; k<nz-1; k++) for (int j=1; j<ny-1; j++) for (int i=1; i<nx-1; i++)
      yee_edges_.push_back({i,j,k,2});
   if (mfem::Mpi::WorldRank()==0)
      std::cout << "[yee_to_iga] Yee grid " << nx << "x" << ny << "x" << nz
                << ": " << yee_edges_.size() << " edges, dx=" << dx_ << "\n";
   built_ = false;
}

int YeeToIGATransfer::GetCoarseTrueVSize() const { Build(); return coarse_fespace_->GetTrueVSize(); }

void YeeToIGATransfer::Build() const {
   if (built_) return;
   const_cast<YeeToIGATransfer*>(this)->BuildCoarseSpace();
   const_cast<YeeToIGATransfer*>(this)->BuildPi1();
   const_cast<YeeToIGATransfer*>(this)->BuildProlongation();
   built_ = true;
}

void YeeToIGATransfer::BuildCoarseSpace() {
   mfem::ParMesh *pm = fine_fespace_.GetParMesh();
   coarse_fec_ = std::make_unique<mfem::NURBS_HCurlFECollection>(coarse_order_, pm->Dimension());
   coarse_nurbs_ext_ = std::make_unique<mfem::NURBSExtension>(pm->NURBSext, coarse_order_);
   coarse_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(pm, coarse_nurbs_ext_.get(), coarse_fec_.get());
   coarse_fespace_->StealNURBSext();
   if (mfem::Mpi::WorldRank()==0)
      std::cout << "[yee_to_iga] Coarse o=" << coarse_order_ << " H(curl): tvsize=" << coarse_fespace_->GetTrueVSize() << "\n";
}

void YeeToIGATransfer::BuildPi1() const {
   const int na = yee_edges_.size(), nc = coarse_fespace_->GetTrueVSize();
   Pi1_.SetSize(nc, na); Pi1_ = 0.0;
   if (na==0) return;

   mfem::ParFiniteElementSpace *cfs = const_cast<mfem::ParFiniteElementSpace*>(coarse_fespace_.get());
   int dim = cfs->GetParMesh()->SpaceDimension(), o = coarse_fespace_->GetOrder(0);
   mfem::NURBSExtension &ext = *const_cast<mfem::NURBSExtension*>(cfs->GetNURBSext());

   std::map<std::tuple<int,int,int>,int> ijk2el;
   for (int el=0; el<cfs->GetNE(); el++) { mfem::Array<int> ijk(dim); ext.GetElementIJK(el,ijk); ijk2el[{ijk[0],ijk[1],ijk[2]}]=el; }

   mfem::Array<const mfem::KnotVector*> kv; ext.GetPatchKnotVectors(0, kv);
   const auto &kvx = *kv[0], &kvy = (kv.Size()>1)?*kv[1]:kvx, &kvz = (kv.Size()>2)?*kv[2]:kvx;

   mfem::Array<int> vdofs;
   int matched=0; std::vector<bool> used(nc,false);

   for (int e=0; e<na; e++) {
      const auto &ye = yee_edges_[e]; int ax = ye.axis;
      double xi_x, xi_y, xi_z;
      if (ax==0) { xi_x=(ye.i+0.5)*dx_; xi_y=ye.j*dy_; xi_z=ye.k*dz_; }
      else if (ax==1) { xi_x=ye.i*dx_; xi_y=(ye.j+0.5)*dy_; xi_z=ye.k*dz_; }
      else { xi_x=ye.i*dx_; xi_y=ye.j*dy_; xi_z=(ye.k+0.5)*dz_; }

      int ix=-1,iy=-1,iz=-1;
      for (int i=0; i<kvx.GetNKS(); i++) if (kvx.isElement(i) && xi_x>=kvx[i+o]-1e-12 && xi_x<=kvx[i+o+1]+1e-12) {ix=i; break;}
      for (int j=0; j<kvy.GetNKS(); j++) if (kvy.isElement(j) && xi_y>=kvy[j+o]-1e-12 && xi_y<=kvy[j+o+1]+1e-12) {iy=j; break;}
      for (int k=0; k<kvz.GetNKS(); k++) if (kvz.isElement(k) && xi_z>=kvz[k+o]-1e-12 && xi_z<=kvz[k+o+1]+1e-12) {iz=k; break;}
      if (ix<0||iy<0||iz<0) continue;
      auto it = ijk2el.find({ix,iy,iz}); if (it==ijk2el.end()) continue;

      int el = it->second;
      double u=(xi_x-kvx[ix+o])/(kvx[ix+o+1]-kvx[ix+o]), v=(xi_y-kvy[iy+o])/(kvy[iy+o+1]-kvy[iy+o]), w=(xi_z-kvz[iz+o])/(kvz[iz+o+1]-kvz[iz+o]);
      u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v)); w=std::max(0.0,std::min(1.0,w));

      mfem::ElementTransformation *T = cfs->GetElementTransformation(el);
      const mfem::FiniteElement *fe = cfs->GetFE(el);
      if (!T||!fe) continue;
      mfem::IntegrationPoint ip; ip.Set3(u,v,w); T->SetIntPoint(&ip);

      int nd = fe->GetDof(); mfem::DenseMatrix vs(nd,dim); fe->CalcVShape(*T,vs);
      cfs->GetElementVDofs(el, vdofs);

      double best=0; int bd=-1;
      for (int i=0; i<nd; i++) { double val=std::abs(vs(i,ax)); if (val>best) { best=val; int d=vdofs[i]; bd=(d>=0)?d:(-d-1); } }
      if (bd>=0 && bd<nc && best>0) { Pi1_(bd,e)=1.0; used[bd]=true; matched++; }
   }

   int cov=0; for (bool u:used) if(u) cov++;
   if (mfem::Mpi::WorldRank()==0)
      std::cout << "[yee_to_iga] Pi1: " << matched << "/" << na << " edges matched, " << cov << "/" << nc << " DOFs covered\n";
}

void YeeToIGATransfer::BuildProlongation() const {
   const int nf = fine_fespace_.GetTrueVSize(), nc = coarse_fespace_->GetTrueVSize();
   P_.SetSize(nf, nc); P_ = 0.0;

   mfem::ParFiniteElementSpace *ffs = const_cast<mfem::ParFiniteElementSpace*>(&fine_fespace_);
   mfem::ConstantCoefficient one(1.0);

   // Lumped fine mass
   mfem::ParBilinearForm mf(ffs); mf.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mf.Assemble(); mf.Finalize();
   int nd_f = ffs->GetNDofs(); mfem::Vector md(nd_f); md=0.0;
   const auto &Ms = mf.SpMat(); const int *I=Ms.GetI(); const double *D=Ms.GetData();
   for (int r=0; r<Ms.Height(); r++) { double s=0; for (int p=I[r]; p<I[r+1]; p++) s+=D[p]; md[r]=s; }

   // Cross mass
   mfem::ParFiniteElementSpace *cfs = const_cast<mfem::ParFiniteElementSpace*>(coarse_fespace_.get());
   mfem::ParMixedBilinearForm cm(cfs, ffs); cm.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   cm.Assemble(); cm.Finalize();

   mfem::ParGridFunction gf_c(cfs), gf_f(ffs);
   mfem::Vector col_true(nf);
   for (int j=0; j<nc; j++) {
      mfem::Vector uj(nc); uj=0.0; uj[j]=1.0;
      gf_c = 0.0; gf_c.SetFromTrueDofs(uj);
      cm.Mult(gf_c, gf_f);
      for (int i=0; i<nd_f; i++) gf_f[i] = (std::abs(md[i])>1e-30) ? gf_f[i]/md[i] : 0.0;
      gf_f.GetTrueDofs(col_true);
      for (int i=0; i<nf; i++) P_(i,j)=col_true[i];
   }
   if (mfem::Mpi::WorldRank()==0) {
      double pn=0; for (int i=0; i<std::min(10,nf); i++) for (int j=0; j<std::min(10,nc); j++) pn+=P_(i,j)*P_(i,j);
      std::cout << "[yee_to_iga] Prolongation P: " << nf << "x" << nc << " sample_norm=" << std::sqrt(pn) << "\n";
   }
}

void YeeToIGATransfer::MapYeeToO1(const mfem::Vector &uy, mfem::Vector &u1) const {
   Build(); u1.SetSize(coarse_fespace_->GetTrueVSize()); Pi1_.Mult(uy, u1);
}
void YeeToIGATransfer::ProlongateO1ToOp(const mfem::Vector &u1, mfem::Vector &up) const {
   Build(); up.SetSize(fine_fespace_.GetTrueVSize()); P_.Mult(u1, up);
}
void YeeToIGATransfer::MapYeeToOp(const mfem::Vector &uy, mfem::Vector &up) const {
   Build(); mfem::Vector u1(coarse_fespace_->GetTrueVSize()); MapYeeToO1(uy, u1); ProlongateO1ToOp(u1, up);
}

} // namespace yee_init_guess
