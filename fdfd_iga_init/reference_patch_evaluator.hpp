#ifndef FDFD_IGA_INIT_REFERENCE_PATCH_EVALUATOR_HPP
#define FDFD_IGA_INIT_REFERENCE_PATCH_EVALUATOR_HPP

#include "mfem.hpp"
#include <vector>

namespace fdfd_iga_init
{

class SinglePatchNURBSEvaluator
{
public:
   SinglePatchNURBSEvaluator(mfem::Mesh &mesh,
                             const mfem::NURBSExtension &ext,
                             int patch_id = 0);

   int Dimension() const { return dim_; }

   void LocateElement(const mfem::Vector &xi, int &elem,
                      mfem::IntegrationPoint &ip) const;

   void EvalGeometry(const mfem::Vector &xi,
                     mfem::Vector &x_phys,
                     mfem::DenseMatrix &jac) const;

private:
   struct ElementLocator1D
   {
      int elem = 0;
      double xi = 0.0;
   };

   mfem::Mesh &mesh_;
   const mfem::NURBSExtension &ext_;
   mfem::Array<const mfem::KnotVector *> kv_;
   int dim_ = 0;
   int patch_id_ = 0;
   std::vector<int> element_map_;
   std::array<int, 3> ne_ = {1, 1, 1};

   int FlatElementIndex(int i, int j, int k) const
   {
      return (k * ne_[1] + j) * ne_[0] + i;
   }

   ElementLocator1D Locate1D(const mfem::KnotVector &kv, double u) const;
};

inline SinglePatchNURBSEvaluator::SinglePatchNURBSEvaluator(
   mfem::Mesh &mesh, const mfem::NURBSExtension &ext, int patch_id)
   : mesh_(mesh), ext_(ext), patch_id_(patch_id)
{
   MFEM_VERIFY(patch_id_ == 0 || ext_.GetNP() > patch_id_,
               "Invalid patch index for single-patch evaluator.");
   MFEM_VERIFY(ext_.GetNP() == 1 || patch_id_ == 0,
               "Current prototype only supports a single NURBS patch.");

   ext_.GetPatchKnotVectors(patch_id_, kv_);
   dim_ = kv_.Size();
   for (int d = 0; d < dim_; d++)
   {
      ne_[d] = kv_[d]->GetNKS();
   }

   element_map_.assign(ne_[0] * ne_[1] * ne_[2], -1);
   for (int e = 0; e < ext_.GetNE(); e++)
   {
      mfem::Array<int> ijk(dim_);
      const_cast<mfem::NURBSExtension &>(ext_).GetElementIJK(e, ijk);
      const int i = ijk[0];
      const int j = (dim_ > 1) ? ijk[1] : 0;
      const int k = (dim_ > 2) ? ijk[2] : 0;
      element_map_[FlatElementIndex(i, j, k)] = e;
   }
}

inline SinglePatchNURBSEvaluator::ElementLocator1D
SinglePatchNURBSEvaluator::Locate1D(const mfem::KnotVector &kv, double u) const
{
   const int p = kv.GetOrder();
   double uc = u;
   if (uc <= kv[p]) { uc = kv[p] + 1e-12; }
   if (uc >= kv[kv.GetNCP()]) { uc = kv[kv.GetNCP()] - 1e-12; }
   ElementLocator1D loc;
   const double tol = 1e-12;
   for (int i = 0; i < kv.GetNKS(); i++)
   {
      if (!kv.isElement(i)) { continue; }
      const double u0 = kv[p + i];
      const double u1 = kv[p + i + 1];
      if ((uc >= u0 - tol && uc <= u1 + tol) || (i + 1 == kv.GetNKS() && uc <= u1 + tol))
      {
         loc.elem = i;
         loc.xi = (uc - u0) / (u1 - u0);
         if (loc.xi < 0.0) { loc.xi = 0.0; }
         if (loc.xi > 1.0) { loc.xi = 1.0; }
         return loc;
      }
   }

   MFEM_ABORT("Failed to locate active NURBS span for parameter coordinate.");
}

inline void SinglePatchNURBSEvaluator::LocateElement(const mfem::Vector &xi,
                                                     int &elem,
                                                     mfem::IntegrationPoint &ip) const
{
   MFEM_VERIFY(xi.Size() == dim_, "Reference coordinate has wrong dimension.");

   const ElementLocator1D lx = Locate1D(*kv_[0], xi[0]);
   const ElementLocator1D ly = (dim_ > 1) ? Locate1D(*kv_[1], xi[1])
                                          : ElementLocator1D{};
   const ElementLocator1D lz = (dim_ > 2) ? Locate1D(*kv_[2], xi[2])
                                          : ElementLocator1D{};

   elem = element_map_[FlatElementIndex(lx.elem, ly.elem, lz.elem)];
   MFEM_VERIFY(elem >= 0, "Failed to locate NURBS element from reference point.");
   ip.Set3(lx.xi, ly.xi, lz.xi);
}

inline void SinglePatchNURBSEvaluator::EvalGeometry(const mfem::Vector &xi,
                                                    mfem::Vector &x_phys,
                                                    mfem::DenseMatrix &jac) const
{
   int elem = -1;
   mfem::IntegrationPoint ip;
   LocateElement(xi, elem, ip);

   mfem::ElementTransformation *T = mesh_.GetElementTransformation(elem);
   MFEM_VERIFY(T != nullptr, "Missing element transformation.");
   T->SetIntPoint(&ip);
   T->Transform(ip, x_phys);
   jac = T->Jacobian();
}

} // namespace fdfd_iga_init

#endif
