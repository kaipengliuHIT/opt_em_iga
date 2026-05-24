#ifndef COVARIANT_AUX_SPACE_YEE_TRANSFER_HPP
#define COVARIANT_AUX_SPACE_YEE_TRANSFER_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include <functional>
#include <vector>

namespace covariant_aux_space
{

struct YeeEdgeDof
{
   int i = 0;
   int j = 0;
   int k = 0;
   int axis = 0;
};

class YeeTransferBuilder
{
public:
   YeeTransferBuilder(const mfem::ParFiniteElementSpace &fespace,
                      const fdfd_iga_init::SinglePatchNURBSEvaluator &geom);

   void SetGrid(const fdfd_iga_init::ReferenceGrid &grid);
   const fdfd_iga_init::ReferenceGrid &Grid() const { return grid_; }
   const std::vector<YeeEdgeDof> &GetEdgeDofs() const;
   void BuildProlongation(mfem::DenseMatrix &P) const;
   void BuildProlongationFast(mfem::DenseMatrix &P) const;
   void AssembleYeeCoarseOperator(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      mfem::DenseMatrix &A) const;

private:
   class EdgeBasisCoefficient : public mfem::VectorCoefficient
   {
   public:
      EdgeBasisCoefficient(const YeeTransferBuilder &builder,
                           const YeeEdgeDof &edge);

      void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override;

   private:
      const YeeTransferBuilder &builder_;
      const YeeEdgeDof &edge_;
   };

   const mfem::ParFiniteElementSpace &fespace_;
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   mfem::NURBSExtension &ext_;
   mfem::Array<const mfem::KnotVector *> patch_kv_;
   fdfd_iga_init::ReferenceGrid grid_;
   mutable std::vector<YeeEdgeDof> edge_dofs_;

   void BuildEdgeDofs() const;
   void GetGlobalPatchXi(int elem, const mfem::IntegrationPoint &ip,
                         mfem::Vector &xi) const;
   void EvaluateReferenceBasis(const YeeEdgeDof &edge,
                               const mfem::Vector &xi,
                               mfem::Vector &Ehat) const;
   void EvaluateReferenceCurl(const YeeEdgeDof &edge,
                              const mfem::Vector &xi,
                              mfem::Vector &curlEhat) const;
};

} // namespace covariant_aux_space

#endif
