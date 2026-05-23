#ifndef COVARIANT_AUX_SPACE_YEE_OPERATOR_HPP
#define COVARIANT_AUX_SPACE_YEE_OPERATOR_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include "yee_transfer.hpp"
#include <functional>
#include <vector>

namespace covariant_aux_space
{

struct YeeFaceDof
{
   int i = 0;
   int j = 0;
   int k = 0;
   int axis = 0; // normal direction: 0->yz, 1->xz, 2->xy
};

class YeeOperatorBuilder
{
public:
   YeeOperatorBuilder(const fdfd_iga_init::SinglePatchNURBSEvaluator &geom);

   void SetGrid(const fdfd_iga_init::ReferenceGrid &grid);
   const fdfd_iga_init::ReferenceGrid &Grid() const { return grid_; }

   const std::vector<YeeEdgeDof> &GetEdgeDofs() const;
   const std::vector<YeeFaceDof> &GetFaceDofs() const;

   void BuildCurlIncidence(mfem::DenseMatrix &C) const;

   void AssembleFaceMassMuInv(mfem::DenseMatrix &MmuInv) const;
   void AssembleEdgeMassEps(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      mfem::DenseMatrix &Meps) const;

   void AssembleYeeMaxwellOperator(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      mfem::DenseMatrix &Ayee) const;

   void PrintDiagnostics(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      std::ostream &os) const;

private:
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   fdfd_iga_init::ReferenceGrid grid_;
   mutable std::vector<YeeEdgeDof> edge_dofs_;
   mutable std::vector<YeeFaceDof> face_dofs_;

   void BuildEdgeDofs() const;
   void BuildFaceDofs() const;

   int XEdgeIndex(int i, int j, int k) const;
   int YEdgeIndex(int i, int j, int k) const;
   int ZEdgeIndex(int i, int j, int k) const;
   int YZFaceIndex(int i, int j, int k) const;
   int XZFaceIndex(int i, int j, int k) const;
   int XYFaceIndex(int i, int j, int k) const;
};

} // namespace covariant_aux_space

#endif
