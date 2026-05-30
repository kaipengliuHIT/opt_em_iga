#ifndef COVARIANT_AUX_SPACE_YEE_OPERATOR_HPP
#define COVARIANT_AUX_SPACE_YEE_OPERATOR_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include "yee_transfer.hpp"
#include <functional>
#include <complex>
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
   void SetReferencePML(bool enabled, double thickness = 0.25,
                        double strength = 5.0, double order = 2.0);
   bool IsReferencePMLEnabled() const { return pml_enabled_; }
   void SetComponentScales(double curl_scale, double mass_scale)
   {
      curl_scale_ = curl_scale;
      mass_scale_ = mass_scale;
   }

   const std::vector<YeeEdgeDof> &GetEdgeDofs() const;
   const std::vector<YeeFaceDof> &GetFaceDofs() const;

   void BuildCurlIncidence(mfem::DenseMatrix &C) const;
   void BuildCurlIncidenceComplex(double k0,
                                  mfem::DenseMatrix &Creal,
                                  mfem::DenseMatrix &Cimag) const;

   void AssembleFaceMassMuInv(mfem::DenseMatrix &MmuInv,
                              double k0 = 0.0) const;
   void AssembleEdgeMassEps(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      mfem::DenseMatrix &Meps,
      double k0 = 0.0) const;

   void AssembleYeeMaxwellOperator(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      mfem::DenseMatrix &Ayee) const;
   void AssembleYeeCurlOperator(mfem::DenseMatrix &CtMC,
                                double k0) const;
   void AssembleYeeMassOperator(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      mfem::DenseMatrix &K2Meps,
      double k0) const;
   void AssembleYeeMaxwellOperatorComplex(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      mfem::DenseMatrix &Areal,
      mfem::DenseMatrix &Aimag) const;
   void AssembleYeeMaxwellOperatorComplexSqrtScaled(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      mfem::DenseMatrix &Areal,
      mfem::DenseMatrix &Aimag) const;

   void PrintDiagnostics(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      double k0,
      std::ostream &os) const;

private:
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   fdfd_iga_init::ReferenceGrid grid_;
   bool pml_enabled_ = false;
   double pml_thickness_ = 0.25;
   double pml_strength_ = 5.0;
   double pml_order_ = 2.0;
   double curl_scale_ = 1.0;
   double mass_scale_ = 1.0;
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
   std::complex<double> StretchFactor(double xi, double k0) const;
   double PMLCurlWeight(const mfem::Vector &xi, int axis, double k0) const;
   double PMLMassWeight(const mfem::Vector &xi, int axis, double k0) const;
   std::complex<double> PMLCurlWeightComplex(const mfem::Vector &xi, int axis,
                                             double k0) const;
   std::complex<double> PMLMassWeightComplex(const mfem::Vector &xi, int axis,
                                             double k0) const;
   std::complex<double> EdgeSqrtInvStretchProduct(const YeeEdgeDof &edge,
                                                  double k0) const;
};

} // namespace covariant_aux_space

#endif
