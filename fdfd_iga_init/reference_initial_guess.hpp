#ifndef FDFD_IGA_INIT_REFERENCE_INITIAL_GUESS_HPP
#define FDFD_IGA_INIT_REFERENCE_INITIAL_GUESS_HPP

#include "mfem.hpp"
#include "reference_fdfd_cpu.hpp"

namespace fdfd_iga_init
{

class ReferenceFieldProjector
{
public:
   ReferenceFieldProjector(const mfem::ParFiniteElementSpace &fespace,
                           const SampledReferenceField &field);

   void Project(mfem::ParComplexGridFunction &x);
   void ProjectReal(mfem::ParGridFunction &x) const;
   void ProjectImag(mfem::ParGridFunction &x) const;

private:
   class SampledFieldCoefficient : public mfem::VectorCoefficient
   {
   public:
      SampledFieldCoefficient(const ReferenceFieldProjector &projector,
                             bool imag_part);

      void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override;

   private:
      const ReferenceFieldProjector &projector_;
      bool imag_part_;
   };

   const mfem::ParFiniteElementSpace &fespace_;
   mfem::NURBSExtension &ext_;
   const SampledReferenceField &field_;
   mfem::Array<const mfem::KnotVector *> patch_kv_;

   int SampleIndex(int i, int j, int k) const
   {
      return (k * field_.ny + j) * field_.nx + i;
   }

   void ProjectGridFunction(mfem::ParGridFunction &gf, bool imag_part) const;
   void GetGlobalPatchXi(int elem, const mfem::IntegrationPoint &ip,
                         mfem::Vector &xi) const;
   void SampleReferenceField(const mfem::Vector &xi, bool imag_part,
                             mfem::Vector &Ehat) const;
   int GetLocalDofDirection(const mfem::FiniteElement &fe, int ldof) const;
   void GetReferenceDofTangent(const mfem::FiniteElement &fe, int ldof,
                               mfem::Vector &tref) const;
};

} // namespace fdfd_iga_init

#endif
