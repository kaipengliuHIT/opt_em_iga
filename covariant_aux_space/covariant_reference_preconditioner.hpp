#ifndef COVARIANT_AUX_SPACE_COVARIANT_REFERENCE_PRECONDITIONER_HPP
#define COVARIANT_AUX_SPACE_COVARIANT_REFERENCE_PRECONDITIONER_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include "../fdfd_iga_init/reference_initial_guess.hpp"
#include "yee_transfer.hpp"
#include "yee_operator.hpp"
#include <iosfwd>
#include <memory>
#include <vector>

namespace covariant_aux_space
{

class CovariantReferencePreconditioner : public mfem::Solver
{
public:
   enum class PrototypeMode
   {
      nodal_proto,
      edge_galerkin_proto,
      edge_yee_proto
   };

   CovariantReferencePreconditioner(
      const mfem::ParFiniteElementSpace &fespace,
      const fdfd_iga_init::SinglePatchNURBSEvaluator &geom,
      const std::function<double(const mfem::Vector &)> &eps_fn);

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

   void SetGrid(const fdfd_iga_init::ReferenceGrid &grid);
   void SetWaveNumber(double k0);
   void SetMaxIterations(int max_it);
   void SetDamping(double omega);
   void SetMassShift(double shift);
   void SetPrototypeMode(PrototypeMode mode);
   PrototypeMode GetPrototypeMode() const { return mode_; }
   void SetYeeDiagonalCalibration(bool enable) { yee_diag_calibration_ = enable; }
   void PrintCoarseOperatorDiagnostics(std::ostream &os) const;

private:
   using AuxDof = YeeEdgeDof;

   const mfem::Operator *op_ = nullptr;
   const mfem::ParFiniteElementSpace &fespace_;
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   std::function<double(const mfem::Vector &)> eps_fn_;
   PrototypeMode mode_ = PrototypeMode::nodal_proto;
   double k0_ = 1.0;
   bool yee_diag_calibration_ = false;
   mutable fdfd_iga_init::ReferenceFDFDCPU aux_solver_;
   mutable std::unique_ptr<YeeTransferBuilder> yee_transfer_;
   mutable std::unique_ptr<YeeOperatorBuilder> yee_operator_;
   mutable bool built_ = false;
   mutable std::vector<AuxDof> aux_dofs_;
   mutable mfem::DenseMatrix Pi_;
   mutable mfem::DenseMatrix Aaux_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> AauxInv_;
   mutable mfem::Vector r_re_;
   mutable mfem::Vector r_im_;
   mutable mfem::Vector z_re_;
   mutable mfem::Vector z_im_;
   mutable mfem::Vector aux_rhs_;
   mutable mfem::Vector aux_sol_;

   void MarkDirty();
   void BuildAuxiliaryOperators() const;
   void BuildAuxDofs() const;
   void BuildTransferMatrix() const;
   void BuildAuxiliaryMatrix() const;
   void BuildGalerkinEdgeCoarseMatrix(mfem::DenseMatrix &Agal) const;
   int FullVectorIndex(const AuxDof &dof) const;
   double EdgeLength(const AuxDof &dof) const;
   fdfd_iga_init::SampledReferenceField MakeAuxBasisField(const AuxDof &dof) const;
   double EvaluateAuxFunctional(const std::vector<double> &data,
                                const AuxDof &dof) const;
};

} // namespace covariant_aux_space

#endif
