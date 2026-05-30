#ifndef COVARIANT_AUX_SPACE_COVARIANT_REFERENCE_PRECONDITIONER_HPP
#define COVARIANT_AUX_SPACE_COVARIANT_REFERENCE_PRECONDITIONER_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include "../fdfd_iga_init/reference_initial_guess.hpp"
#include "yee_transfer.hpp"
#include "yee_operator.hpp"
#include <algorithm>
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
   void SetRealBlockMode(bool enable)
   { real_block_mode_ = enable; MarkDirty(); }

   void SetGrid(const fdfd_iga_init::ReferenceGrid &grid);
   void SetWaveNumber(double k0);
   void SetMaxIterations(int max_it);
   void SetDamping(double omega);
   void SetMassShift(double shift);
   void SetPrototypeMode(PrototypeMode mode);
   PrototypeMode GetPrototypeMode() const { return mode_; }
   void SetYeeDiagonalCalibration(bool enable) { yee_diag_calibration_ = enable; }
   void SetYeeLocalDiagonalCalibration(bool enable)
   { yee_local_diag_calibration_ = enable; MarkDirty(); }
   void SetYeeMaskedGalerkinCalibration(bool enable)
   { yee_masked_galerkin_calibration_ = enable; MarkDirty(); }
   void SetYeeGalerkinBlockRadius(int radius)
   { yee_galerkin_block_radius_ = std::max(0, radius); MarkDirty(); }
   void SetYeeCalibrationOperator(const mfem::Operator &op)
   { yee_calibration_op_ = &op; MarkDirty(); }
   void SetYeeReferencePML(bool enabled, double thickness = 0.25,
                           double strength = 5.0, double order = 2.0);
   void SetYeeComponentScales(double curl_scale, double mass_scale);
   void SetYeePMLGalerkinFallback(bool enable)
   { yee_pml_galerkin_fallback_ = enable; MarkDirty(); }
   void SetYeeIterativeAuxiliarySolve(bool enable)
   { yee_iterative_aux_solve_ = enable; MarkDirty(); }
   void SetYeeSparseAuxiliarySolve(bool enable)
   { yee_sparse_aux_solve_ = enable; MarkDirty(); }
   void SetYeeGMRESAuxiliarySolve(bool enable)
   { yee_gmres_aux_solve_ = enable; MarkDirty(); }
   void SetYeeBiCGSTABAuxiliarySolve(bool enable)
   { yee_bicgstab_aux_solve_ = enable; MarkDirty(); }
   void SetYeeComplexAuxiliary(bool enable)
   { yee_complex_auxiliary_ = enable; MarkDirty(); }
   void SetYeeSqrtPMLScaling(bool enable)
   { yee_sqrt_pml_scaling_ = enable; MarkDirty(); }
   void SetYeeIterativeAuxiliaryOptions(int max_iter, double rel_tol)
   {
      yee_iterative_aux_max_iter_ = std::max(1, max_iter);
      yee_iterative_aux_rel_tol_ = rel_tol;
   }
   void SetCoarseCorrectionWeight(double weight)
   { coarse_correction_weight_ = weight; }
   void SetIdentitySmootherWeight(double weight)
   { identity_smoother_weight_ = weight; }
   void SetOperatorJacobiSmootherWeight(double weight)
   { operator_jacobi_smoother_weight_ = weight; }
   void SetOperatorJacobiSmootherIterations(int iterations)
   { operator_jacobi_smoother_iterations_ = std::max(1, iterations); }
   void SetOperatorBlockJacobiSmootherWeight(double weight)
   { operator_block_jacobi_smoother_weight_ = weight; }
   void SetOperatorBlockJacobiSmootherIterations(int iterations)
   { operator_block_jacobi_smoother_iterations_ = std::max(1, iterations); }
   void SetOperatorPatchBlockSmootherWeight(double weight)
   { operator_patch_block_smoother_weight_ = weight; }
   void SetOperatorPatchBlockReferenceOperator(const mfem::Operator &op)
   { operator_patch_block_ref_op_ = &op; MarkDirty(); }
   void SetOperatorPatchBlockUseReference(bool enable)
   { operator_patch_block_use_ref_op_ = enable; MarkDirty(); }
   void SetOperatorElementPatchBlocks(bool enable)
   { operator_element_patch_blocks_ = enable; MarkDirty(); }
   void SetOperatorPatchBlockSize(int block_size)
   { operator_patch_block_size_ = std::max(1, block_size); }
   void SetOperatorPatchBlockSmootherIterations(int iterations)
   { operator_patch_block_smoother_iterations_ = std::max(1, iterations); }
   const fdfd_iga_init::ReferenceGrid &GetYeeGrid() const
   { return yee_operator_->Grid(); }
   void SetKnotAlignGrid(bool enable, int cells_per_span = 1);
   void PrintCoarseOperatorDiagnostics(std::ostream &os) const;
   void PrintYeeGalerkinComparison(std::ostream &os) const;
   void PrintYeeGalerkinComparison(const mfem::Operator &ref_op,
                                   const char *ref_name,
                                   std::ostream &os) const;
   void PrintYeeCandidateComparison(const mfem::Operator &ref_op,
                                    const mfem::DenseMatrix &Acand,
                                    const char *ref_name,
                                    const char *cand_name,
                                    std::ostream &os) const;

private:
   using AuxDof = YeeEdgeDof;

   const mfem::Operator *op_ = nullptr;
   const mfem::ParFiniteElementSpace &fespace_;
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   std::function<double(const mfem::Vector &)> eps_fn_;
   PrototypeMode mode_ = PrototypeMode::nodal_proto;
   double k0_ = 1.0;
   bool yee_diag_calibration_ = false;
   bool yee_local_diag_calibration_ = false;
   bool yee_masked_galerkin_calibration_ = false;
   int yee_galerkin_block_radius_ = 0;
   const mfem::Operator *yee_calibration_op_ = nullptr;
   bool knot_align_enabled_ = false;
   int knot_align_cps_ = 1;
   mutable bool knot_align_applied_ = false;
   mutable fdfd_iga_init::ReferenceFDFDCPU aux_solver_;
   mutable std::unique_ptr<YeeTransferBuilder> yee_transfer_;
   mutable std::unique_ptr<YeeOperatorBuilder> yee_operator_;
   mutable bool built_ = false;
   mutable std::vector<AuxDof> aux_dofs_;
   mutable mfem::DenseMatrix Pi_;
   mutable mfem::DenseMatrix Aaux_;
   mutable mfem::DenseMatrix AauxImag_;
   mutable mfem::DenseMatrix AauxBlock_;
   mutable std::unique_ptr<mfem::SparseMatrix> AauxSparse_;
   mutable std::unique_ptr<mfem::Solver> AauxSparsePrec_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> AauxInv_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> AauxBlockInv_;
   mutable mfem::Vector r_re_;
   mutable mfem::Vector r_im_;
   mutable mfem::Vector z_re_;
   mutable mfem::Vector z_im_;
   mutable mfem::Vector aux_rhs_;
   mutable mfem::Vector aux_sol_;
   mutable mfem::Vector aux_rhs_block_;
   mutable mfem::Vector aux_sol_block_;
   mutable mfem::Vector op_jacobi_inv_diag_;
   mutable mfem::Vector op_jacobi_work_;
   mutable mfem::Vector op_jacobi_res_;
   mutable mfem::Vector op_jacobi_Awork_;
   mutable mfem::Vector op_block_jacobi_inv00_;
   mutable mfem::Vector op_block_jacobi_inv01_;
   mutable mfem::Vector op_block_jacobi_inv10_;
   mutable mfem::Vector op_block_jacobi_inv11_;
   mutable mfem::Vector op_block_jacobi_work_;
   mutable mfem::Vector op_block_jacobi_res_;
   mutable mfem::Vector op_block_jacobi_Awork_;
   mutable std::vector<int> op_patch_block_starts_;
   mutable std::vector<int> op_patch_block_sizes_;
   mutable std::vector<std::vector<int>> op_patch_block_indices_;
   mutable std::vector<std::unique_ptr<mfem::DenseMatrix>> op_patch_block_mats_;
   mutable std::vector<std::unique_ptr<mfem::DenseMatrixInverse>> op_patch_block_invs_;
   mutable bool op_patch_block_scalar_ = false;
   mutable mfem::Vector op_patch_block_work_;
   mutable mfem::Vector op_patch_block_res_;
   mutable mfem::Vector op_patch_block_Awork_;
   bool yee_complex_auxiliary_ = false;
   bool real_block_mode_ = false;
   bool yee_pml_galerkin_fallback_ = true;
   bool yee_iterative_aux_solve_ = false;
   bool yee_sparse_aux_solve_ = false;
   bool yee_gmres_aux_solve_ = false;
   bool yee_bicgstab_aux_solve_ = false;
   bool yee_sqrt_pml_scaling_ = false;
   int yee_iterative_aux_max_iter_ = 80;
   double yee_iterative_aux_rel_tol_ = 1e-8;
   double coarse_correction_weight_ = 1.0;
   double identity_smoother_weight_ = 0.0;
   double operator_jacobi_smoother_weight_ = 0.0;
   int operator_jacobi_smoother_iterations_ = 1;
   double operator_block_jacobi_smoother_weight_ = 0.0;
   int operator_block_jacobi_smoother_iterations_ = 1;
   double operator_patch_block_smoother_weight_ = 0.0;
   const mfem::Operator *operator_patch_block_ref_op_ = nullptr;
   bool operator_patch_block_use_ref_op_ = false;
   bool operator_element_patch_blocks_ = false;
   int operator_patch_block_size_ = 8;
   int operator_patch_block_smoother_iterations_ = 1;

   void MarkDirty();
   void BuildAuxiliaryOperators() const;
   void BuildOperatorJacobiSmoother() const;
   void AddOperatorJacobiSmoother(const mfem::Vector &r,
                                  mfem::Vector &z) const;
   void BuildOperatorBlockJacobiSmoother() const;
   void AddOperatorBlockJacobiSmoother(const mfem::Vector &r,
                                       mfem::Vector &z) const;
   void BuildOperatorPatchBlockSmoother() const;
   void AddOperatorPatchBlockSmoother(const mfem::Vector &r,
                                      mfem::Vector &z) const;
   void BuildAuxDofs() const;
   void BuildTransferMatrix() const;
   void BuildAuxiliaryMatrix() const;
   void SolveAuxiliarySystem(const mfem::Vector &rhs,
                             mfem::Vector &sol) const;
   void SolveAuxiliaryBlockSystem(const mfem::Vector &rhs,
                                  mfem::Vector &sol) const;
   void BuildGalerkinEdgeCoarseMatrix(mfem::DenseMatrix &Agal) const;
   void BuildGalerkinEdgeCoarseMatrix(const mfem::Operator &ref_op,
                                      mfem::DenseMatrix &Agal) const;
   int FullVectorIndex(const AuxDof &dof) const;
   double EdgeLength(const AuxDof &dof) const;
   fdfd_iga_init::SampledReferenceField MakeAuxBasisField(const AuxDof &dof) const;
   double EvaluateAuxFunctional(const std::vector<double> &data,
                                const AuxDof &dof) const;
};

} // namespace covariant_aux_space

#endif
