#ifndef PML_SPLIT_PREC_HPP
#define PML_SPLIT_PREC_HPP

#include "mfem.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include "../fdfd_iga_init/reference_fdfd_cpu.hpp"
#include "../covariant_aux_space/yee_transfer.hpp"
#include "../covariant_aux_space/yee_operator.hpp"
#include <functional>
#include <memory>

namespace pml_split
{

class SplitPMLPreconditioner : public mfem::Solver
{
public:
   enum class CoarseMode
   {
      none,
      yee_curl_independent,
      yee_curl_galerkin,
      yee_full_galerkin
   };

   enum class CombineMode
   {
      jacobi_only,
      coarse_only,
      additive,
      multiplicative
   };

   SplitPMLPreconditioner(
      const mfem::ParFiniteElementSpace &fespace,
      const fdfd_iga_init::SinglePatchNURBSEvaluator &geom,
      const std::function<double(const mfem::Vector &)> &eps_fn);

   void SetGrid(const fdfd_iga_init::ReferenceGrid &grid);
   void SetWaveNumber(double k0);
   void SetCoarseMode(CoarseMode mode) { coarse_mode_ = mode; built_ = false; }
   void SetCombineMode(CombineMode mode) { combine_mode_ = mode; }
   void SetCoarseWeight(double w) { coarse_weight_ = w; }
   void SetJacobiWeight(double w) { jacobi_weight_ = w; }
   void SetCurlOperator(const mfem::Operator *op) { curl_op_ = op; built_ = false; }
   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;
   int GetAuxDofs() const { return na_; }
   void ApplyCoarsePublic(const mfem::Vector &r, mfem::Vector &z) const
   { if (!built_) { Build(); } ApplyCoarse(r, z); }

private:
   const mfem::ParFiniteElementSpace &fespace_;
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom_;
   std::function<double(const mfem::Vector &)> eps_fn_;
   CoarseMode coarse_mode_  = CoarseMode::yee_curl_independent;
   CombineMode combine_mode_ = CombineMode::multiplicative;
   double coarse_weight_ = 1.0, jacobi_weight_ = 1.0, k0_ = 1.0;
   mutable int na_ = 0;
   const mfem::Operator *op_ = nullptr, *curl_op_ = nullptr;
   std::unique_ptr<covariant_aux_space::YeeTransferBuilder> yee_transfer_;
   std::unique_ptr<covariant_aux_space::YeeOperatorBuilder> yee_operator_;
   mutable bool built_ = false;
   mutable mfem::DenseMatrix Pi_, Aaux_, AauxBlock_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> AauxInv_, AauxBlockInv_;
   mutable mfem::Vector bj00_, bj01_, bj10_, bj11_;
   mutable mfem::Vector aux_rhs_, aux_sol_, z1_, r1_, z2_;
   void BuildTransfer() const;
   void BuildCoarse() const;
   void BuildBlockJacobi() const;
   void ApplyCoarse(const mfem::Vector &r, mfem::Vector &z) const;
   void ApplyJacobi(const mfem::Vector &r, mfem::Vector &z) const;
   void Build() const;
};

// Two-level p-multigrid bridge: o=1 IGA H(curl) -> o=p IGA H(curl).
//
// o=2 A_p  <-- 2x2 block Jacobi smoother
//    | P_{1->p} = M_p^{-1} M_{p,1}  (mass projection)
// o=1 A_1  <-- edge_yee-preconditioned GMRES
//    | Pi  (Yee transfer)
//  Yee FDFD grid (cheap 5-point A_yee)
class PLevelGalerkinPreconditioner : public mfem::Solver
{
public:
   enum class CombineMode { coarse_only, additive, multiplicative };

   enum class CoarseSolveMode
   {
      dense_lu,   // Dense LU (current small-system diagnostic)
      edge_yee    // GMRES + Pi A_yee^{-1} Pi^T + block Jacobi on o=1
   };

   enum class CoarseMatrixMode
   {
      galerkin,        // A_1 = P^T A_p P
      direct_assemble  // A_1 assembled directly on coarse fespace (TODO)
   };

   explicit PLevelGalerkinPreconditioner(const mfem::ParFiniteElementSpace &fine_fespace,
                                         int coarse_order = 1);
   ~PLevelGalerkinPreconditioner() override;

   void SetCombineMode(CombineMode mode) { combine_mode_ = mode; }
   void SetCoarseWeight(double w) { coarse_weight_ = w; }
   void SetJacobiWeight(double w) { jacobi_weight_ = w; }
   void SetVerbose(bool v) { verbose_ = v; }

   // Coarse solve configuration
   void SetCoarseSolveMode(CoarseSolveMode mode) { coarse_solve_mode_ = mode; built_ = false; }
   void SetCoarseMatrixMode(CoarseMatrixMode mode) { coarse_matrix_mode_ = mode; built_ = false; }
   void SetCoarseYeeGrid(const fdfd_iga_init::ReferenceGrid &grid);
   void SetCoarseGeometry(const fdfd_iga_init::SinglePatchNURBSEvaluator &geom);
   void SetWaveNumber(double k0) { k0_ = k0; built_ = false; }
   void SetEpsFunc(std::function<double(const mfem::Vector &)> fn)
   { eps_fn_ = std::move(fn); }

   /// Get pre-computed Yee solution (for initial guess construction)
   const mfem::Vector &GetYeeSolution() const { return yee_sol_block_; }
   int GetYeeDofs() const { return na1_; }
   void SetCoarseGMRESParams(double tol, int max_iter, int kdim)
   { coarse_gmres_tol_ = tol; coarse_gmres_max_iter_ = max_iter; coarse_gmres_kdim_ = kdim; }

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;
   int GetCoarseTrueVSize() const;

private:
   const mfem::ParFiniteElementSpace &fine_fespace_;
   int coarse_order_ = 1;
   CombineMode combine_mode_ = CombineMode::multiplicative;
   double coarse_weight_ = 1.0, jacobi_weight_ = 1.0;
   bool verbose_ = true;
   CoarseSolveMode  coarse_solve_mode_  = CoarseSolveMode::dense_lu;
   CoarseMatrixMode coarse_matrix_mode_ = CoarseMatrixMode::galerkin;
   const mfem::Operator *op_ = nullptr;

   // Coarse FE space
   std::unique_ptr<mfem::FiniteElementCollection> coarse_fec_;
   std::unique_ptr<mfem::NURBSExtension>          coarse_nurbs_ext_;
   std::unique_ptr<mfem::ParFiniteElementSpace>   coarse_fespace_;

   // Coarse Yee infrastructure
   double k0_ = 1.0;
   std::function<double(const mfem::Vector &)> eps_fn_;
   fdfd_iga_init::ReferenceGrid coarse_yee_grid_;
   const fdfd_iga_init::SinglePatchNURBSEvaluator *coarse_geom_ = nullptr;
   mutable std::unique_ptr<covariant_aux_space::YeeTransferBuilder> coarse_yee_transfer_;
   mutable std::unique_ptr<covariant_aux_space::YeeOperatorBuilder> coarse_yee_operator_;
   mutable mfem::DenseMatrix Pi1_, Ayee1Block_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> Ayee1BlockInv_;
   mutable int na1_ = 0;
   mutable mfem::Vector c_bj00_, c_bj01_, c_bj10_, c_bj11_;
   mutable mfem::Vector yee_rhs_block_, yee_sol_block_;
   double coarse_gmres_tol_ = 1e-3;
   int coarse_gmres_max_iter_ = 80, coarse_gmres_kdim_ = 40;

   // p-level structures
   mutable bool built_ = false;
   mutable mfem::DenseMatrix P_, AcoarseBlock_;
   mutable std::unique_ptr<mfem::DenseMatrixInverse> AcoarseInv_;
   mutable mfem::Vector bj00_, bj01_, bj10_, bj11_;
   mutable mfem::Vector z1_, r1_, z2_;

   void Build() const;
   void BuildCoarseSpace();
   void BuildMassProjectionProlongation() const;
   void BuildGalerkinCoarse() const;
   void BuildBlockJacobi() const;
   void BuildCoarseYee() const;
   void BuildCoarseBlockJacobi() const;
   void ApplyCoarse(const mfem::Vector &r, mfem::Vector &z) const;
   void ApplyCoarseDenseLU(const mfem::Vector &r_coarse, mfem::Vector &z_coarse) const;
   void ApplyCoarseEdgeYee(const mfem::Vector &r_coarse, mfem::Vector &z_coarse) const;
   void ApplyJacobi(const mfem::Vector &r, mfem::Vector &z) const;
};

} // namespace pml_split

#endif
