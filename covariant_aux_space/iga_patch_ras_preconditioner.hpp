#ifndef COVARIANT_AUX_SPACE_IGA_PATCH_RAS_PRECONDITIONER_HPP
#define COVARIANT_AUX_SPACE_IGA_PATCH_RAS_PRECONDITIONER_HPP

#include "mfem.hpp"
#include <memory>
#include <vector>

namespace covariant_aux_space
{

class IGAPatchRASPreconditioner : public mfem::Solver
{
public:
   struct Options
   {
      enum class Assembly
      {
         Auto,
         LocalSparse,
         DenseProbe
      };

      int overlap_layers = 0;
      int iterations = 1;
      mfem::real_t damping = 1.0;
      bool verbose = true;
      Assembly assembly = Assembly::Auto;
   };

   IGAPatchRASPreconditioner(const mfem::Operator &A,
                             const mfem::ParFiniteElementSpace &fespace,
                             const Options &options);

   IGAPatchRASPreconditioner(const mfem::Operator &A,
                             const mfem::ParFiniteElementSpace &fespace,
                             int overlap_layers,
                             mfem::real_t damping,
                             int iterations);

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

   int NumPatches() const;
   int MaxPatchSize() const;

   struct PatchData
   {
      int num_patches;
      int max_patch_size;
      double damping;
      std::vector<std::vector<int>> ids;
      std::vector<mfem::DenseMatrix> inv_mats;
      mfem::Vector weights;  // effective: 1/coverage, scatter uses damping*weights[id]
   };
   PatchData ExtractPatchData() const;

private:
   const mfem::Operator &A_;
   const mfem::ParFiniteElementSpace &fespace_;
   Options options_;

   mutable bool built_ = false;
   mutable bool have_local_matrix_ = false;
   mutable int max_patch_size_ = 0;
   mutable mfem::DenseMatrix A_dense_;
   mutable mfem::SparseMatrix A_local_;
   mutable std::unique_ptr<mfem::HypreParMatrix> A_system_hypre_;
   mutable std::unique_ptr<mfem::SparseMatrix> A_system_sparse_;
   mutable mfem::Vector weights_;
   mutable mfem::Vector work_;
   mutable mfem::Vector residual_;
   mutable mfem::Vector Awork_;
   mutable std::vector<std::vector<int>> patches_;
   mutable std::vector<std::unique_ptr<mfem::DenseMatrix>> patch_mats_;
   mutable std::vector<std::unique_ptr<mfem::DenseMatrixInverse>> patch_invs_;

   static int AbsDof(int dof);
   void Build() const;
   void BuildDenseOperator() const;
   bool BuildLocalSparseOperator() const;
   void BuildElementPatches() const;
   void BuildPatchSolvers() const;
   void BuildPatchSolverFromDense(const std::vector<int> &ids) const;
   void BuildPatchSolverFromLocalSparse(const std::vector<int> &ids) const;
};

} // namespace covariant_aux_space

#endif
