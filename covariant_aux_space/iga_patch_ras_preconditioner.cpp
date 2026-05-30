#include "iga_patch_ras_preconditioner.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace covariant_aux_space
{

namespace
{
using mfem::DenseMatrix;
using mfem::DenseMatrixInverse;
using mfem::ComplexHypreParMatrix;
using mfem::ComplexSparseMatrix;
using mfem::HypreParMatrix;
using mfem::Mpi;
using mfem::ParFiniteElementSpace;
using mfem::SparseMatrix;
using mfem::Vector;
}

IGAPatchRASPreconditioner::IGAPatchRASPreconditioner(
   const mfem::Operator &A,
   const ParFiniteElementSpace &fespace,
   const Options &options)
   : mfem::Solver(A.Height(), A.Width()),
     A_(A),
     fespace_(fespace),
     options_(options)
{
   options_.overlap_layers = std::max(0, options_.overlap_layers);
   options_.iterations = std::max(1, options_.iterations);
}

IGAPatchRASPreconditioner::IGAPatchRASPreconditioner(
   const mfem::Operator &A,
   const ParFiniteElementSpace &fespace,
   int overlap_layers,
   mfem::real_t damping,
   int iterations)
   : IGAPatchRASPreconditioner(
        A, fespace,
        Options{std::max(0, overlap_layers), std::max(1, iterations),
                damping, true, Options::Assembly::Auto})
{
}

void IGAPatchRASPreconditioner::SetOperator(const mfem::Operator &op)
{
   MFEM_VERIFY(&op == &A_, "IGAPatchRASPreconditioner is bound to a fixed A.");
}

int IGAPatchRASPreconditioner::NumPatches() const
{
   Build();
   return static_cast<int>(patches_.size());
}

int IGAPatchRASPreconditioner::MaxPatchSize() const
{
   Build();
   return max_patch_size_;
}

int IGAPatchRASPreconditioner::AbsDof(int dof)
{
   return (dof >= 0) ? dof : (-1 - dof);
}

void IGAPatchRASPreconditioner::BuildDenseOperator() const
{
   const int n = A_.Height();
   A_dense_.SetSize(n, n);
   Vector e(n), Ae(n);
   for (int j = 0; j < n; j++)
   {
      e = 0.0;
      e[j] = 1.0;
      A_.Mult(e, Ae);
      for (int i = 0; i < n; i++) { A_dense_(i, j) = Ae[i]; }
   }
}

bool IGAPatchRASPreconditioner::BuildLocalSparseOperator() const
{
   if (const auto *complex_hypre =
          dynamic_cast<const ComplexHypreParMatrix *>(&A_))
   {
      A_system_hypre_.reset(complex_hypre->GetSystemMatrix());
      A_system_hypre_->GetDiag(A_local_);
      have_local_matrix_ = true;
      return true;
   }

   if (const auto *hypre = dynamic_cast<const HypreParMatrix *>(&A_))
   {
      hypre->GetDiag(A_local_);
      have_local_matrix_ = true;
      return true;
   }

   if (const auto *complex_sparse =
          dynamic_cast<const ComplexSparseMatrix *>(&A_))
   {
      A_system_sparse_.reset(complex_sparse->GetSystemMatrix());
      A_local_.MakeRef(*A_system_sparse_);
      have_local_matrix_ = true;
      return true;
   }

   if (const auto *sparse = dynamic_cast<const SparseMatrix *>(&A_))
   {
      A_local_.MakeRef(*sparse);
      have_local_matrix_ = true;
      return true;
   }

   have_local_matrix_ = false;
   return false;
}

void IGAPatchRASPreconditioner::BuildElementPatches() const
{
   const int n = fespace_.GetTrueVSize();
   const int ne = fespace_.GetNE();
   std::vector<std::vector<int>> elem_tdofs(ne);
   std::vector<std::vector<int>> dof_to_elems(n);
   mfem::Array<int> vdofs;

   for (int e = 0; e < ne; e++)
   {
      fespace_.GetElementVDofs(e, vdofs);
      for (int q = 0; q < vdofs.Size(); q++)
      {
         const int ldof = AbsDof(vdofs[q]);
         const int tdof = fespace_.GetLocalTDofNumber(ldof);
         if (tdof >= 0 && tdof < n)
         {
            elem_tdofs[e].push_back(tdof);
         }
      }
      std::sort(elem_tdofs[e].begin(), elem_tdofs[e].end());
      elem_tdofs[e].erase(std::unique(elem_tdofs[e].begin(),
                                      elem_tdofs[e].end()),
                          elem_tdofs[e].end());
      for (int tdof : elem_tdofs[e]) { dof_to_elems[tdof].push_back(e); }
   }

   std::vector<int> coverage(n, 0);
   std::vector<char> selected(ne), next_selected(ne);
   for (int seed = 0; seed < ne; seed++)
   {
      std::fill(selected.begin(), selected.end(), 0);
      selected[seed] = 1;
      for (int layer = 0; layer < options_.overlap_layers; layer++)
      {
         next_selected = selected;
         for (int e = 0; e < ne; e++)
         {
            if (!selected[e]) { continue; }
            for (int tdof : elem_tdofs[e])
            {
               for (int nbr : dof_to_elems[tdof]) { next_selected[nbr] = 1; }
            }
         }
         selected.swap(next_selected);
      }

      std::vector<int> scalar_ids;
      for (int e = 0; e < ne; e++)
      {
         if (!selected[e]) { continue; }
         scalar_ids.insert(scalar_ids.end(), elem_tdofs[e].begin(),
                           elem_tdofs[e].end());
      }
      std::sort(scalar_ids.begin(), scalar_ids.end());
      scalar_ids.erase(std::unique(scalar_ids.begin(), scalar_ids.end()),
                       scalar_ids.end());
      if (scalar_ids.empty()) { continue; }

      std::vector<int> block_ids;
      block_ids.reserve(2 * scalar_ids.size());
      for (int id : scalar_ids) { block_ids.push_back(id); }
      for (int id : scalar_ids) { block_ids.push_back(n + id); }
      for (int id : scalar_ids) { coverage[id]++; }
      patches_.push_back(std::move(block_ids));
   }

   for (int i = 0; i < n; i++)
   {
      if (coverage[i] == 0)
      {
         patches_.push_back(std::vector<int>{i, n + i});
         coverage[i] = 1;
      }
   }

   weights_.SetSize(2 * n);
   weights_ = 0.0;
   for (int i = 0; i < n; i++)
   {
      const mfem::real_t w = 1.0 / mfem::real_t(coverage[i]);
      weights_[i] = w;
      weights_[n + i] = w;
   }
}

void IGAPatchRASPreconditioner::BuildPatchSolverFromDense(
   const std::vector<int> &ids) const
{
   const int m = static_cast<int>(ids.size());
   max_patch_size_ = std::max(max_patch_size_, m);
   auto block = std::make_unique<DenseMatrix>(m, m);
   for (int j = 0; j < m; j++)
   {
      for (int i = 0; i < m; i++)
      {
         (*block)(i, j) = A_dense_(ids[i], ids[j]);
      }
   }

   mfem::real_t max_diag = 0.0;
   for (int i = 0; i < m; i++)
   {
      max_diag = std::max(max_diag, std::abs((*block)(i, i)));
   }
   const mfem::real_t eps_reg =
      (max_diag > 0.0) ?
      std::max(mfem::real_t(1e-12), mfem::real_t(1e-10) * max_diag) :
      mfem::real_t(1e-12);
   for (int i = 0; i < m; i++) { (*block)(i, i) += eps_reg; }

   patch_invs_.push_back(std::make_unique<DenseMatrixInverse>(*block));
   patch_mats_.push_back(std::move(block));
}

void IGAPatchRASPreconditioner::BuildPatchSolverFromLocalSparse(
   const std::vector<int> &ids) const
{
   const int m = static_cast<int>(ids.size());
   max_patch_size_ = std::max(max_patch_size_, m);
   auto block = std::make_unique<DenseMatrix>(m, m);
   *block = 0.0;

   const int width = A_local_.Width();
   std::vector<int> local_pos(width, -1);
   for (int i = 0; i < m; i++)
   {
      MFEM_VERIFY(ids[i] >= 0 && ids[i] < width,
                  "Patch index outside the local sparse matrix width.");
      local_pos[ids[i]] = i;
   }

   mfem::Array<int> cols;
   Vector vals;
   for (int row_pos = 0; row_pos < m; row_pos++)
   {
      const int row = ids[row_pos];
      MFEM_VERIFY(row >= 0 && row < A_local_.Height(),
                  "Patch index outside the local sparse matrix height.");
      A_local_.GetRow(row, cols, vals);
      for (int q = 0; q < cols.Size(); q++)
      {
         const int col = cols[q];
         if (col >= 0 && col < width)
         {
            const int col_pos = local_pos[col];
            if (col_pos >= 0)
            {
               (*block)(row_pos, col_pos) = vals[q];
            }
         }
      }
   }

   for (int id : ids) { local_pos[id] = -1; }

   mfem::real_t max_diag = 0.0;
   for (int i = 0; i < m; i++)
   {
      max_diag = std::max(max_diag, std::abs((*block)(i, i)));
   }
   const mfem::real_t eps_reg =
      (max_diag > 0.0) ?
      std::max(mfem::real_t(1e-12), mfem::real_t(1e-10) * max_diag) :
      mfem::real_t(1e-12);
   for (int i = 0; i < m; i++) { (*block)(i, i) += eps_reg; }

   patch_invs_.push_back(std::make_unique<DenseMatrixInverse>(*block));
   patch_mats_.push_back(std::move(block));
}

void IGAPatchRASPreconditioner::BuildPatchSolvers() const
{
   max_patch_size_ = 0;
   for (const auto &ids : patches_)
   {
      if (have_local_matrix_)
      {
         BuildPatchSolverFromLocalSparse(ids);
      }
      else
      {
         BuildPatchSolverFromDense(ids);
      }
   }
}

void IGAPatchRASPreconditioner::Build() const
{
   if (built_) { return; }

   MFEM_VERIFY(A_.Height() == A_.Width(), "RAS requires a square operator.");
   MFEM_VERIFY(A_.Height() == 2 * fespace_.GetTrueVSize(),
               "RAS expects the full 2x2 real Maxwell system.");

   if (options_.verbose && Mpi::WorldRank() == 0)
   {
      mfem::out << "[iga_ras] building overlapping patches..."
                << std::endl;
   }

   BuildElementPatches();
   const bool force_dense =
      options_.assembly == Options::Assembly::DenseProbe;
   const bool force_sparse =
      options_.assembly == Options::Assembly::LocalSparse;

   if (!force_dense)
   {
      BuildLocalSparseOperator();
   }

   if (!have_local_matrix_)
   {
      MFEM_VERIFY(!force_sparse,
                  "IGA RAS local sparse extraction requested, but A is not a "
                  "supported SparseMatrix/HypreParMatrix/Complex matrix type.");
      if (options_.verbose && Mpi::WorldRank() == 0)
      {
         mfem::out << "[iga_ras] local sparse extraction unavailable; "
                   << "falling back to dense operator probing." << std::endl;
      }
      BuildDenseOperator();
   }
   BuildPatchSolvers();

   if (options_.verbose && Mpi::WorldRank() == 0)
   {
      mfem::out << "[iga_ras] patches=" << patches_.size()
                << ", overlap_layers=" << options_.overlap_layers
                << ", max_patch_dim=" << max_patch_size_
                << ", damping=" << options_.damping
                << ", iterations=" << options_.iterations
                << ", assembly="
                << (have_local_matrix_ ? "local_sparse" : "dense_probe")
                << std::endl;
   }

   built_ = true;
}

void IGAPatchRASPreconditioner::Mult(const Vector &r, Vector &z) const
{
   Build();
   const int n = A_.Height();
   z.SetSize(n);
   z = 0.0;
   work_.SetSize(n);
   residual_.SetSize(n);
   Awork_.SetSize(n);
   work_ = 0.0;

   for (int it = 0; it < options_.iterations; it++)
   {
      residual_ = r;
      if (it > 0)
      {
         A_.Mult(work_, Awork_);
         residual_.Add(-1.0, Awork_);
      }

      for (size_t p = 0; p < patches_.size(); p++)
      {
         const auto &ids = patches_[p];
         const int m = static_cast<int>(ids.size());
         Vector rhs(m), sol(m);
         for (int i = 0; i < m; i++) { rhs[i] = residual_[ids[i]]; }
         patch_invs_[p]->Mult(rhs, sol);
         for (int i = 0; i < m; i++)
         {
            work_[ids[i]] += options_.damping * weights_[ids[i]] * sol[i];
         }
      }
   }

   z = work_;
}

IGAPatchRASPreconditioner::PatchData
IGAPatchRASPreconditioner::ExtractPatchData() const
{
   Build();
   PatchData data;
   data.num_patches    = static_cast<int>(patches_.size());
   data.max_patch_size = max_patch_size_;
   data.damping        = options_.damping;
   data.ids            = patches_;
   data.weights        = weights_;

   data.inv_mats.reserve(data.num_patches);
   for (int p = 0; p < data.num_patches; p++)
   {
      const int m = static_cast<int>(patches_[p].size());
      mfem::DenseMatrix inv(m, m);
      Vector col_in(m), col_out(m);
      for (int j = 0; j < m; j++)
      {
         col_in = 0.0;
         col_in[j] = 1.0;
         patch_invs_[p]->Mult(col_in, col_out);
         for (int i = 0; i < m; i++) { inv(i, j) = col_out[i]; }
      }
      data.inv_mats.push_back(std::move(inv));
   }
   return data;
}

} // namespace covariant_aux_space
