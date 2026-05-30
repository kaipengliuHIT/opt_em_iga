#include "covariant_reference_preconditioner.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>

namespace covariant_aux_space
{

namespace
{

const char *ModeName(CovariantReferencePreconditioner::PrototypeMode mode)
{
   switch (mode)
   {
      case CovariantReferencePreconditioner::PrototypeMode::nodal_proto:
         return "nodal_proto";
      case CovariantReferencePreconditioner::PrototypeMode::edge_galerkin_proto:
         return "edge_galerkin_proto";
      case CovariantReferencePreconditioner::PrototypeMode::edge_yee_proto:
         return "edge_yee_proto";
   }
   return "unknown";
}

class DiagonalAbsInverseSolver : public mfem::Solver
{
public:
   explicit DiagonalAbsInverseSolver(const mfem::Vector &inv_diag)
      : mfem::Solver(inv_diag.Size(), inv_diag.Size()),
        inv_diag_(inv_diag)
   {
   }

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      const int n = inv_diag_.Size();
      y.SetSize(n);
      for (int i = 0; i < n; i++)
      {
         y[i] = inv_diag_[i] * x[i];
      }
   }

   void SetOperator(const mfem::Operator &op) override
   {
      MFEM_VERIFY(op.Height() == height && op.Width() == width,
                  "DiagonalAbsInverseSolver operator size mismatch.");
   }

private:
   mfem::Vector inv_diag_;
};

}

CovariantReferencePreconditioner::CovariantReferencePreconditioner(
   const mfem::ParFiniteElementSpace &fespace,
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom,
   const std::function<double(const mfem::Vector &)> &eps_fn)
   : mfem::Solver(2 * fespace.GetTrueVSize(), 2 * fespace.GetTrueVSize()),
     fespace_(fespace),
     geom_(geom),
     eps_fn_(eps_fn),
     aux_solver_(geom),
     yee_transfer_(std::make_unique<YeeTransferBuilder>(fespace, geom)),
     yee_operator_(std::make_unique<YeeOperatorBuilder>(geom)),
     r_re_(fespace.GetTrueVSize()),
     r_im_(fespace.GetTrueVSize()),
     z_re_(fespace.GetTrueVSize()),
     z_im_(fespace.GetTrueVSize())
{
}

void CovariantReferencePreconditioner::MarkDirty()
{
   built_ = false;
   aux_dofs_.clear();
   Pi_.SetSize(0, 0);
   Aaux_.SetSize(0, 0);
   AauxImag_.SetSize(0, 0);
   AauxBlock_.SetSize(0, 0);
   AauxSparse_.reset();
   AauxSparsePrec_.reset();
   AauxInv_.reset();
   AauxBlockInv_.reset();
   aux_rhs_.SetSize(0);
   aux_sol_.SetSize(0);
   aux_rhs_block_.SetSize(0);
   aux_sol_block_.SetSize(0);
   op_jacobi_inv_diag_.SetSize(0);
   op_jacobi_work_.SetSize(0);
   op_jacobi_res_.SetSize(0);
   op_jacobi_Awork_.SetSize(0);
   op_block_jacobi_inv00_.SetSize(0);
   op_block_jacobi_inv01_.SetSize(0);
   op_block_jacobi_inv10_.SetSize(0);
   op_block_jacobi_inv11_.SetSize(0);
   op_block_jacobi_work_.SetSize(0);
   op_block_jacobi_res_.SetSize(0);
   op_block_jacobi_Awork_.SetSize(0);
   op_patch_block_starts_.clear();
   op_patch_block_sizes_.clear();
   op_patch_block_indices_.clear();
   op_patch_block_mats_.clear();
   op_patch_block_invs_.clear();
   op_patch_block_scalar_ = false;
   op_patch_block_work_.SetSize(0);
   op_patch_block_res_.SetSize(0);
   op_patch_block_Awork_.SetSize(0);
}

void CovariantReferencePreconditioner::SetOperator(const mfem::Operator &op)
{
   op_ = &op;
   height = op.Height();
   width = op.Width();
}

void CovariantReferencePreconditioner::SetGrid(
   const fdfd_iga_init::ReferenceGrid &grid)
{
   fdfd_iga_init::ReferenceGrid effective_grid = grid;
   if (knot_align_enabled_)
   {
      // Build knot-aligned grid from NURBS patch knot vectors
      const mfem::NURBSExtension *ext = fespace_.GetNURBSext();
      if (ext)
      {
         mfem::Array<const mfem::KnotVector *> kv;
         const_cast<mfem::NURBSExtension *>(ext)->GetPatchKnotVectors(0, kv);
         effective_grid = fdfd_iga_init::ComputeKnotAlignedGrid(kv, knot_align_cps_);
         knot_align_applied_ = true;
         if (mfem::Mpi::WorldRank() == 0)
         {
            mfem::out << "[knot_align] knot-span aligned grid: "
                      << effective_grid.nx << "x" << effective_grid.ny
                      << "x" << effective_grid.nz
                      << " (cps=" << knot_align_cps_ << ")" << std::endl;
         }
      }
   }
   aux_solver_.SetGrid(effective_grid);
   yee_transfer_->SetGrid(effective_grid);
   yee_operator_->SetGrid(effective_grid);
   MarkDirty();
}

void CovariantReferencePreconditioner::SetWaveNumber(double k0)
{
   k0_ = k0;
   aux_solver_.SetWaveNumber(k0);
   MarkDirty();
}

void CovariantReferencePreconditioner::SetMaxIterations(int max_it)
{
   aux_solver_.SetMaxIterations(max_it);
}

void CovariantReferencePreconditioner::SetDamping(double omega)
{
   aux_solver_.SetDamping(omega);
}

void CovariantReferencePreconditioner::SetMassShift(double shift)
{
   aux_solver_.SetMassShift(shift);
   MarkDirty();
}

void CovariantReferencePreconditioner::SetPrototypeMode(PrototypeMode mode)
{
   mode_ = mode;
   MarkDirty();
}

void CovariantReferencePreconditioner::SetYeeReferencePML(
   bool enabled, double thickness, double strength, double order)
{
   yee_operator_->SetReferencePML(enabled, thickness, strength, order);
   MarkDirty();
}

void CovariantReferencePreconditioner::SetYeeComponentScales(
   double curl_scale, double mass_scale)
{
   yee_operator_->SetComponentScales(curl_scale, mass_scale);
   MarkDirty();
}

void CovariantReferencePreconditioner::SetKnotAlignGrid(
   bool enable, int cells_per_span)
{
   knot_align_enabled_ = enable;
   knot_align_cps_ = (cells_per_span > 0) ? cells_per_span : 1;
   knot_align_applied_ = false;
   MarkDirty();
}

int CovariantReferencePreconditioner::FullVectorIndex(const AuxDof &dof) const
{
   const auto &grid = aux_solver_.Grid();
   const int node = (dof.k * grid.ny + dof.j) * grid.nx + dof.i;
   const int nnode = grid.nx * grid.ny * grid.nz;
   return dof.axis * nnode + node;
}

double CovariantReferencePreconditioner::EdgeLength(const AuxDof &dof) const
{
   const auto &grid = aux_solver_.Grid();
   const double hx = 1.0 / double(grid.nx - 1);
   const double hy = 1.0 / double(grid.ny - 1);
   const double hz = 1.0 / double(grid.nz - 1);
   return (dof.axis == 0) ? hx : ((dof.axis == 1) ? hy : hz);
}

void CovariantReferencePreconditioner::BuildAuxDofs() const
{
   if (!aux_dofs_.empty())
   {
      return;
   }

   const auto &grid = aux_solver_.Grid();
   MFEM_VERIFY(grid.nx > 2 && grid.ny > 2 && grid.nz > 2,
               "Auxiliary grid must have at least one interior node.");
   if (mode_ == PrototypeMode::nodal_proto)
   {
      for (int k = 1; k < grid.nz - 1; k++)
      {
         for (int j = 1; j < grid.ny - 1; j++)
         {
            for (int i = 1; i < grid.nx - 1; i++)
            {
               for (int c = 0; c < 3; c++)
               {
                  aux_dofs_.push_back({i, j, k, c});
               }
            }
         }
      }
      return;
   }
   aux_dofs_ = yee_transfer_->GetEdgeDofs();
}

fdfd_iga_init::SampledReferenceField
CovariantReferencePreconditioner::MakeAuxBasisField(const AuxDof &dof) const
{
   const auto &grid = aux_solver_.Grid();
   const int nnode = grid.nx * grid.ny * grid.nz;
   fdfd_iga_init::SampledReferenceField field;
   field.nx = grid.nx;
   field.ny = grid.ny;
   field.nz = grid.nz;
   field.dim = 3;
   field.real.assign(3 * nnode, 0.0);
   field.imag.assign(3 * nnode, 0.0);

   if (mode_ == PrototypeMode::nodal_proto)
   {
      field.real[FullVectorIndex(dof)] = 1.0;
      return field;
   }

   const double scale = 1.0 / EdgeLength(dof);
   auto node_id = [&](int i, int j, int k)
   {
      return (k * grid.ny + j) * grid.nx + i;
   };

   if (dof.axis == 0)
   {
      field.real[node_id(dof.i, dof.j, dof.k)] = scale;
      field.real[node_id(dof.i + 1, dof.j, dof.k)] = scale;
   }
   else if (dof.axis == 1)
   {
      field.real[nnode + node_id(dof.i, dof.j, dof.k)] = scale;
      field.real[nnode + node_id(dof.i, dof.j + 1, dof.k)] = scale;
   }
   else
   {
      field.real[2 * nnode + node_id(dof.i, dof.j, dof.k)] = scale;
      field.real[2 * nnode + node_id(dof.i, dof.j, dof.k + 1)] = scale;
   }

   return field;
}

double CovariantReferencePreconditioner::EvaluateAuxFunctional(
   const std::vector<double> &data, const AuxDof &dof) const
{
   const auto &grid = aux_solver_.Grid();
   const int nnode = grid.nx * grid.ny * grid.nz;
   auto node_id = [&](int i, int j, int k)
   {
      return (k * grid.ny + j) * grid.nx + i;
   };

   if (mode_ == PrototypeMode::nodal_proto)
   {
      return data[FullVectorIndex(dof)];
   }

   if (dof.axis == 0)
   {
      const double v0 = data[node_id(dof.i, dof.j, dof.k)];
      const double v1 = data[node_id(dof.i + 1, dof.j, dof.k)];
      return 0.5 * (v0 + v1) * EdgeLength(dof);
   }
   if (dof.axis == 1)
   {
      const double v0 = data[nnode + node_id(dof.i, dof.j, dof.k)];
      const double v1 = data[nnode + node_id(dof.i, dof.j + 1, dof.k)];
      return 0.5 * (v0 + v1) * EdgeLength(dof);
   }

   const double v0 = data[2 * nnode + node_id(dof.i, dof.j, dof.k)];
   const double v1 = data[2 * nnode + node_id(dof.i, dof.j, dof.k + 1)];
   return 0.5 * (v0 + v1) * EdgeLength(dof);
}

void CovariantReferencePreconditioner::BuildTransferMatrix() const
{
   BuildAuxDofs();
   const int tvsize = fespace_.GetTrueVSize();
   const int na = static_cast<int>(aux_dofs_.size());
   if (Pi_.Height() == tvsize && Pi_.Width() == na)
   {
      return;
   }
   Pi_.SetSize(tvsize, na);
   Pi_ = 0.0;

   if (mode_ != PrototypeMode::nodal_proto)
   {
      yee_transfer_->BuildProlongation(Pi_);
      return;
   }

   mfem::ParGridFunction gf(const_cast<mfem::ParFiniteElementSpace *>(&fespace_));
   mfem::Vector col(tvsize);
   for (int j = 0; j < na; j++)
   {
      fdfd_iga_init::SampledReferenceField basis_field =
         MakeAuxBasisField(aux_dofs_[j]);
      fdfd_iga_init::ReferenceFieldProjector projector(fespace_, basis_field);
      gf = 0.0;
      projector.ProjectReal(gf);
      gf.GetTrueDofs(col);
      for (int i = 0; i < tvsize; i++)
      {
         Pi_(i, j) = col[i];
      }
   }
}

void CovariantReferencePreconditioner::BuildAuxiliaryMatrix() const
{
   BuildAuxDofs();
   const int na = static_cast<int>(aux_dofs_.size());
   Aaux_.SetSize(na, na);
   Aaux_ = 0.0;

   if (mode_ == PrototypeMode::nodal_proto)
   {
      mfem::DenseMatrix Afull;
      aux_solver_.AssembleSystemMatrix(eps_fn_, Afull);
      for (int j = 0; j < na; j++)
      {
         const int full_j = FullVectorIndex(aux_dofs_[j]);
         for (int i = 0; i < na; i++)
         {
            const int full_i = FullVectorIndex(aux_dofs_[i]);
            Aaux_(i, j) = Afull(full_i, full_j);
         }
      }
   }
   else if (mode_ == PrototypeMode::edge_galerkin_proto)
   {
      BuildGalerkinEdgeCoarseMatrix(Aaux_);
   }
   else if (mode_ == PrototypeMode::edge_yee_proto)
   {
      // PML auto-fallback: when the Yee operator is configured with PML
      // stretching, the FDFD stencil cannot approximate the dense
      // long-range mass coupling of NURBS H(curl) basis in PML regions
      const bool pml_active = yee_operator_->IsReferencePMLEnabled();
      if (pml_active && yee_pml_galerkin_fallback_)
      {
         const mfem::Operator &fallback_op =
            yee_calibration_op_ ? *yee_calibration_op_ : *op_;
         BuildGalerkinEdgeCoarseMatrix(fallback_op, Aaux_);
         if (mfem::Mpi::WorldRank() == 0)
         {
            mfem::out << "[covariant_aux_space] edge_yee_proto: PML active, "
                      << "using Pi^T A_abs Pi Galerkin restriction (SPD)\n";
         }
      }
      else if (yee_complex_auxiliary_)
      {
         if (yee_sqrt_pml_scaling_)
         {
            yee_operator_->AssembleYeeMaxwellOperatorComplexSqrtScaled(
               eps_fn_, k0_, Aaux_, AauxImag_);
         }
         else
         {
            yee_operator_->AssembleYeeMaxwellOperatorComplex(eps_fn_, k0_,
                                                             Aaux_, AauxImag_);
         }
      }
      else
      {
         yee_operator_->AssembleYeeMaxwellOperator(eps_fn_, k0_, Aaux_);
      }
      if (yee_diag_calibration_)
      {
         mfem::DenseMatrix Agal;
         const mfem::Operator &calib_op =
            yee_calibration_op_ ? *yee_calibration_op_ : *op_;
         BuildGalerkinEdgeCoarseMatrix(calib_op, Agal);
         const int n = Aaux_.Height();
         double gal_diag_mean = 0.0;
         double yee_diag_mean = 0.0;
         for (int i = 0; i < n; i++)
         {
            gal_diag_mean += std::abs(Agal(i, i));
            yee_diag_mean += std::abs(Aaux_(i, i));
         }
         gal_diag_mean /= double(n);
         yee_diag_mean /= double(n);
         const double alpha = (yee_diag_mean > 0.0) ? gal_diag_mean / yee_diag_mean : 1.0;
         Aaux_ *= alpha;
         if (AauxImag_.Height() == n) { AauxImag_ *= alpha; }
      }
      if (yee_local_diag_calibration_)
      {
         mfem::DenseMatrix Agal;
         const mfem::Operator &calib_op =
            yee_calibration_op_ ? *yee_calibration_op_ : *op_;
         BuildGalerkinEdgeCoarseMatrix(calib_op, Agal);
         const int n = Aaux_.Height();
         mfem::Vector scale(n);
         int zero_yee_diag = 0;
         double min_scale = std::numeric_limits<double>::infinity();
         double max_scale = 0.0;
         for (int i = 0; i < n; i++)
         {
            const double gd = std::abs(Agal(i, i));
            const double yd = std::abs(Aaux_(i, i));
            if (yd > 1e-30)
            {
               scale[i] = std::sqrt(gd / yd);
               min_scale = std::min(min_scale, scale[i]);
               max_scale = std::max(max_scale, scale[i]);
            }
            else
            {
               scale[i] = 1.0;
               zero_yee_diag++;
            }
         }
         for (int i = 0; i < n; i++)
         {
            for (int j = 0; j < n; j++)
            {
               const double sij = scale[i] * scale[j];
               Aaux_(i, j) *= sij;
               if (AauxImag_.Height() == n)
               {
                  AauxImag_(i, j) *= sij;
               }
            }
         }
         if (mfem::Mpi::WorldRank() == 0)
         {
            std::cout << "[covariant_aux_space] Yee local diagonal calibration: "
                      << "scale_range=[" << min_scale << ", " << max_scale
                      << "], zero_yee_diag=" << zero_yee_diag << std::endl;
         }
      }
      if (yee_masked_galerkin_calibration_)
      {
         mfem::DenseMatrix Agal;
         const mfem::Operator &calib_op =
            yee_calibration_op_ ? *yee_calibration_op_ : *op_;
         BuildGalerkinEdgeCoarseMatrix(calib_op, Agal);
         const int n = Aaux_.Height();
         int mask_nnz = 0;
         double gal_frob2 = 0.0;
         double masked_frob2 = 0.0;
         for (int i = 0; i < n; i++)
         {
            for (int j = 0; j < n; j++)
            {
               const double g = Agal(i, j);
               gal_frob2 += g * g;
               if (std::abs(Aaux_(i, j)) > 1e-14 || i == j)
               {
                  Aaux_(i, j) = g;
                  masked_frob2 += g * g;
                  mask_nnz++;
               }
               else
               {
                  Aaux_(i, j) = 0.0;
                  if (AauxImag_.Height() == n)
                  {
                     AauxImag_(i, j) = 0.0;
                  }
               }
            }
         }
         if (mfem::Mpi::WorldRank() == 0)
         {
            const double energy_ratio =
               (gal_frob2 > 0.0) ? std::sqrt(masked_frob2 / gal_frob2) : 0.0;
            std::cout << "[covariant_aux_space] Yee masked-Galerkin calibration: "
                      << "mask_nnz=" << mask_nnz
                      << ", density=" << double(mask_nnz) / double(n * n)
                      << ", retained_frob_ratio=" << energy_ratio
                      << std::endl;
         }
      }
      if (yee_galerkin_block_radius_ > 0)
      {
         mfem::DenseMatrix Agal;
         const mfem::Operator &calib_op =
            yee_calibration_op_ ? *yee_calibration_op_ : *op_;
         BuildGalerkinEdgeCoarseMatrix(calib_op, Agal);
         const int n = Aaux_.Height();
         int block_nnz = 0;
         double gal_frob2 = 0.0;
         double block_frob2 = 0.0;
         for (int i = 0; i < n; i++)
         {
            const AuxDof &di = aux_dofs_[i];
            for (int j = 0; j < n; j++)
            {
               const AuxDof &dj = aux_dofs_[j];
               const int dist = std::max(std::abs(di.i - dj.i),
                                  std::max(std::abs(di.j - dj.j),
                                           std::abs(di.k - dj.k)));
               const double g = Agal(i, j);
               gal_frob2 += g * g;
               if (dist <= yee_galerkin_block_radius_)
               {
                  Aaux_(i, j) = g;
                  block_frob2 += g * g;
                  block_nnz++;
               }
               else
               {
                  Aaux_(i, j) = 0.0;
                  if (AauxImag_.Height() == n)
                  {
                     AauxImag_(i, j) = 0.0;
                  }
               }
            }
         }
         if (mfem::Mpi::WorldRank() == 0)
         {
            const double energy_ratio =
               (gal_frob2 > 0.0) ? std::sqrt(block_frob2 / gal_frob2) : 0.0;
            std::cout << "[covariant_aux_space] Yee local Galerkin block: "
                      << "radius=" << yee_galerkin_block_radius_
                      << ", block_nnz=" << block_nnz
                      << ", density=" << double(block_nnz) / double(n * n)
                      << ", retained_frob_ratio=" << energy_ratio
                      << std::endl;
         }
      }
   }
   else
   {
      MFEM_ABORT("Unknown prototype mode in coarse operator assembly.");
   }

   int inactive_transfer_cols = 0;
   if (mode_ != PrototypeMode::nodal_proto &&
       Pi_.Height() > 0 && Pi_.Width() == na)
   {
      for (int j = 0; j < na; j++)
      {
         double col_norm2 = 0.0;
         for (int i = 0; i < Pi_.Height(); i++)
         {
            col_norm2 += Pi_(i, j) * Pi_(i, j);
         }
         if (col_norm2 < 1e-24)
         {
            inactive_transfer_cols++;
            // Instead of zeroing, set diagonal to 1.0 for inactive columns
            // (they will get further regularized below)
            for (int k = 0; k < na; k++)
            {
               if (k != j)
               {
                  Aaux_(j, k) = 0.0;
                  Aaux_(k, j) = 0.0;
               }
               if (AauxImag_.Height() == na)
               {
                  if (k != j)
                  {
                     AauxImag_(j, k) = 0.0;
                     AauxImag_(k, j) = 0.0;
                  }
               }
            }
            Aaux_(j, j) = 0.0;  // Will be filled by regularization below
         }
      }
      if (inactive_transfer_cols > 0 && mfem::Mpi::WorldRank() == 0)
      {
         std::cout << "[covariant_aux_space] inactive transfer columns masked="
                   << inactive_transfer_cols << std::endl;
      }
   }

   if (yee_complex_auxiliary_ && mode_ == PrototypeMode::edge_yee_proto)
   {
      AauxBlock_.SetSize(2 * na, 2 * na);
      AauxBlock_ = 0.0;
      double max_diag = 0.0;
      for (int i = 0; i < na; i++) { max_diag = std::max(max_diag, std::abs(Aaux_(i,i))); }
      const double eps_reg = (max_diag > 0.0) ? std::max(1e-12, 1e-3 * max_diag) : 1.0;
      for (int i = 0; i < na; i++) { Aaux_(i, i) += eps_reg; }
      for (int i = 0; i < na; i++)
      {
         for (int j = 0; j < na; j++)
         {
            AauxBlock_(i, j) = Aaux_(i, j);
            AauxBlock_(i, na + j) = -AauxImag_(i, j);
            AauxBlock_(na + i, j) = AauxImag_(i, j);
            AauxBlock_(na + i, na + j) = Aaux_(i, j);
         }
      }
      if (!yee_iterative_aux_solve_)
      {
         AauxBlockInv_ = std::make_unique<mfem::DenseMatrixInverse>(AauxBlock_);
      }
      if (mfem::Mpi::WorldRank() == 0)
      {
         std::cout << "[covariant_aux_space] Yee complex auxiliary solve mode: "
                   << (yee_iterative_aux_solve_ ?
                       (yee_bicgstab_aux_solve_ ? "inner BiCGSTAB" :
                        (yee_gmres_aux_solve_ ? "inner GMRES" : "inner GMRES")) :
                       "dense direct")
                   << ", sqrt_pml_scaling=" << yee_sqrt_pml_scaling_
                   << ", max_iter=" << yee_iterative_aux_max_iter_
                   << ", rel_tol=" << yee_iterative_aux_rel_tol_
                   << std::endl;
      }
   }
   else
   {
      double max_diag = 0.0;
      for (int i = 0; i < na; i++) { max_diag = std::max(max_diag, std::abs(Aaux_(i,i))); }
      // Stronger regularization for robustness with non-uniform grids
      const double eps_reg = (max_diag > 0.0) ? std::max(1e-12, 1e-3 * max_diag) : 1.0;
      for (int i = 0; i < na; i++) { Aaux_(i, i) += eps_reg; }
      if (!yee_iterative_aux_solve_)
      {
         AauxInv_ = std::make_unique<mfem::DenseMatrixInverse>(Aaux_);
      }
      else
      {
         if (yee_sparse_aux_solve_)
         {
            double max_abs = 0.0;
            for (int j = 0; j < na; j++)
            {
               for (int i = 0; i < na; i++)
               {
                  max_abs = std::max(max_abs, std::abs(Aaux_(i, j)));
               }
            }
            const double drop_tol = std::max(1e-14, 1e-14 * max_abs);
            int nnz_inserted = 0;
            int zero_diag = 0;
            mfem::Vector inv_abs_diag(na);
            AauxSparse_ = std::make_unique<mfem::SparseMatrix>(na);
            for (int i = 0; i < na; i++)
            {
               const double diag = std::abs(Aaux_(i, i));
               if (diag > 1e-30)
               {
                  inv_abs_diag[i] = 1.0 / diag;
               }
               else
               {
                  inv_abs_diag[i] = 0.0;
                  zero_diag++;
               }
               for (int j = 0; j < na; j++)
               {
                  const double v = Aaux_(i, j);
                  if (std::abs(v) > drop_tol)
                  {
                     AauxSparse_->Set(i, j, v);
                     nnz_inserted++;
                  }
               }
            }
            AauxSparse_->Finalize(1);
            AauxSparsePrec_ =
               std::make_unique<DiagonalAbsInverseSolver>(inv_abs_diag);
            if (mfem::Mpi::WorldRank() == 0)
            {
               std::cout << "[covariant_aux_space] Yee auxiliary sparse matrix: "
                         << "nnz=" << AauxSparse_->NumNonZeroElems()
                         << ", inserted=" << nnz_inserted
                         << ", density="
                         << double(AauxSparse_->NumNonZeroElems()) /
                            double(na * na)
                         << ", drop_tol=" << drop_tol
                         << ", zero_diag=" << zero_diag << std::endl;
            }
         }
         if (mfem::Mpi::WorldRank() == 0)
         {
            std::cout << "[covariant_aux_space] Yee auxiliary solve mode: "
                      << (yee_gmres_aux_solve_ ? "inner GMRES" : "inner CG")
                      << (AauxSparse_ ? " on sparse matrix" : " on dense matrix")
                      << ", max_iter=" << yee_iterative_aux_max_iter_
                      << ", rel_tol=" << yee_iterative_aux_rel_tol_
                      << std::endl;
         }
      }
   }
}

void CovariantReferencePreconditioner::BuildGalerkinEdgeCoarseMatrix(
   mfem::DenseMatrix &Agal) const
{
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before building Galerkin coarse matrix.");
   BuildGalerkinEdgeCoarseMatrix(*op_, Agal);
}

void CovariantReferencePreconditioner::SolveAuxiliarySystem(
   const mfem::Vector &rhs,
   mfem::Vector &sol) const
{
   if (!yee_iterative_aux_solve_)
   {
      AauxInv_->Mult(rhs, sol);
      return;
   }

   sol.SetSize(rhs.Size());
   sol = 0.0;
   if (yee_gmres_aux_solve_)
   {
      mfem::GMRESSolver gmres;
      if (AauxSparse_)
      {
         gmres.SetOperator(*AauxSparse_);
         if (AauxSparsePrec_) { gmres.SetPreconditioner(*AauxSparsePrec_); }
      }
      else
      {
         gmres.SetOperator(Aaux_);
      }
      gmres.SetRelTol(yee_iterative_aux_rel_tol_);
      gmres.SetAbsTol(0.0);
      gmres.SetMaxIter(yee_iterative_aux_max_iter_);
      gmres.SetKDim(std::min(50, yee_iterative_aux_max_iter_));
      gmres.SetPrintLevel(-1);
      gmres.Mult(rhs, sol);
      return;
   }

   mfem::CGSolver cg;
   if (AauxSparse_)
   {
      cg.SetOperator(*AauxSparse_);
      if (AauxSparsePrec_) { cg.SetPreconditioner(*AauxSparsePrec_); }
   }
   else
   {
      cg.SetOperator(Aaux_);
   }
   cg.SetRelTol(yee_iterative_aux_rel_tol_);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(yee_iterative_aux_max_iter_);
   cg.SetPrintLevel(-1);
   cg.Mult(rhs, sol);
}

void CovariantReferencePreconditioner::SolveAuxiliaryBlockSystem(
   const mfem::Vector &rhs,
   mfem::Vector &sol) const
{
   if (!yee_iterative_aux_solve_)
   {
      AauxBlockInv_->Mult(rhs, sol);
      return;
   }

   sol.SetSize(rhs.Size());
   sol = 0.0;
   if (yee_bicgstab_aux_solve_)
   {
      mfem::BiCGSTABSolver bicgstab;
      bicgstab.SetOperator(AauxBlock_);
      bicgstab.SetRelTol(yee_iterative_aux_rel_tol_);
      bicgstab.SetAbsTol(0.0);
      bicgstab.SetMaxIter(yee_iterative_aux_max_iter_);
      bicgstab.SetPrintLevel(-1);
      bicgstab.Mult(rhs, sol);
      return;
   }

   mfem::GMRESSolver gmres;
   gmres.SetOperator(AauxBlock_);
   gmres.SetRelTol(yee_iterative_aux_rel_tol_);
   gmres.SetAbsTol(0.0);
   gmres.SetMaxIter(yee_iterative_aux_max_iter_);
   gmres.SetKDim(std::min(50, yee_iterative_aux_max_iter_));
   gmres.SetPrintLevel(-1);
   gmres.Mult(rhs, sol);
}

void CovariantReferencePreconditioner::BuildGalerkinEdgeCoarseMatrix(
   const mfem::Operator &ref_op,
   mfem::DenseMatrix &Agal) const
{
   BuildAuxDofs();
   BuildTransferMatrix();
   const int na = static_cast<int>(aux_dofs_.size());
   const int tvsize = fespace_.GetTrueVSize();
   MFEM_VERIFY(ref_op.Height() == tvsize || ref_op.Height() == 2 * tvsize,
               "Galerkin reference operator has incompatible height.");
   if (ref_op.Width() != tvsize && ref_op.Width() != 2 * tvsize)
   {
      std::cerr << "[covariant_aux_space] WARNING: Galerkin reference operator"
                << " has incompatible width=" << ref_op.Width() << "\n";
      Agal.SetSize(0, 0);
      return;
   }
   Agal.SetSize(na, na);
   Agal = 0.0;

   mfem::Vector coarse_col(tvsize);
   mfem::Vector ref_in(ref_op.Width()), ref_out(ref_op.Height());
   mfem::Vector real_in, real_out;
   if (ref_op.Width() == 2 * tvsize)
   {
      real_in.MakeRef(ref_in, 0, tvsize);
   }
   else
   {
      real_in.MakeRef(ref_in, 0, tvsize);
   }
   if (ref_op.Height() == 2 * tvsize)
   {
      real_out.MakeRef(ref_out, 0, tvsize);
   }
   else
   {
      real_out.MakeRef(ref_out, 0, tvsize);
   }

   for (int j = 0; j < na; j++)
   {
      coarse_col.SetSize(tvsize);
      for (int i = 0; i < tvsize; i++)
      {
         coarse_col[i] = Pi_(i, j);
      }
      ref_in = 0.0;
      real_in = coarse_col;
      ref_op.Mult(ref_in, ref_out);
      // NaN-safe: filter NaN from IGA operator output (non-uniform NURBS can produce NaN)
      for (int k = 0; k < real_out.Size(); k++)
      {
         if (std::isnan(real_out[k])) { real_out[k] = 0.0; }
      }
      for (int i = 0; i < na; i++)
      {
         double val = 0.0;
         for (int k = 0; k < tvsize; k++)
         {
            val += Pi_(k, i) * real_out[k];
         }
         Agal(i, j) = val;
      }
   }

   // Regularize: add small diagonal shift for zero Π-columns.
   // A zero column in Π produces a zero row+col in Agal = Πᵀ A Π,
   // making DenseMatrix::Invert() fail.
   double max_diag = 0.0;
   for (int i = 0; i < na; i++)
   {
      max_diag = std::max(max_diag, std::abs(Agal(i, i)));
   }
   const double eps_reg = std::max(1e-12, 1e-10 * max_diag);
   for (int i = 0; i < na; i++)
   {
      Agal(i, i) += eps_reg;
   }
}

void CovariantReferencePreconditioner::BuildAuxiliaryOperators() const
{
   if (built_)
   {
      return;
   }

   if (mfem::Mpi::WorldRank() == 0)
   {
      const auto &grid = aux_solver_.Grid();
      std::cout << "[covariant_aux_space] building explicit auxiliary operators on "
                << grid.nx << "x" << grid.ny << "x" << grid.nz
                << " reference grid, mode=" << ModeName(mode_) << std::endl;
   }
   const auto t0 = std::chrono::steady_clock::now();
   BuildTransferMatrix();
   const auto t1 = std::chrono::steady_clock::now();
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] transfer build seconds="
                << std::chrono::duration<double>(t1 - t0).count()
                << std::endl;
   }
   BuildAuxiliaryMatrix();
   const auto t2 = std::chrono::steady_clock::now();
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] coarse matrix/inverse build seconds="
                << std::chrono::duration<double>(t2 - t1).count()
                << std::endl;
   }

   const int na = static_cast<int>(aux_dofs_.size());
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] auxiliary dofs=" << na
                << ", true_vsize=" << fespace_.GetTrueVSize() << std::endl;
   }
   aux_rhs_.SetSize(na);
   aux_sol_.SetSize(na);
   aux_rhs_block_.SetSize(2 * na);
   aux_sol_block_.SetSize(2 * na);
   BuildOperatorJacobiSmoother();
   BuildOperatorBlockJacobiSmoother();
   BuildOperatorPatchBlockSmoother();
   built_ = true;
}

void CovariantReferencePreconditioner::BuildOperatorJacobiSmoother() const
{
   if (operator_jacobi_smoother_weight_ == 0.0) { return; }
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before building Jacobi smoother.");

   const int n = op_->Height();
   MFEM_VERIFY(op_->Width() == n, "Jacobi smoother requires a square operator.");
   if (op_jacobi_inv_diag_.Size() == n) { return; }

   op_jacobi_inv_diag_.SetSize(n);
   mfem::Vector e(n), Ae(n);
   int zero_diag = 0;
   double min_abs = std::numeric_limits<double>::infinity();
   double max_abs = 0.0;
   for (int j = 0; j < n; j++)
   {
      e = 0.0;
      e[j] = 1.0;
      op_->Mult(e, Ae);
      const double d = Ae[j];
      const double ad = std::abs(d);
      if (ad > 1e-30)
      {
         op_jacobi_inv_diag_[j] = 1.0 / d;
         min_abs = std::min(min_abs, ad);
         max_abs = std::max(max_abs, ad);
      }
      else
      {
         op_jacobi_inv_diag_[j] = 0.0;
         zero_diag++;
      }
   }
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] operator Jacobi smoother built: "
                << "weight=" << operator_jacobi_smoother_weight_
                << ", iterations=" << operator_jacobi_smoother_iterations_
                << ", zero_diag=" << zero_diag
                << ", diag_abs_range=[" << min_abs << ", " << max_abs << "]"
                << std::endl;
   }
}

void CovariantReferencePreconditioner::AddOperatorJacobiSmoother(
   const mfem::Vector &r,
   mfem::Vector &z) const
{
   if (operator_jacobi_smoother_weight_ == 0.0) { return; }

   const int n = r.Size();
   op_jacobi_work_.SetSize(n);
   op_jacobi_res_.SetSize(n);
   op_jacobi_Awork_.SetSize(n);
   op_jacobi_work_ = 0.0;

   for (int it = 0; it < operator_jacobi_smoother_iterations_; it++)
   {
      op_->Mult(op_jacobi_work_, op_jacobi_Awork_);
      op_jacobi_res_ = r;
      op_jacobi_res_.Add(-1.0, op_jacobi_Awork_);
      for (int i = 0; i < n; i++)
      {
         op_jacobi_work_[i] += operator_jacobi_smoother_weight_ *
                               op_jacobi_inv_diag_[i] *
                               op_jacobi_res_[i];
      }
   }

   z += op_jacobi_work_;
}

void CovariantReferencePreconditioner::BuildOperatorBlockJacobiSmoother() const
{
   if (operator_block_jacobi_smoother_weight_ == 0.0) { return; }
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before building block Jacobi smoother.");

   const int n2 = op_->Height();
   MFEM_VERIFY(op_->Width() == n2, "Block Jacobi smoother requires a square operator.");
   MFEM_VERIFY(n2 % 2 == 0, "Block Jacobi smoother requires a 2x2 real block system.");
   const int n = n2 / 2;
   if (op_block_jacobi_inv00_.Size() == n) { return; }

   op_block_jacobi_inv00_.SetSize(n);
   op_block_jacobi_inv01_.SetSize(n);
   op_block_jacobi_inv10_.SetSize(n);
   op_block_jacobi_inv11_.SetSize(n);

   mfem::Vector e(n2), Ae0(n2), Ae1(n2);
   int singular_blocks = 0;
   double min_det_abs = std::numeric_limits<double>::infinity();
   double max_det_abs = 0.0;
   for (int j = 0; j < n; j++)
   {
      e = 0.0;
      e[j] = 1.0;
      op_->Mult(e, Ae0);

      e = 0.0;
      e[j + n] = 1.0;
      op_->Mult(e, Ae1);

      const double a00 = Ae0[j];
      const double a10 = Ae0[j + n];
      const double a01 = Ae1[j];
      const double a11 = Ae1[j + n];
      const double det = a00 * a11 - a01 * a10;
      const double adet = std::abs(det);
      if (adet > 1e-30)
      {
         op_block_jacobi_inv00_[j] =  a11 / det;
         op_block_jacobi_inv01_[j] = -a01 / det;
         op_block_jacobi_inv10_[j] = -a10 / det;
         op_block_jacobi_inv11_[j] =  a00 / det;
         min_det_abs = std::min(min_det_abs, adet);
         max_det_abs = std::max(max_det_abs, adet);
      }
      else
      {
         op_block_jacobi_inv00_[j] = 0.0;
         op_block_jacobi_inv01_[j] = 0.0;
         op_block_jacobi_inv10_[j] = 0.0;
         op_block_jacobi_inv11_[j] = 0.0;
         singular_blocks++;
      }
   }

   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] operator 2x2 block Jacobi smoother built: "
                << "weight=" << operator_block_jacobi_smoother_weight_
                << ", iterations=" << operator_block_jacobi_smoother_iterations_
                << ", singular_blocks=" << singular_blocks
                << ", det_abs_range=[" << min_det_abs << ", " << max_det_abs << "]"
                << std::endl;
   }
}

void CovariantReferencePreconditioner::AddOperatorBlockJacobiSmoother(
   const mfem::Vector &r,
   mfem::Vector &z) const
{
   if (operator_block_jacobi_smoother_weight_ == 0.0) { return; }

   const int n2 = r.Size();
   MFEM_VERIFY(n2 % 2 == 0, "Block Jacobi smoother requires a 2x2 real block residual.");
   const int n = n2 / 2;
   op_block_jacobi_work_.SetSize(n2);
   op_block_jacobi_res_.SetSize(n2);
   op_block_jacobi_Awork_.SetSize(n2);
   op_block_jacobi_work_ = 0.0;

   for (int it = 0; it < operator_block_jacobi_smoother_iterations_; it++)
   {
      op_->Mult(op_block_jacobi_work_, op_block_jacobi_Awork_);
      op_block_jacobi_res_ = r;
      op_block_jacobi_res_.Add(-1.0, op_block_jacobi_Awork_);
      for (int i = 0; i < n; i++)
      {
         const double rr = op_block_jacobi_res_[i];
         const double ri = op_block_jacobi_res_[i + n];
         op_block_jacobi_work_[i] +=
            operator_block_jacobi_smoother_weight_ *
            (op_block_jacobi_inv00_[i] * rr + op_block_jacobi_inv01_[i] * ri);
         op_block_jacobi_work_[i + n] +=
            operator_block_jacobi_smoother_weight_ *
            (op_block_jacobi_inv10_[i] * rr + op_block_jacobi_inv11_[i] * ri);
      }
   }

   z += op_block_jacobi_work_;
}

void CovariantReferencePreconditioner::BuildOperatorPatchBlockSmoother() const
{
   if (operator_patch_block_smoother_weight_ == 0.0) { return; }
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before building patch block smoother.");

   const int op_size = op_->Height();
   MFEM_VERIFY(op_->Width() == op_size,
               "Patch block smoother requires a square operator.");
   if (!real_block_mode_)
   {
      MFEM_VERIFY(op_size % 2 == 0,
                  "Patch block smoother requires a 2x2 real block system.");
   }
   const int n = real_block_mode_ ? op_size : op_size / 2;
   if (!op_patch_block_invs_.empty()) { return; }
   const mfem::Operator *patch_op =
      (operator_patch_block_use_ref_op_ && operator_patch_block_ref_op_) ?
      operator_patch_block_ref_op_ : op_;
   MFEM_VERIFY(patch_op != nullptr,
               "Patch block smoother reference operator is missing.");
   op_patch_block_scalar_ =
      real_block_mode_ || (patch_op->Height() == n && patch_op->Width() == n);
   MFEM_VERIFY(op_patch_block_scalar_ ||
               (patch_op->Height() == op_size && patch_op->Width() == op_size),
               "Patch block smoother operator has incompatible dimensions.");

   const int block_size = std::max(1, operator_patch_block_size_);
   if (operator_element_patch_blocks_)
   {
      std::vector<char> assigned(n, 0);
      mfem::Array<int> vdofs;
      for (int eidx = 0; eidx < fespace_.GetNE(); eidx++)
      {
         fespace_.GetElementVDofs(eidx, vdofs);
         std::vector<int> block_ids;
         for (int q = 0; q < vdofs.Size(); q++)
         {
            const int ldof = (vdofs[q] < 0) ? -1 - vdofs[q] : vdofs[q];
            const int tdof = fespace_.GetLocalTDofNumber(ldof);
            if (tdof >= 0 && tdof < n && !assigned[tdof])
            {
               assigned[tdof] = 1;
               block_ids.push_back(tdof);
            }
         }
         std::sort(block_ids.begin(), block_ids.end());
         block_ids.erase(std::unique(block_ids.begin(), block_ids.end()),
                         block_ids.end());
         if (!block_ids.empty())
         {
            op_patch_block_indices_.push_back(std::move(block_ids));
         }
      }
      for (int i = 0; i < n; i++)
      {
         if (!assigned[i])
         {
            op_patch_block_indices_.push_back(std::vector<int>(1, i));
         }
      }
   }
   else
   {
      for (int start = 0; start < n; start += block_size)
      {
         const int m = std::min(block_size, n - start);
         std::vector<int> block_ids(m);
         for (int i = 0; i < m; i++) { block_ids[i] = start + i; }
         op_patch_block_indices_.push_back(std::move(block_ids));
      }
   }

   mfem::Vector e(patch_op->Width()), Ae(patch_op->Height());
   int max_dim = 0;
   for (const auto &ids : op_patch_block_indices_)
   {
      const int m = static_cast<int>(ids.size());
      const int dim = op_patch_block_scalar_ ? m : 2 * m;
      max_dim = std::max(max_dim, dim);
      op_patch_block_sizes_.push_back(m);
      auto block = std::make_unique<mfem::DenseMatrix>(dim, dim);
      *block = 0.0;

      for (int col = 0; col < dim; col++)
      {
         const int global_col = op_patch_block_scalar_ ?
            ids[col] : ((col < m) ? ids[col] : (n + ids[col - m]));
         e = 0.0;
         e[global_col] = 1.0;
         patch_op->Mult(e, Ae);
         if (op_patch_block_scalar_)
         {
            for (int row = 0; row < m; row++)
            {
               (*block)(row, col) = Ae[ids[row]];
            }
         }
         else
         {
            for (int row = 0; row < m; row++)
            {
               (*block)(row, col) = Ae[ids[row]];
               (*block)(m + row, col) = Ae[n + ids[row]];
            }
         }
      }

      double max_diag = 0.0;
      for (int i = 0; i < dim; i++)
      {
         max_diag = std::max(max_diag, std::abs((*block)(i, i)));
      }
      const double eps_reg =
         (max_diag > 0.0) ? std::max(1e-12, 1e-10 * max_diag) : 1e-12;
      for (int i = 0; i < dim; i++)
      {
         (*block)(i, i) += eps_reg;
      }

      op_patch_block_invs_.push_back(
         std::make_unique<mfem::DenseMatrixInverse>(*block));
      op_patch_block_mats_.push_back(std::move(block));
   }

   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] operator patch block smoother built: "
                << "weight=" << operator_patch_block_smoother_weight_
                << ", mode=" << (operator_element_patch_blocks_ ?
                                  "element" : "contiguous")
                << ", operator=" << (op_patch_block_scalar_ ? "scalar_ref" :
                                      "full_block")
                << ", block_size=" << block_size
                << ", iterations=" << operator_patch_block_smoother_iterations_
                << ", blocks=" << op_patch_block_invs_.size()
                << ", max_block_dim=" << max_dim
                << std::endl;
   }
}

void CovariantReferencePreconditioner::AddOperatorPatchBlockSmoother(
   const mfem::Vector &r,
   mfem::Vector &z) const
{
   if (operator_patch_block_smoother_weight_ == 0.0) { return; }

   const int rsize = r.Size();
   if (!real_block_mode_)
   {
      MFEM_VERIFY(rsize % 2 == 0,
                  "Patch block smoother requires a 2x2 real block residual.");
   }
   const int n = real_block_mode_ ? rsize : rsize / 2;
   op_patch_block_work_.SetSize(rsize);
   op_patch_block_res_.SetSize(rsize);
   op_patch_block_Awork_.SetSize(rsize);
   op_patch_block_work_ = 0.0;

   for (int it = 0; it < operator_patch_block_smoother_iterations_; it++)
   {
      op_->Mult(op_patch_block_work_, op_patch_block_Awork_);
      op_patch_block_res_ = r;
      op_patch_block_res_.Add(-1.0, op_patch_block_Awork_);

      for (int b = 0; b < (int)op_patch_block_invs_.size(); b++)
      {
         const auto &ids = op_patch_block_indices_[b];
         const int m = op_patch_block_sizes_[b];
         if (real_block_mode_)
         {
            mfem::Vector rhs(m), sol(m);
            for (int i = 0; i < m; i++)
            {
               rhs[i] = op_patch_block_res_[ids[i]];
            }
            op_patch_block_invs_[b]->Mult(rhs, sol);
            for (int i = 0; i < m; i++)
            {
               op_patch_block_work_[ids[i]] +=
                  operator_patch_block_smoother_weight_ * sol[i];
            }
         }
         else if (op_patch_block_scalar_)
         {
            mfem::Vector rhs(m), sol(m);
            for (int i = 0; i < m; i++)
            {
               rhs[i] = op_patch_block_res_[ids[i]];
            }
            op_patch_block_invs_[b]->Mult(rhs, sol);
            for (int i = 0; i < m; i++)
            {
               op_patch_block_work_[ids[i]] +=
                  operator_patch_block_smoother_weight_ * sol[i];
            }
            for (int i = 0; i < m; i++)
            {
               rhs[i] = op_patch_block_res_[n + ids[i]];
            }
            op_patch_block_invs_[b]->Mult(rhs, sol);
            for (int i = 0; i < m; i++)
            {
               op_patch_block_work_[n + ids[i]] +=
                  operator_patch_block_smoother_weight_ * sol[i];
            }
         }
         else
         {
            mfem::Vector rhs(2 * m), sol(2 * m);
            for (int i = 0; i < m; i++)
            {
               rhs[i] = op_patch_block_res_[ids[i]];
               rhs[m + i] = op_patch_block_res_[n + ids[i]];
            }
            op_patch_block_invs_[b]->Mult(rhs, sol);
            for (int i = 0; i < m; i++)
            {
               op_patch_block_work_[ids[i]] +=
                  operator_patch_block_smoother_weight_ * sol[i];
               op_patch_block_work_[n + ids[i]] +=
                  operator_patch_block_smoother_weight_ * sol[m + i];
            }
         }
      }
   }

   z += op_patch_block_work_;
}

void CovariantReferencePreconditioner::Mult(const mfem::Vector &r,
                                            mfem::Vector &z) const
{
   MFEM_VERIFY(op_ != nullptr, "Operator must be set before Mult().");
   const int tvsize = fespace_.GetTrueVSize();
   if (real_block_mode_)
   {
      MFEM_VERIFY(r.Size() == tvsize, "Real-block residual size mismatch.");
      BuildAuxiliaryOperators();

      Pi_.MultTranspose(r, aux_rhs_);
      SolveAuxiliarySystem(aux_rhs_, aux_sol_);
      Pi_.Mult(aux_sol_, z);
      z *= coarse_correction_weight_;
      if (identity_smoother_weight_ != 0.0)
      {
         z.Add(identity_smoother_weight_, r);
      }
      AddOperatorJacobiSmoother(r, z);
      AddOperatorPatchBlockSmoother(r, z);
      return;
   }

   MFEM_VERIFY(r.Size() == 2 * tvsize, "Residual size mismatch.");

   BuildAuxiliaryOperators();

   mfem::Vector r_r, r_i;
   r_r.MakeRef(const_cast<mfem::Vector &>(r), 0, tvsize);
   r_i.MakeRef(const_cast<mfem::Vector &>(r), tvsize, tvsize);
   r_re_ = r_r;
   r_im_ = r_i;

   if (yee_complex_auxiliary_ && mode_ == PrototypeMode::edge_yee_proto)
   {
      const int na = static_cast<int>(aux_dofs_.size());
      mfem::Vector rhs_re, rhs_im, sol_re, sol_im;
      rhs_re.MakeRef(aux_rhs_block_, 0, na);
      rhs_im.MakeRef(aux_rhs_block_, na, na);
      sol_re.MakeRef(aux_sol_block_, 0, na);
      sol_im.MakeRef(aux_sol_block_, na, na);

      Pi_.MultTranspose(r_re_, rhs_re);
      Pi_.MultTranspose(r_im_, rhs_im);
      SolveAuxiliaryBlockSystem(aux_rhs_block_, aux_sol_block_);
      Pi_.Mult(sol_re, z_re_);
      Pi_.Mult(sol_im, z_im_);

      z.SetSize(2 * tvsize);
      mfem::Vector z_r, z_i;
      z_r.MakeRef(z, 0, tvsize);
      z_i.MakeRef(z, tvsize, tvsize);
      z_r = z_re_;
      z_i = z_im_;
      z *= coarse_correction_weight_;
      if (identity_smoother_weight_ != 0.0)
      {
         z_r.Add(identity_smoother_weight_, r_r);
         z_i.Add(identity_smoother_weight_, r_i);
      }
      AddOperatorJacobiSmoother(r, z);
      AddOperatorBlockJacobiSmoother(r, z);
      AddOperatorPatchBlockSmoother(r, z);
      return;
   }

   Pi_.MultTranspose(r_re_, aux_rhs_);
   SolveAuxiliarySystem(aux_rhs_, aux_sol_);
   Pi_.Mult(aux_sol_, z_re_);

   Pi_.MultTranspose(r_im_, aux_rhs_);
   SolveAuxiliarySystem(aux_rhs_, aux_sol_);
   Pi_.Mult(aux_sol_, z_im_);

   z.SetSize(2 * tvsize);
   mfem::Vector z_r, z_i;
   z_r.MakeRef(z, 0, tvsize);
   z_i.MakeRef(z, tvsize, tvsize);
   z_r = z_re_;
   z_i = z_im_;
   z *= coarse_correction_weight_;
   if (identity_smoother_weight_ != 0.0)
   {
      z_r.Add(identity_smoother_weight_, r_r);
      z_i.Add(identity_smoother_weight_, r_i);
   }
   AddOperatorJacobiSmoother(r, z);
   AddOperatorBlockJacobiSmoother(r, z);
   AddOperatorPatchBlockSmoother(r, z);
}

void CovariantReferencePreconditioner::PrintCoarseOperatorDiagnostics(
   std::ostream &os) const
{
   MFEM_VERIFY(mode_ != PrototypeMode::nodal_proto,
               "Coarse operator diagnostics are intended for edge-based prototypes.");
   BuildAuxiliaryOperators();

   mfem::DenseMatrix Agal, Ayee;
   BuildGalerkinEdgeCoarseMatrix(Agal);
   yee_operator_->AssembleYeeMaxwellOperator(eps_fn_, k0_, Ayee);

   const int n = Agal.Height();
   int inactive_transfer_cols = 0;
   if (Pi_.Height() > 0 && Pi_.Width() == n)
   {
      for (int j = 0; j < n; j++)
      {
         double col_norm2 = 0.0;
         for (int i = 0; i < Pi_.Height(); i++)
         {
            col_norm2 += Pi_(i, j) * Pi_(i, j);
         }
         if (col_norm2 < 1e-24)
         {
            inactive_transfer_cols++;
            for (int k = 0; k < n; k++)
            {
               Ayee(j, k) = 0.0;
               Ayee(k, j) = 0.0;
            }
         }
      }
   }
   double gal_frob2 = 0.0;
   double yee_frob2 = 0.0;
   double diff_frob2 = 0.0;
   double gal_diag_min = std::numeric_limits<double>::infinity();
   double gal_diag_max = 0.0;
   double yee_diag_min = std::numeric_limits<double>::infinity();
   double yee_diag_max = 0.0;
   double gal_diag_mean = 0.0;
   double yee_diag_mean = 0.0;
   int yee_zero_diag_count = 0;
   std::vector<int> yee_zero_diag_ids;

   for (int i = 0; i < n; i++)
   {
      gal_diag_min = std::min(gal_diag_min, std::abs(Agal(i, i)));
      gal_diag_max = std::max(gal_diag_max, std::abs(Agal(i, i)));
      yee_diag_min = std::min(yee_diag_min, std::abs(Ayee(i, i)));
      yee_diag_max = std::max(yee_diag_max, std::abs(Ayee(i, i)));
      if (std::abs(Ayee(i, i)) < 1e-14)
      {
         yee_zero_diag_count++;
         if ((int)yee_zero_diag_ids.size() < 12)
         {
            yee_zero_diag_ids.push_back(i);
         }
      }
      gal_diag_mean += std::abs(Agal(i, i));
      yee_diag_mean += std::abs(Ayee(i, i));
      for (int j = 0; j < n; j++)
      {
         const double g = Agal(i, j);
         const double y = Ayee(i, j);
         gal_frob2 += g * g;
         yee_frob2 += y * y;
         const double d = y - g;
         diff_frob2 += d * d;
      }
   }

   const double gal_frob = std::sqrt(gal_frob2);
   const double yee_frob = std::sqrt(yee_frob2);
   const double diff_frob = std::sqrt(diff_frob2);
   const double rel_diff = (gal_frob > 0.0) ? diff_frob / gal_frob : 0.0;
   gal_diag_mean /= double(n);
   yee_diag_mean /= double(n);
   const double diag_scale = (yee_diag_mean > 0.0) ? gal_diag_mean / yee_diag_mean : 1.0;

   os << "[covariant_aux_space] coarse operator diagnostics\n"
      << "  mode=" << ModeName(mode_) << '\n'
      << "  coarse_dofs=" << n << '\n'
      << "  inactive_transfer_cols=" << inactive_transfer_cols << '\n'
      << "  ||A_gal||_F=" << gal_frob << '\n'
      << "  ||A_yee||_F=" << yee_frob << '\n'
      << "  ||A_yee - A_gal||_F=" << diff_frob << '\n'
      << "  relative_F_error=" << rel_diff << '\n'
      << "  gal_diag_abs_mean=" << gal_diag_mean << '\n'
      << "  yee_diag_abs_mean=" << yee_diag_mean << '\n'
      << "  suggested_diag_scale=" << diag_scale << '\n'
      << "  gal_diag_abs_range=[" << gal_diag_min << ", " << gal_diag_max << "]\n"
      << "  yee_diag_abs_range=[" << yee_diag_min << ", " << yee_diag_max << "]\n"
      << "  yee_zero_diag_count=" << yee_zero_diag_count;

   if (!yee_zero_diag_ids.empty())
   {
      os << "\n  yee_zero_diag_examples=";
      for (int idx = 0; idx < (int)yee_zero_diag_ids.size(); idx++)
      {
         const int dof_id = yee_zero_diag_ids[idx];
         const auto &dof = aux_dofs_[dof_id];
         os << " #" << dof_id
            << "(i=" << dof.i
            << ",j=" << dof.j
            << ",k=" << dof.k
            << ",axis=" << dof.axis << ")";
      }
   }

   os << std::endl;

   if (mode_ == PrototypeMode::edge_yee_proto)
   {
      yee_operator_->PrintDiagnostics(eps_fn_, k0_, os);
   }
}

void CovariantReferencePreconditioner::PrintYeeGalerkinComparison(
   std::ostream &os) const
{
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before comparing Yee and Galerkin coarse matrices.");
   PrintYeeGalerkinComparison(*op_, "system_real_block", os);
}

void CovariantReferencePreconditioner::PrintYeeGalerkinComparison(
   const mfem::Operator &ref_op,
   const char *ref_name,
   std::ostream &os) const
{
   MFEM_VERIFY(mode_ == PrototypeMode::edge_yee_proto,
               "Yee/Galerkin comparison is only meaningful for edge_yee_proto.");
   BuildAuxDofs();
   BuildTransferMatrix();

   mfem::DenseMatrix Ayee;
   yee_operator_->AssembleYeeMaxwellOperator(eps_fn_, k0_, Ayee);
   PrintYeeCandidateComparison(ref_op, Ayee, ref_name, "A_yee", os);
}

void CovariantReferencePreconditioner::PrintYeeCandidateComparison(
   const mfem::Operator &ref_op,
   const mfem::DenseMatrix &Acand_in,
   const char *ref_name,
   const char *cand_name,
   std::ostream &os) const
{
   MFEM_VERIFY(mode_ == PrototypeMode::edge_yee_proto,
               "Yee candidate comparison is only meaningful for edge_yee_proto.");

   const int tvsize = fespace_.GetTrueVSize();
   const bool compatible_height =
      (ref_op.Height() == tvsize || ref_op.Height() == 2 * tvsize);
   const bool compatible_width =
      (ref_op.Width() == tvsize || ref_op.Width() == 2 * tvsize);
   if (!compatible_height || !compatible_width)
   {
      os << "[covariant_aux_space] yee-galerkin comparison\n"
         << "  reference=" << (ref_name ? ref_name : "unnamed") << '\n'
         << "  candidate=" << (cand_name ? cand_name : "unnamed") << '\n'
         << "  skipped=1\n"
         << "  reference_dims=" << ref_op.Height() << "x" << ref_op.Width() << '\n'
         << "  expected_height_or_width=" << tvsize << " or " << (2 * tvsize)
         << std::endl;
      return;
   }

   mfem::DenseMatrix Agal, Acand = Acand_in;
   BuildGalerkinEdgeCoarseMatrix(ref_op, Agal);

   const int n = Agal.Height();
   int inactive_transfer_cols = 0;
   if (Pi_.Height() > 0 && Pi_.Width() == n)
   {
      for (int j = 0; j < n; j++)
      {
         double col_norm2 = 0.0;
         for (int i = 0; i < Pi_.Height(); i++)
         {
            col_norm2 += Pi_(i, j) * Pi_(i, j);
         }
         if (col_norm2 < 1e-24)
         {
            inactive_transfer_cols++;
            for (int k = 0; k < n; k++)
            {
               Acand(j, k) = 0.0;
               Acand(k, j) = 0.0;
            }
         }
      }
   }
   double gal_frob2 = 0.0;
   double yee_frob2 = 0.0;
   double diff_frob2 = 0.0;
   double dot_y_g = 0.0;
   double gal_diag_mean = 0.0;
   double yee_diag_mean = 0.0;
   double gal_diag_min = std::numeric_limits<double>::infinity();
   double yee_diag_min = std::numeric_limits<double>::infinity();
   double gal_diag_max = 0.0;
   double yee_diag_max = 0.0;

   for (int i = 0; i < n; i++)
   {
      const double gd = std::abs(Agal(i, i));
      const double yd = std::abs(Acand(i, i));
      gal_diag_mean += gd;
      yee_diag_mean += yd;
      gal_diag_min = std::min(gal_diag_min, gd);
      yee_diag_min = std::min(yee_diag_min, yd);
      gal_diag_max = std::max(gal_diag_max, gd);
      yee_diag_max = std::max(yee_diag_max, yd);
      for (int j = 0; j < n; j++)
      {
         const double g = Agal(i, j);
         const double y = Acand(i, j);
         gal_frob2 += g * g;
         yee_frob2 += y * y;
         dot_y_g += y * g;
         const double d = y - g;
         diff_frob2 += d * d;
      }
   }

   const double alpha = (yee_frob2 > 0.0) ? dot_y_g / yee_frob2 : 1.0;
   double scaled_diff_frob2 = 0.0;
   for (int i = 0; i < n; i++)
   {
      for (int j = 0; j < n; j++)
      {
         const double d = alpha * Acand(i, j) - Agal(i, j);
         scaled_diff_frob2 += d * d;
      }
   }

   const double gal_frob = std::sqrt(gal_frob2);
   const double yee_frob = std::sqrt(yee_frob2);
   const double diff_frob = std::sqrt(diff_frob2);
   const double scaled_diff_frob = std::sqrt(scaled_diff_frob2);
   gal_diag_mean /= double(n);
   yee_diag_mean /= double(n);

   os << "[covariant_aux_space] yee-galerkin comparison\n"
      << "  reference=" << (ref_name ? ref_name : "unnamed") << '\n'
      << "  candidate=" << (cand_name ? cand_name : "unnamed") << '\n'
      << "  coarse_dofs=" << n << '\n'
      << "  inactive_transfer_cols=" << inactive_transfer_cols << '\n'
      << "  ||A_gal||_F=" << gal_frob << '\n'
      << "  ||A_cand||_F=" << yee_frob << '\n'
      << "  ||A_cand-A_gal||_F/||A_gal||_F="
      << ((gal_frob > 0.0) ? diff_frob / gal_frob : 0.0) << '\n'
      << "  best_frobenius_scale_alpha=" << alpha << '\n'
      << "  ||alpha*A_cand-A_gal||_F/||A_gal||_F="
      << ((gal_frob > 0.0) ? scaled_diff_frob / gal_frob : 0.0) << '\n'
      << "  diag_abs_mean: gal=" << gal_diag_mean
      << ", yee=" << yee_diag_mean
      << ", ratio=" << ((yee_diag_mean > 0.0) ? gal_diag_mean / yee_diag_mean : 0.0)
      << '\n'
      << "  diag_abs_range_gal=[" << gal_diag_min << ", " << gal_diag_max << "]\n"
      << "  diag_abs_range_yee=[" << yee_diag_min << ", " << yee_diag_max << "]"
      << std::endl;
}

} // namespace covariant_aux_space
