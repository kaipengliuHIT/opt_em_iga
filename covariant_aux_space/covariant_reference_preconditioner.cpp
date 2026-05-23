#include "covariant_reference_preconditioner.hpp"
#include <iostream>
#include <limits>
#include <cmath>

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
   AauxInv_.reset();
   aux_rhs_.SetSize(0);
   aux_sol_.SetSize(0);
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
   aux_solver_.SetGrid(grid);
   yee_transfer_->SetGrid(grid);
   yee_operator_->SetGrid(grid);
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
      yee_operator_->AssembleYeeMaxwellOperator(eps_fn_, k0_, Aaux_);
      if (yee_diag_calibration_)
      {
         mfem::DenseMatrix Agal;
         BuildGalerkinEdgeCoarseMatrix(Agal);
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
      }
   }
   else
   {
      MFEM_ABORT("Unknown prototype mode in coarse operator assembly.");
   }

   for (int i = 0; i < na; i++)
   {
      Aaux_(i, i) += 1e-8;
   }
   AauxInv_ = std::make_unique<mfem::DenseMatrixInverse>(Aaux_);
}

void CovariantReferencePreconditioner::BuildGalerkinEdgeCoarseMatrix(
   mfem::DenseMatrix &Agal) const
{
   MFEM_VERIFY(op_ != nullptr,
               "Operator must be set before building Galerkin coarse matrix.");
   BuildAuxDofs();
   BuildTransferMatrix();
   const int na = static_cast<int>(aux_dofs_.size());
   const int tvsize = fespace_.GetTrueVSize();
   Agal.SetSize(na, na);
   Agal = 0.0;

   mfem::Vector coarse_col(tvsize);
   mfem::Vector full_in(2 * tvsize), full_out(2 * tvsize);
   mfem::Vector real_in, real_out;
   real_in.MakeRef(full_in, 0, tvsize);
   real_out.MakeRef(full_out, 0, tvsize);

   for (int j = 0; j < na; j++)
   {
      coarse_col.SetSize(tvsize);
      for (int i = 0; i < tvsize; i++)
      {
         coarse_col[i] = Pi_(i, j);
      }
      full_in = 0.0;
      real_in = coarse_col;
      op_->Mult(full_in, full_out);
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
   BuildTransferMatrix();
   BuildAuxiliaryMatrix();

   const int na = static_cast<int>(aux_dofs_.size());
   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[covariant_aux_space] auxiliary dofs=" << na
                << ", true_vsize=" << fespace_.GetTrueVSize() << std::endl;
   }
   aux_rhs_.SetSize(na);
   aux_sol_.SetSize(na);
   built_ = true;
}

void CovariantReferencePreconditioner::Mult(const mfem::Vector &r,
                                            mfem::Vector &z) const
{
   MFEM_VERIFY(op_ != nullptr, "Operator must be set before Mult().");
   const int tvsize = fespace_.GetTrueVSize();
   MFEM_VERIFY(r.Size() == 2 * tvsize, "Residual size mismatch.");

   BuildAuxiliaryOperators();

   mfem::Vector r_r, r_i;
   r_r.MakeRef(const_cast<mfem::Vector &>(r), 0, tvsize);
   r_i.MakeRef(const_cast<mfem::Vector &>(r), tvsize, tvsize);
   r_re_ = r_r;
   r_im_ = r_i;

   Pi_.MultTranspose(r_re_, aux_rhs_);
   AauxInv_->Mult(aux_rhs_, aux_sol_);
   Pi_.Mult(aux_sol_, z_re_);

   Pi_.MultTranspose(r_im_, aux_rhs_);
   AauxInv_->Mult(aux_rhs_, aux_sol_);
   Pi_.Mult(aux_sol_, z_im_);

   z.SetSize(2 * tvsize);
   mfem::Vector z_r, z_i;
   z_r.MakeRef(z, 0, tvsize);
   z_i.MakeRef(z, tvsize, tvsize);
   z_r = z_re_;
   z_i = z_im_;
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

} // namespace covariant_aux_space
