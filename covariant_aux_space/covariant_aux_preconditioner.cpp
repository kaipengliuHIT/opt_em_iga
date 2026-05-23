#include "covariant_aux_preconditioner.hpp"

namespace covariant_aux_space
{

CovariantAuxPreconditioner::CovariantAuxPreconditioner(
   const mfem::Operator &op,
   const AuxiliaryRestriction &restriction,
   const AuxiliarySolver &aux_solver)
   : mfem::Solver(op.Height(), op.Width()),
     op_(&op),
     restriction_(&restriction),
     aux_solver_(&aux_solver),
     r_aux_(restriction.AuxSize()),
     z_aux_(restriction.AuxSize())
{
}

void CovariantAuxPreconditioner::SetOperator(const mfem::Operator &op)
{
   op_ = &op;
   height = op.Height();
   width = op.Width();
}

void CovariantAuxPreconditioner::Mult(const mfem::Vector &r, mfem::Vector &z) const
{
   MFEM_VERIFY(op_ != nullptr, "Operator must be set before Mult().");
   MFEM_VERIFY(restriction_ != nullptr, "Auxiliary restriction is missing.");
   MFEM_VERIFY(aux_solver_ != nullptr, "Auxiliary solver is missing.");

   restriction_->Restrict(r, r_aux_);
   aux_solver_->Solve(r_aux_, z_aux_);
   restriction_->Prolongate(z_aux_, z);
}

void IdentityAuxRestriction::Restrict(const mfem::Vector &r, mfem::Vector &r_aux) const
{
   MFEM_VERIFY(r.Size() == size_, "Identity restriction size mismatch.");
   r_aux.SetSize(size_);
   r_aux = r;
}

void IdentityAuxRestriction::Prolongate(const mfem::Vector &z_aux, mfem::Vector &z) const
{
   MFEM_VERIFY(z_aux.Size() == size_, "Identity prolongation size mismatch.");
   z.SetSize(size_);
   z = z_aux;
}

DiagonalAuxSolver::DiagonalAuxSolver(const mfem::Vector &diag)
   : diag_(diag)
{
}

void DiagonalAuxSolver::Solve(const mfem::Vector &rhs_aux, mfem::Vector &sol_aux) const
{
   MFEM_VERIFY(rhs_aux.Size() == diag_.Size(), "Auxiliary diagonal size mismatch.");
   sol_aux.SetSize(diag_.Size());
   for (int i = 0; i < diag_.Size(); i++)
   {
      const double d = std::abs(diag_[i]) > 0.0 ? diag_[i] : 1.0;
      sol_aux[i] = rhs_aux[i] / d;
   }
}

} // namespace covariant_aux_space
