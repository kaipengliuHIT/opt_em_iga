#ifndef COVARIANT_AUX_SPACE_COVARIANT_AUX_PRECONDITIONER_HPP
#define COVARIANT_AUX_SPACE_COVARIANT_AUX_PRECONDITIONER_HPP

#include "mfem.hpp"
#include <functional>

namespace covariant_aux_space
{

class AuxiliaryRestriction
{
public:
   virtual ~AuxiliaryRestriction() = default;

   virtual int AuxSize() const = 0;
   virtual void Restrict(const mfem::Vector &r, mfem::Vector &r_aux) const = 0;
   virtual void Prolongate(const mfem::Vector &z_aux, mfem::Vector &z) const = 0;
};

class AuxiliarySolver
{
public:
   virtual ~AuxiliarySolver() = default;

   virtual int Size() const = 0;
   virtual void Solve(const mfem::Vector &rhs_aux, mfem::Vector &sol_aux) const = 0;
};

class CovariantAuxPreconditioner : public mfem::Solver
{
public:
   CovariantAuxPreconditioner(const mfem::Operator &op,
                             const AuxiliaryRestriction &restriction,
                             const AuxiliarySolver &aux_solver);

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

private:
   const mfem::Operator *op_;
   const AuxiliaryRestriction *restriction_;
   const AuxiliarySolver *aux_solver_;
   mutable mfem::Vector r_aux_;
   mutable mfem::Vector z_aux_;
};

class IdentityAuxRestriction : public AuxiliaryRestriction
{
public:
   explicit IdentityAuxRestriction(int size) : size_(size) {}

   int AuxSize() const override { return size_; }
   void Restrict(const mfem::Vector &r, mfem::Vector &r_aux) const override;
   void Prolongate(const mfem::Vector &z_aux, mfem::Vector &z) const override;

private:
   int size_;
};

class DiagonalAuxSolver : public AuxiliarySolver
{
public:
   explicit DiagonalAuxSolver(const mfem::Vector &diag);

   int Size() const override { return diag_.Size(); }
   void Solve(const mfem::Vector &rhs_aux, mfem::Vector &sol_aux) const override;

private:
   mfem::Vector diag_;
};

} // namespace covariant_aux_space

#endif
