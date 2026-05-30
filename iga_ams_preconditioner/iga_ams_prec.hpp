#ifndef IGA_AMS_PRECONDITIONER_HPP
#define IGA_AMS_PRECONDITIONER_HPP

#include "mfem.hpp"
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <complex>

namespace iga_ams
{

// ─── Variant 1: AMS on o=1 IGA Galerkin coarse operator ─────────────────────
//
//   B_p^{-1} r = P * AMS(A_1)^{-1} * P^T r + S_p r
//
// where A_1 is the Galerkin coarse operator (P^T A_p P or directly assembled),
// P = M_p^{-1} M_{p,1} is the H(curl) mass-projection prolongation,
// S_p is a block-Jacobi or Jacobi smoother on the fine level.
class CoarseAMS_GalerkinPreconditioner : public mfem::Solver
{
public:
   enum class CoarseMatrixMode { galerkin_proj, direct_assemble };
   enum class CombineMode { coarse_only, additive, multiplicative };

   CoarseAMS_GalerkinPreconditioner(
      const mfem::ParFiniteElementSpace &fine_fespace,
      int coarse_order = 1);
   ~CoarseAMS_GalerkinPreconditioner() override;

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

   void SetCoarseMatrixMode(CoarseMatrixMode m) { cm_mode_ = m; built_ = false; }
   void SetCombineMode(CombineMode m) { combine_mode_ = m; }
   void SetCoarseWeight(double w) { coarse_weight_ = w; }
   void SetJacobiWeight(double w) { jacobi_weight_ = w; }
   void SetVerbose(bool v) { verbose_ = v; }
   void SetAMSPrintLevel(int lvl) { ams_print_level_ = lvl; }

private:
   const mfem::ParFiniteElementSpace &fine_fespace_;
   int coarse_order_ = 1;
   CoarseMatrixMode cm_mode_ = CoarseMatrixMode::direct_assemble;
   CombineMode combine_mode_ = CombineMode::multiplicative;
   double coarse_weight_ = 1.0, jacobi_weight_ = 1.0;
   bool verbose_ = true;
   int ams_print_level_ = 1;

   const mfem::Operator *op_ = nullptr;

   // Coarse IGA H(curl) space (order = coarse_order_)
   std::unique_ptr<mfem::FiniteElementCollection> coarse_fec_;
   std::unique_ptr<mfem::NURBSExtension>          coarse_nurbs_ext_;
   std::unique_ptr<mfem::ParFiniteElementSpace>   coarse_fespace_;

   // Prolongation P: o=1 → o=p
   mutable bool built_ = false;
   mutable mfem::DenseMatrix P_;

   // Fine smoother (2x2 block Jacobi)
   mutable mfem::Vector bj00_, bj01_, bj10_, bj11_;

   // Coarse AMS solver
   mutable std::unique_ptr<mfem::HypreParMatrix> A1_hypre_;
   mutable std::unique_ptr<mfem::HypreAMS>       ams_solver_;

   // Work vectors
   mutable mfem::Vector z1_, r1_, z2_;

   void Build();
   void BuildCoarseSpace();
   void BuildMassProlongation();
   void BuildCoarseAMS();
   void BuildBlockJacobi();
   void ApplyCoarse(const mfem::Vector &r, mfem::Vector &z) const;
   void ApplyJacobi(const mfem::Vector &r, mfem::Vector &z) const;
};

// ─── Variant 2: IGA-exact gradient correction ───────────────────────────────
//
//   B_G^{-1} r = G * A_phi^{-1} * G^T r
//
// where G is the discrete gradient from H1 to H(curl) IGA,
// A_phi = G^T A_p G is the scalar auxiliary operator.
class IGA_GradientCorrectionPreconditioner : public mfem::Solver
{
public:
   IGA_GradientCorrectionPreconditioner(
      const mfem::ParFiniteElementSpace &hcurl_fespace,
      const mfem::ParFiniteElementSpace *h1_fespace = nullptr);
   ~IGA_GradientCorrectionPreconditioner() override;

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

   /// Use BoomerAMG to solve A_phi (default: true)
   void SetUseAMG(bool use) { use_amg_ = use; built_ = false; }
   /// Use lumped mass + Chebyshev instead of full A_phi
   void SetUseLumpedMass(bool use) { use_lumped_mass_ = use; built_ = false; }
   void SetPrintLevel(int lvl) { print_level_ = lvl; }
   void SetVerbose(bool v) { verbose_ = v; }

   /// Get the H1 scalar space
   const mfem::ParFiniteElementSpace &GetH1Space() const { return *h1_fespace_; }
   /// Get the H(curl) space
   const mfem::ParFiniteElementSpace &GetHCurlSpace() const { return hcurl_fespace_; }

private:
   const mfem::ParFiniteElementSpace &hcurl_fespace_;
   std::unique_ptr<mfem::FiniteElementCollection> h1_fec_own_;
   std::unique_ptr<mfem::NURBSExtension> h1_nurbs_ext_own_;
   std::unique_ptr<mfem::ParFiniteElementSpace> h1_fespace_own_;
   const mfem::ParFiniteElementSpace *h1_fespace_;
   const mfem::Operator *op_ = nullptr;
   bool use_amg_ = true;
   bool use_lumped_mass_ = false;
   int print_level_ = 1;
   bool verbose_ = true;

   // Discrete gradient G: H1 → H(curl)
   mutable std::unique_ptr<mfem::HypreParMatrix> G_;

   // Scalar operator A_phi = G^T A_p G
   mutable std::unique_ptr<mfem::HypreParMatrix> A_phi_;

   // Solver for A_phi
   mutable std::unique_ptr<mfem::HypreBoomerAMG> amg_;

   // Work vectors
   mutable mfem::Vector r_grad_, z_grad_;
   mutable mfem::Vector local_r_, local_z_;

   mutable bool built_ = false;
   void Build();
};

// ─── Variant 3: Shifted AMS ─────────────────────────────────────────────────
//
// Constructs A_shift = A_p + eta * M_p
// Applies AMS to A_shift and uses it as preconditioner for A_p.
class ShiftedAMS_Preconditioner : public mfem::Solver
{
public:
   enum class ShiftType { additive_mass, complex_shift };

   ShiftedAMS_Preconditioner(const mfem::ParFiniteElementSpace &fespace,
                              double freq, double eps_val = 1.0);
   ~ShiftedAMS_Preconditioner() override;

   void SetOperator(const mfem::Operator &op) override;
   void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

   void SetEta(double eta) { eta_ = eta; built_ = false; }
   void SetShiftType(ShiftType t) { shift_type_ = t; built_ = false; }
   void SetSingularProblem(bool sp) { singular_ = sp; built_ = false; }
   void SetPrintLevel(int lvl) { print_level_ = lvl; }
   void SetVerbose(bool v) { verbose_ = v; }

private:
   const mfem::ParFiniteElementSpace &fespace_;
   double freq_ = 5.0, eps_val_ = 1.0, eta_ = 0.3;
   ShiftType shift_type_ = ShiftType::additive_mass;
   bool singular_ = false;
   int print_level_ = 1;
   bool verbose_ = true;

   const mfem::Operator *op_ = nullptr;

   // Shifted operator: A_shift = A_p + eta*Mass
   mutable std::unique_ptr<mfem::HypreParMatrix> A_shift_;

   // AMS solver on the shifted operator
   mutable std::unique_ptr<mfem::HypreAMS> ams_;

   mutable bool built_ = false;
   void Build();
};

// ─── Diagnostic utilities ───────────────────────────────────────────────────

double GlobalL2Norm(const mfem::Vector &v, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace iga_ams

#endif // IGA_AMS_PRECONDITIONER_HPP
