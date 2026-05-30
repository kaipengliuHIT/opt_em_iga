#ifndef YEE_INIT_GUESS_YEE_TO_IGA_TRANSFER_HPP
#define YEE_INIT_GUESS_YEE_TO_IGA_TRANSFER_HPP

#include "mfem.hpp"
#include <functional>
#include <vector>

namespace yee_init_guess
{

/// Edge DOF descriptor for the Yee grid in parameter domain.
struct YeeEdge
{
   int i = 0, j = 0, k = 0;   // grid indices
   int axis = 0;               // 0=x, 1=y, 2=z
};

/// Builds the Yee→o=1→o=p transfer chain.
///
/// Usage:
///   1. Build Yee grid edges: BuildYeeEdges(nx, ny, nz)
///   2. Solve Yee system externally → u_yee
///   3. Map to o=1 IGA:      MapYeeToO1(u_yee, u1)
///   4. Prolongate to o=p:   ProlongateO1ToOp(u1, up)
///   5. up is the IGA initial guess
class YeeToIGATransfer
{
public:
   YeeToIGATransfer(mfem::ParFiniteElementSpace &fine_fespace,
                    int coarse_order = 1);

   /// Build Yee edge list on uniform reference grid.
   void BuildYeeEdges(int nx, int ny, int nz);

   int GetNumYeeEdges() const { return yee_edges_.size(); }
   const std::vector<YeeEdge> &GetYeeEdges() const { return yee_edges_; }
   double DX() const { return dx_; }
   double DY() const { return dy_; }
   double DZ() const { return dz_; }

   /// Edge-integral mapping: u_yee (size na) → u1 (size nc_true, o=1 IGA).
   /// Pi1_int(dof, e) = 1.0 if Yee edge e is a sub-edge of IGA edge dof.
   void MapYeeToO1(const mfem::Vector &u_yee, mfem::Vector &u1) const;

   /// H(curl) mass projection: u1 (o=1) → up (o=p fine space).
   /// P = M_p^{-1} * M_{p,1}  (lumped-mass approximation).
   void ProlongateO1ToOp(const mfem::Vector &u1, mfem::Vector &up) const;

   /// Full chain: u_yee → u1 → up.
   void MapYeeToOp(const mfem::Vector &u_yee, mfem::Vector &up) const;

   int GetFineTrueVSize()  const { return fine_fespace_.GetTrueVSize(); }
   int GetCoarseTrueVSize() const;

   /// Accessors for diagnostic
   const mfem::DenseMatrix &GetPi1() const { return Pi1_; }
   const mfem::DenseMatrix &GetProlongation() const { return P_; }

private:
   mfem::ParFiniteElementSpace &fine_fespace_;
   int coarse_order_ = 1;

   // Coarse o=1 IGA H(curl) space
   std::unique_ptr<mfem::FiniteElementCollection> coarse_fec_;
   std::unique_ptr<mfem::NURBSExtension>          coarse_nurbs_ext_;
   std::unique_ptr<mfem::ParFiniteElementSpace>   coarse_fespace_;

   // Yee grid
   std::vector<YeeEdge> yee_edges_;
   double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;

   // Transfer matrices
   mutable mfem::DenseMatrix Pi1_;    // Yee→o=1 edge-integral (nc × na)
   mutable mfem::DenseMatrix P_;      // o=1→o=p mass projection (nf × nc)
   mutable bool built_ = false;

   void Build() const;
   void BuildCoarseSpace();
   void BuildPi1() const;
   void BuildProlongation() const;

   // Helpers
   void ProbeDOFLocation(int dof, int &axis, double &xi_lo, double &xi_hi,
                         double &c0, double &c1) const;
};

} // namespace yee_init_guess

#endif
