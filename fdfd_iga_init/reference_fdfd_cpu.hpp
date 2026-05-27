#ifndef FDFD_IGA_INIT_REFERENCE_FDFD_CPU_HPP
#define FDFD_IGA_INIT_REFERENCE_FDFD_CPU_HPP

#include "mfem.hpp"
#include "reference_patch_evaluator.hpp"
#include <functional>
#include <vector>

namespace fdfd_iga_init
{

struct ReferenceGrid
{
   int nx = 17;
   int ny = 17;
   int nz = 17;

   /// Optional: non-uniform knot positions in parameter space [0,1].
   /// When empty, uniform spacing is used: x_i = i / (nx-1).
   std::vector<double> knot_x;
   std::vector<double> knot_y;
   std::vector<double> knot_z;

   bool HasNonUniformSpacing() const { return !knot_x.empty(); }
};

struct SampledReferenceField
{
   int nx = 0, ny = 0, nz = 0, dim = 0;
   std::vector<double> real;
   std::vector<double> imag;

   int NodeCount() const { return nx * ny * nz; }
};

class ReferenceFDFDCPU
{
public:
   explicit ReferenceFDFDCPU(const SinglePatchNURBSEvaluator &geom);

   void SetGrid(const ReferenceGrid &grid) { grid_ = grid; }
   const ReferenceGrid &Grid() const { return grid_; }
   void SetWaveNumber(double k0) { k0_ = k0; }
   void SetMaxIterations(int max_it) { max_it_ = max_it; }
   void SetDamping(double omega) { damping_ = omega; }
   void SetMassShift(double shift) { mass_shift_ = shift; }
   int VectorSize() const { return 3 * grid_.nx * grid_.ny * grid_.nz; }

   SampledReferenceField Solve(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_re_fn,
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_im_fn) const;

   void ApplyDiscreteOperator(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      const std::vector<double> &x,
      std::vector<double> &y) const;
   void AssembleSystemMatrix(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      mfem::DenseMatrix &A) const;

   /// Precomputed spacing data for non-uniform grids.
   /// Call after SetGrid if you need spacing info externally.
   struct GridSpacing
   {
      std::vector<double> cell_width_x;  // [nx-1]: cell_width[i] = x_{i+1} - x_i
      std::vector<double> cell_width_y;
      std::vector<double> cell_width_z;
      std::vector<double> node_pos_x;    // [nx]: x_i for each grid node
      std::vector<double> node_pos_y;
      std::vector<double> node_pos_z;
   };
   GridSpacing ComputeSpacing() const;

private:
   struct MetricSample
   {
      double curl_metric[9];
      double mass_metric[9];
   };

   const SinglePatchNURBSEvaluator &geom_;
   ReferenceGrid grid_;
   double k0_ = 1.0;
   int max_it_ = 250;
   double damping_ = 0.6;
   double mass_shift_ = 0.2;

   int NodeId(int i, int j, int k) const
   {
      return (k * grid_.ny + j) * grid_.nx + i;
   }

   void SampleMetric(const std::function<double(const mfem::Vector &)> &eps_fn,
                     std::vector<MetricSample> &metric) const;

   void SampleSource(
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_fn,
      std::vector<double> &rhs) const;

   void ApplyCurlCurlMinusMass(const std::vector<MetricSample> &metric,
                               const std::vector<double> &x,
                               std::vector<double> &y) const;

   void BuildDiagonalPreconditioner(const std::vector<MetricSample> &metric,
                                    std::vector<double> &diag) const;

   void JacobiSolve(const std::vector<MetricSample> &metric,
                    const std::vector<double> &rhs,
                    std::vector<double> &x) const;
};

/// Build a grid whose nodes fall on unique knot positions.
/// \param kv  Knot vectors (one per parametric direction).
/// \param cells_per_span  Sub-cells per knot span (>= 1).
/// \return  ReferenceGrid with nx = nspans*cells_per_span + 1,
///          and knot_x/y/z populated with actual knot positions.
inline ReferenceGrid ComputeKnotAlignedGrid(
   const mfem::Array<const mfem::KnotVector *> &kv,
   int cells_per_span = 1)
{
   ReferenceGrid g{0, 0, 0};
   const int dim = kv.Size();

   // Helper: extract unique knot positions and check uniformity
   auto extract_knots = [&](const mfem::KnotVector &k,
                              std::vector<double> &knots, int &n_span) -> bool
   {
      const int order = k.GetOrder();
      const int ncp = k.GetNCP();
      std::vector<double> unique;
      double prev = k[order];
      unique.push_back(prev);
      for (int i = order + 1; i <= ncp; i++)
      {
         if (k[i] > prev + 1e-14) { unique.push_back(k[i]); prev = k[i]; }
      }
      n_span = static_cast<int>(unique.size()) - 1;
      if (n_span <= 0) { n_span = 1; unique = {0.0, 1.0}; }

      // Check uniformity
      bool is_uniform = true;
      if (n_span > 1)
      {
         double ref_spacing = unique[1] - unique[0];
         for (size_t s = 1; s < unique.size(); s++)
         {
            double sp = unique[s] - unique[s-1];
            if (std::abs(sp - ref_spacing) > 1e-12 * std::max(1.0, ref_spacing))
            {
               is_uniform = false;
               break;
            }
            ref_spacing = sp;
         }
      }
      knots = std::move(unique);
      return is_uniform;
   };

   auto subdivide_knots = [cells_per_span](const std::vector<double> &knots,
                                             std::vector<double> &out)
   {
      out.clear();
      out.reserve((knots.size() - 1) * cells_per_span + 1);
      for (size_t s = 0; s < knots.size() - 1; s++)
      {
         double a = knots[s], b = knots[s+1];
         for (int c = 0; c < cells_per_span; c++)
         {
            out.push_back(a + (b - a) * double(c) / double(cells_per_span));
         }
      }
      out.push_back(knots.back());
   };

   // X direction
   if (dim >= 1)
   {
      std::vector<double> kx;
      int nspan_x = 1;
      bool unif_x = extract_knots(*kv[0], kx, nspan_x);
      subdivide_knots(kx, g.knot_x);
      g.nx = static_cast<int>(g.knot_x.size());
      if (unif_x && nspan_x == 1)
      {
         // Single span uniform → don't store knot positions (backward compat)
      }
   }
   else { g.knot_x = {0.0, 1.0}; g.nx = 2; }

   // Y direction
   if (dim >= 2)
   {
      std::vector<double> ky;
      int nspan_y = 1;
      bool unif_y = extract_knots(*kv[1], ky, nspan_y);
      subdivide_knots(ky, g.knot_y);
      g.ny = static_cast<int>(g.knot_y.size());
   }
   else { g.knot_y = {0.0, 1.0}; g.ny = 2; }

   // Z direction
   if (dim >= 3)
   {
      std::vector<double> kz;
      int nspan_z = 1;
      bool unif_z = extract_knots(*kv[2], kz, nspan_z);
      subdivide_knots(kz, g.knot_z);
      g.nz = static_cast<int>(g.knot_z.size());
   }
   else { g.knot_z = {0.0, 1.0}; g.nz = 2; }

   // Minimum sizes
   if (g.nx < 3) { g.knot_x = {0.0, 0.5, 1.0}; g.nx = 3; }
   if (g.ny < 3) { g.knot_y = {0.0, 0.5, 1.0}; g.ny = 3; }
   if (g.nz < 3) { g.knot_z = {0.0, 0.5, 1.0}; g.nz = 3; }

   return g;
}

/// Return true if all knot arrays represent uniform spacing within tolerance.
inline bool IsUniformGrid(const ReferenceGrid &g)
{
   if (g.knot_x.empty()) return true;
   if (g.nx <= 2) return true;
   double hx = g.knot_x[1] - g.knot_x[0];
   for (int i = 1; i < g.nx; i++)
      if (std::abs((g.knot_x[i] - g.knot_x[i-1]) - hx) > 1e-12) return false;
   if (g.ny > 2)
   {
      double hy = g.knot_y[1] - g.knot_y[0];
      for (int i = 1; i < g.ny; i++)
         if (std::abs((g.knot_y[i] - g.knot_y[i-1]) - hy) > 1e-12) return false;
   }
   if (g.nz > 2)
   {
      double hz = g.knot_z[1] - g.knot_z[0];
      for (int i = 1; i < g.nz; i++)
         if (std::abs((g.knot_z[i] - g.knot_z[i-1]) - hz) > 1e-12) return false;
   }
   return true;
}

} // namespace fdfd_iga_init

#endif
