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

} // namespace fdfd_iga_init

#endif
