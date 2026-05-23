#ifndef FDFD_IGA_INIT_REFERENCE_FDFD_CUDA_HPP
#define FDFD_IGA_INIT_REFERENCE_FDFD_CUDA_HPP

#include "reference_fdfd_cpu.hpp"

namespace fdfd_iga_init
{

class ReferenceFDFDCUDA
{
public:
   explicit ReferenceFDFDCUDA(const SinglePatchNURBSEvaluator &geom) : geom_(geom) {}

   void SetGrid(const ReferenceGrid &grid) { grid_ = grid; }
   void SetWaveNumber(double k0) { k0_ = k0; }

   SampledReferenceField Solve(
      const std::function<double(const mfem::Vector &)> &eps_fn,
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_re_fn,
      const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_im_fn) const;

private:
   const SinglePatchNURBSEvaluator &geom_;
   ReferenceGrid grid_;
   double k0_ = 1.0;
};

} // namespace fdfd_iga_init

#endif
