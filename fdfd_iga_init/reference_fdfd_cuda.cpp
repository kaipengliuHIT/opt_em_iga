#include "reference_fdfd_cuda.hpp"

namespace fdfd_iga_init
{

SampledReferenceField ReferenceFDFDCUDA::Solve(
   const std::function<double(const mfem::Vector &)> &,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &) const
{
   MFEM_ABORT(
      "ReferenceFDFDCUDA is a stub in the first prototype. "
      "The intended next step is to port the CPU metric-aware stencil into a "
      "Maxwell-style GPU backend derived from maxwellsolver.");
}

} // namespace fdfd_iga_init
