#include "reference_fdfd_cpu.hpp"
#include <algorithm>

namespace fdfd_iga_init
{

namespace
{

inline void MatVec3(const double A[9], const double x[3], double y[3])
{
   for (int i = 0; i < 3; i++)
   {
      y[i] = 0.0;
      for (int j = 0; j < 3; j++)
      {
         y[i] += A[3 * i + j] * x[j];
      }
   }
}

inline double GetComp(const std::vector<double> &v, int nnode, int c, int id)
{
   return v[c * nnode + id];
}

inline void SetComp(std::vector<double> &v, int nnode, int c, int id, double val)
{
   v[c * nnode + id] = val;
}

} // namespace

ReferenceFDFDCPU::ReferenceFDFDCPU(const SinglePatchNURBSEvaluator &geom)
   : geom_(geom)
{
}

void ReferenceFDFDCPU::SampleMetric(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   std::vector<MetricSample> &metric) const
{
   const int n = grid_.nx * grid_.ny * grid_.nz;
   metric.resize(n);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J, invJ, JTJ, invJJT;
   const int dim = geom_.Dimension();
   for (int k = 0; k < grid_.nz; k++)
   {
      xi[2] = (grid_.nz == 1) ? 0.0 : double(k) / double(grid_.nz - 1);
      for (int j = 0; j < grid_.ny; j++)
      {
         xi[1] = (grid_.ny == 1) ? 0.0 : double(j) / double(grid_.ny - 1);
         for (int i = 0; i < grid_.nx; i++)
         {
            xi[0] = (grid_.nx == 1) ? 0.0 : double(i) / double(grid_.nx - 1);
            geom_.EvalGeometry(xi, x_phys, J);

            const double detJ = J.Det();
            invJ = J;
            invJ.Invert();
            JTJ.SetSize(dim, dim);
            invJJT.SetSize(dim, dim);
            mfem::MultAtB(J, J, JTJ);
            mfem::MultABt(invJ, invJ, invJJT);
            MetricSample ms {};
            for (int r = 0; r < 3; r++)
            {
               for (int c = 0; c < 3; c++)
               {
                  const double curl_val =
                     (r < dim && c < dim) ? JTJ(r, c) / detJ : 0.0;
                  const double mass_val =
                     (r < dim && c < dim) ?
                     detJ * eps_fn(x_phys) * invJJT(r, c) : 0.0;
                  ms.curl_metric[3 * r + c] = curl_val;
                  ms.mass_metric[3 * r + c] = mass_val;
               }
            }
            metric[NodeId(i, j, k)] = ms;
         }
      }
   }
}

void ReferenceFDFDCPU::SampleSource(
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_fn,
   std::vector<double> &rhs) const
{
   const int nnode = grid_.nx * grid_.ny * grid_.nz;
   rhs.assign(3 * nnode, 0.0);
   mfem::Vector xi(3), val(3);
   for (int k = 0; k < grid_.nz; k++)
   {
      xi[2] = (grid_.nz == 1) ? 0.0 : double(k) / double(grid_.nz - 1);
      for (int j = 0; j < grid_.ny; j++)
      {
         xi[1] = (grid_.ny == 1) ? 0.0 : double(j) / double(grid_.ny - 1);
         for (int i = 0; i < grid_.nx; i++)
         {
            xi[0] = (grid_.nx == 1) ? 0.0 : double(i) / double(grid_.nx - 1);
            val.SetSize(3);
            val = 0.0;
            src_fn(xi, val);
            const int id = NodeId(i, j, k);
            for (int c = 0; c < 3; c++)
            {
               rhs[c * nnode + id] = val[c];
            }
         }
      }
   }
}

void ReferenceFDFDCPU::ApplyCurlCurlMinusMass(
   const std::vector<MetricSample> &metric,
   const std::vector<double> &x,
   std::vector<double> &y) const
{
   const int nnode = grid_.nx * grid_.ny * grid_.nz;
   const double hx = 1.0 / double(grid_.nx - 1);
   const double hy = 1.0 / double(grid_.ny - 1);
   const double hz = 1.0 / double(grid_.nz - 1);
   y.assign(3 * nnode, 0.0);

   auto diff_x = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(x, nnode, c, NodeId(i + 1, j, k)) -
              GetComp(x, nnode, c, NodeId(i - 1, j, k))) / (2.0 * hx);
   };
   auto diff_y = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(x, nnode, c, NodeId(i, j + 1, k)) -
              GetComp(x, nnode, c, NodeId(i, j - 1, k))) / (2.0 * hy);
   };
   auto diff_z = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(x, nnode, c, NodeId(i, j, k + 1)) -
              GetComp(x, nnode, c, NodeId(i, j, k - 1))) / (2.0 * hz);
   };

   std::vector<double> flux(3 * nnode, 0.0);
   for (int k = 1; k < grid_.nz - 1; k++)
   {
      for (int j = 1; j < grid_.ny - 1; j++)
      {
         for (int i = 1; i < grid_.nx - 1; i++)
         {
            const int id = NodeId(i, j, k);
            const double curlE[3] = {
               diff_y(2, i, j, k) - diff_z(1, i, j, k),
               diff_z(0, i, j, k) - diff_x(2, i, j, k),
               diff_x(1, i, j, k) - diff_y(0, i, j, k)
            };
            double weighted[3];
            MatVec3(metric[id].curl_metric, curlE, weighted);
            for (int c = 0; c < 3; c++)
            {
               SetComp(flux, nnode, c, id, weighted[c]);
            }
         }
      }
   }

   auto flux_dx = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(flux, nnode, c, NodeId(i + 1, j, k)) -
              GetComp(flux, nnode, c, NodeId(i - 1, j, k))) / (2.0 * hx);
   };
   auto flux_dy = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(flux, nnode, c, NodeId(i, j + 1, k)) -
              GetComp(flux, nnode, c, NodeId(i, j - 1, k))) / (2.0 * hy);
   };
   auto flux_dz = [&](int c, int i, int j, int k) -> double
   {
      return (GetComp(flux, nnode, c, NodeId(i, j, k + 1)) -
              GetComp(flux, nnode, c, NodeId(i, j, k - 1))) / (2.0 * hz);
   };

   for (int k = 1; k < grid_.nz - 1; k++)
   {
      for (int j = 1; j < grid_.ny - 1; j++)
      {
         for (int i = 1; i < grid_.nx - 1; i++)
         {
            const int id = NodeId(i, j, k);
            const double curlFlux[3] = {
               flux_dy(2, i, j, k) - flux_dz(1, i, j, k),
               flux_dz(0, i, j, k) - flux_dx(2, i, j, k),
               flux_dx(1, i, j, k) - flux_dy(0, i, j, k)
            };
            const double Ein[3] = {
               GetComp(x, nnode, 0, id),
               GetComp(x, nnode, 1, id),
               GetComp(x, nnode, 2, id)
            };
            double massE[3];
            MatVec3(metric[id].mass_metric, Ein, massE);
            for (int c = 0; c < 3; c++)
            {
               SetComp(y, nnode, c, id, curlFlux[c] - k0_ * k0_ * massE[c]);
            }
         }
      }
   }
}

void ReferenceFDFDCPU::BuildDiagonalPreconditioner(
   const std::vector<MetricSample> &metric,
   std::vector<double> &diag) const
{
   const int nnode = grid_.nx * grid_.ny * grid_.nz;
   const double hx = 1.0 / double(grid_.nx - 1);
   const double hy = 1.0 / double(grid_.ny - 1);
   const double hz = 1.0 / double(grid_.nz - 1);
   const double scale = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz);
   diag.assign(3 * nnode, 1.0);

   for (int k = 0; k < grid_.nz; k++)
   {
      for (int j = 0; j < grid_.ny; j++)
      {
         for (int i = 0; i < grid_.nx; i++)
         {
            const int id = NodeId(i, j, k);
            if (i == 0 || j == 0 || k == 0 ||
                i == grid_.nx - 1 || j == grid_.ny - 1 || k == grid_.nz - 1)
            {
               diag[id] = diag[nnode + id] = diag[2 * nnode + id] = 1.0;
               continue;
            }

            const double curl_trace =
               metric[id].curl_metric[0] + metric[id].curl_metric[4] +
               metric[id].curl_metric[8];
            for (int c = 0; c < 3; c++)
            {
               const double mass_cc = metric[id].mass_metric[3 * c + c];
               diag[c * nnode + id] =
                  std::max(1.0, scale * std::max(1e-8, curl_trace)
                                    + k0_ * k0_ * std::abs(mass_cc));
            }
         }
      }
   }
}

void ReferenceFDFDCPU::JacobiSolve(const std::vector<MetricSample> &metric,
                                   const std::vector<double> &rhs,
                                   std::vector<double> &x) const
{
   const int nnode = grid_.nx * grid_.ny * grid_.nz;
   std::vector<double> diag, Ax, residual;
   BuildDiagonalPreconditioner(metric, diag);
   x.assign(3 * nnode, 0.0);
   residual.assign(3 * nnode, 0.0);

   for (int it = 0; it < max_it_; it++)
   {
      ApplyCurlCurlMinusMass(metric, x, Ax);
      for (int c = 0; c < 3; c++)
      {
         for (int k = 0; k < grid_.nz; k++)
         {
            for (int j = 0; j < grid_.ny; j++)
            {
               for (int i = 0; i < grid_.nx; i++)
               {
                  const int id = NodeId(i, j, k);
                  if (i == 0 || j == 0 || k == 0 ||
                      i == grid_.nx - 1 || j == grid_.ny - 1 || k == grid_.nz - 1)
                  {
                     x[c * nnode + id] = 0.0;
                     continue;
                  }
                  const int idx = c * nnode + id;
                  residual[idx] = rhs[idx] - Ax[idx];
                  x[idx] += damping_ * residual[idx] / diag[idx];
               }
            }
         }
      }
   }
}

SampledReferenceField ReferenceFDFDCPU::Solve(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_re_fn,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &src_im_fn) const
{
   SampledReferenceField field;
   field.nx = grid_.nx;
   field.ny = grid_.ny;
   field.nz = grid_.nz;
   field.dim = 3;

   std::vector<MetricSample> metric;
   std::vector<double> rhs_re, rhs_im;
   SampleMetric(eps_fn, metric);
   SampleSource(src_re_fn, rhs_re);
   SampleSource(src_im_fn, rhs_im);
   JacobiSolve(metric, rhs_re, field.real);
   JacobiSolve(metric, rhs_im, field.imag);
   return field;
}

void ReferenceFDFDCPU::ApplyDiscreteOperator(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   const std::vector<double> &x,
   std::vector<double> &y) const
{
   std::vector<MetricSample> metric;
   SampleMetric(eps_fn, metric);
   ApplyCurlCurlMinusMass(metric, x, y);
   if (mass_shift_ != 0.0)
   {
      for (int i = 0; i < VectorSize(); i++)
      {
         y[i] += mass_shift_ * x[i];
      }
   }
}

void ReferenceFDFDCPU::AssembleSystemMatrix(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   mfem::DenseMatrix &A) const
{
   std::vector<MetricSample> metric;
   SampleMetric(eps_fn, metric);

   const int ndof = VectorSize();
   std::vector<double> e(ndof, 0.0);
   std::vector<double> Ae;
   A.SetSize(ndof, ndof);
   A = 0.0;

   for (int j = 0; j < ndof; j++)
   {
      std::fill(e.begin(), e.end(), 0.0);
      e[j] = 1.0;
      ApplyCurlCurlMinusMass(metric, e, Ae);
      for (int i = 0; i < ndof; i++)
      {
         A(i, j) = Ae[i];
      }
      A(j, j) += mass_shift_;
   }
}

} // namespace fdfd_iga_init
