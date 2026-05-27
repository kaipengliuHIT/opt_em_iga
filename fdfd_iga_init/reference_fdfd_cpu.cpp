#include "reference_fdfd_cpu.hpp"
#include <algorithm>
#include <cmath>

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

// Precomputed spacing for fast access
struct SpacingData
{
   bool is_uniform;
   double hx, hy, hz;                          // uniform spacing (if uniform)
   std::vector<double> cx, cy, cz;             // cell widths [nx-1], [ny-1], [nz-1]
   std::vector<double> px, py, pz;             // node positions [nx], [ny], [nz]
};

SpacingData MakeSpacing(const ReferenceGrid &grid)
{
   SpacingData s;
   const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
   s.is_uniform = IsUniformGrid(grid);

   if (s.is_uniform && grid.knot_x.empty())
   {
      s.hx = 1.0 / double(nx - 1);
      s.hy = 1.0 / double(ny - 1);
      s.hz = 1.0 / double(nz - 1);
      return s;
   }

   // Populate node positions
   s.px = grid.knot_x;
   s.py = grid.knot_y;
   s.pz = grid.knot_z;

   // Cell widths
   s.cx.resize(nx - 1);
   for (int i = 0; i < nx - 1; i++) s.cx[i] = s.px[i+1] - s.px[i];
   s.cy.resize(ny - 1);
   for (int i = 0; i < ny - 1; i++) s.cy[i] = s.py[i+1] - s.py[i];
   s.cz.resize(nz - 1);
   for (int i = 0; i < nz - 1; i++) s.cz[i] = s.pz[i+1] - s.pz[i];

   // For is_uniform with stored knots
   if (s.is_uniform)
   {
      s.hx = s.cx[0];
      s.hy = (ny > 2) ? s.cy[0] : 1.0;
      s.hz = (nz > 2) ? s.cz[0] : 1.0;
   }
   return s;
}

} // namespace

ReferenceFDFDCPU::GridSpacing ReferenceFDFDCPU::ComputeSpacing() const
{
   auto s = MakeSpacing(grid_);
   GridSpacing gs;
   gs.cell_width_x = std::move(s.cx);
   gs.cell_width_y = std::move(s.cy);
   gs.cell_width_z = std::move(s.cz);
   gs.node_pos_x = std::move(s.px);
   gs.node_pos_y = std::move(s.py);
   gs.node_pos_z = std::move(s.pz);
   return gs;
}

ReferenceFDFDCPU::ReferenceFDFDCPU(const SinglePatchNURBSEvaluator &geom)
   : geom_(geom)
{
}

void ReferenceFDFDCPU::SampleMetric(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   std::vector<MetricSample> &metric) const
{
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
   const int n = nx * ny * nz;
   metric.resize(n);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J, invJ, JTJ, invJJT;
   const int dim = geom_.Dimension();

   auto s = MakeSpacing(grid_);

   for (int k = 0; k < nz; k++)
   {
      xi[2] = s.is_uniform ? double(k) / double(nz - 1) : s.pz[k];
      for (int j = 0; j < ny; j++)
      {
         xi[1] = s.is_uniform ? double(j) / double(ny - 1) : s.py[j];
         for (int i = 0; i < nx; i++)
         {
            xi[0] = s.is_uniform ? double(i) / double(nx - 1) : s.px[i];
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
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
   const int nnode = nx * ny * nz;
   rhs.assign(3 * nnode, 0.0);
   mfem::Vector xi(3), val(3);
   auto s = MakeSpacing(grid_);

   for (int k = 0; k < nz; k++)
   {
      xi[2] = s.is_uniform ? double(k) / double(nz - 1) : s.pz[k];
      for (int j = 0; j < ny; j++)
      {
         xi[1] = s.is_uniform ? double(j) / double(ny - 1) : s.py[j];
         for (int i = 0; i < nx; i++)
         {
            xi[0] = s.is_uniform ? double(i) / double(nx - 1) : s.px[i];
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
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
   const int nnode = nx * ny * nz;
   y.assign(3 * nnode, 0.0);

   auto s = MakeSpacing(grid_);

   // Centered difference helpers. For uniform: denominator = 2*h.
   // For non-uniform: denominator = x_{i+1} - x_{i-1}.
   auto diff_x = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i + 1, j, k);
      const int id_m = NodeId(i - 1, j, k);
      double denom = s.is_uniform ? 2.0 * s.hx :
                     (s.px[i+1] - s.px[i-1]);
      return (GetComp(x, nnode, c, id_p) - GetComp(x, nnode, c, id_m)) / denom;
   };
   auto diff_y = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i, j + 1, k);
      const int id_m = NodeId(i, j - 1, k);
      double denom = s.is_uniform ? 2.0 * s.hy :
                     (s.py[j+1] - s.py[j-1]);
      return (GetComp(x, nnode, c, id_p) - GetComp(x, nnode, c, id_m)) / denom;
   };
   auto diff_z = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i, j, k + 1);
      const int id_m = NodeId(i, j, k - 1);
      double denom = s.is_uniform ? 2.0 * s.hz :
                     (s.pz[k+1] - s.pz[k-1]);
      return (GetComp(x, nnode, c, id_p) - GetComp(x, nnode, c, id_m)) / denom;
   };

   // Step 1: compute curl E → flux
   std::vector<double> flux(3 * nnode, 0.0);
   for (int k = 1; k < nz - 1; k++)
   {
      for (int j = 1; j < ny - 1; j++)
      {
         for (int i = 1; i < nx - 1; i++)
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

   // Step 2: compute curl of flux (same stencil)
   auto flux_dx = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i + 1, j, k);
      const int id_m = NodeId(i - 1, j, k);
      double denom = s.is_uniform ? 2.0 * s.hx :
                     (s.px[i+1] - s.px[i-1]);
      return (GetComp(flux, nnode, c, id_p) - GetComp(flux, nnode, c, id_m)) / denom;
   };
   auto flux_dy = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i, j + 1, k);
      const int id_m = NodeId(i, j - 1, k);
      double denom = s.is_uniform ? 2.0 * s.hy :
                     (s.py[j+1] - s.py[j-1]);
      return (GetComp(flux, nnode, c, id_p) - GetComp(flux, nnode, c, id_m)) / denom;
   };
   auto flux_dz = [&](int c, int i, int j, int k) -> double
   {
      const int id_p = NodeId(i, j, k + 1);
      const int id_m = NodeId(i, j, k - 1);
      double denom = s.is_uniform ? 2.0 * s.hz :
                     (s.pz[k+1] - s.pz[k-1]);
      return (GetComp(flux, nnode, c, id_p) - GetComp(flux, nnode, c, id_m)) / denom;
   };

   for (int k = 1; k < nz - 1; k++)
   {
      for (int j = 1; j < ny - 1; j++)
      {
         for (int i = 1; i < nx - 1; i++)
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
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
   const int nnode = nx * ny * nz;
   diag.assign(3 * nnode, 1.0);

   auto s = MakeSpacing(grid_);

   // Gershgorin-like diagonal scaling
   for (int k = 0; k < nz; k++)
   {
      for (int j = 0; j < ny; j++)
      {
         for (int i = 0; i < nx; i++)
         {
            const int id = NodeId(i, j, k);
            if (i == 0 || j == 0 || k == 0 ||
                i == nx - 1 || j == ny - 1 || k == nz - 1)
            {
               diag[id] = diag[nnode + id] = diag[2 * nnode + id] = 1.0;
               continue;
            }

            // Laplacian scaling: 1/h² per direction
            double hx_inv2, hy_inv2, hz_inv2;
            if (s.is_uniform)
            {
               hx_inv2 = 1.0 / (s.hx * s.hx);
               hy_inv2 = 1.0 / (s.hy * s.hy);
               hz_inv2 = 1.0 / (s.hz * s.hz);
            }
            else
            {
               // Average of left and right cell inverse squares
               double dx_l = s.px[i] - s.px[i-1], dx_r = s.px[i+1] - s.px[i];
               double dy_l = s.py[j] - s.py[j-1], dy_r = s.py[j+1] - s.py[j];
               double dz_l = s.pz[k] - s.pz[k-1], dz_r = s.pz[k+1] - s.pz[k];
               hx_inv2 = 0.5 * (1.0/(dx_l*dx_l) + 1.0/(dx_r*dx_r));
               hy_inv2 = 0.5 * (1.0/(dy_l*dy_l) + 1.0/(dy_r*dy_r));
               hz_inv2 = 0.5 * (1.0/(dz_l*dz_l) + 1.0/(dz_r*dz_r));
            }
            const double scale = 2.0 * (hx_inv2 + hy_inv2 + hz_inv2);

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
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
   const int nnode = nx * ny * nz;
   std::vector<double> diag, Ax, residual;
   BuildDiagonalPreconditioner(metric, diag);
   x.assign(3 * nnode, 0.0);
   residual.assign(3 * nnode, 0.0);

   for (int it = 0; it < max_it_; it++)
   {
      ApplyCurlCurlMinusMass(metric, x, Ax);
      for (int c = 0; c < 3; c++)
      {
         for (int k = 0; k < nz; k++)
         {
            for (int j = 0; j < ny; j++)
            {
               for (int i = 0; i < nx; i++)
               {
                  const int id = NodeId(i, j, k);
                  if (i == 0 || j == 0 || k == 0 ||
                      i == nx - 1 || j == ny - 1 || k == nz - 1)
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
