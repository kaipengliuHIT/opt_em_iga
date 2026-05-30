#include "yee_operator.hpp"
#include <complex>
#include <limits>

namespace covariant_aux_space
{

namespace
{

struct LocalEdgeDef
{
   int axis;
   int a;
   int b;
};

struct LocalFaceDef
{
   int axis;
   int side;
};

const LocalEdgeDef kCellEdges[12] =
{
   {0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 1},
   {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1},
   {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}
};

const LocalFaceDef kCellFaces[6] =
{
   {0, 0}, {0, 1},
   {1, 0}, {1, 1},
   {2, 0}, {2, 1}
};

double L0(double t) { return 1.0 - t; }
double L1(double t) { return t; }
double dL0() { return -1.0; }
double dL1() { return 1.0; }

void EvalLocalEdgeBasis(int edge, double u, double v, double w,
                        double hx, double hy, double hz,
                        mfem::Vector &E, mfem::Vector &curlE)
{
   E.SetSize(3);
   curlE.SetSize(3);
   E = 0.0;
   curlE = 0.0;
   const auto le = kCellEdges[edge];

   const double Lu = le.a ? L1(u) : L0(u);
   const double Lv = le.a ? L1(v) : L0(v);
   const double Lw = le.b ? L1(w) : L0(w);
   const double dU = le.a ? dL1() : dL0();
   const double dV = le.a ? dL1() : dL0();
   const double dW = le.b ? dL1() : dL0();

   if (le.axis == 0)
   {
      const double lv = le.a ? L1(v) : L0(v);
      const double lw = le.b ? L1(w) : L0(w);
      const double dv = le.a ? dL1() : dL0();
      const double dw = le.b ? dL1() : dL0();
      E[0] = lv * lw / hx;
      curlE[1] = lv * dw / (hx * hz);
      curlE[2] = -dv * lw / (hx * hy);
      return;
   }
   if (le.axis == 1)
   {
      const double lu = le.a ? L1(u) : L0(u);
      const double lw = le.b ? L1(w) : L0(w);
      const double du = le.a ? dL1() : dL0();
      const double dw = le.b ? dL1() : dL0();
      E[1] = lu * lw / hy;
      curlE[0] = -lu * dw / (hy * hz);
      curlE[2] = du * lw / (hx * hy);
      return;
   }

   const double lu = le.a ? L1(u) : L0(u);
   const double lv = le.b ? L1(v) : L0(v);
   const double du = le.a ? dL1() : dL0();
   const double dv = le.b ? dL1() : dL0();
   E[2] = lu * lv / hz;
   curlE[0] = lu * dv / (hy * hz);
   curlE[1] = -du * lv / (hx * hz);
}

void EvalLocalFaceBasis(int face, double u, double v, double w,
                        double hx, double hy, double hz,
                        mfem::Vector &F)
{
   F.SetSize(3);
   F = 0.0;
   const auto lf = kCellFaces[face];
   const double Lu = lf.side ? L1(u) : L0(u);
   const double Lv = lf.side ? L1(v) : L0(v);
   const double Lw = lf.side ? L1(w) : L0(w);

   if (lf.axis == 0)
   {
      F[0] = Lu / (hy * hz);
   }
   else if (lf.axis == 1)
   {
      F[1] = Lv / (hx * hz);
   }
   else
   {
      F[2] = Lw / (hx * hy);
   }
}

double PhysicalEdgeMetric(const mfem::DenseMatrix &J, int axis)
{
   return std::sqrt(J(0, axis) * J(0, axis) +
                    J(1, axis) * J(1, axis) +
                    J(2, axis) * J(2, axis));
}

double PhysicalFaceMetric(const mfem::DenseMatrix &J, int axis)
{
   // Compute |J * (e_b × e_c)| = |col_b × col_c|  (cross-product formula)
   // This avoids matrix inversion which can fail for degenerate Jacobians.
   // For face normal along `axis`, the area is the magnitude of the
   // cross product of the two columns of J orthogonal to `axis`.
   const int b = (axis + 1) % 3;
   const int c = (axis + 2) % 3;
   const double cb0 = J(0, b), cb1 = J(1, b), cb2 = J(2, b);
   const double cc0 = J(0, c), cc1 = J(1, c), cc2 = J(2, c);
   const double cx = cb1 * cc2 - cb2 * cc1;
   const double cy = cb2 * cc0 - cb0 * cc2;
   const double cz = cb0 * cc1 - cb1 * cc0;
   const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
   // Guard against NaN from degenerate geometry
   if (!std::isfinite(area) || area < 1e-30) { return 0.0; }
   return area;
}

} // namespace

YeeOperatorBuilder::YeeOperatorBuilder(
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom)
   : geom_(geom)
{
}


// ── Non-uniform grid helpers ──────────────────────────────────────────
namespace {

/// Compute position of a Yee node/edge/face center using knot positions.
/// When grid is uniform, falls back to i * h.
struct GridMapper
{
   const fdfd_iga_init::ReferenceGrid &grid;
   bool has_knots;

   GridMapper(const fdfd_iga_init::ReferenceGrid &g)
      : grid(g), has_knots(!g.knot_x.empty()) {}

   double node_x(int i) const {
      return has_knots ? grid.knot_x[i] : double(i) / (grid.nx - 1);
   }
   double node_y(int j) const {
      return has_knots ? grid.knot_y[j] : double(j) / (grid.ny - 1);
   }
   double node_z(int k) const {
      return has_knots ? grid.knot_z[k] : double(k) / (grid.nz - 1);
   }

   /// Edge center in param space (midpoint between two nodes)
   double edge_x(int i) const {
      return has_knots ? 0.5 * (grid.knot_x[i] + grid.knot_x[i+1])
                       : (i + 0.5) / (grid.nx - 1);
   }
   double edge_y(int j) const {
      return has_knots ? 0.5 * (grid.knot_y[j] + grid.knot_y[j+1])
                       : (j + 0.5) / (grid.ny - 1);
   }
   double edge_z(int k) const {
      return has_knots ? 0.5 * (grid.knot_z[k] + grid.knot_z[k+1])
                       : (k + 0.5) / (grid.nz - 1);
   }

   /// Cell width (h) at index i
   double cell_hx(int i) const {
      return has_knots ? (grid.knot_x[i+1] - grid.knot_x[i])
                       : 1.0 / (grid.nx - 1);
   }
   double cell_hy(int j) const {
      return has_knots ? (grid.knot_y[j+1] - grid.knot_y[j])
                       : 1.0 / (grid.ny - 1);
   }
   double cell_hz(int k) const {
      return has_knots ? (grid.knot_z[k+1] - grid.knot_z[k])
                       : 1.0 / (grid.nz - 1);
   }
};

} // namespace

void YeeOperatorBuilder::SetGrid(const fdfd_iga_init::ReferenceGrid &grid)
{
   grid_ = grid;
   edge_dofs_.clear();
   face_dofs_.clear();
}

void YeeOperatorBuilder::SetReferencePML(bool enabled, double thickness,
                                         double strength, double order)
{
   pml_enabled_ = enabled;
   pml_thickness_ = thickness;
   pml_strength_ = strength;
   pml_order_ = order;
}

const std::vector<YeeEdgeDof> &YeeOperatorBuilder::GetEdgeDofs() const
{
   BuildEdgeDofs();
   return edge_dofs_;
}

const std::vector<YeeFaceDof> &YeeOperatorBuilder::GetFaceDofs() const
{
   BuildFaceDofs();
   return face_dofs_;
}

void YeeOperatorBuilder::BuildEdgeDofs() const
{
   if (!edge_dofs_.empty()) { return; }

   MFEM_VERIFY(grid_.nx > 1 && grid_.ny > 1 && grid_.nz > 1,
               "Yee operator grid is not initialized.");

   for (int k = 1; k < grid_.nz - 1; k++)
   {
      for (int j = 1; j < grid_.ny - 1; j++)
      {
         for (int i = 0; i < grid_.nx - 1; i++)
         {
            edge_dofs_.push_back({i, j, k, 0});
         }
      }
   }
   for (int k = 1; k < grid_.nz - 1; k++)
   {
      for (int j = 0; j < grid_.ny - 1; j++)
      {
         for (int i = 1; i < grid_.nx - 1; i++)
         {
            edge_dofs_.push_back({i, j, k, 1});
         }
      }
   }
   for (int k = 0; k < grid_.nz - 1; k++)
   {
      for (int j = 1; j < grid_.ny - 1; j++)
      {
         for (int i = 1; i < grid_.nx - 1; i++)
         {
            edge_dofs_.push_back({i, j, k, 2});
         }
      }
   }
}

void YeeOperatorBuilder::BuildFaceDofs() const
{
   if (!face_dofs_.empty()) { return; }

   MFEM_VERIFY(grid_.nx > 1 && grid_.ny > 1 && grid_.nz > 1,
               "Yee operator grid is not initialized.");

   for (int k = 0; k < grid_.nz - 1; k++)
   {
      for (int j = 0; j < grid_.ny - 1; j++)
      {
         for (int i = 1; i < grid_.nx - 1; i++)
         {
            face_dofs_.push_back({i, j, k, 0});
         }
      }
   }
   for (int k = 0; k < grid_.nz - 1; k++)
   {
      for (int j = 1; j < grid_.ny - 1; j++)
      {
         for (int i = 0; i < grid_.nx - 1; i++)
         {
            face_dofs_.push_back({i, j, k, 1});
         }
      }
   }
   for (int k = 1; k < grid_.nz - 1; k++)
   {
      for (int j = 0; j < grid_.ny - 1; j++)
      {
         for (int i = 0; i < grid_.nx - 1; i++)
         {
            face_dofs_.push_back({i, j, k, 2});
         }
      }
   }
}

int YeeOperatorBuilder::XEdgeIndex(int i, int j, int k) const
{
   if (i < 0 || i >= grid_.nx - 1 || j < 1 || j >= grid_.ny - 1 ||
       k < 1 || k >= grid_.nz - 1)
   {
      return -1;
   }
   return (k - 1) * (grid_.ny - 2) * (grid_.nx - 1) + (j - 1) * (grid_.nx - 1) + i;
}

int YeeOperatorBuilder::YEdgeIndex(int i, int j, int k) const
{
   if (i < 1 || i >= grid_.nx - 1 || j < 0 || j >= grid_.ny - 1 ||
       k < 1 || k >= grid_.nz - 1)
   {
      return -1;
   }
   const int offset = (grid_.nz - 2) * (grid_.ny - 2) * (grid_.nx - 1);
   return offset + (k - 1) * (grid_.ny - 1) * (grid_.nx - 2) + j * (grid_.nx - 2) + (i - 1);
}

int YeeOperatorBuilder::ZEdgeIndex(int i, int j, int k) const
{
   if (i < 1 || i >= grid_.nx - 1 || j < 1 || j >= grid_.ny - 1 ||
       k < 0 || k >= grid_.nz - 1)
   {
      return -1;
   }
   const int x_count = (grid_.nz - 2) * (grid_.ny - 2) * (grid_.nx - 1);
   const int y_count = (grid_.nz - 2) * (grid_.ny - 1) * (grid_.nx - 2);
   return x_count + y_count +
          k * (grid_.ny - 2) * (grid_.nx - 2) + (j - 1) * (grid_.nx - 2) + (i - 1);
}

int YeeOperatorBuilder::YZFaceIndex(int i, int j, int k) const
{
   if (i < 1 || i >= grid_.nx - 1 || j < 0 || j >= grid_.ny - 1 ||
       k < 0 || k >= grid_.nz - 1)
   {
      return -1;
   }
   return k * (grid_.ny - 1) * (grid_.nx - 2) + j * (grid_.nx - 2) + (i - 1);
}

int YeeOperatorBuilder::XZFaceIndex(int i, int j, int k) const
{
   if (i < 0 || i >= grid_.nx - 1 || j < 1 || j >= grid_.ny - 1 ||
       k < 0 || k >= grid_.nz - 1)
   {
      return -1;
   }
   const int yz_count = (grid_.nz - 1) * (grid_.ny - 1) * (grid_.nx - 2);
   return yz_count + k * (grid_.ny - 2) * (grid_.nx - 1) + (j - 1) * (grid_.nx - 1) + i;
}

int YeeOperatorBuilder::XYFaceIndex(int i, int j, int k) const
{
   if (i < 0 || i >= grid_.nx - 1 || j < 0 || j >= grid_.ny - 1 ||
       k < 1 || k >= grid_.nz - 1)
   {
      return -1;
   }
   const int yz_count = (grid_.nz - 1) * (grid_.ny - 1) * (grid_.nx - 2);
   const int xz_count = (grid_.nz - 1) * (grid_.ny - 2) * (grid_.nx - 1);
   return yz_count + xz_count +
          (k - 1) * (grid_.ny - 1) * (grid_.nx - 1) + j * (grid_.nx - 1) + i;
}

std::complex<double> YeeOperatorBuilder::StretchFactor(double xi, double k0) const
{
   if (!pml_enabled_ || pml_thickness_ <= 0.0 || k0 <= 0.0)
   {
      return {1.0, 0.0};
   }

   double d = 0.0;
   if (xi < pml_thickness_)
   {
      d = pml_thickness_ - xi;
   }
   else if (xi > 1.0 - pml_thickness_)
   {
      d = xi - (1.0 - pml_thickness_);
   }

   if (d <= 0.0)
   {
      return {1.0, 0.0};
   }

   const double n = pml_order_;
   const double coeff = n * pml_strength_ / k0 / std::pow(pml_thickness_, n);
   const double sigma = coeff * std::pow(d, n - 1.0);
   return {1.0, sigma};
}

double YeeOperatorBuilder::PMLCurlWeight(const mfem::Vector &xi, int axis,
                                         double k0) const
{
   return std::abs(PMLCurlWeightComplex(xi, axis, k0));
}

double YeeOperatorBuilder::PMLMassWeight(const mfem::Vector &xi, int axis,
                                         double k0) const
{
   return std::abs(PMLMassWeightComplex(xi, axis, k0));
}

std::complex<double> YeeOperatorBuilder::PMLCurlWeightComplex(
   const mfem::Vector &xi, int axis, double k0) const
{
   if (!pml_enabled_) { return {1.0, 0.0}; }
   std::complex<double> s[3] =
   {
      StretchFactor(xi[0], k0),
      StretchFactor(xi[1], k0),
      StretchFactor(xi[2], k0)
   };
   const std::complex<double> det = s[0] * s[1] * s[2];
   return (s[axis] * s[axis]) / det;
}

std::complex<double> YeeOperatorBuilder::PMLMassWeightComplex(
   const mfem::Vector &xi, int axis, double k0) const
{
   if (!pml_enabled_) { return {1.0, 0.0}; }
   std::complex<double> s[3] =
   {
      StretchFactor(xi[0], k0),
      StretchFactor(xi[1], k0),
      StretchFactor(xi[2], k0)
   };
   const std::complex<double> det = s[0] * s[1] * s[2];
   return det / (s[axis] * s[axis]);
}

std::complex<double> YeeOperatorBuilder::EdgeSqrtInvStretchProduct(
   const YeeEdgeDof &edge, double k0) const
{
   if (!pml_enabled_) { return {1.0, 0.0}; }

   GridMapper map(grid_);
   double xi[3];
   if (edge.axis == 0)
   {
      xi[0] = map.edge_x(edge.i);
      xi[1] = map.node_y(edge.j);
      xi[2] = map.node_z(edge.k);
   }
   else if (edge.axis == 1)
   {
      xi[0] = map.node_x(edge.i);
      xi[1] = map.edge_y(edge.j);
      xi[2] = map.node_z(edge.k);
   }
   else
   {
      xi[0] = map.node_x(edge.i);
      xi[1] = map.node_y(edge.j);
      xi[2] = map.edge_z(edge.k);
   }

   std::complex<double> dinv(1.0, 0.0);
   for (int d = 0; d < 3; d++)
   {
      dinv *= std::sqrt(1.0 / StretchFactor(xi[d], k0));
   }
   return dinv;
}

void YeeOperatorBuilder::BuildCurlIncidence(mfem::DenseMatrix &C) const
{
   BuildEdgeDofs();
   BuildFaceDofs();

   const int ne = static_cast<int>(edge_dofs_.size());
   const int nf = static_cast<int>(face_dofs_.size());
   C.SetSize(nf, ne);
   C = 0.0;

   for (int row = 0; row < nf; row++)
   {
      const auto &f = face_dofs_[row];
      if (f.axis == 0)
      {
         const int ez0 = ZEdgeIndex(f.i, f.j, f.k);
         const int ez1 = ZEdgeIndex(f.i, f.j + 1, f.k);
         const int ey0 = YEdgeIndex(f.i, f.j, f.k);
         const int ey1 = YEdgeIndex(f.i, f.j, f.k + 1);
         if (ez0 >= 0) { C(row, ez0) += -1.0; }
         if (ez1 >= 0) { C(row, ez1) +=  1.0; }
         if (ey0 >= 0) { C(row, ey0) +=  1.0; }
         if (ey1 >= 0) { C(row, ey1) += -1.0; }
      }
      else if (f.axis == 1)
      {
         const int ex0 = XEdgeIndex(f.i, f.j, f.k);
         const int ex1 = XEdgeIndex(f.i, f.j, f.k + 1);
         const int ez0 = ZEdgeIndex(f.i, f.j, f.k);
         const int ez1 = ZEdgeIndex(f.i + 1, f.j, f.k);
         if (ex0 >= 0) { C(row, ex0) +=  1.0; }
         if (ex1 >= 0) { C(row, ex1) += -1.0; }
         if (ez0 >= 0) { C(row, ez0) +=  1.0; }
         if (ez1 >= 0) { C(row, ez1) += -1.0; }
      }
      else
      {
         const int ey0 = YEdgeIndex(f.i, f.j, f.k);
         const int ey1 = YEdgeIndex(f.i + 1, f.j, f.k);
         const int ex0 = XEdgeIndex(f.i, f.j, f.k);
         const int ex1 = XEdgeIndex(f.i, f.j + 1, f.k);
         if (ey0 >= 0) { C(row, ey0) += -1.0; }
         if (ey1 >= 0) { C(row, ey1) +=  1.0; }
         if (ex0 >= 0) { C(row, ex0) += -1.0; }
         if (ex1 >= 0) { C(row, ex1) +=  1.0; }
      }
   }
}

void YeeOperatorBuilder::BuildCurlIncidenceComplex(double k0,
                                                   mfem::DenseMatrix &Creal,
                                                   mfem::DenseMatrix &Cimag) const
{
   BuildEdgeDofs();
   BuildFaceDofs();

   const int ne = static_cast<int>(edge_dofs_.size());
   const int nf = static_cast<int>(face_dofs_.size());
   Creal.SetSize(nf, ne);
   Cimag.SetSize(nf, ne);
   Creal = 0.0;
   Cimag = 0.0;

   GridMapper map(grid_);
   mfem::Vector xi(3);

   auto add_entry = [&](int row, int col, double sign, int deriv_axis)
   {
      if (col < 0) { return; }
      const std::complex<double> inv_s =
         1.0 / StretchFactor(xi[deriv_axis], k0);
      Creal(row, col) += sign * inv_s.real();
      Cimag(row, col) += sign * inv_s.imag();
   };

   for (int row = 0; row < nf; row++)
   {
      const auto &f = face_dofs_[row];
      if (f.axis == 0)
      {
         xi[0] = map.node_x(f.i);
         xi[1] = map.edge_y(f.j);
         xi[2] = map.edge_z(f.k);
         add_entry(row, ZEdgeIndex(f.i, f.j, f.k), -1.0, 1);
         add_entry(row, ZEdgeIndex(f.i, f.j + 1, f.k), 1.0, 1);
         add_entry(row, YEdgeIndex(f.i, f.j, f.k), 1.0, 2);
         add_entry(row, YEdgeIndex(f.i, f.j, f.k + 1), -1.0, 2);
      }
      else if (f.axis == 1)
      {
         xi[0] = map.edge_x(f.i);
         xi[1] = map.node_y(f.j);
         xi[2] = map.edge_z(f.k);
         add_entry(row, XEdgeIndex(f.i, f.j, f.k), 1.0, 2);
         add_entry(row, XEdgeIndex(f.i, f.j, f.k + 1), -1.0, 2);
         add_entry(row, ZEdgeIndex(f.i, f.j, f.k), 1.0, 0);
         add_entry(row, ZEdgeIndex(f.i + 1, f.j, f.k), -1.0, 0);
      }
      else
      {
         xi[0] = map.edge_x(f.i);
         xi[1] = map.edge_y(f.j);
         xi[2] = map.node_z(f.k);
         add_entry(row, YEdgeIndex(f.i, f.j, f.k), -1.0, 0);
         add_entry(row, YEdgeIndex(f.i + 1, f.j, f.k), 1.0, 0);
         add_entry(row, XEdgeIndex(f.i, f.j, f.k), -1.0, 1);
         add_entry(row, XEdgeIndex(f.i, f.j + 1, f.k), 1.0, 1);
      }
   }
}

void YeeOperatorBuilder::AssembleFaceMassMuInv(mfem::DenseMatrix &MmuInv,
                                               double k0) const
{
   BuildFaceDofs();
   const int nf = static_cast<int>(face_dofs_.size());
   MmuInv.SetSize(nf, nf);
   MmuInv = 0.0;

   GridMapper map(grid_);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J;
   for (int row = 0; row < nf; row++)
   {
      const auto &f = face_dofs_[row];
      if (f.axis == 0)
      {
         xi[0] = map.node_x(f.i);
         xi[1] = map.edge_y(f.j);
         xi[2] = map.edge_z(f.k);
      }
      else if (f.axis == 1)
      {
         xi[0] = map.edge_x(f.i);
         xi[1] = map.node_y(f.j);
         xi[2] = map.edge_z(f.k);
      }
      else
      {
         xi[0] = map.edge_x(f.i);
         xi[1] = map.edge_y(f.j);
         xi[2] = map.node_z(f.k);
      }
      geom_.EvalGeometry(xi, x_phys, J);

      const double h_axis = (f.axis == 0) ? map.cell_hx(f.i)
                          : (f.axis == 1) ? map.cell_hy(f.j)
                                          : map.cell_hz(f.k);
      const double h_orth1 = (f.axis == 0) ? map.cell_hy(f.j)
                           : (f.axis == 1) ? map.cell_hx(f.i)
                                           : map.cell_hx(f.i);
      const double h_orth2 = (f.axis == 0) ? map.cell_hz(f.k)
                           : (f.axis == 1) ? map.cell_hz(f.k)
                                           : map.cell_hy(f.j);
      const double area_ref = h_orth1 * h_orth2;
      const double dual_edge = PhysicalEdgeMetric(J, f.axis) * h_axis;
      const double primal_face = PhysicalFaceMetric(J, f.axis) * area_ref;
      const double pml_weight = PMLCurlWeight(xi, f.axis, k0);
      MmuInv(row, row) = (primal_face > 0.0) ?
                         (pml_weight * dual_edge / primal_face) : 0.0;
   }
}

void YeeOperatorBuilder::AssembleEdgeMassEps(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   mfem::DenseMatrix &Meps,
   double k0) const
{
   BuildEdgeDofs();
   const int ne = static_cast<int>(edge_dofs_.size());
   Meps.SetSize(ne, ne);
   Meps = 0.0;

   GridMapper map(grid_);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J;
   for (int row = 0; row < ne; row++)
   {
      const auto &e = edge_dofs_[row];
      if (e.axis == 0)
      {
         xi[0] = map.edge_x(e.i);
         xi[1] = map.node_y(e.j);
         xi[2] = map.node_z(e.k);
      }
      else if (e.axis == 1)
      {
         xi[0] = map.node_x(e.i);
         xi[1] = map.edge_y(e.j);
         xi[2] = map.node_z(e.k);
      }
      else
      {
         xi[0] = map.node_x(e.i);
         xi[1] = map.node_y(e.j);
         xi[2] = map.edge_z(e.k);
      }
      geom_.EvalGeometry(xi, x_phys, J);

      const double h_axis = (e.axis == 0) ? map.cell_hx(e.i)
                          : (e.axis == 1) ? map.cell_hy(e.j)
                                          : map.cell_hz(e.k);
      const double h_orth1 = (e.axis == 0) ? map.cell_hy(e.j)
                           : (e.axis == 1) ? map.cell_hx(e.i)
                                           : map.cell_hx(e.i);
      const double h_orth2 = (e.axis == 0) ? map.cell_hz(e.k)
                           : (e.axis == 1) ? map.cell_hz(e.k)
                                           : map.cell_hy(e.j);
      const double area_ref = h_orth1 * h_orth2;
      const double primal_edge = PhysicalEdgeMetric(J, e.axis) * h_axis;
      const double dual_face = PhysicalFaceMetric(J, e.axis) * area_ref;
      const double eps = eps_fn(x_phys);
      const double pml_weight = PMLMassWeight(xi, e.axis, k0);
      Meps(row, row) = (primal_edge > 0.0) ?
                       (pml_weight * eps * dual_face / primal_edge) : 0.0;
   }
}

void YeeOperatorBuilder::AssembleYeeMaxwellOperator(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   mfem::DenseMatrix &Ayee) const
{
   mfem::DenseMatrix CtMC, K2Meps;
   AssembleYeeCurlOperator(CtMC, k0);
   AssembleYeeMassOperator(eps_fn, K2Meps, k0);
   Ayee = CtMC;
   Ayee *= curl_scale_;
   K2Meps *= mass_scale_;
   // Form choice (empirical):
   //   No PML  -> indefinite Maxwell form CtMC - k^2 Meps (matches real A_h
   //              in cavity, ratio ~ 1, edge_yee converges 36-66 iters)
   //   PML on  -> ex25p-style PD-like form CtMC + k^2 Meps (matches the abs
   //              prec target that GMRES actually prefers; using the indefinite
   //              form here makes GMRES stall on PML meshes).
   // The two regimes converge to different effective preconditioners; the
   // diagnostics in PrintYeeGalerkinComparison report against both targets.
   const double mass_sign = pml_enabled_ ? 1.0 : -1.0;
   if (mass_sign > 0.0) { Ayee += K2Meps; }
   else                 { Ayee -= K2Meps; }
}

void YeeOperatorBuilder::AssembleYeeCurlOperator(mfem::DenseMatrix &CtMC,
                                                 double k0) const
{
   mfem::DenseMatrix C, MmuInv;
   BuildCurlIncidence(C);
   AssembleFaceMassMuInv(MmuInv, k0);

   CtMC.SetSize(C.Width(), C.Width());
   CtMC = 0.0;
   for (int f = 0; f < C.Height(); f++)
   {
      const double w = MmuInv(f, f);
      if (std::abs(w) == 0.0) { continue; }
      for (int i = 0; i < C.Width(); i++)
      {
         const double cfi = C(f, i);
         if (std::abs(cfi) == 0.0) { continue; }
         for (int j = 0; j < C.Width(); j++)
         {
            const double cfj = C(f, j);
            if (std::abs(cfj) == 0.0) { continue; }
            CtMC(i, j) += w * cfi * cfj;
         }
      }
   }
}

void YeeOperatorBuilder::AssembleYeeMassOperator(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   mfem::DenseMatrix &K2Meps,
   double k0) const
{
   mfem::DenseMatrix Meps;
   AssembleEdgeMassEps(eps_fn, Meps, k0);
   K2Meps = Meps;
   K2Meps *= (k0 * k0);
}

void YeeOperatorBuilder::AssembleYeeMaxwellOperatorComplex(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   mfem::DenseMatrix &Areal,
   mfem::DenseMatrix &Aimag) const
{
   mfem::DenseMatrix Cr, Ci, MmuInv, Meps;
   BuildCurlIncidenceComplex(k0, Cr, Ci);
   AssembleFaceMassMuInv(MmuInv, 0.0);
   AssembleEdgeMassEps(eps_fn, Meps, 0.0);

   const int ne = Cr.Width();
   Areal.SetSize(ne, ne);
   Aimag.SetSize(ne, ne);
   Areal = 0.0;
   Aimag = 0.0;

   for (int f = 0; f < Cr.Height(); f++)
   {
      const double w = MmuInv(f, f);
      if (std::abs(w) == 0.0) { continue; }
      for (int i = 0; i < ne; i++)
      {
         const std::complex<double> cfi(Cr(f, i), Ci(f, i));
         if (std::abs(cfi) == 0.0) { continue; }
         for (int j = 0; j < ne; j++)
         {
            const std::complex<double> cfj(Cr(f, j), Ci(f, j));
            if (std::abs(cfj) == 0.0) { continue; }
            const std::complex<double> val = w * cfi * cfj;
            Areal(i, j) += val.real();
            Aimag(i, j) += val.imag();
         }
      }
   }

   for (int i = 0; i < ne; i++)
   {
      Areal(i, i) -= k0 * k0 * Meps(i, i);
   }
}

void YeeOperatorBuilder::AssembleYeeMaxwellOperatorComplexSqrtScaled(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   mfem::DenseMatrix &Areal,
   mfem::DenseMatrix &Aimag) const
{
   AssembleYeeMaxwellOperatorComplex(eps_fn, k0, Areal, Aimag);
   if (!pml_enabled_) { return; }

   BuildEdgeDofs();
   const int ne = static_cast<int>(edge_dofs_.size());
   std::vector<std::complex<double>> dinv(ne);
   for (int i = 0; i < ne; i++)
   {
      dinv[i] = EdgeSqrtInvStretchProduct(edge_dofs_[i], k0);
   }

   for (int i = 0; i < ne; i++)
   {
      for (int j = 0; j < ne; j++)
      {
         const std::complex<double> a(Areal(i, j), Aimag(i, j));
         const std::complex<double> v = dinv[i] * a * dinv[j];
         Areal(i, j) = v.real();
         Aimag(i, j) = v.imag();
      }
   }
}

void YeeOperatorBuilder::PrintDiagnostics(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   std::ostream &os) const
{
   mfem::DenseMatrix C, MmuInv, Meps, Ayee;
   mfem::DenseMatrix CtMC;
   BuildCurlIncidence(C);
   AssembleFaceMassMuInv(MmuInv, k0);
   AssembleEdgeMassEps(eps_fn, Meps, k0);
   CtMC.SetSize(C.Width(), C.Width());
   CtMC = 0.0;
   for (int f = 0; f < C.Height(); f++)
   {
      const double w = MmuInv(f, f);
      if (std::abs(w) == 0.0) { continue; }
      for (int i = 0; i < C.Width(); i++)
      {
         const double cfi = C(f, i);
         if (std::abs(cfi) == 0.0) { continue; }
         for (int j = 0; j < C.Width(); j++)
         {
            const double cfj = C(f, j);
            if (std::abs(cfj) == 0.0) { continue; }
            CtMC(i, j) += w * cfi * cfj;
         }
      }
   }
   AssembleYeeMaxwellOperator(eps_fn, k0, Ayee);

   double c_frob2 = 0.0;
   int c_nnz = 0;
   for (int i = 0; i < C.Height(); i++)
   {
      for (int j = 0; j < C.Width(); j++)
      {
         const double v = C(i, j);
         c_frob2 += v * v;
         if (std::abs(v) > 0.0) { c_nnz++; }
      }
   }

   auto diag_stats = [](const mfem::DenseMatrix &M,
                        double &mean, double &minv, double &maxv)
   {
      mean = 0.0;
      minv = std::numeric_limits<double>::infinity();
      maxv = 0.0;
      for (int i = 0; i < M.Height(); i++)
      {
         const double a = std::abs(M(i, i));
         mean += a;
         minv = std::min(minv, a);
         maxv = std::max(maxv, a);
      }
      if (M.Height() > 0) { mean /= double(M.Height()); }
   };

   double mumu_mean, mumu_min, mumu_max;
   double meps_mean, meps_min, meps_max;
    double ctmc_mean, ctmc_min, ctmc_max;
   double ayee_mean, ayee_min, ayee_max;
   diag_stats(MmuInv, mumu_mean, mumu_min, mumu_max);
   diag_stats(Meps, meps_mean, meps_min, meps_max);
   diag_stats(CtMC, ctmc_mean, ctmc_min, ctmc_max);
   diag_stats(Ayee, ayee_mean, ayee_min, ayee_max);

   os << "[covariant_aux_space] yee operator internals\n"
      << "  edge_dofs=" << C.Width() << '\n'
      << "  face_dofs=" << C.Height() << '\n'
      << "  k0=" << k0 << '\n'
      << "  curl_incidence_nnz=" << c_nnz << '\n'
      << "  ||C_Y||_F=" << std::sqrt(c_frob2) << '\n'
      << "  MmuInv_diag_abs_mean=" << mumu_mean
      << " range=[" << mumu_min << ", " << mumu_max << "]\n"
      << "  Meps_diag_abs_mean=" << meps_mean
      << " range=[" << meps_min << ", " << meps_max << "]\n"
      << "  CtMC_diag_abs_mean=" << ctmc_mean
      << " range=[" << ctmc_min << ", " << ctmc_max << "]\n"
      << "  k0^2*Meps_diag_abs_mean=" << (k0 * k0 * meps_mean)
      << " range=[" << (k0 * k0 * meps_min) << ", "
      << (k0 * k0 * meps_max) << "]\n"
      << "  Ayee_diag_abs_mean=" << ayee_mean
      << " range=[" << ayee_min << ", " << ayee_max << "]"
      << std::endl;
}

} // namespace covariant_aux_space
