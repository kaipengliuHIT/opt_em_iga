#include "yee_operator.hpp"
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
   mfem::DenseMatrix invJ = J;
   invJ.Invert();
   const double nx = invJ(axis, 0);
   const double ny = invJ(axis, 1);
   const double nz = invJ(axis, 2);
   return J.Det() * std::sqrt(nx * nx + ny * ny + nz * nz);
}

} // namespace

YeeOperatorBuilder::YeeOperatorBuilder(
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom)
   : geom_(geom)
{
}

void YeeOperatorBuilder::SetGrid(const fdfd_iga_init::ReferenceGrid &grid)
{
   grid_ = grid;
   edge_dofs_.clear();
   face_dofs_.clear();
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

void YeeOperatorBuilder::AssembleFaceMassMuInv(mfem::DenseMatrix &MmuInv) const
{
   BuildFaceDofs();
   const int nf = static_cast<int>(face_dofs_.size());
   MmuInv.SetSize(nf, nf);
   MmuInv = 0.0;

   const double hx = 1.0 / double(grid_.nx - 1);
   const double hy = 1.0 / double(grid_.ny - 1);
   const double hz = 1.0 / double(grid_.nz - 1);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J;
   for (int row = 0; row < nf; row++)
   {
      const auto &f = face_dofs_[row];
      if (f.axis == 0)
      {
         xi[0] = f.i * hx;
         xi[1] = (f.j + 0.5) * hy;
         xi[2] = (f.k + 0.5) * hz;
      }
      else if (f.axis == 1)
      {
         xi[0] = (f.i + 0.5) * hx;
         xi[1] = f.j * hy;
         xi[2] = (f.k + 0.5) * hz;
      }
      else
      {
         xi[0] = (f.i + 0.5) * hx;
         xi[1] = (f.j + 0.5) * hy;
         xi[2] = f.k * hz;
      }
      geom_.EvalGeometry(xi, x_phys, J);

      const double h_axis = (f.axis == 0) ? hx : ((f.axis == 1) ? hy : hz);
      const double area_ref = (f.axis == 0) ? (hy * hz)
                            : (f.axis == 1) ? (hx * hz)
                                            : (hx * hy);
      const double dual_edge = PhysicalEdgeMetric(J, f.axis) * h_axis;
      const double primal_face = PhysicalFaceMetric(J, f.axis) * area_ref;
      MmuInv(row, row) = (primal_face > 0.0) ? (dual_edge / primal_face) : 0.0;
   }
}

void YeeOperatorBuilder::AssembleEdgeMassEps(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   mfem::DenseMatrix &Meps) const
{
   BuildEdgeDofs();
   const int ne = static_cast<int>(edge_dofs_.size());
   Meps.SetSize(ne, ne);
   Meps = 0.0;

   const double hx = 1.0 / double(grid_.nx - 1);
   const double hy = 1.0 / double(grid_.ny - 1);
   const double hz = 1.0 / double(grid_.nz - 1);
   mfem::Vector xi(3), x_phys;
   mfem::DenseMatrix J;
   for (int row = 0; row < ne; row++)
   {
      const auto &e = edge_dofs_[row];
      if (e.axis == 0)
      {
         xi[0] = (e.i + 0.5) * hx;
         xi[1] = e.j * hy;
         xi[2] = e.k * hz;
      }
      else if (e.axis == 1)
      {
         xi[0] = e.i * hx;
         xi[1] = (e.j + 0.5) * hy;
         xi[2] = e.k * hz;
      }
      else
      {
         xi[0] = e.i * hx;
         xi[1] = e.j * hy;
         xi[2] = (e.k + 0.5) * hz;
      }
      geom_.EvalGeometry(xi, x_phys, J);

      const double h_axis = (e.axis == 0) ? hx : ((e.axis == 1) ? hy : hz);
      const double area_ref = (e.axis == 0) ? (hy * hz)
                            : (e.axis == 1) ? (hx * hz)
                                            : (hx * hy);
      const double primal_edge = PhysicalEdgeMetric(J, e.axis) * h_axis;
      const double dual_face = PhysicalFaceMetric(J, e.axis) * area_ref;
      const double eps = eps_fn(x_phys);
      Meps(row, row) = (primal_edge > 0.0) ? (eps * dual_face / primal_edge) : 0.0;
   }
}

void YeeOperatorBuilder::AssembleYeeMaxwellOperator(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   mfem::DenseMatrix &Ayee) const
{
   mfem::DenseMatrix C, MmuInv, Meps;
   BuildCurlIncidence(C);
   AssembleFaceMassMuInv(MmuInv);
   AssembleEdgeMassEps(eps_fn, Meps);

   mfem::DenseMatrix CtMC(C.Width(), C.Width());
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

   Ayee.SetSize(C.Width(), C.Width());
   Ayee = CtMC;
   for (int i = 0; i < Ayee.Height(); i++)
   {
      for (int j = 0; j < Ayee.Width(); j++)
      {
         Ayee(i, j) -= k0 * k0 * Meps(i, j);
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
   AssembleFaceMassMuInv(MmuInv);
   AssembleEdgeMassEps(eps_fn, Meps);
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
