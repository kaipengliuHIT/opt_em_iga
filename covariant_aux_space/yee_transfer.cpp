#include "yee_transfer.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace covariant_aux_space
{

namespace
{

struct LocalEdgeRef
{
   int axis;
   int ay;
   int az;
};

const LocalEdgeRef kLocalEdges[12] =
{
   {0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 1},
   {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1},
   {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}
};

double Lagrange01(int bit, double x)
{
   return bit ? x : (1.0 - x);
}

double Lagrange01Grad(int bit)
{
   return bit ? 1.0 : -1.0;
}

void EvalCellEdgeBasis(int local_edge, double u, double v, double w,
                       double hx, double hy, double hz,
                       mfem::Vector &E, mfem::Vector &curlE)
{
   E.SetSize(3);
   curlE.SetSize(3);
   E = 0.0;
   curlE = 0.0;

   const LocalEdgeRef le = kLocalEdges[local_edge];

   if (le.axis == 0)
   {
      const double lv = Lagrange01(le.ay, v);
      const double lw = Lagrange01(le.az, w);
      const double dlv = Lagrange01Grad(le.ay);
      const double dlw = Lagrange01Grad(le.az);
      E[0] = lv * lw / hx;
      curlE[1] = lv * dlw / (hx * hz);
      curlE[2] = -dlv * lw / (hx * hy);
      return;
   }

   if (le.axis == 1)
   {
      const double lu = Lagrange01(le.ay, u);
      const double lw = Lagrange01(le.az, w);
      const double dlu = Lagrange01Grad(le.ay);
      const double dlw = Lagrange01Grad(le.az);
      E[1] = lu * lw / hy;
      curlE[0] = -lu * dlw / (hy * hz);
      curlE[2] = dlu * lw / (hx * hy);
      return;
   }

   const double lu = Lagrange01(le.ay, u);
   const double lv = Lagrange01(le.az, v);
   const double dlu = Lagrange01Grad(le.ay);
   const double dlv = Lagrange01Grad(le.az);
   E[2] = lu * lv / hz;
   curlE[0] = lu * dlv / (hy * hz);
   curlE[1] = -dlu * lv / (hx * hz);
}

} // namespace

YeeTransferBuilder::EdgeBasisCoefficient::EdgeBasisCoefficient(
   const YeeTransferBuilder &builder, const YeeEdgeDof &edge)
   : mfem::VectorCoefficient(builder.fespace_.GetParMesh()->SpaceDimension()),
     builder_(builder),
     edge_(edge)
{
}

void YeeTransferBuilder::EdgeBasisCoefficient::Eval(
   mfem::Vector &V, mfem::ElementTransformation &T,
   const mfem::IntegrationPoint &ip)
{
   mfem::Vector xi, Ehat;
   builder_.GetGlobalPatchXi(T.ElementNo, ip, xi);
   builder_.EvaluateReferenceBasis(edge_, xi, Ehat);

   const mfem::DenseMatrix &J = T.Jacobian();
   mfem::DenseMatrix invJ(J);
   // Safe Invert: skip degenerate Jacobians
   const double detJ = invJ.Det();
   if (std::abs(detJ) < 1e-30) { V.SetSize(T.GetSpaceDim()); V = 0.0; return; }
   invJ.Invert();

   V.SetSize(T.GetSpaceDim());
   V = 0.0;
   for (int c = 0; c < T.GetSpaceDim(); c++)
   {
      for (int r = 0; r < Ehat.Size(); r++)
      {
         V[c] += invJ(r, c) * Ehat[r];
      }
   }
}

YeeTransferBuilder::YeeTransferBuilder(
   const mfem::ParFiniteElementSpace &fespace,
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom)
   : fespace_(fespace),
     geom_(geom),
     ext_(*const_cast<mfem::NURBSExtension *>(fespace.GetNURBSext()))
{
   ext_.GetPatchKnotVectors(0, patch_kv_);
}

void YeeTransferBuilder::SetGrid(const fdfd_iga_init::ReferenceGrid &grid)
{
   grid_ = grid;
   edge_dofs_.clear();
}

const std::vector<YeeEdgeDof> &YeeTransferBuilder::GetEdgeDofs() const
{
   BuildEdgeDofs();
   return edge_dofs_;
}

void YeeTransferBuilder::BuildEdgeDofs() const
{
   if (!edge_dofs_.empty())
   {
      return;
   }

   MFEM_VERIFY(grid_.nx > 2 && grid_.ny > 2 && grid_.nz > 2,
               "Yee grid must contain interior lines.");

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

void YeeTransferBuilder::GetGlobalPatchXi(
   int elem, const mfem::IntegrationPoint &ip, mfem::Vector &xi) const
{
   mfem::Array<int> ijk(ext_.Dimension());
   ext_.GetElementIJK(elem, ijk);

   xi.SetSize(ext_.Dimension());
   for (int d = 0; d < ext_.Dimension(); d++)
   {
      const int order = patch_kv_[d]->GetOrder();
      const int knot_idx = ijk[d] + order;
      const double xloc = (d == 0) ? ip.x : ((d == 1) ? ip.y : ip.z);
      xi[d] = patch_kv_[d]->getKnotLocation(xloc, knot_idx);
   }
}

void YeeTransferBuilder::EvaluateReferenceBasis(
   const YeeEdgeDof &edge, const mfem::Vector &xi, mfem::Vector &Ehat) const
{
   // Use knot positions for cell edges when available (non-uniform grid)
   const bool has_knots = !grid_.knot_x.empty();
   const auto& kx = has_knots ? grid_.knot_x : std::vector<double>{};
   const auto& ky = has_knots ? grid_.knot_y : std::vector<double>{};
   const auto& kz = has_knots ? grid_.knot_z : std::vector<double>{};

   const double hx0 = 1.0 / double(grid_.nx - 1);
   const double hy0 = 1.0 / double(grid_.ny - 1);
   const double hz0 = 1.0 / double(grid_.nz - 1);

   Ehat.SetSize(3);
   Ehat = 0.0;
   mfem::Vector Eloc, curl_dummy;

   auto accumulate = [&](int ci, int cj, int ck, int local_edge)
   {
      if (ci < 0 || cj < 0 || ck < 0 ||
          ci >= grid_.nx - 1 || cj >= grid_.ny - 1 || ck >= grid_.nz - 1)
      {
         return;
      }
      double cell_hx, cell_hy, cell_hz;
      double u, v, w;
      if (has_knots)
      {
         cell_hx = kx[ci+1] - kx[ci];
         cell_hy = ky[cj+1] - ky[cj];
         cell_hz = kz[ck+1] - kz[ck];
         u = (xi[0] - kx[ci]) / cell_hx;
         v = (xi[1] - ky[cj]) / cell_hy;
         w = (xi[2] - kz[ck]) / cell_hz;
      }
      else
      {
         cell_hx = hx0; cell_hy = hy0; cell_hz = hz0;
         u = (xi[0] - ci * hx0) / hx0;
         v = (xi[1] - cj * hy0) / hy0;
         w = (xi[2] - ck * hz0) / hz0;
      }
      if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || w < 0.0 || w > 1.0)
      {
         return;
      }
      EvalCellEdgeBasis(local_edge, u, v, w, cell_hx, cell_hy, cell_hz,
                        Eloc, curl_dummy);
      Ehat += Eloc;
   };

   if (edge.axis == 0)
   {
      for (int dj = 0; dj <= 1; dj++)
      {
         for (int dk = 0; dk <= 1; dk++)
         {
            const int ci = edge.i;
            const int cj = edge.j - dj;
            const int ck = edge.k - dk;
            const int local_edge = dj + 2 * dk;
            accumulate(ci, cj, ck, local_edge);
         }
      }
      return;
   }

   if (edge.axis == 1)
   {
      for (int di = 0; di <= 1; di++)
      {
         for (int dk = 0; dk <= 1; dk++)
         {
            const int ci = edge.i - di;
            const int cj = edge.j;
            const int ck = edge.k - dk;
            const int local_edge = 4 + di + 2 * dk;
            accumulate(ci, cj, ck, local_edge);
         }
      }
      return;
   }

   for (int di = 0; di <= 1; di++)
   {
      for (int dj = 0; dj <= 1; dj++)
      {
         const int ci = edge.i - di;
         const int cj = edge.j - dj;
         const int ck = edge.k;
         const int local_edge = 8 + di + 2 * dj;
         accumulate(ci, cj, ck, local_edge);
      }
   }
}

void YeeTransferBuilder::EvaluateReferenceCurl(
   const YeeEdgeDof &edge, const mfem::Vector &xi, mfem::Vector &curlEhat) const
{
   // Use knot positions for cell edges when available (non-uniform grid)
   const bool has_knots = !grid_.knot_x.empty();
   const auto& kx = has_knots ? grid_.knot_x : std::vector<double>{};
   const auto& ky = has_knots ? grid_.knot_y : std::vector<double>{};
   const auto& kz = has_knots ? grid_.knot_z : std::vector<double>{};

   const double hx0 = 1.0 / double(grid_.nx - 1);
   const double hy0 = 1.0 / double(grid_.ny - 1);
   const double hz0 = 1.0 / double(grid_.nz - 1);

   curlEhat.SetSize(3);
   curlEhat = 0.0;
   mfem::Vector Eloc, curlLoc;

   auto accumulate = [&](int ci, int cj, int ck, int local_edge)
   {
      if (ci < 0 || cj < 0 || ck < 0 ||
          ci >= grid_.nx - 1 || cj >= grid_.ny - 1 || ck >= grid_.nz - 1)
      {
         return;
      }
      double cell_hx, cell_hy, cell_hz;
      double u, v, w;
      if (has_knots)
      {
         cell_hx = kx[ci+1] - kx[ci];
         cell_hy = ky[cj+1] - ky[cj];
         cell_hz = kz[ck+1] - kz[ck];
         u = (xi[0] - kx[ci]) / cell_hx;
         v = (xi[1] - ky[cj]) / cell_hy;
         w = (xi[2] - kz[ck]) / cell_hz;
      }
      else
      {
         cell_hx = hx0; cell_hy = hy0; cell_hz = hz0;
         u = (xi[0] - ci * hx0) / hx0;
         v = (xi[1] - cj * hy0) / hy0;
         w = (xi[2] - ck * hz0) / hz0;
      }
      if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || w < 0.0 || w > 1.0)
      {
         return;
      }
      EvalCellEdgeBasis(local_edge, u, v, w, cell_hx, cell_hy, cell_hz,
                        Eloc, curlLoc);
      curlEhat += curlLoc;
   };

   if (edge.axis == 0)
   {
      for (int dj = 0; dj <= 1; dj++)
      {
         for (int dk = 0; dk <= 1; dk++)
         {
            const int ci = edge.i;
            const int cj = edge.j - dj;
            const int ck = edge.k - dk;
            const int local_edge = dj + 2 * dk;
            accumulate(ci, cj, ck, local_edge);
         }
      }
      return;
   }

   if (edge.axis == 1)
   {
      for (int di = 0; di <= 1; di++)
      {
         for (int dk = 0; dk <= 1; dk++)
         {
            const int ci = edge.i - di;
            const int cj = edge.j;
            const int ck = edge.k - dk;
            const int local_edge = 4 + di + 2 * dk;
            accumulate(ci, cj, ck, local_edge);
         }
      }
      return;
   }

   for (int di = 0; di <= 1; di++)
   {
      for (int dj = 0; dj <= 1; dj++)
      {
         const int ci = edge.i - di;
         const int cj = edge.j - dj;
         const int ck = edge.k;
         const int local_edge = 8 + di + 2 * dj;
         accumulate(ci, cj, ck, local_edge);
      }
   }
}

void YeeTransferBuilder::BuildProlongation(mfem::DenseMatrix &P) const
{
   BuildProlongationFast(P);
}

void YeeTransferBuilder::BuildProlongationFast(mfem::DenseMatrix &P) const
{
   BuildEdgeDofs();
   const int tvsize = fespace_.GetTrueVSize();
   const int ndofs = fespace_.GetNDofs();
   const int na = static_cast<int>(edge_dofs_.size());
   P.SetSize(tvsize, na);
   P = 0.0;

   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[yee_transfer] BuildProlongationFast start: true_vsize="
                << tvsize << ", edge_dofs=" << na << std::endl;
   }
   const auto t0 = std::chrono::steady_clock::now();

   mfem::ParFiniteElementSpace *pfes =
      const_cast<mfem::ParFiniteElementSpace *>(&fespace_);

   mfem::ConstantCoefficient one(1.0);
   mfem::ParBilinearForm mass(pfes);
   mass.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mass.Assemble();
   mass.Finalize();
   mfem::SparseMatrix &Ms = mass.SpMat();

   // Use row-sum lumped (diagonal) mass for robust L2 projection.
   // Full mass matrix inversion can be corrupted by NaN entries from
   // NURBS element transformations on non-uniform meshes.
   mfem::Vector mass_diag(ndofs);
   mass_diag = 0.0;
   const int *I = Ms.GetI();
   const int *J = Ms.GetJ();
   const double *Data = Ms.GetData();
   for (int r = 0; r < Ms.Height(); r++)
   {
      double row_sum = 0.0;
      for (int p = I[r]; p < I[r + 1]; p++)
      {
         row_sum += Data[p];
      }
      mass_diag[r] = row_sum;
   }
   // Lumped mass diagonal stats
   {
      double diag_min = 1e100, diag_max = 0.0;
      int zero_diag = 0;
      for (int r = 0; r < ndofs; r++) {
         double d = std::abs(mass_diag[r]);
         if (d < 1e-30) zero_diag++;
         else { diag_min = std::min(diag_min, d); diag_max = std::max(diag_max, d); }
      }
      if (mfem::Mpi::WorldRank() == 0) {
         std::cout << "[yee_transfer] Lumped mass diag: ndofs=" << ndofs
                   << " zero_diag=" << zero_diag
                   << " range=[" << diag_min << ", " << diag_max << "]" << std::endl;
      }
   }

   mfem::DenseMatrix B(ndofs, na);
   B = 0.0;

   // Non-uniform knot-aware h values (like EvaluateReferenceBasis)
   const bool has_knots = !grid_.knot_x.empty();
   const auto& kx = has_knots ? grid_.knot_x : std::vector<double>{};
   const auto& ky = has_knots ? grid_.knot_y : std::vector<double>{};
   const auto& kz = has_knots ? grid_.knot_z : std::vector<double>{};
   const double hx0 = 1.0 / double(grid_.nx - 1);  // uniform fallback
   const double hy0 = 1.0 / double(grid_.ny - 1);
   const double hz0 = 1.0 / double(grid_.nz - 1);
   const int nx = grid_.nx;
   const int ny = grid_.ny;
   const int nz = grid_.nz;

   std::vector<int> x_edge_map((nx - 1) * ny * nz, -1);
   std::vector<int> y_edge_map(nx * (ny - 1) * nz, -1);
   std::vector<int> z_edge_map(nx * ny * (nz - 1), -1);

   auto x_id = [&](int i, int j, int k)
   {
      return (k * ny + j) * (nx - 1) + i;
   };
   auto y_id = [&](int i, int j, int k)
   {
      return (k * (ny - 1) + j) * nx + i;
   };
   auto z_id = [&](int i, int j, int k)
   {
      return (k * ny + j) * nx + i;
   };

   for (int idx = 0; idx < na; idx++)
   {
      const auto &e = edge_dofs_[idx];
      if (e.axis == 0) { x_edge_map[x_id(e.i, e.j, e.k)] = idx; }
      else if (e.axis == 1) { y_edge_map[y_id(e.i, e.j, e.k)] = idx; }
      else { z_edge_map[z_id(e.i, e.j, e.k)] = idx; }
   }

   mfem::Array<int> vdofs;
   mfem::DenseMatrix vshape;
   mfem::Vector xi, Ehat, Ephys;
   mfem::DenseMatrix invJ;
   mfem::Vector curl_dummy;
   int local_to_global[12];

   for (int e = 0; e < fespace_.GetNE(); e++)
   {
      mfem::ElementTransformation *T = pfes->GetElementTransformation(e);
      const mfem::FiniteElement *fe = pfes->GetFE(e);
      pfes->GetElementVDofs(e, vdofs);
      const int dof = fe->GetDof();
      const int dim = T->GetSpaceDim();
      const mfem::IntegrationRule *ir =
         &mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder());
      vshape.SetSize(dof, dim);

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcVShape(*T, vshape);
         GetGlobalPatchXi(e, ip, xi);

         // Cell lookup using knot positions for non-uniform grids
         double cell_hx, cell_hy, cell_hz, u, v, w;
         int ci, cj, ck;
         if (has_knots)
         {
            // Binary-search style: find cell index where knot[ci] <= xi < knot[ci+1]
            ci = 0;
            while (ci < nx - 2 && kx[ci + 1] <= xi[0] + 1e-14) { ci++; }
            cj = 0;
            while (cj < ny - 2 && ky[cj + 1] <= xi[1] + 1e-14) { cj++; }
            ck = 0;
            while (ck < nz - 2 && kz[ck + 1] <= xi[2] + 1e-14) { ck++; }
            cell_hx = kx[ci + 1] - kx[ci];
            cell_hy = ky[cj + 1] - ky[cj];
            cell_hz = kz[ck + 1] - kz[ck];
            u = (xi[0] - kx[ci]) / cell_hx;
            v = (xi[1] - ky[cj]) / cell_hy;
            w = (xi[2] - kz[ck]) / cell_hz;
         }
         else
         {
            ci = std::min(std::max(int(std::floor(xi[0] / hx0)), 0), nx - 2);
            cj = std::min(std::max(int(std::floor(xi[1] / hy0)), 0), ny - 2);
            ck = std::min(std::max(int(std::floor(xi[2] / hz0)), 0), nz - 2);
            cell_hx = hx0;
            cell_hy = hy0;
            cell_hz = hz0;
            u = (xi[0] - ci * hx0) / hx0;
            v = (xi[1] - cj * hy0) / hy0;
            w = (xi[2] - ck * hz0) / hz0;
         }

         // Clamp local coordinates to [0,1] to handle floating-point drift
         // at knot boundaries (essential for non-uniform grids)
         u = std::max(0.0, std::min(1.0, u));
         v = std::max(0.0, std::min(1.0, v));
         w = std::max(0.0, std::min(1.0, w));

         local_to_global[0]  = x_edge_map[x_id(ci,     cj,     ck    )];
         local_to_global[1]  = x_edge_map[x_id(ci,     cj + 1, ck    )];
         local_to_global[2]  = x_edge_map[x_id(ci,     cj,     ck + 1)];
         local_to_global[3]  = x_edge_map[x_id(ci,     cj + 1, ck + 1)];
         local_to_global[4]  = y_edge_map[y_id(ci,     cj,     ck    )];
         local_to_global[5]  = y_edge_map[y_id(ci + 1, cj,     ck    )];
         local_to_global[6]  = y_edge_map[y_id(ci,     cj,     ck + 1)];
         local_to_global[7]  = y_edge_map[y_id(ci + 1, cj,     ck + 1)];
         local_to_global[8]  = z_edge_map[z_id(ci,     cj,     ck    )];
         local_to_global[9]  = z_edge_map[z_id(ci + 1, cj,     ck    )];
         local_to_global[10] = z_edge_map[z_id(ci,     cj + 1, ck    )];
         local_to_global[11] = z_edge_map[z_id(ci + 1, cj + 1, ck    )];

         const double weight = ip.weight * T->Weight();
         invJ = T->Jacobian();
         // Safe Invert: skip degenerate Jacobians (possible with non-uniform knots)
         const double detJ = invJ.Det();
         if (std::abs(detJ) < 1e-30) { continue; }
         invJ.Invert();

         for (int a = 0; a < 12; a++)
         {
            const int col = local_to_global[a];
            if (col < 0) { continue; }
            EvalCellEdgeBasis(a, u, v, w, cell_hx, cell_hy, cell_hz, Ehat, curl_dummy);
            Ephys.SetSize(dim);
            Ephys = 0.0;
            for (int c = 0; c < dim; c++)
            {
               for (int r = 0; r < dim; r++)
               {
                  Ephys[c] += invJ(r, c) * Ehat[r];
               }
            }

            for (int i = 0; i < dof; i++)
            {
               int row = vdofs[i];
               double sgn = 1.0;
               if (row < 0)
               {
                  row = -row - 1;
                  sgn = -1.0;
               }
               double dot = 0.0;
               for (int d = 0; d < dim; d++)
               {
                  dot += vshape(i, d) * Ephys[d];
               }
               if (row >= 0 && row < ndofs)
               {
                  B(row, col) += sgn * weight * dot;
               }
            }
         }
      }
   }

   // NaN diagnostics: check B matrix
   {
      double b_max = 0.0, b_min = 1e100;
      int b_nan = 0, b_inf = 0, b_nnz = 0;
      for (int r = 0; r < ndofs; r++) {
         for (int c = 0; c < na; c++) {
            double v = B(r, c);
            if (std::isnan(v)) b_nan++;
            else if (std::isinf(v)) b_inf++;
            else if (std::abs(v) > 0) {
               b_nnz++;
               b_max = std::max(b_max, std::abs(v));
               b_min = std::min(b_min, std::abs(v));
            }
         }
      }
      if (mfem::Mpi::WorldRank() == 0) {
         std::cout << "[yee_transfer] B matrix: nnz=" << b_nnz
                   << " nan=" << b_nan << " inf=" << b_inf
                   << " range=[" << b_min << ", " << b_max << "]" << std::endl;
      }
   }

   mfem::Vector rhs(ndofs), sol(ndofs), col_true(tvsize);
   mfem::ParGridFunction gf(pfes);
   for (int j = 0; j < na; j++)
   {
      for (int i = 0; i < ndofs; i++) { rhs[i] = B(i, j); }
      for (int i = 0; i < ndofs; i++)
      {
         sol[i] = (std::abs(mass_diag[i]) > 1e-30) ? rhs[i] / mass_diag[i] : 0.0;
      }
      for (int i = 0; i < ndofs; i++) { gf[i] = sol[i]; }
      gf.GetTrueDofs(col_true);
      for (int i = 0; i < tvsize; i++) { P(i, j) = col_true[i]; }
   }

   if (mfem::Mpi::WorldRank() == 0)
   {
      const auto t1 = std::chrono::steady_clock::now();

      // NaN diagnostics for P
      int p_nan = 0, p_zero_cols = 0;
      for (int c = 0; c < na; c++) {
         double col_norm2 = 0.0;
         for (int r = 0; r < tvsize; r++) {
            if (std::isnan(P(r,c))) { p_nan++; col_norm2 = NAN; break; }
            col_norm2 += P(r,c) * P(r,c);
         }
         if (col_norm2 == 0.0) p_zero_cols++;
      }
      std::cout << "[yee_transfer] P nan_entries=" << p_nan
                << " zero_cols=" << p_zero_cols << std::endl;

      double p0norm = 0.0, p1norm = 0.0;
      if (na > 0)
      {
         for (int i = 0; i < tvsize; i++) { p0norm += P(i,0) * P(i,0); }
      }
      if (na > 1)
      {
         for (int i = 0; i < tvsize; i++) { p1norm += P(i,1) * P(i,1); }
      }
      std::cout << "[yee_transfer] P col0 norm=" << std::sqrt(p0norm)
                << " P col1 norm=" << std::sqrt(p1norm) << std::endl;
      std::cout << "[yee_transfer] BuildProlongationFast done seconds="
                << std::chrono::duration<double>(t1 - t0).count()
                << std::endl;
   }
}

void YeeTransferBuilder::AssembleYeeCoarseOperator(
   const std::function<double(const mfem::Vector &)> &eps_fn,
   double k0,
   mfem::DenseMatrix &A) const
{
   BuildEdgeDofs();
   const int na = static_cast<int>(edge_dofs_.size());
   A.SetSize(na, na);
   A = 0.0;

   // Non-uniform knot-aware h values
   const bool has_knots_asm = !grid_.knot_x.empty();
   const auto& kx_asm = has_knots_asm ? grid_.knot_x : std::vector<double>{};
   const auto& ky_asm = has_knots_asm ? grid_.knot_y : std::vector<double>{};
   const auto& kz_asm = has_knots_asm ? grid_.knot_z : std::vector<double>{};
   const double hx0 = 1.0 / double(grid_.nx - 1);  // uniform fallback
   const double hy0 = 1.0 / double(grid_.ny - 1);
   const double hz0 = 1.0 / double(grid_.nz - 1);
   const double qx[2] = {0.2113248654051871, 0.7886751345948129};
   const double qw[2] = {0.5, 0.5};

   mfem::Vector xi(3), x_phys, Ei, Ej, curlEi, curlEj;
   mfem::DenseMatrix J, JTJ, invJ, invJJT;

   const int nx = grid_.nx;
   const int ny = grid_.ny;
   const int nz = grid_.nz;
   std::vector<int> x_edge_map((nx - 1) * ny * nz, -1);
   std::vector<int> y_edge_map(nx * (ny - 1) * nz, -1);
   std::vector<int> z_edge_map(nx * ny * (nz - 1), -1);

   auto x_id = [&](int i, int j, int k)
   {
      return (k * ny + j) * (nx - 1) + i;
   };
   auto y_id = [&](int i, int j, int k)
   {
      return (k * (ny - 1) + j) * nx + i;
   };
   auto z_id = [&](int i, int j, int k)
   {
      return (k * ny + j) * nx + i;
   };

   for (int idx = 0; idx < na; idx++)
   {
      const auto &e = edge_dofs_[idx];
      if (e.axis == 0) { x_edge_map[x_id(e.i, e.j, e.k)] = idx; }
      else if (e.axis == 1) { y_edge_map[y_id(e.i, e.j, e.k)] = idx; }
      else { z_edge_map[z_id(e.i, e.j, e.k)] = idx; }
   }

   int local_to_global[12];
   for (int ck = 0; ck < grid_.nz - 1; ck++)
   {
      for (int cj = 0; cj < grid_.ny - 1; cj++)
      {
         for (int ci = 0; ci < grid_.nx - 1; ci++)
         {
            local_to_global[0]  = x_edge_map[x_id(ci,     cj,     ck    )];
            local_to_global[1]  = x_edge_map[x_id(ci,     cj + 1, ck    )];
            local_to_global[2]  = x_edge_map[x_id(ci,     cj,     ck + 1)];
            local_to_global[3]  = x_edge_map[x_id(ci,     cj + 1, ck + 1)];
            local_to_global[4]  = y_edge_map[y_id(ci,     cj,     ck    )];
            local_to_global[5]  = y_edge_map[y_id(ci + 1, cj,     ck    )];
            local_to_global[6]  = y_edge_map[y_id(ci,     cj,     ck + 1)];
            local_to_global[7]  = y_edge_map[y_id(ci + 1, cj,     ck + 1)];
            local_to_global[8]  = z_edge_map[z_id(ci,     cj,     ck    )];
            local_to_global[9]  = z_edge_map[z_id(ci + 1, cj,     ck    )];
            local_to_global[10] = z_edge_map[z_id(ci,     cj + 1, ck    )];
            local_to_global[11] = z_edge_map[z_id(ci + 1, cj + 1, ck    )];

            for (int qk = 0; qk < 2; qk++)
            {
               for (int qj = 0; qj < 2; qj++)
               {
                  for (int qi = 0; qi < 2; qi++)
                  {
                     const double u = qx[qi];
                     const double v = qx[qj];
                     const double w = qx[qk];
                     double cell_hx_asm, cell_hy_asm, cell_hz_asm;
                     if (has_knots_asm)
                     {
                        cell_hx_asm = kx_asm[ci+1] - kx_asm[ci];
                        cell_hy_asm = ky_asm[cj+1] - ky_asm[cj];
                        cell_hz_asm = kz_asm[ck+1] - kz_asm[ck];
                        xi[0] = kx_asm[ci] + qx[qi] * cell_hx_asm;
                        xi[1] = ky_asm[cj] + qx[qj] * cell_hy_asm;
                        xi[2] = kz_asm[ck] + qx[qk] * cell_hz_asm;
                     }
                     else
                     {
                        cell_hx_asm = hx0;
                        cell_hy_asm = hy0;
                        cell_hz_asm = hz0;
                        xi[0] = (ci + qx[qi]) * hx0;
                        xi[1] = (cj + qx[qj]) * hy0;
                        xi[2] = (ck + qx[qk]) * hz0;
                     }
                     const double wq = qw[qi] * qw[qj] * qw[qk] * cell_hx_asm * cell_hy_asm * cell_hz_asm;

                     geom_.EvalGeometry(xi, x_phys, J);
                     const double detJ = J.Det();
                     if (std::abs(detJ) < 1e-30) { continue; }
                     invJ = J;
                     invJ.Invert();
                     JTJ.SetSize(3, 3);
                     invJJT.SetSize(3, 3);
                     mfem::MultAtB(J, J, JTJ);
                     mfem::MultABt(invJ, invJ, invJJT);

                     mfem::DenseMatrix muhat_inv(3), epshat(3);
                     muhat_inv = 0.0;
                     epshat = 0.0;
                     const double eps = eps_fn(x_phys);
                     for (int r = 0; r < 3; r++)
                     {
                        for (int c = 0; c < 3; c++)
                        {
                           muhat_inv(r, c) = JTJ(r, c) / detJ;
                           epshat(r, c) = detJ * eps * invJJT(r, c);
                        }
                     }

                     for (int a = 0; a < 12; a++)
                     {
                        const int ia = local_to_global[a];
                        if (ia < 0) { continue; }
                        EvalCellEdgeBasis(a, u, v, w, cell_hx_asm, cell_hy_asm, cell_hz_asm, Ei, curlEi);
                        for (int b = 0; b < 12; b++)
                        {
                           const int ib = local_to_global[b];
                           if (ib < 0) { continue; }
                           EvalCellEdgeBasis(b, u, v, w, cell_hx_asm, cell_hy_asm, cell_hz_asm, Ej, curlEj);

                           double curl_term = 0.0;
                           double mass_term = 0.0;
                           for (int r = 0; r < 3; r++)
                           {
                              double tmp_curl = 0.0;
                              double tmp_mass = 0.0;
                              for (int c = 0; c < 3; c++)
                              {
                                 tmp_curl += muhat_inv(r, c) * curlEj[c];
                                 tmp_mass += epshat(r, c) * Ej[c];
                              }
                              curl_term += curlEi[r] * tmp_curl;
                              mass_term += Ei[r] * tmp_mass;
                           }
                           A(ia, ib) += wq * (curl_term - k0 * k0 * mass_term);
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void YeeTransferBuilder::BuildProlongationExact(mfem::DenseMatrix &P) const
{
   BuildEdgeDofs();
   const int tvsize = fespace_.GetTrueVSize();
   const int ndofs = fespace_.GetNDofs();
   const int na = static_cast<int>(edge_dofs_.size());
   P.SetSize(tvsize, na);
   P = 0.0;

   if (mfem::Mpi::WorldRank() == 0)
   {
      std::cout << "[yee_transfer] BuildProlongationExact start: true_vsize="
                << tvsize << ", edge_dofs=" << na << std::endl;
   }
   const auto t0 = std::chrono::steady_clock::now();

   mfem::ParFiniteElementSpace *pfes =
      const_cast<mfem::ParFiniteElementSpace *>(&fespace_);

   // --- Assemble mass bilinear form for CG ---
   mfem::ConstantCoefficient one(1.0);
   mfem::ParBilinearForm mass_form(pfes);
   mass_form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mass_form.Assemble();
   mass_form.Finalize();

   // --- Build B matrix (same element-driven assembly) ---
   mfem::DenseMatrix B(ndofs, na);
   B = 0.0;

   const bool has_knots = !grid_.knot_x.empty();
   const auto& kx = has_knots ? grid_.knot_x : std::vector<double>{};
   const auto& ky = has_knots ? grid_.knot_y : std::vector<double>{};
   const auto& kz = has_knots ? grid_.knot_z : std::vector<double>{};
   const double hx0 = 1.0 / double(grid_.nx - 1);
   const double hy0 = 1.0 / double(grid_.ny - 1);
   const double hz0 = 1.0 / double(grid_.nz - 1);
   const int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;

   std::vector<int> x_edge_map((nx-1)*ny*nz, -1), y_edge_map(nx*(ny-1)*nz, -1), z_edge_map(nx*ny*(nz-1), -1);
   auto x_id = [&](int i,int j,int k){ return (k*ny+j)*(nx-1)+i; };
   auto y_id = [&](int i,int j,int k){ return (k*(ny-1)+j)*nx+i; };
   auto z_id = [&](int i,int j,int k){ return (k*ny+j)*nx+i; };

   for (int idx = 0; idx < na; idx++)
   {
      const auto &e = edge_dofs_[idx];
      if (e.axis == 0) x_edge_map[x_id(e.i,e.j,e.k)] = idx;
      else if (e.axis == 1) y_edge_map[y_id(e.i,e.j,e.k)] = idx;
      else z_edge_map[z_id(e.i,e.j,e.k)] = idx;
   }

   mfem::Array<int> vdofs;
   mfem::DenseMatrix vshape;
   mfem::Vector xi, Ehat, Ephys;
   mfem::DenseMatrix invJ;
   mfem::Vector curl_dummy;
   int local_to_global[12];

   for (int e = 0; e < fespace_.GetNE(); e++)
   {
      mfem::ElementTransformation *T = pfes->GetElementTransformation(e);
      const mfem::FiniteElement *fe = pfes->GetFE(e);
      pfes->GetElementVDofs(e, vdofs);
      const int dof = fe->GetDof();
      const int dim = T->GetSpaceDim();
      const mfem::IntegrationRule *ir =
         &mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder());
      vshape.SetSize(dof, dim);

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcVShape(*T, vshape);
         GetGlobalPatchXi(e, ip, xi);

         double cell_hx, cell_hy, cell_hz, u, v, w;
         int ci, cj, ck;
         if (has_knots)
         {
            ci = 0; while (ci < nx - 2 && kx[ci + 1] <= xi[0] + 1e-14) ci++;
            cj = 0; while (cj < ny - 2 && ky[cj + 1] <= xi[1] + 1e-14) cj++;
            ck = 0; while (ck < nz - 2 && kz[ck + 1] <= xi[2] + 1e-14) ck++;
            cell_hx = kx[ci+1]-kx[ci]; cell_hy = ky[cj+1]-ky[cj]; cell_hz = kz[ck+1]-kz[ck];
            u = (xi[0]-kx[ci])/cell_hx; v = (xi[1]-ky[cj])/cell_hy; w = (xi[2]-kz[ck])/cell_hz;
         }
         else
         {
            ci = std::max(0, std::min(nx-2, (int)(xi[0]/hx0)));
            cj = std::max(0, std::min(ny-2, (int)(xi[1]/hy0)));
            ck = std::max(0, std::min(nz-2, (int)(xi[2]/hz0)));
            cell_hx = hx0; cell_hy = hy0; cell_hz = hz0;
            u = (xi[0]-ci*hx0)/hx0; v = (xi[1]-cj*hy0)/hy0; w = (xi[2]-ck*hz0)/hz0;
         }
         u = std::max(0.0, std::min(1.0, u));
         v = std::max(0.0, std::min(1.0, v));
         w = std::max(0.0, std::min(1.0, w));

         local_to_global[0]  = x_edge_map[x_id(ci,     cj,     ck    )];
         local_to_global[1]  = x_edge_map[x_id(ci,     cj + 1, ck    )];
         local_to_global[2]  = x_edge_map[x_id(ci,     cj,     ck + 1)];
         local_to_global[3]  = x_edge_map[x_id(ci,     cj + 1, ck + 1)];
         local_to_global[4]  = y_edge_map[y_id(ci,     cj,     ck    )];
         local_to_global[5]  = y_edge_map[y_id(ci + 1, cj,     ck    )];
         local_to_global[6]  = y_edge_map[y_id(ci,     cj,     ck + 1)];
         local_to_global[7]  = y_edge_map[y_id(ci + 1, cj,     ck + 1)];
         local_to_global[8]  = z_edge_map[z_id(ci,     cj,     ck    )];
         local_to_global[9]  = z_edge_map[z_id(ci + 1, cj,     ck    )];
         local_to_global[10] = z_edge_map[z_id(ci,     cj + 1, ck    )];
         local_to_global[11] = z_edge_map[z_id(ci + 1, cj + 1, ck    )];

         const double weight = ip.weight * T->Weight();
         invJ = T->Jacobian();
         if (std::abs(invJ.Det()) < 1e-30) continue;
         invJ.Invert();

         for (int a = 0; a < 12; a++)
         {
            const int col = local_to_global[a];
            if (col < 0) continue;
            EvalCellEdgeBasis(a, u, v, w, cell_hx, cell_hy, cell_hz, Ehat, curl_dummy);
            Ephys.SetSize(dim);
            Ephys = 0.0;
            for (int c = 0; c < dim; c++)
               for (int r = 0; r < dim; r++)
                  Ephys[c] += invJ(r, c) * Ehat[r];

            for (int i = 0; i < dof; i++)
            {
               int row = vdofs[i];
               double sgn = 1.0;
               if (row < 0) { row = -row - 1; sgn = -1.0; }
               double dot = 0.0;
               for (int d = 0; d < dim; d++) dot += vshape(i, d) * Ephys[d];
               if (row >= 0 && row < ndofs) B(row, col) += sgn * weight * dot;
            }
         }
      }
   }

   // --- Per-column exact mass solve: M_h * Pi_col = B_col ---
   // Solve in LOCAL DOF space using serial SparseMatrix + CG
   // with Jacobi (diagonal) preconditioner for fast convergence
   mfem::SparseMatrix &Ms_local = const_cast<mfem::SparseMatrix &>(mass_form.SpMat());

   // Build diagonal preconditioner from mass matrix
   mfem::Vector mass_diag(ndofs);
   mass_diag = 0.0;
   {
      const int *I = Ms_local.GetI();
      const int *J = Ms_local.GetJ();
      const double *Data = Ms_local.GetData();
      for (int r = 0; r < Ms_local.Height(); r++) {
         double row_sum = 0.0;
         for (int p = I[r]; p < I[r+1]; p++) row_sum += Data[p];
         mass_diag[r] = row_sum;
      }
   }
   mfem::DSmoother jacobi(Ms_local, 0);

   mfem::CGSolver cg;
   cg.SetOperator(Ms_local);
   cg.SetPreconditioner(jacobi);
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(300);
   cg.SetPrintLevel(0);

   mfem::ParGridFunction gf(pfes);
   mfem::Vector rhs_ldof(ndofs), sol_ldof(ndofs);
   mfem::Vector col_true(tvsize);
   int cg_total_iters = 0;

   for (int j = 0; j < na; j++)
   {
      for (int i = 0; i < ndofs; i++) rhs_ldof[i] = B(i, j);
      sol_ldof = 0.0;
      cg.Mult(rhs_ldof, sol_ldof);
      cg_total_iters += cg.GetNumIterations();
      for (int i = 0; i < ndofs; i++) gf[i] = sol_ldof[i];
      gf.GetTrueDofs(col_true);
      for (int i = 0; i < tvsize; i++) P(i, j) = col_true[i];
   }

   if (mfem::Mpi::WorldRank() == 0) {
      std::cout << "[yee_transfer] CG avg iters/col=" 
                << double(cg_total_iters)/double(na) << std::endl;
   }

   if (mfem::Mpi::WorldRank() == 0)
   {
      const auto t1 = std::chrono::steady_clock::now();
      double p0norm = 0.0;
      if (na > 0) for (int i = 0; i < tvsize; i++) p0norm += P(i,0)*P(i,0);
      std::cout << "[yee_transfer] BuildProlongationExact done seconds="
                << std::chrono::duration<double>(t1 - t0).count()
                << " P col0 norm=" << std::sqrt(p0norm) << std::endl;
   }
}

} // namespace covariant_aux_space
