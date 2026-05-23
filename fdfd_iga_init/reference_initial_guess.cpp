#include "reference_initial_guess.hpp"
#include "fem/fe/fe_nurbs.hpp"
#include <iostream>

namespace fdfd_iga_init
{

ReferenceFieldProjector::SampledFieldCoefficient::SampledFieldCoefficient(
   const ReferenceFieldProjector &projector, bool imag_part)
   : mfem::VectorCoefficient(projector.fespace_.GetParMesh()->SpaceDimension()),
     projector_(projector),
     imag_part_(imag_part)
{
}

void ReferenceFieldProjector::SampledFieldCoefficient::Eval(
   mfem::Vector &V, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
{
   mfem::Vector xi, Ehat;
   projector_.GetGlobalPatchXi(T.ElementNo, ip, xi);
   projector_.SampleReferenceField(xi, imag_part_, Ehat);

   const mfem::DenseMatrix &J = T.Jacobian();
   mfem::DenseMatrix invJ(J);
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

ReferenceFieldProjector::ReferenceFieldProjector(
   const mfem::ParFiniteElementSpace &fespace,
   const SampledReferenceField &field)
   : fespace_(fespace),
     ext_(*const_cast<mfem::NURBSExtension *>(fespace.GetNURBSext())),
     field_(field)
{
   ext_.GetPatchKnotVectors(0, patch_kv_);
}

void ReferenceFieldProjector::GetGlobalPatchXi(
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

void ReferenceFieldProjector::SampleReferenceField(const mfem::Vector &xi,
                                                   bool imag_part,
                                                   mfem::Vector &Ehat) const
{
   Ehat.SetSize(3);
   Ehat = 0.0;

   const std::vector<double> &data = imag_part ? field_.imag : field_.real;
   const double gx = xi[0] * double(field_.nx - 1);
   const double gy = xi[1] * double(field_.ny - 1);
   const double gz = xi[2] * double(field_.nz - 1);
   const int i0 = std::max(0, std::min(field_.nx - 2, int(std::floor(gx))));
   const int j0 = std::max(0, std::min(field_.ny - 2, int(std::floor(gy))));
   const int k0 = std::max(0, std::min(field_.nz - 2, int(std::floor(gz))));
   const double tx = gx - i0;
   const double ty = gy - j0;
   const double tz = gz - k0;
   const int nnode = field_.NodeCount();

   for (int dz = 0; dz <= 1; dz++)
   {
      const double wz = dz ? tz : (1.0 - tz);
      for (int dy = 0; dy <= 1; dy++)
      {
         const double wy = dy ? ty : (1.0 - ty);
         for (int dx = 0; dx <= 1; dx++)
         {
            const double wx = dx ? tx : (1.0 - tx);
            const double w = wx * wy * wz;
            const int nid = SampleIndex(i0 + dx, j0 + dy, k0 + dz);
            for (int c = 0; c < 3; c++)
            {
               Ehat[c] += w * data[c * nnode + nid];
            }
         }
      }
   }
}

int ReferenceFieldProjector::GetLocalDofDirection(const mfem::FiniteElement &fe,
                                                  int ldof) const
{
   if (const auto *hc3 =
          dynamic_cast<const mfem::NURBS_HCurl3DFiniteElement *>(&fe))
   {
      (void)hc3;
      const int px = patch_kv_[0]->GetOrder();
      const int py = patch_kv_[1]->GetOrder();
      const int pz = patch_kv_[2]->GetOrder();
      const int ndof_x = (px + 1) * (py + 2) * (pz + 2);
      const int ndof_y = (px + 2) * (py + 1) * (pz + 2);
      if (ldof < ndof_x) { return 0; }
      if (ldof < ndof_x + ndof_y) { return 1; }
      return 2;
   }

   if (const auto *hc2 =
          dynamic_cast<const mfem::NURBS_HCurl2DFiniteElement *>(&fe))
   {
      (void)hc2;
      const int px = patch_kv_[0]->GetOrder();
      const int py = patch_kv_[1]->GetOrder();
      const int ndof_x = (px + 1) * (py + 2);
      return (ldof < ndof_x) ? 0 : 1;
   }

   MFEM_ABORT("ReferenceFieldProjector only supports NURBS_HCurl elements.");
}

void ReferenceFieldProjector::GetReferenceDofTangent(const mfem::FiniteElement &fe,
                                                     int ldof,
                                                     mfem::Vector &tref) const
{
   tref.SetSize(fe.GetDim());
   tref = 0.0;
   tref[GetLocalDofDirection(fe, ldof)] = 1.0;
}

void ReferenceFieldProjector::ProjectGridFunction(mfem::ParGridFunction &gf,
                                                  bool imag_part) const
{
   mfem::ParFiniteElementSpace *pfes = const_cast<mfem::ParFiniteElementSpace *>(&fespace_);
   SampledFieldCoefficient coeff(*this, imag_part);
   mfem::ConstantCoefficient one(1.0);

   mfem::ParLinearForm rhs(pfes);
   rhs.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(coeff));
   rhs.Assemble();

   mfem::ParBilinearForm mass(pfes);
   mass.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
   mass.Assemble();

   mfem::Array<int> empty_tdofs;
   mfem::OperatorPtr A;
   mfem::Vector X, B;
   gf = 0.0;
   mass.FormLinearSystem(empty_tdofs, gf, rhs, A, X, B);

   mfem::CGSolver cg(pfes->GetComm());
   cg.SetOperator(*A);
   std::unique_ptr<mfem::HypreSmoother> prec;
   if (auto *Ah = dynamic_cast<mfem::HypreParMatrix *>(A.Ptr()))
   {
      prec = std::make_unique<mfem::HypreSmoother>(*Ah, mfem::HypreSmoother::Jacobi);
      cg.SetPreconditioner(*prec);
   }
   cg.SetRelTol(1e-10);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   cg.Mult(B, X);
   mass.RecoverFEMSolution(X, rhs, gf);
}

void ReferenceFieldProjector::Project(mfem::ParComplexGridFunction &x)
{
   ProjectGridFunction(x.real(), false);
   ProjectGridFunction(x.imag(), true);

   mfem::Vector x_re_true, x_im_true;
   x.real().GetTrueDofs(x_re_true);
   x.imag().GetTrueDofs(x_im_true);
   x.real().SetFromTrueDofs(x_re_true);
   x.imag().SetFromTrueDofs(x_im_true);
}

void ReferenceFieldProjector::ProjectReal(mfem::ParGridFunction &x) const
{
   ProjectGridFunction(x, false);
}

void ReferenceFieldProjector::ProjectImag(mfem::ParGridFunction &x) const
{
   ProjectGridFunction(x, true);
}

} // namespace fdfd_iga_init
