#ifndef COVARIANT_AUX_SPACE_PRECONDITIONER_FACTORY_HPP
#define COVARIANT_AUX_SPACE_PRECONDITIONER_FACTORY_HPP

#include "mfem.hpp"
#include "covariant_reference_preconditioner.hpp"
#include "../fdfd_iga_init/reference_patch_evaluator.hpp"
#include <memory>
#include <functional>
#include <string>

namespace covariant_aux_space {

/// One-stop configuration for the covariant auxiliary-space preconditioner.
/// Fill this struct and pass to CreateCovariantPreconditioner().
struct PreconditionerConfig
{
   /// Preconditioner variant.
   /// "edge_yee"      — Yee FDFD auxiliary operator (fastest, default)
   /// "edge_galerkin" — Πᵀ A_h Π exact Galerkin projection
   /// "ams"           — Hypre AMS (baseline, often fails on NURBS)
   std::string mode = "edge_yee";

   /// Auxiliary grid points per direction (used when knot_align=false).
   /// For cube-nurbs r=2 o=2, optimal is 5.
   int aux_n = 5;

   /// Wave number k₀ = 2πf/c.
   double wave_number = 1.0;

   /// ── Knot-align options ──────────────────────────────────────────
   /// Align Yee grid lines to NURBS knot positions.
   /// Strongly recommended: produces optimal aux space with cps=1.
   bool knot_align = true;

   /// Cells per knot span (≥1). Default 1 = one Yee cell per Bézier element.
   int cells_per_span = 1;

   /// ── Yee FDFD PML options ────────────────────────────────────────
   /// Enable PML stretching inside the Yee coarse operator.
   /// Enable PML stretching inside Yee coarse operator (for PML meshes).
   bool yee_pml = false;

   /// Disable PML Galerkin fallback; force pure Yee FDFD even with PML.
   /// This does not disable the Yee PML stretching weights.
   bool no_pml_fallback = false;

   /// Calibrate Yee diagonal mean to match IGA Galerkin operator.
   bool calibrate_yee_diag = true;

   /// Full-rank fallback smoother term: z = beta*r + Pi Aaux^{-1} Pi^T r.
   /// Keep at 0 for pure two-level coarse correction.
   double identity_smoother_weight = 0.0;

   /// Diagnostic full-rank operator-Jacobi smoother term:
   /// z += omega * diag(A)^{-1} r.
   double operator_jacobi_smoother_weight = 0.0;
   int operator_jacobi_smoother_iterations = 1;

   /// ── PML parameters (only used when yee_pml=true) ────────────────
   double pml_thickness = 0.25;
   double pml_strength  = 5.0;
   double pml_order     = 2.0;

   /// ── GMRES / FGMRES options ──────────────────────────────────────
   int    gmres_max_iter  = 500;
   int    gmres_print     = 0;
   double gmres_rel_tol   = 1e-5;
   double gmres_abs_tol   = 0.0;
};

/// Return a short human-readable label for the configuration,
/// useful for benchmark table column headers.
inline std::string PreconditionerLabel(const PreconditionerConfig &cfg)
{
   if (cfg.mode == "ams") return "AMS";
   std::string label = (cfg.mode == "edge_yee") ? "edge_yee" : "edge_galerkin";
   if (cfg.knot_align) {
      label += "_ka" + std::to_string(cfg.cells_per_span);
   } else {
      label += "_an" + std::to_string(cfg.aux_n);
   }
   if (cfg.no_pml_fallback) label += "_nopf";  // no PML fallback
   return label;
}

/// Factory: create a covariant auxiliary-space preconditioner from a config.
///
/// Usage:
///   PreconditionerConfig cfg;
///   cfg.mode = "edge_yee";
///   cfg.knot_align = true;
///   cfg.cells_per_span = 1;
///   auto prec = CreateCovariantPreconditioner(fespace, geom, eps_fn, cfg);
///   gmres.SetPreconditioner(*prec);
///
inline std::unique_ptr<CovariantReferencePreconditioner>
CreateCovariantPreconditioner(
   const mfem::ParFiniteElementSpace &fespace,
   const fdfd_iga_init::SinglePatchNURBSEvaluator &geom,
   const std::function<double(const mfem::Vector &)> &eps_fn,
   const PreconditionerConfig &cfg = PreconditionerConfig{})
{
   using Mode = CovariantReferencePreconditioner::PrototypeMode;

   Mode pmode;
   if (cfg.mode == "edge_galerkin") {
      pmode = Mode::edge_galerkin_proto;
   } else {
      pmode = Mode::edge_yee_proto;  // default
   }

   auto prec = std::make_unique<CovariantReferencePreconditioner>(
      fespace, geom, eps_fn);

   prec->SetPrototypeMode(pmode);
   prec->SetWaveNumber(cfg.wave_number);
   prec->SetYeeDiagonalCalibration(cfg.calibrate_yee_diag);
   prec->SetIdentitySmootherWeight(cfg.identity_smoother_weight);
   prec->SetOperatorJacobiSmootherWeight(cfg.operator_jacobi_smoother_weight);
   prec->SetOperatorJacobiSmootherIterations(cfg.operator_jacobi_smoother_iterations);

   // Knot-aligned grid
   if (cfg.knot_align && cfg.cells_per_span > 0) {
      prec->SetKnotAlignGrid(true, cfg.cells_per_span);
   }

   // Grid (aux_n is fallback when knot_align is off)
   prec->SetGrid({cfg.aux_n, cfg.aux_n, cfg.aux_n});

   // PML
   if (cfg.mode == "edge_yee") {
      prec->SetYeeReferencePML(cfg.yee_pml,
                               cfg.pml_thickness,
                               cfg.pml_strength,
                               cfg.pml_order);
      prec->SetYeePMLGalerkinFallback(!cfg.no_pml_fallback);
   }

   return prec;
}

/// Convenience: create and configure a GMRES solver with the preconditioner.
inline mfem::GMRESSolver
CreatePreconditionedGMRES(
   const mfem::Operator &A,
   CovariantReferencePreconditioner &prec,
   const PreconditionerConfig &cfg = PreconditionerConfig{})
{
   mfem::GMRESSolver gmres;
   gmres.SetAbsTol(cfg.gmres_abs_tol);
   gmres.SetRelTol(cfg.gmres_rel_tol);
   gmres.SetMaxIter(cfg.gmres_max_iter);
   gmres.SetPrintLevel(cfg.gmres_print);
   gmres.SetOperator(A);
   gmres.SetPreconditioner(prec);
   return gmres;
}

} // namespace covariant_aux_space

#endif
