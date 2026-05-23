# Edge Yee Stage Summary

## Scope

This note records the current stage of the `edge_yee_proto` branch in
`covariant_aux_space/`.

The target problem is a single-patch NURBS `H(curl)` Maxwell IGA system in MFEM.
The target auxiliary-space design is:

`M^{-1} = S^{-1} + P A_Y^{-1} P^T`

with:

- `P`: transfer from parameter-domain Yee edge space to spline `H(curl)` space
- `A_Y`: independent parameter-domain Yee Maxwell operator
- `A_Y = C_Y^T M_{muhat^{-1}} C_Y - omega^2 M_{epshat}`

## Prototype Roles

Three prototype modes are now clearly separated:

- `nodal_proto`
  - transitional baseline
  - not the final theory target
- `edge_galerkin_proto`
  - uses the same edge transfer `P`
  - coarse operator is `P^T A P`
  - serves as an ideal coarse-space upper bound and transfer validation tool
- `edge_yee_proto`
  - uses the same edge transfer `P`
  - coarse operator is an independently assembled Yee-type `A_Y`
  - this is the actual theory-facing branch

## Main Findings So Far

### 1. The transfer `P` is strong

`edge_galerkin_proto` gives near-ideal coarse correction on the tested
single-patch case. This does **not** prove the full Yee auxiliary operator, but
it strongly supports that the edge-oriented transfer `P` captures the dominant
coarse / low-energy error content of the spline `H(curl)` discretization.

### 2. The first independent `A_Y` attempts failed for structural reasons

Early `edge_yee_proto` variants used surrogate coarse operators. They exhibited:

- large mismatch from `A_gal = P^T A P`
- bad GMRES behavior
- zero-diagonal or near-zero coarse-operator pathologies

This established that the main difficulty is not only the transfer `P`, but the
construction quality of the independent coarse Yee operator.

### 3. Explicit Yee topology is now in place

The current `edge_yee_proto` now uses:

- explicit Yee edge dofs
- explicit Yee face dofs
- explicit discrete curl/incidence matrix `C_Y`
- explicit coarse operator assembly of the form
  `A_Y = C_Y^T M C_Y - omega^2 M`

This is implemented in:

- `yee_transfer.hpp/.cpp`
- `yee_operator.hpp/.cpp`

### 4. Switching to cochain-style Hodge stars materially improved the method

The current Hodge assembly uses primal/dual metric ratios instead of the older
FE-like surrogate basis integrations:

- face Hodge for `muhat^{-1}` based on `dual_edge_length / primal_face_area`
- edge Hodge for `epshat` based on `eps * dual_face_area / primal_edge_length`

This moved `edge_yee_proto` from unstable / ineffective behavior to a genuinely
useful preconditioner on the current benchmark.

## Current Reference Result

For the present benchmark:

- mesh: `mfem/data/cube-nurbs.mesh`
- single patch
- `order = 2`
- `ref_levels = 2`
- `fd_n = 41`
- `aux_n = 7`
- `wavelength = 0.2`
- `eps_r = 8`
- source: `gaussian_y`

Observed iteration counts:

- zero initial guess: `79`
- direct FDFD initial guess: `88`
- `nodal_proto`: about `40`
- `edge_galerkin_proto`: about `1`
- `edge_yee_proto`: `36`

Interpretation:

- direct FDFD initial guess remains inferior in this setup
- `edge_galerkin_proto` remains the idealized upper bound
- `edge_yee_proto` is now meaningfully effective and has become better than the
  current `nodal_proto` baseline on this case

## Important Diagnostic Interpretation

For the current cube-like single-patch test:

- geometry is close to affine
- `eps` is constant

Therefore, approximately constant diagonal ranges in `M_{muhat^{-1}}` and
`M_{epshat}` are **not** by themselves a bug signal in this specific case.

The key signal is instead whether:

- `A_Y` is non-degenerate
- zero-diagonal pathologies are gone
- the coarse operator produces an actual reduction in GMRES iterations

At the current stage, these conditions are satisfied.

## What Has Been Established

This stage supports the following claims:

1. A strong edge-oriented transfer `P` has been constructed and validated in the
   Galerkin coarse-space setting.
2. An explicit Yee-topology coarse operator branch is now implemented.
3. Replacing surrogate Hodge assembly with cochain-style primal/dual metric
   ratios substantially improves the independent `edge_yee_proto`.
4. On the current benchmark, `edge_yee_proto` is already a competitive and
   effective auxiliary-space preconditioner.

## What Has Not Been Established Yet

This stage does **not** yet establish:

1. `edge_yee_proto` is robust with respect to:
   - non-affine geometry
   - variable material coefficients
   - higher spline order
   - stronger refinement
2. the current transfer `P` satisfies a full commuting-property theory
3. the current `A_Y` is already close to optimal relative to `P^T A P`
4. superiority over AMS on a broad benchmark set

## Immediate Next Steps

The next stage should validate `edge_yee_proto` in settings where Hodge-star
quality matters more clearly:

1. non-affine single-patch NURBS geometry
2. spatially varying `eps(x)`
3. larger `order` / `ref_levels`
4. side-by-side comparison of:
   - `nodal_proto`
   - `edge_galerkin_proto`
   - `edge_yee_proto`

The main goal of the next stage is to determine whether the new explicit Yee
coarse operator retains its advantage when geometry and material variation are
no longer trivial.
