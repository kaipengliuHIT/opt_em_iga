# Covariant Auxiliary Space Stage Summary

## Context

This prototype studies a single-patch isogeometric `H(curl)` Maxwell solve in
MFEM and tests whether a reference-space structured auxiliary problem can help
the Krylov convergence.

The main discrete space is a NURBS `H(curl)` space:

- `NURBS_HCurlFECollection`
- `ParFiniteElementSpace` built on `NURBSExtension`

So this is already an electromagnetic IGA setting, not a scalar IGA test. The
discrete unknowns are the NURBS `H(curl)` analogue of Nedelec-type tangential
electromagnetic degrees of freedom.

The PDE used in the demo is the frequency-domain Maxwell curl-curl model:

`curl(mu^{-1} curl E) - k0^2 eps E = f`

assembled through:

- `CurlCurlIntegrator`
- `VectorFEMassIntegrator`

This is a simplified Maxwell model for algorithm verification:

- single patch
- constant `mu`
- constant `eps_r`
- volumetric Gaussian source
- no PML
- no multipatch coupling

## Goal

The original goal was to test whether a reference-space finite-difference
Maxwell-type solve can help MFEM electromagnetic IGA convergence, first as an
initial guess and then as an auxiliary-space preconditioner.

## What Was Tried

### 1. Direct reference-space initial guess

A reference-space `curl-curl` finite-difference prototype was built in
`fdfd_iga_init/`, sampled on a structured grid and projected back to the MFEM
`H(curl)` space.

Important result:

- direct FDFD initial guess was not helpful
- in multiple cases it was worse than the zero initial guess

Typical result:

- zero initial guess: `79` GMRES iterations
- direct reference-space initial guess: `88` GMRES iterations

Conclusion:

- using the reference solve only as an initial guess is not the right route

### 2. Incorrect residual-as-field auxiliary attempt

An intermediate prototype interpreted the algebraic Krylov residual as if it
were directly an `H(curl)` field, then sampled and pull-backed it to the
reference domain.

This failed badly.

Conclusion:

- the residual of the linear system does not live as a physical field
- this path is algorithmically incorrect

### 3. Explicit discrete auxiliary-space preconditioner

The auxiliary-space prototype was then rewritten into an explicit discrete form:

`P^{-1} ~= Pi A_aux^{-1} Pi^T`

where:

- `Pi` is a discrete prolongation from reference auxiliary unknowns to MFEM
  true dofs
- `A_aux` is the reference-space auxiliary operator
- `Pi^T` restricts residuals algebraically into the auxiliary space

This change is the main algorithmic breakthrough of the current stage.

## Main Findings

### Finding 1. The algorithmic form matters more than the initial guess

Direct initial guesses were weak or harmful, but the explicit auxiliary-space
preconditioner was effective.

Representative case:

- order `2`
- refinement level `2`
- `fd_n = 31`
- `aux_n = 5`
- true dofs `882`

Results:

- zero initial guess: `79`
- direct FDFD initial guess: `88`
- covariant auxiliary preconditioner: `8`

So the auxiliary preconditioner reduced iterations by almost `10x` relative to
the zero initial guess.

### Finding 2. The reference-space grid for the initial FDFD solve is not the dominant factor

For the tested cases, changing `fd_n` from `31` to `41` had very little impact.

Conclusion:

- the dominant issue is not the resolution of the reference-space FDFD grid
- the dominant issue is the auxiliary-space transfer/operator design

### Finding 3. Auxiliary-space size strongly affects performance

The most important experimental trend is that the auxiliary-space size must be
chosen carefully. Bigger is not better.

Current auxiliary dof count is approximately:

`aux_dofs ~= 3 * (aux_n - 2)^3`

Scan results showed:

- small auxiliary space can be very effective
- medium auxiliary space can still help, but much less
- overly large auxiliary space can become unstable or ineffective

Representative results:

1. `order=2, ref_levels=2, true_vsize=882`
   - `aux_n=5`, `aux_dofs=81`, `aux_ratio=0.0918`: `79 -> 8`
   - `aux_n=7`, `aux_dofs=375`, `aux_ratio=0.425`: `79 -> 40`

2. `order=3, ref_levels=2, true_vsize=1344`
   - `aux_n=5`, `aux_dofs=81`, `aux_ratio=0.0603`: `254 -> 8`
   - `aux_n=7`, `aux_dofs=375`, `aux_ratio=0.279`: `254 -> 41`

3. cases with large ratios can fail
   - e.g. `aux_ratio ~ 0.69` or larger was often poor or nonconvergent
   - `aux_ratio > 1` is clearly not appropriate for this prototype

## Interpretation of the Auxiliary-Space Size Hypothesis

The experiments support the following hypothesis:

- the useful auxiliary-space size is related to the size of the IGA problem
- but not through a one-to-one scaling with the full number of MFEM true dofs
- what matters is the ratio `aux_dofs / true_vsize`
- a relatively small auxiliary space is enough to capture the troublesome
  low-frequency / near-kernel error modes

What the current data suggests:

- `aux_ratio ~ 0.06 - 0.15` is very promising
- `aux_ratio ~ 0.27 - 0.43` is still usable, but weaker
- `aux_ratio ~ 0.69` and above is risky for the current prototype

This does not yet prove a universal law, but it is strong evidence for the
practical design rule:

- choose the auxiliary space significantly smaller than the IGA true-dof space
- do not assume that increasing auxiliary-space size always improves
  preconditioning

## What `ok` Means in the Scan Output

In the scan CSV, the `status` column currently has a limited meaning:

- `ok`: the auxiliary-space ratio did not exceed the configured threshold
- `over_ratio`: the auxiliary-space ratio exceeded the configured threshold

`ok` does **not** mean that the case is numerically good. A case can still be
`ok` and converge poorly.

## Current Limitations

The current prototype is still a research prototype, not a full production
Maxwell IGA preconditioner. Important limitations are:

- only single patch
- no PML
- no material discontinuity study
- no multipatch interface treatment
- `Pi` is still built by column-wise mass projection, which is practical but
  not yet a fully commuting `H(curl)` transfer
- the reference-space auxiliary operator is still a simplified structured
  prototype, not yet a full Yee-style edge-based discretization

## Code Locations

Main current files:

- `fdfd_iga_init/reference_patch_evaluator.hpp`
- `fdfd_iga_init/reference_fdfd_cpu.hpp`
- `fdfd_iga_init/reference_fdfd_cpu.cpp`
- `fdfd_iga_init/reference_initial_guess.hpp`
- `fdfd_iga_init/reference_initial_guess.cpp`
- `fdfd_iga_init/single_patch_demo.cpp`
- `covariant_aux_space/covariant_reference_preconditioner.hpp`
- `covariant_aux_space/covariant_reference_preconditioner.cpp`

## Recommended Next Steps

1. Keep the explicit auxiliary-space form `Pi A_aux^{-1} Pi^T`.
2. Continue studying the relation between `aux_ratio` and convergence quality.
3. Improve `Pi` from a projected-column prototype toward a more structure-aware
   `H(curl)` transfer.
4. Test larger single-patch problems to see whether the good `aux_ratio` range
   remains stable.
5. After that, extend to multipatch and more realistic electromagnetic cases.

## Stage Conclusion

The main stage conclusion is:

- the original idea is viable, but only when formulated as a true discrete
  auxiliary-space preconditioner
- using the reference-space solve directly as an initial guess is not the right
  strategy
- the correct discrete transfer operator is the key
- auxiliary-space size matters, and a small, well-chosen auxiliary space can
  produce very strong acceleration for electromagnetic IGA `H(curl)` solves
