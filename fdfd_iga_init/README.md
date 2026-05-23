# Reference-Space FDFD Initial Guess Prototype

This directory contains a first-stage prototype for using a reference-space
finite-difference solve as an initial guess for single-patch NURBS H(curl)
Maxwell problems in MFEM.

Scope of this prototype:

- Single patch only.
- CPU reference implementation is complete.
- CUDA implementation is a stub interface for later integration.
- Geometry mapping is included in two places:
  - reference-space operator coefficients use the pulled-back metric
    `det(J) * J^{-1} J^{-T}` sampled from the NURBS patch;
  - projection from the reference-space field into the physical H(curl) space
    uses the covariant Piola map `E = J^{-T} E_hat`.

Important limitation:

- The current CPU operator is a geometry-aware vector Helmholtz initializer,
  not yet a full Yee-type curl-curl FDFD discretization.
- This is intentional for the first prototype: it is designed to test whether a
  geometry-aware reference-space solve provides a materially better initial
  guess for MFEM IGA than the zero vector.

Files:

- `reference_patch_evaluator.hpp`
  Geometry and Jacobian evaluation on a single NURBS patch.
- `reference_fdfd_cpu.hpp/.cpp`
  Geometry-aware reference-space finite-difference initializer.
- `reference_fdfd_cuda.hpp/.cpp`
  CUDA-oriented interface stub for later Maxwell-style acceleration.
- `reference_initial_guess.hpp/.cpp`
  Projection of the sampled reference field into MFEM's H(curl) grid function.
- `single_patch_demo.cpp`
  Minimal end-to-end driver.
