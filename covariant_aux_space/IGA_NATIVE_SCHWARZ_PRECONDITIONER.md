# IGA-native full-rank Schwarz preconditioner for PML Maxwell systems

## Motivation

The Yee/FDFD auxiliary-space preconditioner has a useful low-order regime, but
the PML high-order case exposed a structural weakness: the preconditioned
operator depends on a transfer between the IGA H(curl) space and an auxiliary
Yee edge space. If the transferred Yee energy is not operator-equivalent to the
IGA Galerkin PML energy, the method may reduce the left-preconditioned residual
without reducing the unpreconditioned true residual.

The IGA-native Schwarz route removes that transfer from the main solver. The
preconditioner is built directly from the real 2x2 IGA-PML operator

```text
A_h x = b,
```

where the real and imaginary unknowns are represented as two coupled real
blocks. Every local subproblem is extracted from the same assembled operator
that GMRES solves. This is the main reason the method is a full-rank
preconditioner instead of a low-rank auxiliary correction.

## Patch construction

Let `V_h` be the true H(curl) IGA space after essential boundary elimination and
parallel true-dof constraints. For each knot element `K`, MFEM provides the
vector dofs whose NURBS basis functions have support on that element:

```text
D_K = GetElementVDofs(K).
```

The current prototype creates one seed patch per element. With `ell` overlap
layers, the patch grows through the element graph induced by shared true dofs:

```text
omega_K^(0) = {K},
omega_K^(ell+1) = omega_K^(ell) union {K' : D_K' intersects D_J
                                      for some J in omega_K^(ell)}.
```

The scalar true-dof set of a patch is

```text
I_K = union_{J in omega_K^(ell)} D_J.
```

For the real 2x2 complex representation, the actual patch indices are

```text
I_K^C = I_K union (n + I_K),
```

where `n = dim(V_h)`. This keeps the local real/imaginary coupling inside each
patch and follows MFEM's block representation of complex Maxwell operators.

## Restricted additive Schwarz operator

Let `R_i` restrict a global vector to patch indices `I_i^C`. The local matrix is
not a surrogate:

```text
A_i = R_i A_h R_i^T.
```

The weighted RAS preconditioner is

```text
B^{-1} r = sum_i R_i^T W_i A_i^{-1} R_i r.
```

`W_i` is diagonal and implements partition-of-unity weighting. In the current
implementation, if a scalar true dof is covered by `c_j` scalar patches, both
its real and imaginary entries receive weight `1 / c_j`.

The prototype also supports repeated residual-updated Schwarz sweeps inside one
preconditioner application:

```text
z_0 = 0,
rho_k = r - A_h z_k,
z_{k+1} = z_k + damping * sum_i R_i^T W_i A_i^{-1} R_i rho_k.
```

One sweep is the default.

## Full-rank property

Assume:

1. Every global true dof belongs to at least one patch.
2. Every retained patch weight is positive.
3. Each local matrix `A_i` is nonsingular, or is replaced by a nonsingular
   regularized local solver.

The patch construction enforces the first condition because every true dof that
appears on a knot element is included in the seed patch of that element; any
unexpected uncovered dof is added as a two-entry real/imag fallback patch.

For the diagonal coverage weights,

```text
sum_i R_i^T W_i R_i = I.
```

Thus the restriction/prolongation structure itself has no missing coordinate
direction. Unlike an auxiliary correction of the form

```text
Pi A_aux^{-1} Pi^T,
```

whose rank is at most `rank(Pi)`, the Schwarz operator touches every degree of
freedom in the IGA system. This eliminates the structural nullspace caused by a
dimension- or transfer-limited auxiliary space.

For coercive or Hermitian positive definite model problems, standard overlapping
Schwarz theory then gives spectral equivalence bounds controlled by stable
decomposition, strengthened Cauchy-Schwarz constants, overlap, and local solver
quality. For the indefinite complex-stretched Maxwell-PML operator, the same
SPD theorem does not directly prove GMRES convergence. The rigorous statement
we can make is narrower but still important: the preconditioner is an
IGA-native full-rank approximate inverse assembled from `A_h` itself, so any
failure is no longer caused by loss of operator equivalence between IGA and Yee
spaces.

## Relation to the Yee/FDFD auxiliary space

The Yee route remains useful as a diagnostic and as a low-order cheap method.
Its correction has the algebraic form

```text
z = Pi A_Yee^{-1} Pi^T r,
```

possibly with smoothers added around it. When `Pi^T A_h Pi` and `A_Yee` are not
operator-equivalent under PML stretching, the correction can be energetically
mis-scaled. Adding Yee dofs or solving the Yee system more accurately does not
fix that mismatch.

The IGA-native Schwarz route instead uses:

```text
A_i = R_i A_h R_i^T.
```

There is no Yee Hodge approximation, no PML transfer calibration, and no
auxiliary-space rank bottleneck in the primary preconditioner.

## Current implementation

The reusable interface is:

```cpp
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"

covariant_aux_space::IGAPatchRASPreconditioner::Options opts;
opts.overlap_layers = 0;
opts.damping = 0.8;
opts.iterations = 1;

covariant_aux_space::IGAPatchRASPreconditioner B(A, fespace, opts);
gmres.SetPreconditioner(B);
```

The same object is also available through the common factory header:

```cpp
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"

covariant_aux_space::IGAPatchRASConfig cfg;
cfg.overlap_layers = 0;
cfg.damping = 0.8;

auto B = covariant_aux_space::CreateIGAPatchRASPreconditioner(A, fespace, cfg);
auto gmres = covariant_aux_space::CreatePreconditionedGMRES(A, *B, cfg);
```

The `pml_point_source_demo` exposes the same preconditioner through:

```text
-prec iga_ras -rasov <overlap_layers> -rasw <damping> -rasit <sweeps>
```

Prototype limitation: the current implementation materializes the full dense
operator only when sparse matrix extraction is unavailable or explicitly forced
with `-rasasm dense_probe`. The default `auto` path extracts the rank-local CSR
block from MFEM matrix types such as `ComplexHypreParMatrix` and
`HypreParMatrix`, then forms dense local patch matrices from that sparse block.
This avoids global dense probing and is the intended production-oriented path.
Patch solves are still dense LU on each patch; future larger-scale runs should
replace those by sparse patch ILU/direct solvers when patch dimensions become
too large.

MPI status: the implementation supports rank-local RAS on parallel MFEM
matrices. Patches are built from locally owned true dofs and the local diagonal
block of the parallel matrix. Overlap does not yet cross MPI partition
boundaries.

## Benchmark evidence

The focused PML case is:

```text
cube-nurbs.mesh, refine=2, order=3, frequency=4.0, true residual control.
```

Observed prototype results:

```text
iga_ras, overlap=0, damping=0.8:
  patches=64, max_patch_dim=600, 20 GMRES iterations,
  final true relative residual = 2.55864e-07 in a run with target 1e-7
  (well below the original 1e-5 target).

iga_ras, overlap=1, damping=0.8:
  patches=64, max_patch_dim=2688, 1 GMRES iteration,
  final true relative residual = 4.85410e-08 with target 1e-5.
```

The second result is close to a global solve on this small single-patch mesh,
so it should be treated as an upper-bound diagnostic rather than a scalable
method. The first result is the more meaningful prototype: no extra overlap,
still full-rank, and already able to drive the unpreconditioned true residual
below the original 1e-5 target on the case where the transferred Yee-PML
auxiliary operator failed.
