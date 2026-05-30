# MFEM IGA H(curl) preconditioner module

This module contains three solver-side preconditioner families for MFEM
IGA/NURBS H(curl) Maxwell simulations:

```text
edge_galerkin
edge_yee
iga_ras
```

The first two are auxiliary-space methods based on a reference-grid edge space.
The third is an IGA-native full-rank Schwarz method built directly from the true
IGA operator.

## Files

```text
covariant_aux_space/covariant_reference_preconditioner.hpp
covariant_aux_space/covariant_reference_preconditioner.cpp
covariant_aux_space/iga_patch_ras_preconditioner.hpp
covariant_aux_space/iga_patch_ras_preconditioner.cpp
covariant_aux_space/covariant_preconditioner_factory.hpp
covariant_aux_space/yee_transfer.hpp
covariant_aux_space/yee_operator.hpp
```

`covariant_preconditioner_factory.hpp` is the intended public entry point for
new demos.

## edge_galerkin

`edge_galerkin` uses the Yee/reference edge transfer but builds the auxiliary
operator by Galerkin projection:

```text
A_aux = Pi^T A_h Pi.
```

This is the cleanest diagnostic for the transfer space itself. If this method
fails, the issue is usually not the independent Yee Hodge but the transfer
space/range or the lack of a full-rank smoother.

Usage:

```cpp
covariant_aux_space::PreconditionerConfig cfg;
cfg.mode = "edge_galerkin";
cfg.knot_align = true;
cfg.cells_per_span = 2;
cfg.wave_number = k0;

auto prec = covariant_aux_space::CreateCovariantPreconditioner(
   fespace, geom, eps_fn, cfg);
prec->SetOperator(A);
gmres.SetPreconditioner(*prec);
```

## edge_yee

`edge_yee` uses the same transfer space but replaces the Galerkin projected
operator by an independent Yee/FDFD-style edge operator. This is the cheap
method that worked well for non-PML and lower-order IGA cases.

Usage:

```cpp
covariant_aux_space::PreconditionerConfig cfg;
cfg.mode = "edge_yee";
cfg.knot_align = true;
cfg.cells_per_span = 1;
cfg.wave_number = k0;
cfg.yee_pml = true;
cfg.no_pml_fallback = true;

auto prec = covariant_aux_space::CreateCovariantPreconditioner(
   fespace, geom, eps_fn, cfg);
prec->SetOperator(A);
gmres.SetPreconditioner(*prec);
```

Current paper-facing interpretation:

- For non-PML cavity and `p <= 2`, `edge_yee` is the cheap successful method.
- For high-order PML, especially `r=2,o=3`, it should be presented as a
  limitation/diagnostic unless a new stable equivalence result is obtained.

## iga_ras

`iga_ras` is the robust IGA-native path. It does not use a Yee auxiliary grid.
Patches are built from NURBS H(curl) basis support and local matrices are
extracted from the true real 2x2 IGA-PML system:

```text
A_i = R_i A_h R_i^T.
```

Usage:

```cpp
covariant_aux_space::IGAPatchRASConfig cfg;
cfg.overlap_layers = 0;
cfg.damping = 0.8;
cfg.iterations = 1;

auto prec = covariant_aux_space::CreateIGAPatchRASPreconditioner(
   A, fespace, cfg);
gmres.SetPreconditioner(*prec);
```

`overlap_layers=0` already means a NURBS-support-aware element patch: the dof
set is obtained from `GetElementVDofs()`. Larger overlap expands through the
element graph induced by shared true dofs.

Current implementation limitation:

- The default `auto` assembly path extracts the local CSR block from
  `ComplexHypreParMatrix`, `HypreParMatrix`, `ComplexSparseMatrix`, or
  `SparseMatrix`, then builds dense patch matrices from that sparse local
  operator.
- The old dense global probing path is still available as
  `assembly = DenseProbe` or `-rasasm dense_probe`, but should only be used for
  debugging unsupported operator types.
- Patch solves are still dense LU on each local patch. This is acceptable for
  support-sized NURBS patches; the next scalability improvement is sparse
  local ILU/direct patch solvers.
- MPI is supported as rank-local RAS: patches are built from dofs owned by the
  current MPI rank and the local diagonal block of the parallel matrix. Current
  overlap does not cross MPI partition boundaries.

## Demo command-line interface

`pml_point_source_demo` exposes:

```text
-prec none
-prec ams
-prec edge_galerkin -ka -cps <n>
-prec edge_yee -ka -cps <n> [-npf] [-mfbaux]
  -prec iga_ras -rasov <overlap> -rasw <damping> -rasit <sweeps>
  -rasasm auto|local_sparse|dense_probe
```

The recommended hard PML diagnostic command for the current IGA-native path is:

```bash
./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -o 3 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 \
  -trc -gmi 500 -gpl 0 -grt 1e-7 -no-vis
```

For MPI smoke testing:

```bash
bash run_iga_ras_mpi_smoke.sh
```

In MPI runs, `-trc` (true-residual early-stop) is now collective-safe.
`TrueResidualController` uses `MPI_Allreduce` to compute the global L2 norm, so
all ranks set the `converged` flag at the same iteration and no hang occurs.
The final unpreconditioned true residual is also printed after GMRES in all cases.

## Comprehensive benchmark

Run:

```bash
bash run_full_preconditioner_benchmark.sh
```

The script writes:

```text
benchmark_results/full_preconditioners_<timestamp>/results.csv
benchmark_results/full_preconditioners_<timestamp>/summary.md
```

Current focused result on `cube-nurbs.mesh`, `r=2`, `o=3`, PML, `f=4.0`:

```text
no_preconditioner:      500 iters, true rel 3.99660e-03
ams:                    500 iters, true rel 8.34136e-03
edge_galerkin_cps2:     500 iters, true rel 1.74578e-01
edge_yee_cps2_nopf:     500 iters, true rel 1.98237e-01
iga_ras_overlap0:        20 iters, true rel 2.55864e-07
```

The FDFD-initial-guess diagnostic is currently not a valid PML row. It belongs
to the older cavity/reference-grid initializer demo, and in the current tree it
does not complete cleanly under the benchmark script. Keep it as a separate
diagnostic until a PML-compatible initial-guess mode is implemented in
`pml_point_source_demo`.
