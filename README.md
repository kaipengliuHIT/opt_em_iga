# opt_em_iga

Optimization of electromagnetic IGA simulations — fast auxiliary-space
preconditioning and reference-space initial guesses for single-patch NURBS
`H(curl)` Maxwell problems in [MFEM](https://mfem.org).

## Overview

This repository contains two complementary prototypes:

| Component | Directory | Purpose |
|---|---|---|
| Reference-space FDFD initial guess | `fdfd_iga_init/` | Geometry-aware finite-difference solve on the parameter domain, projected into the IGA `H(curl)` space as a GMRES initial guess. |
| Covariant auxiliary-space preconditioner | `covariant_aux_space/` | Matrix-based two-level preconditioner `P⁻¹ ≈ Π A_aux⁻¹ Πᵀ` with explicit prolongation/restriction between a structured auxiliary grid and the IGA true-DOF space. |

Three auxiliary prototype modes are supported:

- `nodal_proto`
- `edge_galerkin_proto`
- `edge_yee_proto` — Yee-edge auxiliary space with fast element-driven prolongation assembly

## Dependencies

- **MFEM** (with H(curl), NURBS, and parallel support)
- **Hypre**
- **METIS**
- **OpenMPI**
- **CUDA 12** + **cuDSS** (optional, for GPU-accelerated coarse solves)

The Makefile expects the following layout (override with `WORKSPACE_DIR`):

```
/mnt/f/optemcode/
  mfem/              # MFEM source
  mfem-cpu-build/    # MFEM CPU build
  opt/
    hypre/
    metis/
    openmpi/
```

## Build

```bash
make fdfd_iga_init_demo
```

To build without cuDSS:

```bash
make fdfd_iga_init_demo LDFLAGS="$(make -p | grep LDFLAGS_NO_CUDSS | head -1 | cut -d= -f2-)"
```

## Quick start

### Auxiliary preconditioner (edge_yee_proto)

```bash
LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH \
./fdfd_iga_init_demo \
  --proto-mode edge_yee_proto \
  --epsilon-mode constant \
  --mesh /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -an 7
```

Expected output:

```
[yee_transfer] BuildProlongationFast done seconds=0.7~0.8
Covariant-aux-preconditioned GMRES iterations: 36, converged=1
```

### Reference initial guess only

```bash
./fdfd_iga_init_demo \
  --proto-mode none \
  --mesh /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 1
```

## Key options

| Flag | Description |
|---|---|
| `--proto-mode` | Auxiliary prototype: `nodal_proto`, `edge_galerkin_proto`, `edge_yee_proto`, or `none` |
| `--epsilon-mode` | Material profile: `constant`, `layered_x` |
| `--mesh` | NURBS mesh file |
| `-r N` | Refinement levels |
| `-an N` | Auxiliary grid subdivisions per direction |
| `--order N` | IGA polynomial order |
| `--gmres-max-iter N` | Max GMRES iterations per solve |

## Directory structure

```
opt_em_iga/
  README.md
  Makefile
  fdfd_iga_init/           # Reference-space FDFD initial guess
    README.md
    single_patch_demo.cpp  # End-to-end driver
    reference_fdfd_cpu.*   # CPU finite-difference solver
    reference_fdfd_cuda.*  # CUDA stub interface
    reference_initial_guess.*  # Field projection into H(curl)
    reference_patch_evaluator.*  # NURBS geometry evaluation
  covariant_aux_space/     # Auxiliary-space preconditioner
    README.md
    covariant_reference_preconditioner.*  # Preconditioner orchestrator
    covariant_aux_preconditioner.*        # Preconditioner interface
    yee_transfer.*          # Yee prolongation (fast local assembly)
    yee_operator.*          # Yee coarse operator assembly
  meshes/                  # Example NURBS meshes
  run_proto_benchmark_matrix.sh   # Benchmark script
  run_hp_proto_scan.sh            # HP parameter scan
  run_hp_proto_scan_safe.sh       # HP scan with per-case timeout & logging
```

## Performance note

The `edge_yee_proto` prolongation matrix `Π` is built via element-driven local
assembly of the coupling matrix `B_ij = ∫ φ_i · w_j dx`, followed by a single
mass-matrix backsolve. This avoids per-column global projections and achieves
~700× speedup over the original per-column CG approach while preserving
identical GMRES convergence.
