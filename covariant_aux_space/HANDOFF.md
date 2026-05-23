# Handoff Note

## Purpose

This file is the shortest practical handoff for continuing the current
`covariant_aux_space` research on another machine or in a fresh Codex session.

If you are a new Codex session, read this file first, then read:

1. `covariant_aux_space/PROTO_MODES_NOTE.md`
2. `covariant_aux_space/EDGE_YEE_STAGE_SUMMARY.md`
3. `covariant_aux_space/CORE_BENCHMARK_RESULTS.md`

## Project Context

We are working in:

- `/mnt/d/code/code_opt_em/opt_em_iga`

Main goal:

- develop an auxiliary-space preconditioner for spline-based NURBS `H(curl)`
  Maxwell IGA
- the intended theory target is an independent mapped Yee auxiliary operator:

`M^{-1} = S^{-1} + P A_Y^{-1} P^T`

with:

- `P`: transfer from parameter-domain Yee edge space to spline `H(curl)` space
- `A_Y`: parameter-domain mapped Yee Maxwell operator

## Important Directories

- `fdfd_iga_init/`
  - demo driver
  - reference FDFD solver
  - reference-field projector
- `covariant_aux_space/`
  - auxiliary-space preconditioner code
  - Yee transfer and Yee operator code
- `meshes/`
  - custom single-patch warped test mesh

## Important Files

### Driver

- `fdfd_iga_init/single_patch_demo.cpp`

### Auxiliary-space implementation

- `covariant_aux_space/covariant_reference_preconditioner.hpp`
- `covariant_aux_space/covariant_reference_preconditioner.cpp`
- `covariant_aux_space/yee_transfer.hpp`
- `covariant_aux_space/yee_transfer.cpp`
- `covariant_aux_space/yee_operator.hpp`
- `covariant_aux_space/yee_operator.cpp`

### Documentation

- `covariant_aux_space/PROTO_MODES_NOTE.md`
- `covariant_aux_space/EDGE_YEE_STAGE_SUMMARY.md`
- `covariant_aux_space/CORE_BENCHMARK_RESULTS.md`
- `covariant_aux_space/TRANSFER_VALIDATION_NOTE.md`
- `covariant_aux_space/STAGE_SUMMARY.md`

### Helper script

- `run_proto_benchmark_matrix.sh`

### Custom mesh

- `meshes/warped-cube-singlepatch-nurbs.mesh`

## Prototype Modes

There are three prototype modes:

- `nodal_proto`
  - transitional baseline
- `edge_galerkin_proto`
  - uses the same edge transfer `P`
  - coarse operator is `P^T A P`
  - serves as ideal coarse-space upper bound and transfer validation
- `edge_yee_proto`
  - main research target
  - uses the same edge transfer `P`
  - uses independently assembled mapped Yee auxiliary operator `A_Y`

## Current Main Conclusion

The strongest current positive result is:

- `edge_yee_proto` is now consistently effective across:
  - flat single-patch cube
  - warped single-patch cube
  - constant permittivity
  - layered permittivity

It outperforms:

- zero initial guess
- direct reference-FDFD initial guess
- `nodal_proto`

`edge_galerkin_proto` remains the idealized upper bound and validates that the
current edge transfer `P` is strong.

## Core Benchmark Results

These are the current reference iteration counts:

| Mesh | Epsilon mode | Zero | FDFD init | `nodal_proto` | `edge_galerkin_proto` | `edge_yee_proto` |
|---|---:|---:|---:|---:|---:|---:|
| `cube-nurbs` | `constant` | 79 | 88 | 40 | 1 | 36 |
| `cube-nurbs` | `layered_x` | 172 | 222 | 93 | 1 | 47 |
| `warped-cube-singlepatch` | `constant` | 279 | 294 | 144 | 1 | 62 |
| `warped-cube-singlepatch` | `layered_x` | 294 | 338 | 163 | 1 | 66 |

Interpretation:

- direct FDFD initial guess is not the right research direction
- auxiliary-space preconditioning is the right direction
- `edge_yee_proto` now shows both material robustness and geometry robustness in
  the tested single-patch setting

## Current Implementation Status

### Transfer `P`

- `P` is built through `YeeTransferBuilder`
- `edge_galerkin_proto` strongly suggests that `P` is already good
- do not spend time trying to revive direct FDFD initialization as the main path

### Yee operator `A_Y`

Current `edge_yee_proto` now uses:

- explicit edge dofs
- explicit face dofs
- explicit curl incidence `C_Y`
- cochain-style diagonal Hodge stars
  - face Hodge for `muhat^{-1}`
  - edge Hodge for `epshat`

This was a major step forward. Earlier surrogate coarse operators were much
worse.

## Most Useful Commands

### Rebuild

```bash
cd /mnt/d/code/code_opt_em/opt_em_iga
make rebuild_fdfd_iga_init_demo
```

### Single case

```bash
./fdfd_iga_init_demo --proto-mode edge_yee_proto --epsilon-mode constant
./fdfd_iga_init_demo --proto-mode edge_yee_proto --epsilon-mode layered_x
./fdfd_iga_init_demo --proto-mode edge_yee_proto --mesh /mnt/d/code/code_opt_em/opt_em_iga/meshes/warped-cube-singlepatch-nurbs.mesh --epsilon-mode constant
./fdfd_iga_init_demo --proto-mode edge_yee_proto --mesh /mnt/d/code/code_opt_em/opt_em_iga/meshes/warped-cube-singlepatch-nurbs.mesh --epsilon-mode layered_x
```

### Core benchmark matrix

```bash
./run_proto_benchmark_matrix.sh
```

### Coarse-operator diagnostics

```bash
./fdfd_iga_init_demo --proto-mode edge_yee_proto --diagnose-coarse
```

## Environment Notes

Runtime may need:

- OpenMPI libs:
  - `/mnt/d/code/code_opt_em/opt/openmpi/lib`
- Hypre libs:
  - `/mnt/d/code/code_opt_em/opt/hypre/lib`
- CUDA runtime libs depending on machine setup

If the binary fails with `libcusparse.so.12` or related errors, fix the runtime
library visibility first.

## Recommended Next Step

The next phase should be a controlled `h/p` study focusing on:

- `nodal_proto`
- `edge_yee_proto`

Use the same benchmark families:

- flat cube
- warped single-patch cube
- constant permittivity
- layered permittivity

Suggested parameters to scan:

- `order = 2, 3`
- `ref_levels = 1, 2, 3`
- keep `fd_n` and `aux_n` fixed initially unless there is a clear reason to
  revisit them

The main goal is to quantify how the gap between `nodal_proto` and
`edge_yee_proto` evolves under refinement and order increase.

## What Not To Redo

Do not spend time re-proving these points unless something broke:

1. direct reference-FDFD initial guess is inferior
2. `edge_galerkin_proto` is an upper bound, not the final method
3. `P` is already strong enough to justify continuing with `edge_yee_proto`

The active research problem is now:

- how far the independent mapped Yee operator can go
- and how robust it remains under harder spline Maxwell IGA settings
