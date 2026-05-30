# opt_em_iga

Adaptive preconditioning and fast solvers for isogeometric analysis (IGA)
of `H(curl)` Maxwell problems in [MFEM](https://mfem.org).

Supports **real SPD** (cavity resonance) and **complex PML** (open-domain
radiation) systems on single-patch NURBS geometries.

## Overview

This repository develops a **diagnostic-driven adaptive preconditioning
framework** for IGA Maxwell equations. Instead of searching for one universal
preconditioner, we probe the system at runtime and automatically select the best
preconditioner for the problem at hand.

### Project components

| Component | Directory / File | System |
|---|---|---|
| **Adaptive preconditioner selector** | `prec_selector.cpp` | SPD + PML |
| IGA AMS (Auxiliary-space Maxwell Solver) preconditioners | `iga_ams_preconditioner/` | SPD |
| PML Pi-lumped auxiliary preconditioner tests | `pml_pi_prec_test.cpp` | PML |
| PML p-multigrid bridge prototype | `pml_split_preconditioner/` | PML |
| PML point-source demo | `pml_point_source_demo.cpp` | PML |
| Covariant Yee auxiliary-space preconditioner | `covariant_aux_space/` | SPD |
| Reference-space FDFD initial guess | `fdfd_iga_init/` | SPD |
| Yee-to-IGA initial guess projection | `yee_init_guess/` | SPD |
| IGA patch-RAS Schwarz preconditioner | `covariant_aux_space/iga_patch_ras_preconditioner.*` | SPD |
| GPU BiCGStab iterative solver | `cuda_iterative_solver/` | SPD + PML |

## Key Scientific Findings

### 1. Adaptive preconditioner selection is feasible

The `prec_selector` probes 4 candidate preconditioners (Jacobi, Pi_lumped+Jacobi,
Jacobi→Pi multiplicative, Pi→Jacobi multiplicative) at 3 damping weights each
(12 total) using two selector schemes:

- **Scheme A (one-step residual probe)**: `ρ = |r₀ − A·B⁻¹·r₀| / |r₀|` — pick smallest ρ
- **Scheme B (10-step warm-up probe)**: `score = log(|r₀|/|r₁₀|) / (t_setup + t_probe)` — pick highest score

**Results across 5 test cases**:

| Case | System | r | o | f | One-step selects | Warmup selects | Best iters | None iters |
|------|--------|---|---|---|-----------------|----------------|------------|------------|
| 1 | SPD | 2 | 2 | 5 | Pi→Jac ω=1.0 (27) | Jacobi ω=0.5 (71) | **27** | 80 |
| 2 | SPD | 3 | 2 | 5 | Pi→Jac ω=1.0 (49) | Jacobi ω=0.7 (186) | **49** | 229 |
| 3 | SPD | 2 | 3 | 5 | Pi→Jac ω=1.0 (46) | Jacobi ω=0.7 (139) | **46** | 159 |
| 4 | SPD | 3 | 3 | 5 | Jac→Pi ω=1.0 (80) | Jacobi ω=1.0 (432) | **80** | 500 |
| 5 | PML | 2 | 2 | 5 | Pi_lumped+Jac (1000✗) | Jacobi(cpx) ω=0.7 (176) | **176** | 574 |

- **One-step probe**: reliable for SPD (correctly selects Pi-based methods),
  unreliable for PML point-source (misled by Pi false-positive ρ < 1)
- **Warmup probe**: reliable for PML, but overweights Pi setup cost for SPD

### 2. Pi_lumped p-auxiliary preconditioner is effective for SPD, ineffective for PML

- **SPD**: `Π_{p−1→p}` auxiliary space correction + Jacobi yields 1.9−3.3× speedup
  - `Π_{p−1→p}` is the correct rule (NOT `Π_{1→p}`) for high-order IGA
- **PML point-source**: Pi_lumped correction is ineffective (488−5000+ iters);
  complex Jacobi alone is the best choice (176 iters, 3.26× vs none)

### 3. Multiplicative composition beats additive

Pi→Jacobi and Jacobi→Pi multiplicative forms consistently give 10−20% fewer
iterations than the additive Pi_lumped+Jacobi form across all SPD cases.

## Build

```bash
# Adaptive preconditioner selector (SPD + PML)
make prec_selector

# IGA AMS sweep
make ams_sweep_v3

# PML Pi preconditioner tests
make pml_pi_prec_test

# PML point-source demo
make pml_point_source_demo

# Covariant Yee auxiliary preconditioner
make fdfd_iga_init_demo

# Yee-to-IGA initial guess
make yee_init_guess_demo

# IGA patch-RAS Schwarz preconditioner
make iga_ras_benchmark
```

To build without cuDSS:

```bash
make prec_selector LDFLAGS="$(make -p | grep LDFLAGS_NO_CUDSS | head -1 | cut -d= -f2-)"
```

## Quick Start

### Environment

```bash
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH
```

### Adaptive preconditioner selector

```bash
# SPD cavity resonance (Cases 1-4)
./prec_selector -system spd -r 2 -o 2 -f 5 -ao 1    # Case 1
./prec_selector -system spd -r 3 -o 2 -f 5 -ao 1    # Case 2
./prec_selector -system spd -r 2 -o 3 -f 5 -ao 2    # Case 3 (Π_{2→3})
./prec_selector -system spd -r 3 -o 3 -f 5 -ao 2    # Case 4 (Π_{2→3})

# PML open-domain radiation (Case 5)
./prec_selector -system pml -r 2 -o 2 -f 5 -ao 1
```

### IGA AMS preconditioners

```bash
./ams_sweep_v3 -r 2 -o 2 -f 5    # SPD Pi_lumped sweep
```

### PML point-source demo

```bash
./pml_point_source_demo -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh -r 2 -o 2 -f 5
```

### Benchmark scripts

```bash
./run_core_paper_cases.sh              # Core paper cases (FDFD init + Yee aux)
./run_full_preconditioner_benchmark.sh  # Full preconditioner benchmark matrix
./run_sweep3.sh                         # SPD Pi_lumped HP sweep
./run_pml_hp_sweep.sh                   # PML HP parameter sweep
./run_iga_ras_benchmark.sh             # IGA patch-RAS Schwarz benchmark
```

## Dependencies

- **MFEM** (with H(curl), NURBS, and parallel support)
- **Hypre**
- **METIS**
- **OpenMPI**
- **CUDA 12** + **cuDSS** (optional, for GPU-accelerated coarse solves)

The Makefile expects the following layout:

```
/mnt/f/optemcode/
  mfem/              # MFEM source
  mfem-cpu-build/    # MFEM CPU build
  opt/
    hypre/
    metis/
    openmpi/
```

## Key Options (prec_selector)

| Flag | Description |
|---|---|
| `-system spd\|pml` | System type: real SPD cavity or complex PML radiation |
| `-r N` | Mesh refinement levels |
| `-o N` | NURBS polynomial order (fine space) |
| `-f N` | Frequency (Hz) |
| `-ao N` | Auxiliary NURBS order (default: 1; use o−1 for Π_{p−1→p}) |
| `-m <file>` | NURBS mesh file (default: cube-nurbs.mesh) |

## Directory Structure

```
opt_em_iga_repo/
  README.md
  PROGRESS.md                         # Detailed research progress log
  Makefile
  prec_selector.cpp                   # ★ Adaptive preconditioner selector (SPD + PML)
  pml_pi_prec_test.cpp                # PML Pi-lumped preconditioner tests
  pml_point_source_demo.cpp           # PML point-source radiation demo
  plot_field.py                       # Field visualization helper
  iga_ams_preconditioner/             # IGA AMS preconditioners
    ams_sweep_v3.cpp                  #   SPD Pi_lumped HP sweep
    ams_algebraic_diag.cpp            #   Algebraic diagonal diagnostics
    ams_diagnostics.cpp               #   AMS diagnostics
    ams_iga_coupled.cpp               #   Coupled IGA AMS
    ams_q_preconditioner.cpp          #   Q-preconditioner variant
    iga_ams_prec.*                    #   IGA AMS preconditioner library
  covariant_aux_space/                # Covariant Yee auxiliary-space preconditioner
    covariant_reference_preconditioner.*
    covariant_aux_preconditioner.*
    yee_transfer.*                    #   Yee prolongation (fast local assembly)
    yee_operator.*                    #   Yee coarse operator assembly
    iga_patch_ras_preconditioner.*    #   IGA patch-RAS Schwarz
    IGA_NATIVE_SCHWARZ_PRECONDITIONER.md
  pml_split_preconditioner/           # PML p-multigrid bridge prototype
    pml_split_demo.cpp
    split_pml_prec.*
  fdfd_iga_init/                      # Reference-space FDFD initial guess
  yee_init_guess/                     # Yee-to-IGA initial guess projection
  cuda_iterative_solver/              # GPU BiCGStab solver sandbox
  reference_precon/                   # Reference preconditioner design docs
  meshes/                             # Example NURBS meshes
  benchmark_results/                  # Benchmark output summaries
  tools/                              # CSV/Markdown summary helpers
  run_*.sh                            # Benchmark and sweep scripts
```

## Documentation

- **[PROGRESS.md](PROGRESS.md)** — Full research progress log with all sessions
- **[HOW_TO_USE.md](HOW_TO_USE.md)** — Usage guide for key targets
- **[SCI_PAPER_STRATEGY.md](SCI_PAPER_STRATEGY.md)** — Paper strategy and narrative
- **[SCI_PAPER_PLAN.md](SCI_PAPER_PLAN.md)** — Detailed paper plan
- **[IGA_RAS_PAPER_AND_API.md](IGA_RAS_PAPER_AND_API.md)** — IGA patch-RAS paper and API design
