# Core Benchmark Results

## Scope

This note summarizes the current core benchmark matrix for the auxiliary-space
prototypes:

- `nodal_proto`
- `edge_galerkin_proto`
- `edge_yee_proto`

The goal is to compare:

1. zero initial guess
2. direct reference-FDFD initial guess
3. transitional nodal auxiliary-space prototype
4. ideal edge-based Galerkin coarse-space prototype
5. independent edge-based Yee auxiliary-space prototype

All runs below use:

- `order = 2`
- `ref_levels = 2`
- `fd_n = 41`
- `aux_n = 7`
- `source_mode = gaussian_y`
- `wavelength = 0.2`
- `eps_r = 8`

## Benchmark Matrix

### Flat single-patch cube

Mesh:

- `/mnt/d/code/code_opt_em/mfem/data/cube-nurbs.mesh`

#### Constant permittivity

- zero initial guess: `79`
- direct FDFD initial guess: `88`
- `nodal_proto`: `40`
- `edge_galerkin_proto`: `1`
- `edge_yee_proto`: `36`

#### Layered permittivity in x

- zero initial guess: `172`
- direct FDFD initial guess: `222`
- `nodal_proto`: `93`
- `edge_galerkin_proto`: `1`
- `edge_yee_proto`: `47`

### Warped single-patch cube

Mesh:

- `/mnt/d/code/code_opt_em/opt_em_iga/meshes/warped-cube-singlepatch-nurbs.mesh`

This is a non-affine single-patch NURBS test geometry derived from
`cube-nurbs.mesh` by perturbing the top control points.

#### Constant permittivity

- zero initial guess: `279`
- direct FDFD initial guess: `294`
- `nodal_proto`: `144`
- `edge_galerkin_proto`: `1`
- `edge_yee_proto`: `62`

#### Layered permittivity in x

- zero initial guess: `294`
- direct FDFD initial guess: `338`
- `nodal_proto`: `163`
- `edge_galerkin_proto`: `1`
- `edge_yee_proto`: `66`

## Compact Table

| Mesh | Epsilon mode | Zero | FDFD init | `nodal_proto` | `edge_galerkin_proto` | `edge_yee_proto` |
|---|---:|---:|---:|---:|---:|---:|
| `cube-nurbs` | `constant` | 79 | 88 | 40 | 1 | 36 |
| `cube-nurbs` | `layered_x` | 172 | 222 | 93 | 1 | 47 |
| `warped-cube-singlepatch` | `constant` | 279 | 294 | 144 | 1 | 62 |
| `warped-cube-singlepatch` | `layered_x` | 294 | 338 | 163 | 1 | 66 |

## Main Observations

### 1. Direct FDFD initial guesses are consistently inferior

For all tested cases, the direct reference-FDFD initial guess requires more
GMRES iterations than a zero initial guess. This confirms that the reference
FDFD solver is not effective as a standalone initialization strategy in the
current formulation.

### 2. `nodal_proto` is useful, but transitional

`nodal_proto` improves all tested cases relative to zero initialization.
However, it is not the final theory-aligned construction and should be treated
as a baseline transitional auxiliary-space prototype.

### 3. `edge_galerkin_proto` remains the ideal upper bound

`edge_galerkin_proto` solves all four benchmark cases in one GMRES iteration.
This does not validate the independent Yee coarse operator directly, but it
strongly validates the quality of the edge-oriented transfer `P` and provides a
clear upper-bound reference for the coarse-space effect.

### 4. `edge_yee_proto` is now consistently effective

`edge_yee_proto` improves all four benchmark cases and outperforms
`nodal_proto` in each one:

- `40 -> 36`
- `93 -> 47`
- `144 -> 62`
- `163 -> 66`

This is the strongest current positive result.

### 5. Material and geometry robustness are both visible

Two trends are especially important:

- Material variation:
  - `cube-nurbs`, `constant`: `79 -> 36`
  - `cube-nurbs`, `layered_x`: `172 -> 47`

- Geometry variation:
  - `cube-nurbs`, `constant`: `79 -> 36`
  - `warped-cube`, `constant`: `279 -> 62`

The independent Yee auxiliary operator remains useful when the problem becomes
substantially harder due to either variable material coefficients or non-affine
geometry.

## Interpretation

At this stage, the evidence supports the following narrative:

1. The edge-based transfer `P` is strong and has been validated in the
   Galerkin coarse-space setting.
2. The explicit Yee-topology branch is no longer merely a diagnostic toy.
3. The current independent `edge_yee_proto` is already a competitive and
   theory-aligned auxiliary-space preconditioner.
4. The method shows promising robustness with respect to both material
   heterogeneity and geometric distortion in the tested single-patch setting.

## Limits of This Benchmark Set

The present results are still limited to:

- single patch
- one spline order (`p = 2`)
- one refinement level family
- one source family
- a small number of material and geometry patterns

These results do **not** yet establish:

- full `h`-robustness
- full `p`-robustness
- broad comparison against AMS
- commuting-property theory

## Recommended Next Step

The next experimental phase should focus on controlled `h/p` studies of:

- `nodal_proto`
- `edge_yee_proto`

with the same benchmark families:

- flat cube
- warped single-patch cube
- constant and layered permittivity

The main goal is to determine how the performance gap between
`nodal_proto` and `edge_yee_proto` evolves under refinement and order increase.
