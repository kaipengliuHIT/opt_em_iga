# Covariant Auxiliary Space

This directory contains a matrix-based auxiliary-space preconditioner prototype
for Maxwell-type IGA `H(curl)` systems.

The intended discrete operator form is

`P^{-1} ~= Pi A_aux^{-1} Pi^T`

where:

- `Pi` is a discrete prolongation from auxiliary-space unknowns to MFEM true
  DOFs.
- `A_aux` is a covariant reference-space Maxwell operator assembled with the
  pull-back geometry tensors induced by the NURBS map.
- `Pi^T` is the algebraic restriction of residuals into the auxiliary space.

## Scope of this prototype

This folder is not another "initial guess" experiment. It is meant for a true
auxiliary-space preconditioner that can later be used inside `GMRES/FGMRES`.

The currently supported prototype modes are:

1. `nodal_proto`
2. `edge_galerkin_proto`
3. `edge_yee_proto`

The `edge_yee_proto` path builds a Yee-edge auxiliary space on the reference
grid and uses an explicit prolongation matrix from Yee edge unknowns into the
IGA `H(curl)` true DOF space.

## Important algorithmic note

The earlier prototype treated a Krylov residual as if it were directly a
physical `H(curl)` field, then sampled and pull-backed that field into the
reference domain. That is not correct.

The residual `r` of the linear system lives in the algebraic dual space. A
correct auxiliary preconditioner must first apply an explicit discrete transfer
operator:

`r_aux = Pi^T r`

Only then does the auxiliary solve make sense:

`z_aux = A_aux^{-1} r_aux`

and the correction is lifted back algebraically:

`z = Pi z_aux`

The implemented preconditioner path provides:

1. A residual-driven entry point `Mult(const Vector &r, Vector &z)`.
2. An explicit transfer operator `Pi`.
3. A reference-space auxiliary solve driven by `Pi^T r`.
4. An algebraic prolongation `Pi z_aux`.

## Fast Yee transfer construction

The `edge_yee_proto` transfer is built in `YeeTransferBuilder` in
`yee_transfer.cpp`.

The earlier implementation formed each column of `Pi` by a separate global
mass projection:

`M_h p_j = b_j`

for every Yee edge basis function `w_j`. This was correct but slow because it
assembled a global right-hand side and ran a global solve for every column.

The current implementation constructs the coupling matrix

`B_ij = int phi_i . w_j dx`

with an element-driven local assembly:

1. Iterate over IGA elements and quadrature points.
2. Map each quadrature point to the reference Yee cell.
3. Use the 12 locally supported Yee edge basis functions of that cell.
4. Accumulate the local contributions into `B`.
5. Apply a single mass-matrix inverse to obtain `Pi = M_h^{-1} B`.

This avoids per-column global projections and preserves the same mass-projected
transfer used by the baseline algorithm.

Example validation command:

```bash
./fdfd_iga_init_demo \
  --proto-mode edge_yee_proto \
  --epsilon-mode constant \
  --mesh /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -an 7
```

Expected behavior on the reference cube test:

- `BuildProlongationFast` completes in about one second or less for
  `true_vsize=882, edge_dofs=450`.
- The printed `P` column norms are nonzero.
- The covariant auxiliary preconditioner converges, e.g. about 36 GMRES
  iterations for the command above.

## Planned next steps

1. Add broader benchmark coverage for higher orders, refinements, and
   heterogeneous material profiles.
2. Replace dense auxiliary inverses with scalable structured or sparse
   reference-space solvers.
3. Add a right-preconditioning wrapper for MFEM complex `H(curl)` systems.

## Paper benchmark workflow

Run the current four core cases with:

```bash
../run_core_paper_cases.sh
```

The script runs the paper-facing `edge_yee_proto` cases on:

- `cube-nurbs.mesh` with `constant`
- `cube-nurbs.mesh` with `layered_x`
- `warped-cube-singlepatch-nurbs.mesh` with `constant`
- `warped-cube-singlepatch-nurbs.mesh` with `layered_x`

It stores the raw CSV, logs, and a compact Markdown table under this directory.
The Markdown table uses the baseline structure `zero`, `FDFD init`, Hypre AMS,
and `edge_yee_proto`.
