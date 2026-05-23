# Covariant Auxiliary Space

This directory is reserved for a matrix-based auxiliary-space preconditioner
prototype for Maxwell-type IGA `H(curl)` systems.

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

The first correct implementation stage should provide:

1. A residual-driven entry point `Mult(const Vector &r, Vector &z)`.
2. An explicit transfer operator `Pi`.
3. A reference-space auxiliary solve driven by `Pi^T r`.
4. An algebraic prolongation `Pi z_aux`.

## Planned next steps

1. Build a discrete auxiliary space on the parameter-domain tensor grid.
2. Assemble or apply an explicit transfer matrix `Pi`.
3. Replace the current placeholder residual-field path with `Pi^T r`.
4. Replace the current placeholder auxiliary solve with a structured
   reference-space `curl-curl` solve.
5. Add a right-preconditioning wrapper for MFEM complex `H(curl)` systems.
