# Mathematical Proof for the Yee-Space Galerkin Coarse Correction

## 1. Algebraic Setting

Let the assembled IGA H(curl) Maxwell system be

```text
A_h x = b,
```

where

```text
A_h in R^{n x n}
```

is the real block matrix used by GMRES. For a complex Maxwell system, `A_h`
may be the real-valued `2 x 2` block representation of the complex operator.

Let the Yee-derived auxiliary edge space have dimension `m`, and let

```text
Pi in R^{n x m}
```

be the prolongation from Yee edge auxiliary unknowns into the IGA true-dof
space.

The Galerkin coarse operator is

```text
A_c = Pi^T A_h Pi  in R^{m x m}.
```

The corresponding coarse correction operator is

```text
B_c = Pi A_c^{-1} Pi^T.
```

When applied to the residual `r`, it produces

```text
delta x = B_c r = Pi (Pi^T A_h Pi)^{-1} Pi^T r.
```

This is the `edge_galerkin` coarse correction.

## 2. Exact Error Removal on Range(Pi)

Assume `A_c = Pi^T A_h Pi` is invertible.

Let the current error be

```text
e = x_* - x,
```

where `x_*` is the exact solution. The residual is

```text
r = b - A_h x = A_h e.
```

The edge Galerkin correction is

```text
delta x = Pi A_c^{-1} Pi^T A_h e.
```

Now suppose the error lies in the Yee-derived auxiliary subspace:

```text
e in Range(Pi).
```

Then there exists `y in R^m` such that

```text
e = Pi y.
```

Substitute this into the correction:

```text
delta x
= Pi A_c^{-1} Pi^T A_h Pi y
= Pi (Pi^T A_h Pi)^{-1} (Pi^T A_h Pi) y
= Pi y
= e.
```

Therefore the new error after applying the correction is

```text
e_new = e - delta x = 0.
```

This proves:

```text
The edge Galerkin coarse correction exactly removes all error components in
Range(Pi).
```

This statement does not require `Pi` to be square. It only requires the coarse
operator `Pi^T A_h Pi` to be invertible on the auxiliary space.

## 3. Error Propagation Operator

The error propagation operator for one coarse correction is

```text
E_c = I - B_c A_h
    = I - Pi (Pi^T A_h Pi)^{-1} Pi^T A_h.
```

For any `e = Pi y`,

```text
E_c e
= Pi y - Pi (Pi^T A_h Pi)^{-1} Pi^T A_h Pi y
= Pi y - Pi y
= 0.
```

Thus:

```text
Range(Pi) subset Null(E_c).
```

This is the precise algebraic meaning of saying that `edge_galerkin` eliminates
the error components represented by the Yee auxiliary space.

## 4. Square and Invertible Pi

Now consider the special case

```text
m = n,
Pi in R^{n x n},
Pi is invertible.
```

Then

```text
A_c = Pi^T A_h Pi.
```

If `A_h` is invertible, then `A_c` is also invertible, and

```text
A_c^{-1}
= (Pi^T A_h Pi)^{-1}
= Pi^{-1} A_h^{-1} Pi^{-T}.
```

Therefore

```text
B_c
= Pi A_c^{-1} Pi^T
= Pi (Pi^{-1} A_h^{-1} Pi^{-T}) Pi^T
= A_h^{-1}.
```

So in the square full-rank case,

```text
Pi (Pi^T A_h Pi)^{-1} Pi^T = A_h^{-1}.
```

This proves:

```text
If Pi is an invertible coordinate transform and A_h is invertible, then the
edge Galerkin correction is exactly the inverse of the original system.
```

In that case, the method is no longer merely a coarse correction. It is
equivalent to solving the original system in a transformed basis.

## 5. What Happens to Nullspaces?

The above result does not mean a coordinate transform can remove a true
mathematical nullspace of the original operator.

If `Pi` is square and invertible, then

```text
A_c = Pi^T A_h Pi
```

is congruent to `A_h`. Congruence by an invertible matrix preserves rank:

```text
rank(A_c) = rank(A_h).
```

Therefore:

```text
A_h is singular  <=>  A_c is singular.
```

So if the original Maxwell matrix has a true nullspace because of the PDE,
boundary conditions, gauge freedom, or an exact resonance, then the Galerkin
coarse operator cannot remove that true singularity.

What it can remove is different:

```text
It removes the near-nullspace / low-energy error components that are represented
inside Range(Pi).
```

This is why it is better to say:

```text
edge_galerkin captures and eliminates Yee-representable low-energy H(curl)
error components.
```

rather than:

```text
edge_galerkin removes the mathematical nullspace of A_h.
```

## 6. Rectangular Pi and the Need for a Smoother

In the practical auxiliary-space method,

```text
m << n.
```

For example, in the current PML test:

```text
n = 882 true dofs per real/imag block
m = 108 Yee auxiliary edge dofs
```

Then `Pi` cannot span the full IGA true-dof space. The correction

```text
B_c = Pi (Pi^T A_h Pi)^{-1} Pi^T
```

is a low-rank coarse correction.

It is exact on `Range(Pi)`, but it does not directly correct error components
outside `Range(Pi)`.

Therefore the complete auxiliary-space preconditioner should include a full-rank
smoother:

```text
M^{-1} = S^{-1} + Pi A_c^{-1} Pi^T.
```

Here:

```text
S^{-1}
```

handles the components not covered by the Yee auxiliary space, while

```text
Pi A_c^{-1} Pi^T
```

handles the low-energy / near-nullspace components represented in the Yee
auxiliary space.

This explains the PML observations:

```text
Pi A_c^{-1} Pi^T alone reduces the preconditioned residual but can leave a
large true residual.

Adding a full-rank smoother reduces the true residual.
```

## 7. Comparison with Independent Yee Coarse Operator

The `edge_galerkin` coarse operator is

```text
A_c = Pi^T A_h Pi.
```

The independent `edge_yee` coarse operator is

```text
A_c approx A_yee.
```

The exact error-removal proof above applies directly to `edge_galerkin` because
its coarse operator is exactly `Pi^T A_h Pi`.

For `edge_yee`, the correction is

```text
B_y = Pi A_yee^{-1} Pi^T.
```

For an error `e = Pi y`,

```text
delta x
= Pi A_yee^{-1} Pi^T A_h Pi y.
```

This equals `e = Pi y` only if

```text
A_yee^{-1} Pi^T A_h Pi y = y,
```

or equivalently

```text
A_yee y = Pi^T A_h Pi y.
```

Thus `edge_yee` exactly removes the same coarse-space errors only when

```text
A_yee = Pi^T A_h Pi
```

on the relevant subspace. In practice, `A_yee` is a cheaper structured
approximation, so this equality is not exact.

This is why:

```text
edge_galerkin validates the auxiliary space and gives an ideal reference.
edge_yee tests whether a cheap Yee/FDFD operator can approximate that ideal
coarse solve.
```

## 8. Consequence for the Paper

The main mathematical contribution can be stated as:

```text
We construct a Yee-derived H(curl) auxiliary edge space and a covariant
prolongation Pi into the IGA true-dof space. The Galerkin coarse correction
Pi (Pi^T A_h Pi)^{-1} Pi^T exactly eliminates all error components in Range(Pi).
```

For a complete preconditioner, especially when `m << n`, the method should be
presented as:

```text
M^{-1} = S^{-1} + Pi (Pi^T A_h Pi)^{-1} Pi^T.
```

The cheaper Yee/FDFD variant is:

```text
M^{-1} = S^{-1} + Pi A_yee^{-1} Pi^T.
```

The first is the robust Galerkin auxiliary-space method.

The second is a structured approximation to the Galerkin coarse solve.

## 9. Summary

1. `edge_galerkin` exactly eliminates all error components in `Range(Pi)`.
2. If `Pi` is square and invertible and `A_h` is nonsingular, then
   `edge_galerkin` is algebraically equivalent to applying `A_h^{-1}`.
3. If `A_h` has a true nullspace, an invertible coordinate transform cannot
   remove that nullspace.
4. In the practical case `m << n`, `edge_galerkin` is a low-rank coarse
   correction and needs a full-rank smoother for the remaining components.
5. `edge_yee` is cheaper because it replaces `Pi^T A_h Pi` by `A_yee`, but then
   the exact error-removal property becomes approximate.

