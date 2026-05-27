# Yee-Space Galerkin and Independent Yee Coarse Operators

## 1. Basic Distinction

There are two different uses of the Yee grid in the current auxiliary-space
preconditioner study.

The first use is to define an auxiliary edge space:

```text
Yee edge dofs  --Pi-->  IGA H(curl) true dofs
```

The second use is to define an independent FDFD/Yee coarse operator:

```text
A_yee : Yee edge dofs -> Yee edge dual dofs
```

These two uses should not be confused. The Yee grid can be useful even when the
independent Yee FDFD operator is not used.

## 2. Edge Galerkin Method

The `edge_galerkin` method uses the Yee grid to define the auxiliary space, but
it does not use the independent Yee FDFD stiffness matrix.

Its coarse operator is

```text
A_gal = Pi^T A_h Pi
```

where:

- `A_h` is the original IGA Maxwell operator.
- `Pi` maps Yee edge auxiliary unknowns into the IGA H(curl) true-dof space.
- `Pi^T` restricts algebraic residuals from the IGA space back to the Yee
  auxiliary space.

Thus `A_gal` is still a matrix on Yee edge dofs, but its entries are computed by
projecting the full IGA operator onto the Yee-derived auxiliary space.

This is why `edge_galerkin` can be described as:

```text
Yee auxiliary edge space + exact Galerkin restriction of the IGA operator
```

It is the expensive but consistent version.

## 3. Independent Edge Yee Method

The `edge_yee` method uses the same Yee edge auxiliary space and the same
transfer `Pi`, but replaces the Galerkin coarse operator by an independently
assembled Yee/FDFD operator:

```text
A_aux = A_yee
```

The correction is then

```text
z = Pi A_yee^{-1} Pi^T r
```

This is the cheap version. It avoids the expensive projection

```text
Pi^T A_h Pi
```

but introduces an additional approximation error: `A_yee` must approximate the
action of the true Galerkin coarse operator on the Yee auxiliary space.

The key comparison is therefore:

```text
ideal:      A_gal = Pi^T A_h Pi
cheap:      A_yee
question:   does A_yee approximate A_gal well enough?
```

## 4. Relation to H(curl) Nullspace and Near-Nullspace

The Yee grid is not only a way to build an FDFD operator. It is also a way to
define a compatible edge-oriented auxiliary space.

In H(curl) Maxwell systems, difficult error components are often associated with
low-energy curl-curl modes, gradient-like components, or other near-nullspace
structures. The role of the Yee-derived auxiliary space is to capture these
components through `Range(Pi)`.

The edge Galerkin correction is

```text
M_coarse^{-1} = Pi (Pi^T A_h Pi)^{-1} Pi^T
```

This can reduce the difficult H(curl) error components if `Range(Pi)` contains a
good approximation to those components.

Therefore, when `edge_galerkin` works well, the conclusion is:

```text
The Yee-derived auxiliary space and Pi transfer are useful.
```

It does not prove that the independent Yee FDFD operator is accurate. It proves
that the auxiliary space itself is meaningful.

## 5. Why Edge Galerkin Is an Upper Bound

`edge_galerkin` removes the error caused by approximating the coarse operator.
It uses the exact restriction of the original IGA operator:

```text
Pi^T A_h Pi
```

Thus it answers the question:

```text
If we choose this Yee-derived auxiliary space, how good can the coarse
correction be?
```

By contrast, `edge_yee` answers:

```text
Can an independently assembled Yee FDFD operator reproduce enough of that ideal
coarse correction?
```

This is why `edge_galerkin` should be treated as an ideal coarse-space reference
or upper bound, not as the final cheap method.

## 6. PML Case

In the PML demo, the current `edge_yee` path has three possible ingredients.

First, the Yee-PML coarse operator:

```text
A_yee,PML = C^T M_mu^{-1,PML} C + k0^2 M_eps,PML
```

This uses PML stretch weights in the Yee Hodge terms. The sign is positive in
the mass term because this follows the positive-definite PML preconditioner
style used by the MFEM ex25p/AMS path.

Second, the optional Galerkin fallback:

```text
A_aux = Pi^T A_h Pi
```

When fallback is enabled, the method still uses the Yee auxiliary space and
transfer `Pi`, but it does not use the independent Yee-PML matrix as the coarse
operator.

Third, the full-rank smoother:

```text
S^{-1} r
```

The pure coarse correction

```text
Pi A_aux^{-1} Pi^T r
```

is low rank. It only corrects components represented in `Range(Pi)`. The PML
experiments showed that this is not enough for the true residual, even when the
preconditioned residual reported by GMRES appears to converge.

The complete auxiliary-space structure should therefore be

```text
M^{-1} = S^{-1} + Pi A_aux^{-1} Pi^T
```

where `A_aux` may be either:

```text
A_aux = Pi^T A_h Pi      expensive Galerkin coarse operator
```

or

```text
A_aux = A_yee            cheap independent Yee/FDFD coarse operator
```

## 7. Current Experimental Interpretation

The current evidence supports the following interpretation.

For non-PML cavity cases:

- The independent Yee coarse operator can be effective.
- The Yee-derived auxiliary space and transfer `Pi` are useful.
- Knot-aligned Yee grids, especially one Yee cell per knot span, are important.

For PML cases:

- Pure low-rank coarse correction is not enough.
- `edge_galerkin` and `edge_yee` can both reduce the preconditioned residual but
  may leave a large true residual without a full-rank smoother.
- Adding a one-step operator-Jacobi smoother can reduce the true residual below
  the requested tolerance in the current cube-nurbs PML test.

Representative corrected PML result:

```text
edge_yee + Yee-PML coarse + one-step operator Jacobi smoother
true residual rel < 1e-5 for sjac around 2.5-3.0
```

This means the paper-facing method should not be described as only a Yee coarse
correction. A more accurate description is:

```text
Yee auxiliary-space coarse correction plus a full-rank smoother.
```

## 8. Recommended Terminology

Use these names consistently:

```text
edge_galerkin
```

Yee auxiliary edge space with Galerkin coarse operator:

```text
A_aux = Pi^T A_h Pi
```

```text
edge_yee
```

Yee auxiliary edge space with independent Yee/FDFD coarse operator:

```text
A_aux = A_yee
```

```text
Yee-space Galerkin coarse operator
```

The expensive but consistent operator on Yee dofs:

```text
Pi^T A_h Pi
```

```text
Independent Yee coarse operator
```

The cheap FDFD/DEC-style operator assembled directly on the Yee grid:

```text
A_yee
```

```text
Full auxiliary-space preconditioner
```

The complete structure:

```text
M^{-1} = S^{-1} + Pi A_aux^{-1} Pi^T
```

## 9. Main Takeaway

The Yee grid contributes two separable ideas:

1. It defines a compatible edge auxiliary space for H(curl) IGA.
2. It can also provide a cheap independent FDFD coarse operator.

The first idea is already strongly supported by `edge_galerkin`.

The second idea is more delicate. It works in some cavity cases, but PML cases
need a full-rank smoother, and the independent Yee-PML operator remains an
approximation to the ideal Galerkin coarse operator.

