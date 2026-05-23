# Prototype Modes Note

## Purpose

This note clarifies the meaning of the three current prototype modes:

- `nodal_proto`
- `edge_galerkin_proto`
- `edge_yee_proto`

These modes do **not** represent equivalent algorithmic objects. They serve
different research roles and should not be interpreted in the same way.

## 1. `nodal_proto`

### Definition

This prototype uses:

- a reference-space auxiliary space based on nodal vector samples
- an explicit algebraic transfer into the MFEM spline `H(curl)` space
- a coarse operator derived from the nodal auxiliary discretization

### Role

This is a transitional prototype.

It is **not** the final theoretical target, because it is not a true
edge-based / Yee-type auxiliary space for Maxwell.

### Why it is still useful

It demonstrates that:

- an auxiliary-space strategy can help the IGA Maxwell solve
- the problem is not limited to the initial-guess idea
- explicit auxiliary correction can already reduce iteration counts

### Interpretation

`nodal_proto` is best viewed as:

- a proof-of-concept auxiliary-space baseline
- a regression baseline for later edge-based work

It is not the intended final method.

## 2. `edge_galerkin_proto`

### Definition

This prototype uses:

- an edge-oriented transfer operator `P`
- a coarse operator defined by Galerkin projection of the main operator:

`A_c = P^T A P`

where `A` is the main IGA Maxwell operator.

### Role

This is an **ideal coarse-space reference**.

It is not yet the final independent Yee auxiliary operator. Instead, it tests
the quality of the transfer operator `P` under the most favorable consistent
coarse-space setting.

### Why it matters

If `edge_galerkin_proto` is very strong, that means:

- the edge-oriented transfer `P` is likely capturing the important error modes
- the coarse space `Range(P)` is relevant to the Maxwell IGA near-nullspace and
  low-energy error components

### Interpretation

`edge_galerkin_proto` should be interpreted as:

- a transfer-validation experiment
- a coarse-space upper bound or ideal reference

It does **not** yet validate the final independent parameter-domain Yee
auxiliary operator.

## 3. `edge_yee_proto`

### Definition

This prototype uses:

- the same edge-oriented transfer `P`
- but replaces the Galerkin coarse operator with an independently assembled
  parameter-domain edge-based auxiliary Maxwell operator

This is the prototype that tries to realize the original theoretical idea:

`M^{-1} = S^{-1} + P A_Y^{-1} P^T`

with `A_Y` built independently in the mapped parameter domain.

### Role

This is the **actual research target** among the current prototypes.

It is the first version that tries to answer the real question:

- can an independently assembled NURBS-mapped Yee auxiliary operator approach
  the ideal coarse-space performance suggested by `edge_galerkin_proto`?

### Interpretation

`edge_yee_proto` is the prototype that should eventually be compared against:

- `AMS`
- `nodal_proto`
- `edge_galerkin_proto`

Its current performance directly reflects the quality of the independently
constructed auxiliary operator `A_Y`.

## Summary of the Differences

### `nodal_proto`

- transfer type: nodal/vector-sample based
- coarse operator: nodal auxiliary operator
- purpose: proof-of-concept baseline

### `edge_galerkin_proto`

- transfer type: edge-oriented
- coarse operator: `P^T A P`
- purpose: validate `P` and provide ideal coarse-space upper bound

### `edge_yee_proto`

- transfer type: edge-oriented
- coarse operator: independently assembled mapped Yee-type auxiliary operator
- purpose: actual target method

## Practical Research Meaning

The current research logic is:

1. `nodal_proto` showed that auxiliary-space preconditioning is viable.
2. `edge_galerkin_proto` showed that the current edge transfer `P` can be very
   powerful when paired with an ideal consistent coarse operator.
3. `edge_yee_proto` now becomes the main object of study, because it tests
   whether an independently built Yee auxiliary operator can reproduce enough of
   the ideal Galerkin coarse-space behavior.

## Current Recommendation

When discussing results:

- use `nodal_proto` as a practical baseline
- use `edge_galerkin_proto` as a transfer-quality / ideal coarse-space
  reference
- treat `edge_yee_proto` as the main theoretical method under development

Only `edge_yee_proto` directly addresses the original NURBS-mapped Yee
auxiliary-space preconditioning idea. `edge_galerkin_proto` is best viewed as a
diagnostic and validation tool for the transfer operator `P`.
