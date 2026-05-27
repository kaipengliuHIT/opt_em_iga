# SCI Paper Plan: Covariant Yee Auxiliary-Space Preconditioning for IGA Maxwell Systems

## 1. Target paper direction

### Tentative title

**A Covariant Yee Auxiliary-Space Preconditioner for Isogeometric Maxwell Systems on NURBS Geometries**

### Core research story

Frequency-domain Maxwell systems discretized by isogeometric `H(curl)` spaces are difficult to solve efficiently because of high-order basis functions, NURBS geometry mappings, and indefinite curl-curl operators. This project develops a reference-space auxiliary preconditioner based on a structured Yee-edge grid and a residual-consistent algebraic transfer between the Yee auxiliary space and the IGA true-DOF space.

The key algorithmic point is the explicit prolongation/restriction pair:

```text
r_aux = Pi^T r
z_aux = A_aux^{-1} r_aux
z     = Pi z_aux
```

where `r` is the algebraic Krylov residual, not a physical field to be directly sampled.

The current implementation includes a fast `edge_yee_proto` transfer construction. Instead of projecting every Yee edge basis function by a separate global mass solve, it assembles the coupling matrix locally:

```text
B_ij = int phi_i . w_j dx
Pi   = M_h^{-1} B
```

This makes the Yee-to-IGA transfer practical and preserves the baseline mass-projected transfer.

## 2. Current status

### Implemented

- Matrix-based auxiliary-space preconditioner framework.
- Prototype modes:
  - `nodal_proto`
  - `edge_galerkin_proto`
  - `edge_yee_proto`
- Residual-consistent algebraic path:
  - restriction by `Pi^T r`
  - auxiliary solve in reference space
  - prolongation by `Pi z_aux`
- Yee-edge auxiliary DOF construction.
- Yee coarse Maxwell operator assembly.
- Fast `BuildProlongationFast` for `edge_yee_proto`:
  - element-driven local assembly of `B`
  - one mass matrix inversion/backsolve
  - no per-column global projection
- Basic validation on NURBS cube examples.

### Verified examples

Small reference cube case:

```text
true_vsize = 300
edge_dofs  = 108
BuildProlongationFast time ~= 0.07 s
aux_prec GMRES iterations = 13
```

Larger reference cube case:

```text
true_vsize = 882
edge_dofs  = 450
BuildProlongationFast time ~= 0.78 s
zero_init GMRES iterations = 79
fdfd_init GMRES iterations = 88
aux_prec GMRES iterations = 36
```

### Current maturity estimate

```text
Method prototype:       70%
Engineering usability:  60%
Paper completeness:     35-45%
```

## 3. Main gaps before SCI submission

### Gap 1: Insufficient numerical study

The current tests are too limited for a SCI paper. The paper needs systematic experiments across:

- mesh refinement levels
- IGA polynomial orders
- auxiliary grid sizes
- geometry distortion
- material contrast
- frequency or wavelength
- comparison against baseline solvers/preconditioners

### Gap 2: Auxiliary solver scalability

If the auxiliary solve currently uses a dense inverse for `A_aux`, reviewers may question scalability. The paper needs either:

1. a scalable sparse/structured auxiliary solver, or
2. a clear limitation statement plus numerical evidence that the current prototype is useful for the target scale.

Preferred direction:

- assemble `A_aux` as sparse matrix
- test sparse direct solve or iterative solve
- explore structured/geometric multigrid for the Yee auxiliary grid

### Gap 3: Missing baseline comparisons

The main paper tables should compare against:

- no preconditioner / zero initial guess
- reference FDFD initial guess only
- CPU Hypre AMS
- `edge_yee_proto`

`nodal_proto`, `edge_galerkin_proto`, Jacobi, and ILU/block Jacobi are useful as
internal diagnostics or optional appendix ablations, but they should not be the
main paper story unless a reviewer asks for more solver context.

### Gap 4: Theory and formulation need packaging

The paper should clearly define:

- physical Maxwell problem
- IGA `H(curl)` discretization
- NURBS geometry mapping
- covariant Piola mapping
- auxiliary Yee basis functions
- mass-projected prolongation
- algebraic residual restriction
- reference-space coarse operator

A commuting-diagram or de Rham compatibility discussion would strengthen the method.

## 4. Recommended experiment matrix

### 4.1 Transfer construction validation

Goal: prove that the fast transfer is correct and much faster.

Compare:

- reference/slow transfer construction
- fast local assembly transfer construction

Suggested table columns:

| mesh | order | refine | true DOFs | Yee DOFs | slow setup time | fast setup time | speedup | relative transfer error | GMRES slow | GMRES fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Metrics:

```text
relative transfer error = ||Pi_fast - Pi_ref||_F / ||Pi_ref||_F
```

If full `Pi` comparison is expensive, compare:

```text
||B_fast[:,j] - B_ref[:,j]|| / ||B_ref[:,j]||
```

for selected columns, plus identical GMRES iteration counts.

### 4.2 Preconditioner effectiveness

Goal: show that `edge_yee_proto` reduces GMRES iterations and solve time.

Suggested table columns:

| mesh | eps mode | order | refine | aux_n | true DOFs | aux DOFs | zero iters | FDFD iters | AMS iters | edge_yee iters | setup time | solve time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Suggested parameter ranges:

```text
mesh:       cube-nurbs, warped-cube-singlepatch-nurbs
order:      1, 2, 3, 4
refine:     1, 2, 3, 4
aux_n:      5, 7, 9, 11, 15
eps mode:   constant, layered_x
```

### 4.3 Auxiliary grid sensitivity

Goal: understand the trade-off between auxiliary grid size and convergence.

Suggested table columns:

| mesh | order | refine | aux_n | aux DOFs | aux ratio | setup time | GMRES iterations | total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Expected trend to investigate:

- too coarse `aux_n`: weak preconditioner
- moderate `aux_n`: best time-to-solution
- too fine `aux_n`: high setup/coarse solve cost

### 4.4 Geometry robustness

Goal: support the covariant formulation.

Cases:

```text
cube-nurbs.mesh
warped-cube-singlepatch-nurbs.mesh
```

If possible, add stronger geometric distortion cases.

Suggested metrics:

- GMRES iterations
- setup time
- solve time
- failure/convergence status

### 4.5 Material contrast robustness

Goal: test whether the method survives heterogeneous coefficients.

Cases:

```text
epsilon_mode = constant
epsilon_mode = layered_x
eps_r = 1, 8, 20, 50, 100
```

If high contrast causes degradation, report it honestly and discuss possible weighted transfer or coefficient-aware auxiliary operators.

### 4.6 Frequency/wavelength sensitivity

Goal: understand behavior for more indefinite systems.

Cases:

```text
wavelength = 2.0, 1.0, 0.5, 0.25
```

Track:

- GMRES iterations
- convergence failures
- relative residual
- effect of auxiliary preconditioning

## 5. Code tasks

### Task A: Benchmark infrastructure

Priority: high

- Add CSV output for all important metrics.
- Include setup-time breakdown:
  - transfer build time
  - auxiliary matrix build time
  - auxiliary inverse/solver setup time
  - GMRES solve time
- Include convergence flags.
- Include command-line parameters in output.
- Ensure benchmark scripts are reproducible on a clean clone.

Suggested output columns:

```text
mesh,proto_mode,epsilon_mode,order,refine,fd_n,aux_n,
true_vsize,aux_dofs,aux_ratio,
transfer_time,coarse_setup_time,total_setup_time,
zero_iters,fdfd_iters,auxprec_iters,
zero_time,fdfd_time,auxprec_time,
zero_conv,fdfd_conv,auxprec_conv
```

### Task B: Keep a reference transfer path

Priority: high

Add or preserve a slow but correct reference implementation:

```text
BuildProlongationReference()
BuildProlongationFast()
```

Use it for validation only, not production runs.

Needed diagnostics:

```text
||Pi_fast - Pi_ref|| / ||Pi_ref||
||B_fast[:,j] - B_ref[:,j]|| / ||B_ref[:,j]||
```

### Task C: Auxiliary solver improvement

Priority: high

Investigate replacing dense inverse with scalable alternatives:

1. sparse direct solve for `A_aux`
2. iterative solve with diagonal/block preconditioner
3. structured geometric multigrid on the Yee grid
4. FFT-like solver if boundary conditions and operator structure allow it

At minimum, document complexity and current limitations.

### Task D: Hypre AMS comparison

Priority: medium-high

Try to configure an AMS baseline for the MFEM `H(curl)` system.

If AMS works:

- report GMRES iterations and solve time
- compare setup time
- compare robustness under geometry/material changes

If AMS is hard to configure for NURBS or complex systems:

- document the issue
- compare against available alternatives

### Task E: Paper-ready figures

Priority: medium

Prepare plots:

1. Transfer setup time vs DOFs.
2. GMRES iterations vs DOFs.
3. Total time-to-solution vs DOFs.
4. Auxiliary grid sensitivity.
5. Geometry distortion comparison.
6. Material contrast comparison.

Suggested plot style:

- log scale for time and DOFs
- grouped bars for iteration counts
- line plots for scaling trends

## 6. Suggested paper outline

### Abstract

Summarize the problem, covariant Yee auxiliary preconditioner, fast transfer assembly, and numerical improvements.

### 1. Introduction

- Frequency-domain Maxwell systems
- IGA/NURBS advantages and linear solver challenges
- Existing preconditioners and limitations
- Motivation for reference-space auxiliary methods
- Contributions

### 2. IGA Maxwell discretization

- Maxwell model problem
- IGA `H(curl)` space
- NURBS geometry mapping
- Covariant Piola transformation
- Algebraic system

### 3. Covariant Yee auxiliary space

- Reference Yee grid
- Edge-based auxiliary unknowns
- Auxiliary curl-curl operator
- Pull-back geometry tensors
- Relationship to de Rham structure

### 4. Algebraic transfer operator

- Mass-projected prolongation
- Residual-consistent restriction
- Why direct residual sampling is incorrect
- Definition of `Pi`, `Pi^T`, and `A_aux`

### 5. Fast transfer construction

- Baseline per-column projection
- Bottleneck analysis
- Element-driven local assembly of `B`
- Complexity discussion
- Numerical equivalence to baseline

### 6. Numerical experiments

- Setup and implementation details
- Transfer validation
- Convergence studies
- Scalability studies
- Geometry/material/frequency robustness
- Baseline comparisons

### 7. Limitations and future work

- Auxiliary solver scalability
- Multi-patch extension
- Boundary condition handling
- GPU acceleration
- More robust high-contrast preconditioning

### 8. Conclusion

Summarize effectiveness and future direction.

## 7. Recommended timeline

### Week 1: Benchmark cleanup and validation

- Finalize CSV benchmark output.
- Keep reference transfer implementation.
- Validate fast vs slow transfer on small cases.
- Run initial benchmark matrix for cube and warped cube.

Deliverables:

- validated benchmark script
- transfer correctness table
- first convergence table

### Week 2: Parameter studies

- Run order/refinement study.
- Run auxiliary grid sensitivity study.
- Run material contrast study.
- Collect setup and solve times.

Deliverables:

- main tables
- draft plots
- identified failure cases

### Week 3: Auxiliary solver and baseline comparison

- Test sparse auxiliary solve alternatives.
- Try Hypre AMS or other baseline preconditioners.
- Compare with `nodal_proto` and `edge_galerkin_proto`.

Deliverables:

- solver comparison table
- scalability discussion

### Week 4: Paper formulation and first draft

- Write method sections.
- Prepare final plots.
- Write numerical experiment section.
- Document limitations clearly.

Deliverables:

- first manuscript draft
- reproducibility notes

### Additional 1-2 months if targeting stronger journals

- Improve auxiliary solver scalability.
- Add larger 3D cases.
- Add more complete theoretical discussion.
- Add multi-patch or more realistic geometry if feasible.

## 8. Possible target venues

### More applied / engineering computation

- Engineering Analysis with Boundary Elements
- Computer Methods in Applied Mechanics and Engineering
- Journal of Computational and Applied Mathematics
- Applied Mathematics and Computation
- Mathematics and Computers in Simulation

### Computational electromagnetics / numerical methods

- IEEE Transactions on Antennas and Propagation, if the electromagnetic application is strengthened
- Journal of Computational Physics, if scalability and theory are strengthened
- Computer Physics Communications, if software and reproducibility are emphasized

### More ambitious numerical analysis direction

- SIAM Journal on Scientific Computing, if theory, scalability, and baseline comparisons are significantly strengthened
- ESAIM: Mathematical Modelling and Numerical Analysis, if analysis is expanded

## 9. Minimum publishable package

A realistic minimum SCI package should include:

- Clear formulation of the covariant Yee auxiliary-space preconditioner.
- Correct residual-based algebraic restriction `Pi^T r`.
- Fast Yee-to-IGA transfer construction.
- Numerical equivalence between fast and reference transfer.
- Systematic tests over at least:
  - two meshes
  - three refinement levels
  - two polynomial orders
  - two material profiles
  - several auxiliary grid sizes
- Comparison against at least zero, FDFD initial guess, and CPU Hypre AMS.
- Honest discussion of auxiliary solver scalability.

## 10. Immediate next actions

1. Clean and commit benchmark-related scripts separately.
2. Add stable CSV timing output to the driver.
3. Restore or preserve a reference slow transfer path for validation.
4. Run the transfer validation table.
5. Run the first benchmark matrix:

```text
mesh = cube-nurbs, warped-cube-singlepatch-nurbs
proto = edge_yee_proto
order = 2, 3
refine = 1, 2, 3
epsilon_mode = constant, layered_x
aux_n = 5, 7, 9
```

6. Generate first plots and decide whether the current method is strong enough or whether the auxiliary solver must be improved before writing.
