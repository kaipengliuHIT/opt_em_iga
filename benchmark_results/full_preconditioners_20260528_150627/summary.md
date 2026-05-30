# Full preconditioner benchmark

Mesh: /mnt/f/optemcode/mfem/data/cube-nurbs.mesh

PML case: refine=2, order=3, frequency=4.0, true residual control.

FDFD initial-guess rows are a separate cavity/reference diagnostic; the current PML demo has no FDFD-initial-guess mode.

| problem | case | status | elapsed_sec | system_size | true_dofs | iters | gmres_converged | true_converged | true_rel_res | aux_dofs | aux_ratio | patches | max_patch_dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pml | no_preconditioner | 0 | 29 | 2688 | 1344 | 500 | 0 | 0 | 0.0039966 | 0 | 0.0000 | NA | NA |
| pml | ams | 0 | 50 | 2688 | 1344 | 500 | 0 | 0 | 0.00834136 | 0 | 0.0000 | NA | NA |
| pml | edge_galerkin_cps2 | 0 | 52 | 2688 | 1344 | 500 | 0 | 0 | 0.174578 | 1176 | 0.8750 | NA | NA |
| pml | edge_yee_cps2_nopf_mfbaux | 0 | 48 | 2688 | 1344 | 500 | 0 | 0 | 0.198237 | 1176 | 0.8750 | NA | NA |
| pml | iga_ras_overlap0 | 0 | 36 | 2688 | 1344 | 20 | 1 | 0 | 2.55864e-07 | 0 | 0.0000 | 64 | 600 |
| cavity | zero_initial_guess | 124 | 933 | NA | NA | NA | NA | NA | NA | 0 | NA | NA | NA |
| cavity | fdfd_initial_guess | 124 | 933 | NA | NA | NA | NA | NA | NA | 0 | NA | NA | NA |
