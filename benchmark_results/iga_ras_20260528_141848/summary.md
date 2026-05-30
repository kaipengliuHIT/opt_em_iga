# IGA-native RAS benchmark

Mesh: /mnt/f/optemcode/mfem/data/cube-nurbs.mesh

Case: refine=2, order=3, frequency=4.0, true residual control.

| case | status | elapsed_sec | system_size | iters | gmres_converged | true_converged | true_rel_res | patches | max_patch_dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ams | 0 | 53 | 2688 | 500 | 0 | 0 | 0.00834136 | NA | NA |
| iga_ras_overlap0 | 0 | 36 | 2688 | 20 | 1 | 0 | 2.55864e-07 | 64 | 600 |
| iga_ras_overlap1 | 0 | 446 | 2688 | 1 | 1 | 1 | 4.8541e-08 | 64 | 2688 |
