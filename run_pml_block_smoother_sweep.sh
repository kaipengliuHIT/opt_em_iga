#!/bin/bash
# Sweep a 2x2 real/imag block-Jacobi smoother for pure independent Yee-PML.
# Usage: bash run_pml_block_smoother_sweep.sh

set -e

REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
OUTFILE=$REPO/pml_block_smoother_sweep.csv

export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "weight,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,true_dofs" > "$OUTFILE"

for W in 0.3 0.5 0.8 1.0 1.2 1.5 2.0 2.5 3.0; do
    echo "--- edge_yee pure Yee-PML sbjac=$W ---"
    out=$("$REPO/pml_point_source_demo" \
        -m "$MESH" \
        -o 2 -r 2 -f 4.0 \
        -prec edge_yee \
        -ka -cps 1 \
        -npf \
        -sbjac "$W" -sbjit 1 \
        -trc \
        -gmi 500 -gpl 0 -grt 1e-5 \
        -no-vis 2>&1)

    done_line=$(echo "$out" | grep "GMRES.*done" | tail -n 1)
    iters=$(echo "$done_line" | sed -n 's/.*iters=\([^, ]*\).*/\1/p')
    gmres_conv=$(echo "$done_line" | sed -n 's/.*, converged=\([^, ]*\).*/\1/p')
    true_conv=$(echo "$done_line" | sed -n 's/.*true_converged=\([^, ]*\).*/\1/p')
    true_rel=$(echo "$done_line" | sed -n 's/.*(rel=\([^)]*\)).*/\1/p')
    adofs=$(echo "$out" | grep "auxiliary dofs" | tail -n 1 | sed 's/.*dofs=//;s/,.*//')
    system_size=$(echo "$out" | grep "Size of linear" | tail -n 1 | sed 's/.*: //')
    tdofs=$((system_size / 2))

    [ -z "$iters" ] && iters=fail
    [ -z "$gmres_conv" ] && gmres_conv=0
    [ -z "$true_conv" ] && true_conv="$gmres_conv"
    [ -z "$true_rel" ] && true_rel=NA
    [ -z "$adofs" ] && adofs=0

    echo "$W,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs" >> "$OUTFILE"
    echo "$W,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs"
done

echo "Wrote $OUTFILE"
