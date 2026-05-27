#!/bin/bash
# Component-wise Yee-PML scaling test for refine=2, order=3.
# Usage: bash run_pml_r2o3_component_scale.sh

set -e

REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
OUTFILE=$REPO/pml_r2o3_component_scale.csv

export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "ycurl,ymass,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,true_dofs,aux_ratio" > "$OUTFILE"

run_case() {
    local YCURL=$1
    local YMASS=$2

    echo "--- r=2 o=3 edge_yee ycurl=$YCURL ymass=$YMASS ---"
    out=$("$REPO/pml_point_source_demo" \
        -m "$MESH" \
        -r 2 -o 3 -f 4.0 \
        -prec edge_yee \
        -ka -cps 1 \
        -npf \
        -no-ycal \
        -ycurl "$YCURL" -ymass "$YMASS" \
        -sbjac 3.0 -sbjit 1 \
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

    [ -z "$iters" ] && iters=fail
    [ -z "$gmres_conv" ] && gmres_conv=0
    [ -z "$true_conv" ] && true_conv="$gmres_conv"
    [ -z "$true_rel" ] && true_rel=NA
    [ -z "$adofs" ] && adofs=0

    tdofs=NA
    ratio=NA
    if [ -n "$system_size" ]; then
        tdofs=$((system_size / 2))
        ratio=$(python3 -c "print(f'{int($adofs)/int($tdofs):.4f}')" 2>/dev/null || echo "NA")
    fi

    echo "$YCURL,$YMASS,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio" >> "$OUTFILE"
    echo "$YCURL,$YMASS,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio"
}

# Diagnostics suggested curl scale ~0.07 and mass scale ~4.7 against the
# PML positive preconditioner target. Nearby values test whether the component
# imbalance is the main high-order failure mode.
run_case 1.0 1.0
run_case 0.25 2.0
run_case 0.10 4.0
run_case 0.07 4.7
run_case 0.05 5.0
run_case 0.03 6.0

echo "Wrote $OUTFILE"
