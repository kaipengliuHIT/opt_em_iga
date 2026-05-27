#!/bin/bash
# Focused scan for the PML h/p failure point: refine=2, order=3.
# Usage: bash run_pml_r2o3_target_scan.sh

set -e

REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
OUTFILE=$REPO/pml_r2o3_target_scan.csv

export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "method,params,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,true_dofs,aux_ratio" > "$OUTFILE"

run_case() {
    local METHOD=$1
    local PARAMS=$2
    local GMI=$3
    local LABEL=$4

    echo "--- r=2 o=3 $METHOD $LABEL ---"
    out=$("$REPO/pml_point_source_demo" \
        -m "$MESH" \
        -r 2 -o 3 -f 4.0 \
        -prec "$METHOD" $PARAMS \
        -trc \
        -gmi "$GMI" -gpl 0 -grt 1e-5 \
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

    echo "$METHOD,$LABEL,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio" >> "$OUTFILE"
    echo "$METHOD,$LABEL,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio"
}

run_case none "-no-ka -an 1" 500 "none"
run_case ams "" 500 "ams"

# Low-rank Galerkin coarse correction alone is a diagnostic, not a complete
# preconditioner. The +sbjac rows test whether the same full-rank smoother fixes
# the true residual.
run_case edge_galerkin "-ka -cps 1" 300 "gal_cps1"
run_case edge_galerkin "-ka -cps 1 -sbjac 3.0 -sbjit 1" 300 "gal_cps1_sbjac30"
run_case edge_galerkin "-ka -cps 2 -sbjac 3.0 -sbjit 1" 300 "gal_cps2_sbjac30"

# Independent Yee-PML coarse operator. cps=1 failed in the h/p sweep; cps=2
# checks whether the failure is mostly auxiliary-space resolution.
run_case edge_yee "-ka -cps 1 -npf -sbjac 3.0 -sbjit 1" 500 "yee_cps1_sbjac30"
run_case edge_yee "-ka -cps 2 -npf -sbjac 2.0 -sbjit 1" 500 "yee_cps2_sbjac20"
run_case edge_yee "-ka -cps 2 -npf -sbjac 3.0 -sbjit 1" 500 "yee_cps2_sbjac30"
run_case edge_yee "-ka -cps 2 -npf -sbjac 4.0 -sbjit 1" 500 "yee_cps2_sbjac40"

echo "Wrote $OUTFILE"
