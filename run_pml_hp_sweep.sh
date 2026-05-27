#!/bin/bash
# PML h/p sweep for paper-facing comparison.
# Usage:
#   bash run_pml_hp_sweep.sh
# Optional environment overrides:
#   R_LIST="1 2 3" O_LIST="1 2 3" RUN_AMS=0 bash run_pml_hp_sweep.sh

set -e

REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
OUTFILE=$REPO/pml_hp_sweep.csv

R_LIST=${R_LIST:-"1 2"}
O_LIST=${O_LIST:-"1 2 3"}
RUN_AMS=${RUN_AMS:-1}

export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "mesh,ref,order,method,params,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,true_dofs,aux_ratio" > "$OUTFILE"

run_case() {
    local R=$1
    local O=$2
    local METHOD=$3
    local PARAMS=$4
    local GMI=$5
    local LABEL=$6

    echo "--- pml r=$R o=$O $METHOD $LABEL ---"
    out=$("$REPO/pml_point_source_demo" \
        -m "$MESH" \
        -r "$R" -o "$O" -f 4.0 \
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

    echo "cube-nurbs,$R,$O,$METHOD,$LABEL,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio" >> "$OUTFILE"
    echo "cube-nurbs,$R,$O,$METHOD,$LABEL,$iters,$gmres_conv,$true_conv,$true_rel,$adofs,$tdofs,$ratio"
}

for R in $R_LIST; do
    for O in $O_LIST; do
        run_case "$R" "$O" none "-no-ka -an 1" 500 "none"
        if [ "$RUN_AMS" = "1" ]; then
            run_case "$R" "$O" ams "" 500 "ams"
        fi
        run_case "$R" "$O" edge_galerkin "-ka -cps 1" 300 "ka_cps1"
        run_case "$R" "$O" edge_yee "-ka -cps 1 -npf -sbjac 3.0 -sbjit 1" 500 "ka_cps1_nopf_sbjac30"
    done
done

echo "Wrote $OUTFILE"
