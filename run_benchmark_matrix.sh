#!/bin/bash
# Systematic benchmark matrix for paper
# Usage: bash run_benchmark_matrix.sh

set -e
REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

OUTFILE=$REPO/benchmark_results.csv
echo "mesh,ref,order,problem,method,params,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,true_dofs,ratio" > $OUTFILE

run_cavity() {
    local R=$1 O=$2 MODE=$3 PARAMS=$4 GMI=$5 LABEL=$6
    echo "--- cavity r=$R o=$O $LABEL ---"
    local out=$($REPO/minimal_demo -m $MESH -r $R -o $O -wl 0.2 -mode $MODE $PARAMS -gmi $GMI 2>&1)
    local done_line=$(echo "$out" | grep "GMRES.*done" | tail -n 1)
    local iters=$(echo "$done_line" | sed -n 's/.*iters=\([^, ]*\).*/\1/p')
    local conv=$(echo "$done_line" | sed -n 's/.*, converged=\([^, ]*\).*/\1/p')
    local truerel=$(echo "$done_line" | sed -n 's/.*(rel=\([^)]*\)).*/\1/p')
    local tdofs=$(echo "$out" | grep "true DOFs" | tail -n 1 | sed 's/.*true DOFs: //')
    local adofs=$(echo "$out" | grep "auxiliary dofs" | tail -n 1 | sed 's/.*auxiliary dofs=//;s/,.*//')
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="fail"
    [ -z "$conv" ] && conv=0
    [ -z "$truerel" ] && truerel="NA"
    local ratio=$(python3 -c "print(f'{int($adofs)/int($tdofs):.4f}')" 2>/dev/null || echo "NA")
    echo "cube-nurbs,$R,$O,cavity,$MODE,$LABEL,$iters,$conv,$conv,$truerel,$adofs,$tdofs,$ratio" >> $OUTFILE
}

run_pml() {
    local R=$1 O=$2 PREC=$3 PARAMS=$4 GMI=$5 LABEL=$6
    echo "--- pml r=$R o=$O $PREC $LABEL ---"
    local out=$($REPO/pml_point_source_demo -m $MESH -r $R -o $O -f 4.0 -prec $PREC $PARAMS -gmi $GMI -no-vis 2>&1)
    local done_line=$(echo "$out" | grep "GMRES.*done" | tail -n 1)
    local iters=$(echo "$done_line" | sed -n 's/.*iters=\([^, ]*\).*/\1/p')
    local conv=$(echo "$done_line" | sed -n 's/.*, converged=\([^, ]*\).*/\1/p')
    local trueconv=$(echo "$done_line" | sed -n 's/.*true_converged=\([^, ]*\).*/\1/p')
    local truerel=$(echo "$done_line" | sed -n 's/.*(rel=\([^)]*\)).*/\1/p')
    local system_size=$(echo "$out" | grep "Size of linear" | tail -n 1 | sed 's/.*: //')
    local tdofs=$((system_size / 2))
    local adofs=$(echo "$out" | grep "auxiliary dofs" | tail -n 1 | sed 's/.*auxiliary dofs=//;s/,.*//')
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="fail"
    [ -z "$conv" ] && conv=0
    [ -z "$trueconv" ] && trueconv="$conv"
    [ -z "$truerel" ] && truerel="NA"
    local ratio=$(python3 -c "print(f'{int($adofs)/int($tdofs):.4f}')" 2>/dev/null || echo "NA")
    echo "cube-nurbs,$R,$O,pml,$PREC,$LABEL,$iters,$conv,$trueconv,$truerel,$adofs,$tdofs,$ratio" >> $OUTFILE
}

# === CAVITY ===
# Baseline: no preconditioner
run_cavity 2 2 edge_yee "-no-ka -an 1" 200 "no_prec_an1"
# edge_yee uniform
run_cavity 2 2 edge_yee "-no-ka -an 5" 200 "an5"
run_cavity 2 2 edge_yee "-no-ka -an 7" 200 "an7"
# edge_yee knot-align
run_cavity 2 2 edge_yee "-ka -cps 1" 200 "ka_cps1"
run_cavity 2 2 edge_yee "-ka -cps 2" 200 "ka_cps2"
# edge_galerkin knot-align
run_cavity 2 2 edge_galerkin "-ka -cps 1" 200 "ka_cps1"
# Higher refinement
run_cavity 3 2 edge_yee "-ka -cps 1" 300 "ka_cps1"
run_cavity 3 2 edge_galerkin "-ka -cps 1" 300 "ka_cps1"

# === PML ===
# Baselines and comparison methods
run_pml 2 2 none "-no-ka -an 1" 500 "none"
run_pml 2 2 ams "" 500 "ams"
run_pml 2 2 edge_galerkin "-ka -cps 1" 200 "ka_cps1"
# Independent Yee-PML candidates
run_pml 2 2 edge_yee "-ka -cps 1 -npf" 500 "ka_cps1_nopf"
run_pml 2 2 edge_yee "-ka -cps 1 -npf -sjac 2.5 -sjit 1" 500 "ka_cps1_nopf_sjac25"
run_pml 2 2 edge_yee "-ka -cps 1 -npf -sjac 3.0 -sjit 1" 500 "ka_cps1_nopf_sjac30"
# Galerkin fallback remains a diagnostic upper-bound path, not the cheap method.
run_pml 2 2 edge_yee "-ka -cps 1" 200 "ka_cps1_fb"

echo "=== Results ==="
cat $OUTFILE
