#!/bin/bash
# Systematic benchmark matrix for paper
# Usage: bash run_benchmark_matrix.sh

set -e
REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

OUTFILE=$REPO/benchmark_results.csv
echo "mesh,ref,order,problem,method,params,iters,converged,true_rel_res,aux_dofs,true_dofs,ratio" > $OUTFILE

run_cavity() {
    local R=$1 O=$2 MODE=$3 PARAMS=$4 GMI=$5 LABEL=$6
    echo "--- cavity r=$R o=$O $LABEL ---"
    local out=$($REPO/minimal_demo -m $MESH -r $R -o $O -wl 0.2 -mode $MODE $PARAMS -gmi $GMI 2>&1)
    local iters=$(echo "$out" | grep "GMRES.*done" | sed 's/.*iters=//;s/,.*//')
    local conv=$(echo "$out" | grep "GMRES.*done" | sed 's/.*converged=//')
    local truerel=$(echo "$out" | grep "GMRES.*done" | sed 's/.*rel=//;s/).*//')
    local tdofs=$(echo "$out" | grep "true DOFs" | sed 's/.*true DOFs: //')
    local adofs=$(echo "$out" | grep "auxiliary dofs" | sed 's/.*auxiliary dofs=//;s/,.*//')
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="fail"
    [ -z "$truerel" ] && truerel="NA"
    local ratio=$(echo "scale=4; $adofs/$tdofs" | bc 2>/dev/null || echo "NA")
    echo "cube-nurbs,$R,$O,cavity,$MODE,$LABEL,$iters,$conv,$truerel,$adofs,$tdofs,$ratio" >> $OUTFILE
}

run_pml() {
    local R=$1 O=$2 PARAMS=$3 GMI=$4 LABEL=$5
    echo "--- pml r=$R o=$O $LABEL ---"
    local out=$($REPO/pml_point_source_demo -m $MESH -r $R -o $O -f 4.0 -prec edge_yee $PARAMS -gmi $GMI -no-vis 2>&1)
    local iters=$(echo "$out" | grep "GMRES.*done" | sed 's/.*iters=//;s/ .*//')
    local truerel=$(echo "$out" | grep "GMRES.*done" | sed 's/.*rel=//;s/).*//')
    local tdofs=$(echo "$out" | grep "Size of linear" | sed 's/.*: //')
    local adofs=$(echo "$out" | grep "auxiliary dofs" | sed 's/.*auxiliary dofs=//;s/,.*//')
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="fail"
    [ -z "$truerel" ] && truerel="NA"
    local ratio=$(echo "scale=4; $adofs/$tdofs" | bc 2>/dev/null || echo "NA")
    echo "cube-nurbs,$R,$O,pml,edge_yee,$LABEL,$iters,1,$truerel,$adofs,$tdofs,$ratio" >> $OUTFILE
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
# Pure FDFD (no fallback)
run_pml 2 2 "-ka -cps 1 -npf" 500 "ka_cps1_nopf"
# Galerkin fallback
run_pml 2 2 "-ka -cps 1" 200 "ka_cps1_fb"
# No knot-align
run_pml 2 2 "-no-ka -an 7" 200 "an7_fb"

echo "=== Results ==="
cat $OUTFILE
