#!/bin/bash
# Paper-ready benchmark runner
REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "======= PAPER BENCHMARK MATRIX ======="
echo ""
echo "Legend: iters(converged) true_rel_res  aux_dofs/true_dofs(ratio%)"
echo ""

run_one() {
    local PROG=$1 M=$2 R=$3 O=$4 EXTRA=$5 GMI=$6
    local out
    if [ "$PROG" = "cavity" ]; then
        out=$($REPO/minimal_demo -m $M -r $R -o $O -wl 0.2 $EXTRA -gmi $GMI 2>&1)
    else
        out=$($REPO/pml_point_source_demo -m $M -r $R -o $O -f 4.0 $EXTRA -gmi $GMI -no-vis 2>&1)
    fi
    local iters=$(echo "$out" | grep "GMRES.*done" | sed 's/.*iters=//;s/ .*//;s/,.*//')
    local conv=$(echo "$out" | grep "GMRES.*done" | sed 's/.*converged=//;s/ .*//')
    local truerel=$(echo "$out" | grep "GMRES.*done" | sed 's/.*rel=//;s/).*//')
    local adofs=$(echo "$out" | grep "auxiliary dofs" | sed 's/.*dofs=//;s/,.*//')
    local tdofs=""
    if [ "$PROG" = "cavity" ]; then
        tdofs=$(echo "$out" | grep "true DOFs" | sed 's/.*DOFs: //')
    else
        tdofs=$(echo "$out" | grep "Size of linear" | sed 's/.*: //')
        tdofs=$((tdofs / 2))
    fi
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="FAIL"
    [ -z "$truerel" ] && truerel="NA"
    local ratio="NA"
    [ "$adofs" != "0" ] && [ -n "$tdofs" ] && ratio=$(python3 -c "print(f'{int($adofs)/int($tdofs)*100:.1f}')" 2>/dev/null)
    local conv_flag="✓"
    [ "$conv" != "1" ] && conv_flag="✗"
    printf "  %-35s  %4s(%s)  %7s  %5s/%5s(%s%%)\n" \
        "$EXTRA" "$iters" "$conv_flag" "$truerel" "$adofs" "$tdofs" "$ratio"
}

echo "=== CAVITY (cube-nurbs, r=2, o=2, tdo=882) ==="
echo "  Method                                  Iters(C)  TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                  --------  -------   -------------------"
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 1" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 3" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 5" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 7" 200
run_one cavity $MESH 2 2 "-mode edge_yee -ka -cps 1" 200
run_one cavity $MESH 2 2 "-mode edge_yee -ka -cps 2" 200
run_one cavity $MESH 2 2 "-mode edge_galerkin -ka -cps 1" 200
echo ""

echo "=== PML (cube-nurbs, r=2, o=2, freq=4.0, tdo=882) ==="
echo "  Method                                  Iters(C)  TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                  --------  -------   -------------------"
run_one pml $MESH 2 2 "-prec none -no-ka -an 1" 500
run_one pml $MESH 2 2 "-prec edge_yee -no-ka -an 7 -npf" 200
run_one pml $MESH 2 2 "-prec edge_yee -no-ka -an 7" 200
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1 -npf" 500
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1" 200
echo ""

echo "=== HIGHER REFINEMENT (cube-nurbs, r=3, o=2) ==="
echo "  Method                                  Iters(C)  TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                  --------  -------   -------------------"
run_one cavity $MESH 3 2 "-mode edge_yee -ka -cps 1" 300
run_one cavity $MESH 3 2 "-mode edge_galerkin -ka -cps 1" 300
echo ""

echo "======= DONE ======="
