#!/bin/bash
# Paper-ready benchmark runner
REPO=/mnt/f/optemcode/opt_em_iga_repo
MESH=/mnt/f/optemcode/mfem/data/cube-nurbs.mesh
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:$LD_LIBRARY_PATH

echo "======= PAPER BENCHMARK MATRIX ======="
echo ""
echo "Legend: iters(gmres/true) true_rel_res  aux_dofs/true_dofs(ratio%)"
echo ""

run_one() {
    local PROG=$1 M=$2 R=$3 O=$4 EXTRA=$5 GMI=$6
    local out
    if [ "$PROG" = "cavity" ]; then
        out=$($REPO/minimal_demo -m $M -r $R -o $O -wl 0.2 $EXTRA -gmi $GMI 2>&1)
    else
        out=$($REPO/pml_point_source_demo -m $M -r $R -o $O -f 4.0 $EXTRA -gmi $GMI -no-vis 2>&1)
    fi
    local done_line=$(echo "$out" | grep "GMRES.*done" | tail -n 1)
    local iters=$(echo "$done_line" | sed -n 's/.*iters=\([^, ]*\).*/\1/p')
    local conv=$(echo "$done_line" | sed -n 's/.*, converged=\([^, ]*\).*/\1/p')
    local trueconv=$(echo "$done_line" | sed -n 's/.*true_converged=\([^, ]*\).*/\1/p')
    local truerel=$(echo "$done_line" | sed -n 's/.*(rel=\([^)]*\)).*/\1/p')
    local adofs=$(echo "$out" | grep "auxiliary dofs" | tail -n 1 | sed 's/.*dofs=//;s/,.*//')
    local tdofs=""
    if [ "$PROG" = "cavity" ]; then
        tdofs=$(echo "$out" | grep "true DOFs" | tail -n 1 | sed 's/.*DOFs: //')
    else
        tdofs=$(echo "$out" | grep "Size of linear" | tail -n 1 | sed 's/.*: //')
        tdofs=$((tdofs / 2))
    fi
    [ -z "$adofs" ] && adofs=0
    [ -z "$iters" ] && iters="FAIL"
    [ -z "$conv" ] && conv="0"
    [ -z "$trueconv" ] && trueconv="$conv"
    [ -z "$truerel" ] && truerel="NA"
    local ratio="NA"
    [ "$adofs" != "0" ] && [ -n "$tdofs" ] && ratio=$(python3 -c "print(f'{int($adofs)/int($tdofs)*100:.1f}')" 2>/dev/null)
    printf "  %-48s  %4s(%s/%s)  %10s  %5s/%5s(%s%%)\n" \
        "$EXTRA" "$iters" "$conv" "$trueconv" "$truerel" "$adofs" "$tdofs" "$ratio"
}

echo "=== CAVITY (cube-nurbs, r=2, o=2, tdo=882) ==="
echo "  Method                                               Iters(G/T)   TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                               ----------   -------   -------------------"
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 1" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 3" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 5" 200
run_one cavity $MESH 2 2 "-mode edge_yee -no-ka -an 7" 200
run_one cavity $MESH 2 2 "-mode edge_yee -ka -cps 1" 200
run_one cavity $MESH 2 2 "-mode edge_yee -ka -cps 2" 200
run_one cavity $MESH 2 2 "-mode edge_galerkin -ka -cps 1" 200
echo ""

echo "=== PML (cube-nurbs, r=2, o=2, freq=4.0, tdo=882) ==="
echo "  Method                                               Iters(G/T)   TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                               ----------   -------   -------------------"
run_one pml $MESH 2 2 "-prec none -no-ka -an 1" 500
run_one pml $MESH 2 2 "-prec ams" 500
run_one pml $MESH 2 2 "-prec edge_galerkin -ka -cps 1" 200
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1 -npf" 500
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1 -npf -sjac 2.5 -sjit 1" 500
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1 -npf -sjac 3.0 -sjit 1" 500
run_one pml $MESH 2 2 "-prec edge_yee -ka -cps 1" 200
echo ""

echo "=== HIGHER REFINEMENT (cube-nurbs, r=3, o=2) ==="
echo "  Method                                               Iters(G/T)   TrueRel   AuxDOFs/True(ratio)"
echo "  ------                                               ----------   -------   -------------------"
run_one cavity $MESH 3 2 "-mode edge_yee -ka -cps 1" 300
run_one cavity $MESH 3 2 "-mode edge_galerkin -ka -cps 1" 300
echo ""

echo "======= DONE ======="
