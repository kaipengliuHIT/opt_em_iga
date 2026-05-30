#!/bin/bash
# Comprehensive solver/preconditioner benchmark table.
#
# The PML block compares the main solver choices on the same IGA-PML Maxwell
# problem. The FDFD-initial-guess block is kept as a separate cavity/reference
# diagnostic because the current PML demo does not yet support an FDFD initial
# guess.
#
# Usage:
#   bash run_full_preconditioner_benchmark.sh
#
# Optional:
#   RUN_FDFD_INIT=0 bash run_full_preconditioner_benchmark.sh
#   RUN_EDGE_YEE_EXPENSIVE=1 bash run_full_preconditioner_benchmark.sh

set -u

REPO=${REPO:-/mnt/f/optemcode/opt_em_iga_repo}
MESH=${MESH:-/mnt/f/optemcode/mfem/data/cube-nurbs.mesh}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-$REPO/benchmark_results/full_preconditioners_$STAMP}
CSV=$OUTDIR/results.csv
SUMMARY=$OUTDIR/summary.md
TIMEOUT_SEC=${TIMEOUT_SEC:-900}
FDFD_TIMEOUT_SEC=${FDFD_TIMEOUT_SEC:-180}
RUN_FDFD_INIT=${RUN_FDFD_INIT:-1}
RUN_EDGE_YEE_EXPENSIVE=${RUN_EDGE_YEE_EXPENSIVE:-1}

export OPAL_PREFIX=${OPAL_PREFIX:-/mnt/f/optemcode/opt/openmpi}
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUTDIR"

make -C "$REPO" pml_point_source_demo fdfd_iga_init_demo

echo "problem,case,status,elapsed_sec,system_size,true_dofs,iters,gmres_converged,true_converged,true_rel_res,aux_dofs,aux_ratio,patches,max_patch_dim,log" > "$CSV"

extract() {
    local text=$1
    local pattern=$2
    echo "$text" | sed -n "$pattern" | tail -n 1
}

append_row() {
    local problem=$1
    local name=$2
    local status=$3
    local elapsed=$4
    local system_size=$5
    local true_dofs=$6
    local iters=$7
    local gmres_conv=$8
    local true_conv=$9
    local true_rel=${10}
    local aux_dofs=${11}
    local aux_ratio=${12}
    local patches=${13}
    local max_patch_dim=${14}
    local log=${15}

    echo "$problem,$name,$status,$elapsed,$system_size,$true_dofs,$iters,$gmres_conv,$true_conv,$true_rel,$aux_dofs,$aux_ratio,$patches,$max_patch_dim,$log" >> "$CSV"
}

run_pml_case() {
    local name=$1
    shift
    local log=$OUTDIR/pml_$name.log
    echo "--- PML $name ---"

    local start end elapsed status
    start=$(date +%s)
    timeout -k 15 "$TIMEOUT_SEC" "$REPO/pml_point_source_demo" \
        -m "$MESH" -r 2 -o 3 -f 4.0 \
        "$@" \
        -trc -gmi 500 -gpl 0 -no-vis > "$log" 2>&1
    status=$?
    end=$(date +%s)
    elapsed=$((end - start))

    local done_line patch_line aux_line size_line
    done_line=$(grep "\\[PML GMRES\\].*done" "$log" | tail -n 1 || true)
    patch_line=$(grep "\\[iga_ras\\] patches=" "$log" | tail -n 1 || true)
    aux_line=$(grep "auxiliary dofs=" "$log" | tail -n 1 || true)
    size_line=$(grep "Size of linear system" "$log" | tail -n 1 || true)

    local system_size true_dofs iters gmres_conv true_conv true_rel aux_dofs aux_ratio patches max_patch_dim
    system_size=$(extract "$size_line" 's/.*: \([0-9][0-9]*\).*/\1/p')
    true_dofs=NA
    if [ -n "$system_size" ]; then
        true_dofs=$((system_size / 2))
    fi
    iters=$(extract "$done_line" 's/.*iters=\([^, ]*\).*/\1/p')
    gmres_conv=$(extract "$done_line" 's/.*, converged=\([^, ]*\).*/\1/p')
    true_conv=$(extract "$done_line" 's/.*true_converged=\([^, ]*\).*/\1/p')
    true_rel=$(extract "$done_line" 's/.*(rel=\([^)]*\)).*/\1/p')
    aux_dofs=$(extract "$aux_line" 's/.*auxiliary dofs=\([^, ]*\).*/\1/p')
    patches=$(extract "$patch_line" 's/.*patches=\([^, ]*\).*/\1/p')
    max_patch_dim=$(extract "$patch_line" 's/.*max_patch_dim=\([^, ]*\).*/\1/p')

    [ -z "$system_size" ] && system_size=NA
    [ -z "$iters" ] && iters=NA
    [ -z "$gmres_conv" ] && gmres_conv=NA
    [ -z "$true_conv" ] && true_conv=NA
    [ -z "$true_rel" ] && true_rel=NA
    [ -z "$aux_dofs" ] && aux_dofs=0
    [ -z "$patches" ] && patches=NA
    [ -z "$max_patch_dim" ] && max_patch_dim=NA

    aux_ratio=NA
    if [ "$true_dofs" != "NA" ]; then
        aux_ratio=$(python3 -c "print(f'{int($aux_dofs)/int($true_dofs):.4f}')" 2>/dev/null || echo "NA")
    fi

    append_row pml "$name" "$status" "$elapsed" "$system_size" "$true_dofs" \
        "$iters" "$gmres_conv" "$true_conv" "$true_rel" "$aux_dofs" \
        "$aux_ratio" "$patches" "$max_patch_dim" "$log"

    echo "$name: status=$status elapsed=${elapsed}s iters=$iters true_rel=$true_rel"
}

run_fdfd_init_case() {
    local log=$OUTDIR/cavity_fdfd_init.log
    echo "--- cavity fdfd_init diagnostic ---"

    local start end elapsed status
    start=$(date +%s)
    timeout -k 15 "$FDFD_TIMEOUT_SEC" "$REPO/fdfd_iga_init_demo" \
        -m "$MESH" -r 2 -o 2 -n 41 -an 7 \
        -pm edge_yee_proto -ka -cps 1 \
        -gmi 300 -no-vis > "$log" 2>&1
    status=$?
    end=$(date +%s)
    elapsed=$((end - start))

    local header zero_line fdfd_line system_size true_dofs aux_dofs aux_ratio
    header=$(grep "\\[single_patch_demo\\]" "$log" | head -n 2 | tr '\n' ' ' || true)
    zero_line=$(grep "\\[GMRES\\] zero_init done" "$log" | tail -n 1 || true)
    fdfd_line=$(grep "\\[GMRES\\] fdfd_init done" "$log" | tail -n 1 || true)

    true_dofs=$(extract "$header" 's/.*true_vsize=\([0-9][0-9]*\).*/\1/p')
    aux_dofs=$(extract "$header" 's/.*aux_dofs=\([0-9][0-9]*\).*/\1/p')
    [ -z "$true_dofs" ] && true_dofs=NA
    [ -z "$aux_dofs" ] && aux_dofs=0
    system_size=NA
    aux_ratio=NA
    if [ "$true_dofs" != "NA" ]; then
        aux_ratio=$(python3 -c "print(f'{int($aux_dofs)/int($true_dofs):.4f}')" 2>/dev/null || echo "NA")
    fi

    local zero_iters zero_conv zero_rel fdfd_iters fdfd_conv fdfd_rel
    zero_iters=$(extract "$zero_line" 's/.*iters=\([^, ]*\).*/\1/p')
    zero_conv=$(extract "$zero_line" 's/.*, converged=\([^, ]*\).*/\1/p')
    zero_rel=$(extract "$zero_line" 's/.*(rel=\([^)]*\)).*/\1/p')
    fdfd_iters=$(extract "$fdfd_line" 's/.*iters=\([^, ]*\).*/\1/p')
    fdfd_conv=$(extract "$fdfd_line" 's/.*, converged=\([^, ]*\).*/\1/p')
    fdfd_rel=$(extract "$fdfd_line" 's/.*(rel=\([^)]*\)).*/\1/p')

    [ -z "$zero_iters" ] && zero_iters=NA
    [ -z "$zero_conv" ] && zero_conv=NA
    [ -z "$zero_rel" ] && zero_rel=NA
    [ -z "$fdfd_iters" ] && fdfd_iters=NA
    [ -z "$fdfd_conv" ] && fdfd_conv=NA
    [ -z "$fdfd_rel" ] && fdfd_rel=NA

    append_row cavity zero_initial_guess "$status" "$elapsed" "$system_size" \
        "$true_dofs" "$zero_iters" "$zero_conv" "$zero_conv" "$zero_rel" \
        "$aux_dofs" "$aux_ratio" NA NA "$log"
    append_row cavity fdfd_initial_guess "$status" "$elapsed" "$system_size" \
        "$true_dofs" "$fdfd_iters" "$fdfd_conv" "$fdfd_conv" "$fdfd_rel" \
        "$aux_dofs" "$aux_ratio" NA NA "$log"

    echo "fdfd_init diagnostic: status=$status zero_iters=$zero_iters fdfd_iters=$fdfd_iters"
}

run_pml_case no_preconditioner "-prec" "none" "-grt" "1e-5"
run_pml_case ams "-prec" "ams" "-grt" "1e-5"
run_pml_case edge_galerkin_cps2 "-prec" "edge_galerkin" "-ka" "-cps" "2" "-grt" "1e-5"

if [ "$RUN_EDGE_YEE_EXPENSIVE" = "1" ]; then
    run_pml_case edge_yee_cps2_nopf_mfbaux "-prec" "edge_yee" "-ka" "-cps" "2" "-npf" "-mfbaux" "-grt" "1e-5"
fi

run_pml_case iga_ras_overlap0 "-prec" "iga_ras" "-rasov" "0" "-rasw" "0.8" "-rasit" "1" "-grt" "1e-7"

if [ "$RUN_FDFD_INIT" = "1" ]; then
    run_fdfd_init_case
fi

{
    echo "# Full preconditioner benchmark"
    echo
    echo "Mesh: $MESH"
    echo
    echo "PML case: refine=2, order=3, frequency=4.0, true residual control."
    echo
    echo "FDFD initial-guess rows are a separate cavity/reference diagnostic; the current PML demo has no FDFD-initial-guess mode."
    echo
    echo "| problem | case | status | elapsed_sec | system_size | true_dofs | iters | gmres_converged | true_converged | true_rel_res | aux_dofs | aux_ratio | patches | max_patch_dim |"
    echo "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    tail -n +2 "$CSV" | while IFS=, read -r problem name status elapsed system_size true_dofs iters gmres_conv true_conv true_rel aux_dofs aux_ratio patches max_patch_dim log; do
        echo "| $problem | $name | $status | $elapsed | $system_size | $true_dofs | $iters | $gmres_conv | $true_conv | $true_rel | $aux_dofs | $aux_ratio | $patches | $max_patch_dim |"
    done
} > "$SUMMARY"

echo "Wrote $CSV"
echo "Wrote $SUMMARY"
