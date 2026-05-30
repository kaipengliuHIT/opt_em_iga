#!/bin/bash
# Focused benchmark for the IGA-native full-rank RAS preconditioner.
# Usage:
#   bash run_iga_ras_benchmark.sh
# Optional:
#   RUN_UPPER_BOUND=1 bash run_iga_ras_benchmark.sh
#   RUN_EDGE=1 bash run_iga_ras_benchmark.sh

set -u

REPO=${REPO:-/mnt/f/optemcode/opt_em_iga_repo}
MESH=${MESH:-/mnt/f/optemcode/mfem/data/cube-nurbs.mesh}
TIMEOUT_SEC=${TIMEOUT_SEC:-900}
RUN_UPPER_BOUND=${RUN_UPPER_BOUND:-0}
RUN_EDGE=${RUN_EDGE:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-$REPO/benchmark_results/iga_ras_$STAMP}
CSV=$OUTDIR/results.csv
SUMMARY=$OUTDIR/summary.md

export OPAL_PREFIX=${OPAL_PREFIX:-/mnt/f/optemcode/opt/openmpi}
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUTDIR"

make -C "$REPO" pml_point_source_demo

echo "case,status,elapsed_sec,system_size,iters,gmres_converged,true_converged,true_rel_res,patches,max_patch_dim,log" > "$CSV"

extract_value() {
    local text=$1
    local pattern=$2
    echo "$text" | sed -n "$pattern" | tail -n 1
}

run_case() {
    local name=$1
    shift
    local log=$OUTDIR/$name.log

    echo "--- running $name ---"
    local start end elapsed status
    start=$(date +%s)
    timeout "$TIMEOUT_SEC" "$REPO/pml_point_source_demo" \
        -m "$MESH" -r 2 -o 3 -f 4.0 \
        "$@" \
        -trc -gmi 500 -gpl 0 -no-vis > "$log" 2>&1
    status=$?
    end=$(date +%s)
    elapsed=$((end - start))

    local done_line patch_line size_line iters gmres_conv true_conv true_rel patches max_patch_dim
    done_line=$(grep "\\[PML GMRES\\].*done" "$log" | tail -n 1 || true)
    patch_line=$(grep "\\[iga_ras\\] patches=" "$log" | tail -n 1 || true)
    size_line=$(grep "Size of linear system" "$log" | tail -n 1 || true)

    iters=$(extract_value "$done_line" 's/.*iters=\([^, ]*\).*/\1/p')
    gmres_conv=$(extract_value "$done_line" 's/.*, converged=\([^, ]*\).*/\1/p')
    true_conv=$(extract_value "$done_line" 's/.*true_converged=\([^, ]*\).*/\1/p')
    true_rel=$(extract_value "$done_line" 's/.*(rel=\([^)]*\)).*/\1/p')
    patches=$(extract_value "$patch_line" 's/.*patches=\([^, ]*\).*/\1/p')
    max_patch_dim=$(extract_value "$patch_line" 's/.*max_patch_dim=\([^, ]*\).*/\1/p')
    system_size=$(extract_value "$size_line" 's/.*: \([0-9][0-9]*\).*/\1/p')

    [ -z "$iters" ] && iters=NA
    [ -z "$gmres_conv" ] && gmres_conv=NA
    [ -z "$true_conv" ] && true_conv=NA
    [ -z "$true_rel" ] && true_rel=NA
    [ -z "$patches" ] && patches=NA
    [ -z "$max_patch_dim" ] && max_patch_dim=NA
    [ -z "$system_size" ] && system_size=NA

    echo "$name,$status,$elapsed,$system_size,$iters,$gmres_conv,$true_conv,$true_rel,$patches,$max_patch_dim,$log" >> "$CSV"
    echo "$name: status=$status elapsed=${elapsed}s iters=$iters true_rel=$true_rel true_converged=$true_conv"
}

run_case ams "-prec" "ams" "-grt" "1e-5"

if [ "$RUN_EDGE" = "1" ]; then
    run_case edge_yee_mfbaux "-prec" "edge_yee" "-ka" "-cps" "2" "-npf" "-mfbaux" "-grt" "1e-5"
fi

run_case iga_ras_overlap0 "-prec" "iga_ras" "-rasov" "0" "-rasw" "0.8" "-rasit" "1" "-grt" "1e-7"

if [ "$RUN_UPPER_BOUND" = "1" ]; then
    run_case iga_ras_overlap1 "-prec" "iga_ras" "-rasov" "1" "-rasw" "0.8" "-rasit" "1" "-grt" "1e-5"
fi

{
    echo "# IGA-native RAS benchmark"
    echo
    echo "Mesh: $MESH"
    echo
    echo "Case: refine=2, order=3, frequency=4.0, true residual control."
    echo
    echo "| case | status | elapsed_sec | system_size | iters | gmres_converged | true_converged | true_rel_res | patches | max_patch_dim |"
    echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    tail -n +2 "$CSV" | while IFS=, read -r name status elapsed system_size iters gmres_conv true_conv true_rel patches max_patch log; do
        echo "| $name | $status | $elapsed | $system_size | $iters | $gmres_conv | $true_conv | $true_rel | $patches | $max_patch |"
    done
} > "$SUMMARY"

echo "Wrote $CSV"
echo "Wrote $SUMMARY"
