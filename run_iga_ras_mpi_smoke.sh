#!/bin/bash
# MPI smoke test for the production-oriented IGA-native RAS path.
# Usage:
#   bash run_iga_ras_mpi_smoke.sh

set -e

REPO=${REPO:-/mnt/f/optemcode/opt_em_iga_repo}
MESH=${MESH:-/mnt/f/optemcode/mfem/data/cube-nurbs.mesh}
NP=${NP:-2}
TIMEOUT_SEC=${TIMEOUT_SEC:-60}

export OPAL_PREFIX=${OPAL_PREFIX:-/mnt/f/optemcode/opt/openmpi}
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib:${LD_LIBRARY_PATH:-}

make -C "$REPO" pml_point_source_demo

timeout -k 10 "$TIMEOUT_SEC" "$OPAL_PREFIX/bin/mpirun" -np "$NP" \
    "$REPO/pml_point_source_demo" \
    -m "$MESH" \
    -r 1 -o 2 -f 4.0 \
    -prec iga_ras \
    -rasov 0 -rasw 0.8 -rasit 1 -rasasm auto \
    -trc \
    -gmi 20 -gpl 0 -grt 1e-5 \
    -no-vis
