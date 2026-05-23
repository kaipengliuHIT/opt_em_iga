#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/d/code/code_opt_em/opt_em_iga"
BIN="$ROOT/fdfd_iga_init_demo"

MESHES=(
  "/mnt/d/code/code_opt_em/mfem/data/cube-nurbs.mesh"
  "$ROOT/meshes/warped-cube-singlepatch-nurbs.mesh"
)

PROTOS=(
  "nodal_proto"
  "edge_galerkin_proto"
  "edge_yee_proto"
)

EPS_MODES=(
  "constant"
  "layered_x"
)

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  echo "run: make rebuild_fdfd_iga_init_demo" >&2
  exit 1
fi

for mesh in "${MESHES[@]}"; do
  for eps_mode in "${EPS_MODES[@]}"; do
    for proto in "${PROTOS[@]}"; do
      echo
      echo "=== mesh=$mesh proto=$proto epsilon_mode=$eps_mode ==="
      LD_LIBRARY_PATH=/mnt/d/code/code_opt_em/opt/openmpi/lib:/mnt/d/code/code_opt_em/opt/hypre/lib:${LD_LIBRARY_PATH:-} \
        "$BIN" \
        --proto-mode "$proto" \
        --mesh "$mesh" \
        --epsilon-mode "$eps_mode"
    done
  done
done
