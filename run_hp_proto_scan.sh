#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/mnt/f/optemcode}"
ROOT="${ROOT:-$WORKSPACE/opt_em_iga_repo}"
BIN="${BIN:-$ROOT/fdfd_iga_init_demo}"
OUT="${OUT:-$ROOT/covariant_aux_space/hp_proto_scan_results.csv}"

ORDERS="${ORDERS:-2,3}"
REFINES="${REFINES:-1,2,3}"
FD_N="${FD_N:-41}"
AUX_N="${AUX_N:-7}"
MAX_RATIO="${MAX_RATIO:-0}"
GMRES_MAX_ITER="${GMRES_MAX_ITER:-250}"
SOURCES="${SOURCES:-gaussian_y}"
EPS_MODES="${EPS_MODES:-constant,layered_x}"
PROTOS="${PROTOS:-nodal_proto,edge_yee_proto}"
MESHES="${MESHES:-$WORKSPACE/mfem/data/cube-nurbs.mesh,$ROOT/meshes/warped-cube-singlepatch-nurbs.mesh}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  echo "build fdfd_iga_init_demo first" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

header_written=0
IFS=',' read -r -a proto_list <<< "$PROTOS"
for proto in "${proto_list[@]}"; do
  tmp="$(mktemp)"
  LD_LIBRARY_PATH="$WORKSPACE/opt/openmpi/lib:$WORKSPACE/opt/hypre/lib:${LD_LIBRARY_PATH:-}" \
    "$BIN" \
      --scan \
      --proto-mode "$proto" \
      --scan-meshes "$MESHES" \
      --scan-eps "$EPS_MODES" \
      --scan-sources "$SOURCES" \
      --scan-orders "$ORDERS" \
      --scan-refines "$REFINES" \
      --scan-fd "$FD_N" \
      --scan-aux "$AUX_N" \
      --gmres-max-iter "$GMRES_MAX_ITER" \
      --scan-max-ratio "$MAX_RATIO" > "$tmp"

  if [[ "$header_written" -eq 0 ]]; then
    grep -E '^(mesh,|/)' "$tmp" >> "$OUT"
    header_written=1
  else
    grep -E '^/' "$tmp" >> "$OUT"
  fi
  rm -f "$tmp"
done

echo "$OUT"
