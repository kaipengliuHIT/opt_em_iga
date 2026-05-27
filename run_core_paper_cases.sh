#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKSPACE="${WORKSPACE:-/mnt/f/optemcode}"
OUT="${OUT:-$ROOT/covariant_aux_space/core_paper_cases.csv}"
LOG_DIR="${LOG_DIR:-$ROOT/covariant_aux_space/core_paper_case_logs}"

export ROOT WORKSPACE OUT LOG_DIR
export ORDERS="${ORDERS:-2}"
export REFINES="${REFINES:-2}"
export FD_N="${FD_N:-41}"
export AUX_N="${AUX_N:-7}"
export GMRES_MAX_ITER="${GMRES_MAX_ITER:-800}"
export CASE_TIMEOUT="${CASE_TIMEOUT:-20m}"
export EPS_MODES="${EPS_MODES:-constant,layered_x}"
export PROTOS="${PROTOS:-ams,edge_yee_proto}"
export MESHES="${MESHES:-$WORKSPACE/mfem/data/cube-nurbs.mesh,$ROOT/meshes/warped-cube-singlepatch-nurbs.mesh}"
export LD_LIBRARY_PATH="$WORKSPACE/opt/openmpi/lib:$WORKSPACE/opt/hypre/lib:${LD_LIBRARY_PATH:-}"

"$ROOT/run_hp_proto_scan_safe.sh"
"$ROOT/tools/summarize_hp_scan.py" "$OUT" -o "$ROOT/covariant_aux_space/core_paper_cases.md"
