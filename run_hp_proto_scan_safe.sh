#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/mnt/f/optemcode}"
ROOT="${ROOT:-$WORKSPACE/opt_em_iga_repo}"
BIN="${BIN:-$ROOT/fdfd_iga_init_demo}"
OUT="${OUT:-$ROOT/covariant_aux_space/hp_proto_scan_safe_results.csv}"
LOG_DIR="${LOG_DIR:-$ROOT/covariant_aux_space/hp_proto_scan_logs}"

ORDERS="${ORDERS:-2,3}"
REFINES="${REFINES:-1,2}"
FD_N="${FD_N:-41}"
AUX_N="${AUX_N:-7}"
GMRES_MAX_ITER="${GMRES_MAX_ITER:-250}"
CASE_TIMEOUT="${CASE_TIMEOUT:-180s}"
EPS_MODES="${EPS_MODES:-constant,layered_x}"
PROTOS="${PROTOS:-nodal_proto,edge_yee_proto}"
MESHES="${MESHES:-$WORKSPACE/mfem/data/cube-nurbs.mesh,$ROOT/meshes/warped-cube-singlepatch-nurbs.mesh}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")" "$LOG_DIR"
echo "mesh,proto_mode,epsilon_mode,order,ref_levels,fd_n,aux_n,true_vsize,aux_dofs,aux_ratio,zero_iters,fdfd_iters,auxprec_iters,zero_conv,fdfd_conv,auxprec_conv,status,log" > "$OUT"

slug() {
  echo "$1" | sed 's#[/: ]#_#g'
}

extract_iters() {
  local label="$1"
  local file="$2"
  grep "\\[GMRES\\] $label done" "$file" | tail -1 | sed -E 's/.*iters=([0-9]+), converged=([0-9]+).*/\1,\2/' || true
}

IFS=',' read -r -a mesh_list <<< "$MESHES"
IFS=',' read -r -a proto_list <<< "$PROTOS"
IFS=',' read -r -a eps_list <<< "$EPS_MODES"
IFS=',' read -r -a order_list <<< "$ORDERS"
IFS=',' read -r -a refine_list <<< "$REFINES"

for mesh in "${mesh_list[@]}"; do
  for eps_mode in "${eps_list[@]}"; do
    for order in "${order_list[@]}"; do
      for refine in "${refine_list[@]}"; do
        for proto in "${proto_list[@]}"; do
          log="$LOG_DIR/$(slug "$mesh")_${eps_mode}_p${order}_r${refine}_${proto}.log"
          status="ok"
          if ! timeout "$CASE_TIMEOUT" env LD_LIBRARY_PATH="$WORKSPACE/opt/openmpi/lib:$WORKSPACE/opt/hypre/lib:${LD_LIBRARY_PATH:-}" \
              "$BIN" \
              --mesh "$mesh" \
              --proto-mode "$proto" \
              --epsilon-mode "$eps_mode" \
              --order "$order" \
              --refine "$refine" \
              --fd-n "$FD_N" \
              --aux-n "$AUX_N" \
              --gmres-max-iter "$GMRES_MAX_ITER" > "$log" 2>&1; then
            status="timeout_or_error"
          fi

          header="$(grep '\[single_patch_demo\] ndofs=' "$log" | tail -1 || true)"
          meta="$(grep '\[single_patch_demo\] order=' "$log" | tail -1 || true)"
          true_vsize="$(echo "$header" | sed -E 's/.*true_vsize=([0-9]+).*/\1/' || true)"
          aux_dofs="$(echo "$meta" | sed -E 's/.*aux_dofs=([0-9]+).*/\1/' || true)"
          aux_ratio="$(echo "$meta" | sed -E 's/.*aux_ratio=([^,]+).*/\1/' || true)"
          zero="$(extract_iters zero_init "$log")"
          fdfd="$(extract_iters fdfd_init "$log")"
          aux="$(extract_iters aux_prec "$log")"

          zero_iters="${zero%,*}"; zero_conv="${zero#*,}"
          fdfd_iters="${fdfd%,*}"; fdfd_conv="${fdfd#*,}"
          aux_iters="${aux%,*}"; aux_conv="${aux#*,}"
          [[ "$zero" == "$zero_iters" ]] && zero_iters="" && zero_conv=""
          [[ "$fdfd" == "$fdfd_iters" ]] && fdfd_iters="" && fdfd_conv=""
          [[ "$aux" == "$aux_iters" ]] && aux_iters="" && aux_conv=""

          echo "$mesh,$proto,$eps_mode,$order,$refine,$FD_N,$AUX_N,$true_vsize,$aux_dofs,$aux_ratio,$zero_iters,$fdfd_iters,$aux_iters,$zero_conv,$fdfd_conv,$aux_conv,$status,$log" >> "$OUT"
          tail -1 "$OUT"
        done
      done
    done
  done
done

echo "$OUT"
