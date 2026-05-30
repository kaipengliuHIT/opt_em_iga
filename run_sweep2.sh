#!/bin/bash
cd /mnt/f/optemcode/opt_em_iga_repo
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib

parse_iters() {
  # Input: output of ams_sweep, label to search
  # Output: iteration count
  local out="$1"
  local label="$2"
  echo "$out" | grep -F "$label" | head -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' '
}

echo "# r o f N dimPi none jac lump07 lump10 exact07 exact10"
for r in 1 2; do
  for o in 2 3; do
    for f in 1 5 10; do
      out=$(./ams_sweep -r $r -o $o -f $f 2>/dev/null)
      N=$(echo "$out" | grep "^│  Sweep:" | sed 's/.*N=[[:space:]]*\([0-9]*\).*/\1/')
      dimPi=$(echo "$out" | grep "^│  Sweep:" | sed 's/.*dim(Pi)=[[:space:]]*\([0-9]*\).*/\1/')
      none_i=$(parse_iters "$out" "│ none")
      jac_i=$(parse_iters "$out" "│ Jacobi")
      lump07_i=$(parse_iters "$out" "Pi_lumped+jac")
      # lump07 is first, lump10 is second
      lump07_i=$(echo "$out" | grep "Pi_lumped+jac" | head -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      lump10_i=$(echo "$out" | grep "Pi_lumped+jac" | tail -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      exact07_i=$(echo "$out" | grep "Pi_exact+jac" | head -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      exact10_i=$(echo "$out" | grep "Pi_exact+jac" | tail -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      echo "$r $o $f $N $dimPi $none_i $jac_i $lump07_i $lump10_i $exact07_i $exact10_i"
      echo "  [r=$r o=$o f=$f] N=$N exact10=$exact10_i" >&2
    done
  done
done
