#!/bin/bash
cd /mnt/f/optemcode/opt_em_iga_repo
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib
export OMPI_MCA_btl=self,sm

parse_col() {
  # Extract iteration count from a specific row
  local out="$1"
  local pattern="$2"
  local nth="${3:-1}"
  echo "$out" | grep "$pattern" | head -"$nth" | tail -1 | \
    sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' '
}

echo "# r o f N dimPi none jac lump07 lump10 exact07 exact10"
echo "# ================================================================"

for r in 1 2 3; do
  for o in 2 3; do
    for f in 1 5 10; do
      rm -rf /tmp/ompi.jf.0 2>/dev/null
      out=$(timeout 180 ./ams_sweep -r $r -o $o -f $f 2>/dev/null)
      
      N=$(echo "$out" | grep "^│  Sweep:" | sed 's/.*N=[[:space:]]*\([0-9]*\).*/\1/')
      dimPi=$(echo "$out" | grep "^│  Sweep:" | sed 's/.*dim(Pi)=[[:space:]]*\([0-9]*\).*/\1/')
      none_i=$(parse_col "$out" "^│ none" 1)
      jac_i=$(parse_col "$out" "Jacobi" 1)
      lump07_i=$(echo "$out" | grep "Pi_lumped+jac" | head -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      lump10_i=$(echo "$out" | grep "Pi_lumped+jac" | tail -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      exact07_i=$(echo "$out" | grep "Pi_exact+jac" | head -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      exact10_i=$(echo "$out" | grep "Pi_exact+jac" | tail -1 | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/' | tr -d ' ')
      
      echo "$r $o $f $N $dimPi $none_i $jac_i $lump07_i $lump10_i $exact07_i $exact10_i"
      echo "  [r=$r o=$o f=$f] N=$N Pi_exact=$exact10_i iters" >&2
    done
  done
done
