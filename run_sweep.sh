#!/bin/bash
# Run full parameter sweep for Pi+Jacobi preconditioner
# Sweeps: o=2,3; r=1,2,3,4; f=1,5,10

cd /mnt/f/optemcode/opt_em_iga_repo
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib

echo "# Sweep Results: Pi_exact + Jacobi Preconditioner"
echo "# Date: $(date)"
echo "# Format: r o f N dim(Pi) none_iters jac_iters lump07 lump10 exact07 exact10"
echo "#"

for r in 1 2 3 4; do
  for o in 2 3; do
    for f in 1 5 10; do
      echo -n "RUNNING: r=$r o=$o f=$f ... " >&2
      output=$(./ams_sweep -r $r -o $o -f $f 2>/dev/null)
      if [ $? -ne 0 ] && [ $? -ne 139 ]; then
        echo "FAILED (exit=$?)" >&2
        continue
      fi
      
      # Parse output
      N=$(echo "$output" | grep "Sweep:" | sed 's/.*N=\s*\([0-9]*\).*/\1/')
      dimPi=$(echo "$output" | grep "Sweep:" | sed 's/.*dim(Pi)=\s*\([0-9]*\).*/\1/')
      none_iters=$(echo "$output" | grep "^│ none" | awk '{print $3}')
      jac_iters=$(echo "$output" | grep "Jacobi ω=1.0" | awk '{print $3}')
      lump07=$(echo "$output" | grep "Pi_lumped+jac ω=0.7" | awk '{print $3}')
      lump10=$(echo "$output" | grep "Pi_lumped+jac ω=1.0" | awk '{print $3}')
      exact07=$(echo "$output" | grep "Pi_exact+jac ω=0.7" | awk '{print $3}')
      exact10=$(echo "$output" | grep "Pi_exact+jac ω=1.0" | awk '{print $3}')
      
      echo "$r $o $f $N $dimPi $none_iters $jac_iters $lump07 $lump10 $exact07 $exact10"
      echo "  done (N=$N, Pi_exact=$exact10 iters)" >&2
    done
  done
done
