#!/bin/sh
for i in {0.0002,0.0005,0.001,0.002,0.003,0.004,0.006,0.008,0.016}; do
    for j in {1..50}; do
          qsub -v "var1=$i, var2=$j" qsub_MCMC_addingLinks.pbs
    done
done
