#!/bin/sh
for i in {1e-4,1e-3,3e-3,1e-2,1e-1}; do
    for j in {1e-4,1e-3,3e-3,1e-2,1e-1}; do
        for k in {1..50}; do
            qsub -v "var1=$i,var2=$j,var3=$k" qsub_MCMC_addingLinks.pbs
        done
    done
done
