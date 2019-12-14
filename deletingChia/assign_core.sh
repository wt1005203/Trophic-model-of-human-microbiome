#!/bin/sh
for i in {1..10}; do
    for j in {1..10}; do
            qsub -v "var1=$i, var2=$j" qsub_MCMC_addingLinks.pbs
    done
done
