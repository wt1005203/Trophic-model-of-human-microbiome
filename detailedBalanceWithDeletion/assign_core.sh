#!/bin/sh
for i in {1..100}; do
          qsub -v "var1=$i" qsub_MCMC_addingLinks.pbs
done
