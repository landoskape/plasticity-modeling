#!/bin/bash

# Run Parameters
nRuns=10

# Simulation Parameters
nDepression = 3

for (( rnIdx=1; rnIdx<=nRuns; rnIdx++ ))
do
    for (( nd=1; nd<=$nDepression; nd++ ))
    do
        sbatch o2Instructions_poiraziBottomTop.sh $rnIdx $nd
    done
done