#!/bin/bash

# Run Parameters
nRuns=1 #5

# Simulation Parameters
nApicDep=1 #4
nEdgeProb=1 #4


for (( rnIdx=1; rnIdx<=nRuns; rnIdx++ ))
do
    for (( nd=1; nd<=$nApicDep; nd++ ))
    do
        for (( ne=1; ne<=$nEdgeProb; ne++ ))
        do
            sbatch o2Instructions_poiraziBottomTop.sh $rnIdx $nd $ne
        done
    done
done