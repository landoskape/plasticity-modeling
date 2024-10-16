#!/bin/bash

# Run Parameters
nRuns=1 #5

# Simulation Parameters
nApicDep=1 #4
nState=1 #4


for (( rnIdx=1; rnIdx<=nRuns; rnIdx++ ))
do
    for (( nd=1; nd<=$nApicDep; nd++ ))
    do
        for (( ns=1; ns<=$nState; ns++ ))
        do
            sbatch o2Instructions_corrD.sh $rnIdx $nd $ns
        done
    done
done