#!/bin/bash

# Run Parameters
nRuns=1 #5

# Simulation Parameters
nApicDep=1 #4
nState=1 #4
basalFollow=0 #0&1

for (( rnIdx=1; rnIdx<=nRuns; rnIdx++ ))
do
    for (( nd=1; nd<=$nApicDep; nd++ ))
    do
        for (( ns=1; ns<=$nState; ns++ ))
        do
            for (( bf=1; bf<=$basalFollow; bf++ ))
            do
                sbatch o2Instructions_corrD.sh $rnIdx $nd $ns $bf
            done
        done
    done
done