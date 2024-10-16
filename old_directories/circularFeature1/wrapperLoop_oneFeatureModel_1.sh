#!/bin/bash

# Run Parameters
nRuns=5

# Simulation Parameters (tuning of inputs)
nTuning=4

for (( ts=1; ts<=$nTuning; t++ ))
do
    for (( run=1; run<=$nRuns; run++ ))
    do
        sbatch oneFeatureModel_1.sh $run $ts
    done
done
