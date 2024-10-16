#!/bin/bash
#SBATCH -c 1                                 # Request one core
#SBATCH -N 1                                 # Request one node (if you request more t$
                                             # -N 1 means all cores will be on the sam$
#SBATCH -t 0-00:10                           # Runtime in D-HH:MM format
#SBATCH -p short                             # Partition to run in
#SBATCH --mem=8G                             # Memory total in MB (for all cores)
#SBATCH -o atl7_%j.out                       # File to which STDOUT will be written, i$
#SBATCH -e atl7_%j.err                       # File to which STDERR will be written, i$
#SBATCH --mail-type=ALL                      # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=atlandau@g.harvard.edu   # Email to which notifications will be se$

module load matlab/2020b
matlab -nodesktop -r "runo2_corrD($1,$2,$3)"



  

