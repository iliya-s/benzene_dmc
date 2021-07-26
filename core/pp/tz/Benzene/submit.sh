#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ilsa8974@colorado.edu
##SBATCH -A ucb-summit-sha
#SBATCH --qos blanca-sha
#SBATCH --job-name Benzene
#SBATCH --nodes 1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --export=NONE

python benzene.py > benzene.out
