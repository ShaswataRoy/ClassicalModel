#!/usr/bin/bash
# FILENAME:  chimera

#SBATCH -t 00:45:00
#SBATCH --nodes=2
#SBATCH --ntasks=256 


timestamp=$(date +%b_%d_%y_%H_%M_%e)

singularity exec parallel.sif bash job.sh>new.out
echo "done."