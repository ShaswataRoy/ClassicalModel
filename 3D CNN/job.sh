#!/bin/sh -l
# FILENAME:  script_train3dCNN

#SBATCH -A cis220051-gpu
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1     # Number of GPUs per node
#SBATCH --time=47:30:00        # Total run time limit (hh:mm:ss)
#SBATCH -J train          # Job name
#SBATCH -o train.o%j          # Name of stdout output file
#SBATCH -e train.e%j          # +Name of stderr error file
#SBATCH -p gpu                # Queue (partition) name

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

module purge # Unload all loaded modules and reset everything to original state.
module load python
module load anaconda
conda activate 3dcnn

# Run file

python main_train.py --output-path outputs_classic/ 
