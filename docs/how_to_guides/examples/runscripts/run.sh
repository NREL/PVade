#!/bin/bash

#SBATCH --ntasks-per-node=104
#SBATCH --time=08:00:00
#SBATCH --account=pvopt
#SBATCH --job-name=jobname
#SBATCH --nodes=1
#SBATCH --error=jobname.err
#SBATCH --output=jobname.out
#SBATCH --mem=0
#SBATCH --exclusive

module purge
module load mamba
mamba activate pvade
export OMP_NUM_THREADS=1

mpirun -np 104 python -u /home/bstanisl/pvade/PVade/pvade_main.py --input_file /home/bstanisl/pvade/PVade/input/<input_yaml_file>
