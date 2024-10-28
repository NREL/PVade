#!/bin/bash

#SBATCH --ntasks-per-node=104
#SBATCH --time=08:00:00
#SBATCH --account=csphfm
#SBATCH --job-name=heliostat_u2
#SBATCH --nodes=1
#SBATCH --error=heliostat-0.err
#SBATCH --output=heliostat-0.out
#SBATCH --mem=0
#SBATCH --exclusive

module purge
module load mamba
mamba activate pvade
export OMP_NUM_THREADS=1

mpirun -np 104 python -u /home/bstanisl/pvade/PVade/ns_main.py --input_file /home/bstanisl/pvade/PVade/input/single_heliostat.yaml
