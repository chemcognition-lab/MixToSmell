#!/bin/bash
#SBATCH --account=aip-aspuru
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-20:00           # time (DD-HH:MM)
#SBATCH --gpus-per-node=1


module load python/3.11.5 StdEnv/2023
module load gcc arrow
module load rdkit/2024.09.6     # required if not using conda
module load cuda/12.6
# module load shared nccl2-cuda12.6-gcc11/2.23.4

source ~/env/dream/bin/activate

wandb agent --count 200 gkwt/{project}/{sweep_id}
