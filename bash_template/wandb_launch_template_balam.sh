#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/log_%a.txt
#SBATCH --array=0-19

module load anaconda3/2023.09-0
source activate mix

wandb agent --count 4 rajaonsonella/{project}/{sweep_id}
