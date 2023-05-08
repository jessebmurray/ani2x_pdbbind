#!/bin/bash
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # 1 tasks
#SBATCH --gres=gpu:1                    # number of gpus
#SBATCH --gpus-per-task=1               # number of gpus per task
#SBATCH --cpus-per-gpu=1                # number of cores per gpu
#SBATCH --mem-per-cpu=10000             # memory/cpu (in MB)
#SBATCH -J gen_frozen
#SBATCH --mail-user=jesse.murray@stats.ox.ac.uk
#S BATCH --nodelist=nagagpu02.cpu.stats.ox.ac.uk
#SBATCH --clusters srf_gpu_01 -w nagagpu02.cpu.stats.ox.ac.uk
#SBATCH --partition=high-opig-gpu

#SBATCH --output=/data/localhost/not-backed-up/jemurray/jobs/slurm_%j.out  # Writes standard output to this file. %j is jobnumber
#SBATCH --error=/data/localhost/not-backed-up/jemurray/errors/slurm_%j.out   # Writes error messages to this file. %j is jobnumber


source ~/.bashrc

conda activate

python /data/localhost/not-backed-up/jemurray/ani2x_pdbbind/cont_frozen.py

