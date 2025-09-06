#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=slurm/%j.out

source ~/.bashrc
conda activate quetzal

wandb offline

srun --nodes=1 --tasks-per-node=4 python train.py --devices=4 --name=geom_run --dataset=geom --vis_every_n_epochs=1 --lr=2e-4 --sigma_data=2.5 --bsz=40 --packlen=512 --packdepth=10
