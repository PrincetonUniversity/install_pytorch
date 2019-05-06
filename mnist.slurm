#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
 
env | grep SLURM
env | grep CUDA

module purge
module load anaconda3
module list

conda activate myenv

python mnist_classify.py
