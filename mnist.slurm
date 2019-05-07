#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
 
module purge
module load anaconda3
module list

conda activate pytorch_tutorial

python mnist_classify.py
