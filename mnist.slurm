#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --job-name="pytorch_tutorial"
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=YourNetID@princeton.edu
 
module purge
module load anaconda3
module list

conda activate torch-env

srun python mnist_classify.py
