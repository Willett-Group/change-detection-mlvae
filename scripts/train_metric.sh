#!/bin/sh

#SBATCH --mail-user=renyi@uchicago.edu
#SBATCH --mail-type=ALL

conda activate env_me
nvidia-smi

cd ..
python train_metric.py --dataset cifar10 --method triplet --epochs 500 --batch_size 96 --num_workers 4