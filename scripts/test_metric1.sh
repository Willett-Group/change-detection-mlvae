#!/bin/sh

#SBATCH --mail-user=renyi@uchicago.edu
#SBATCH --mail-type=ALL

source activate env_me
nvidia-smi
python test_metric.py --dataset cifar10 --method triplet --debug --model_path 'runs/TRAIN/triplet/weights.pt'