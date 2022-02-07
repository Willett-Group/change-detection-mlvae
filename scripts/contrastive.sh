#!/bin/bash
nvidia-smi
conda activate env_darts

python contrastive.py --dataset cifar10 --nepochs 50 --batch_size 32 --model triplet --test 1