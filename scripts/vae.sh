#!/bin/bash
for b1 in 0.1 0.01
do
  python vae.py --dataset cifar10 --model linearvae --cs_dim 50 --train 1 --test 1 --end_epoch 100 --beta1 $b1 --log_file 'log.txt'
done