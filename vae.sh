#!/bin/bash
for b1 in 1
do
  python vae.py --dataset cifar10 --model linearvae --cs_dim 10 --train 1 --test 1 --end_epoch 100 --beta1 $b1 --log_file 'log.txt'
done