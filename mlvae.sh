#!/bin/bash
for b1 in 1
do
  for b2 in 1
  do
    python mlvae.py --dataset cifar10 --model linearmlvae --cs_dim 50 --end_epoch 50 --beta1 $b1 --beta2 $b2 --log_file 'log.txt'
  done
done