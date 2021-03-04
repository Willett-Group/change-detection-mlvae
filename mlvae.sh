#!/bin/bash

for b1 in 0.1 1 10 100
do
  for b2 in 0.1 1 10 100
  if [$b1 == $b2]; then
    do
      python mlvae.py --dataset cifar10 --model linearmlvae --cs_dim 50 --end_epoch 200 --beta1 $b1 --beta2 $b2 --log_file 'log.txt'
    done
done