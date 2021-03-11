#!/bin/bash

for b in 1
do
  python mlvae.py --dataset celeba --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
  --val_period 100 --log_file 'log.txt'
done

for b in 1
do
  python mlvae.py --dataset clevr --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
  --val_period 100 --log_file 'log.txt'
done

for b in 1
do
  python mlvae.py --dataset cifar10 --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
  --val_period 100 --log_file 'log.txt'
done