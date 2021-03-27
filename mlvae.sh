#!/bin/bash

for b in 0.0000001 0.00001 0.001 0.1 1 100
#for b in 0.1
do
  python mlvae.py --dataset cifar10 --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
  --val_period 100 --log_file 'log.txt'
done

#for b in 100 1 0.1 0.001 0.00001 0.0000001
#do
#  python mlvae.py --dataset cifar10 --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
#  --val_period 100 --log_file 'log.txt'
#done
#
#for b in 100 1 0.1 0.001 0.00001 0.0000001
#do
#  python mlvae.py --dataset celeba --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b \
#  --val_period 100 --log_file 'log.txt'
#done
#
#for b in 100 1 0.1 0.001 0.00001 0.0000001
#do
#  python mlvae.py --dataset clevr --model linearmlvae --cs_dim 100 --end_epoch 50 --beta $b \
#  --val_period 100 --log_file 'log.txt'
#done