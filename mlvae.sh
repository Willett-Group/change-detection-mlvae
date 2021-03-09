#!/bin/bash

for b in 0.00001 0.001 0.1 1 10 100
do
  python mlvae.py --dataset celeba --model linearmlvae --cs_dim 50 --end_epoch 50 --beta $b --log_file 'log.txt'
done