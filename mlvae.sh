#!/bin/bash

for b in 1
do
  python mlvae.py --dataset mnist --nepochs 50 --model linearmlvae --cs_dim 50 \
  --beta $b --test 1 --log_file 'log.txt'
done

#for itr in 100
#do
#  python mlvae.py --dataset mnist --nepochs 50 --model linearmlvae --cs_dim 50 \
#   --test 1 --iterations $itr --log_file 'log.txt'
#done