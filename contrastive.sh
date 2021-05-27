#!/bin/bash

python contrastive.py --dataset cifar10 --nepochs 50 --batch_size 256 \
--model contrastive --test 0 --test_method 2