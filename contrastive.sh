#!/bin/bash

python contrastive.py --dataset cifar10 --nepochs 50 \
--model contrastive --nclasses 2 --test 1 --test_method 2