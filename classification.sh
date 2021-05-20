#!/bin/bash

python classification.py --dataset cifar10 --nepochs 50 \
--model linearclassifier --nclasses 10 --test 1 --test_method graph-cut