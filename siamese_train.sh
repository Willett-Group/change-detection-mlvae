python siamese_train.py \
--dataset mnist \
--epochs 5 \
--n_max 1000 \
--t_max 50 \
--p_max 50 \
--method ruslan \
--arch cnn \
--seed 0

python siamese_train.py \
--dataset cifar10 \
--n_max 1000 \
--t_max 50 \
--p_max 50 \
--method ruslan \
--arch cnn \
--seed 0

python siamese_train.py \
--dataset cifar100 \
--n_max 1000 \
--t_max 50 \
--p_max 50 \
--method ruslan \
--arch cnn \
--seed 0